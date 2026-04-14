"""
전체 전략 일괄 백테스트 실행 + 결과 히스토리 저장.

모든 전략 × 모든 데이터(타임프레임) 조합을 백테스트하고,
결과를 reports/ 폴더에 날짜별로 저장한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import (
    BacktestEnv, BasicSlippage, FixedRateCommission,
    PercentSizer, run_backtest, compute_drawdown,
)
from strategies import get_strategy, list_strategies


# =============================
# 설정
# =============================

# 백테스트 대상 데이터 파일들 (4개 코인 x 3개 타임프레임)
COINS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
TIMEFRAMES = ["1h", "4h", "1d"]
DATA_FILES = [
    f"data/ohlcv_full_{coin}_{tf}.parquet"
    for coin in COINS
    for tf in TIMEFRAMES
]

# 백테스트 환경: 자본 비례 사이징 (15%)
DEFAULT_ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=10),
    sizer=PercentSizer(percent=0.15),   # 가용 현금의 15% 투입
)

# 결과 저장 디렉토리
REPORT_DIR = Path("reports")


def load_data(path: str) -> pd.DataFrame:
    """Parquet 데이터 로드."""
    df = pd.read_parquet(path).set_index("time")[
        ["open", "high", "low", "close", "volume"]
    ]
    return df


def run_single_backtest(strategy_name: str, data_path: str, env: BacktestEnv) -> dict:
    """단일 전략/데이터 조합 백테스트 실행."""
    df = load_data(data_path)
    strat = get_strategy(strategy_name)

    equity, report = run_backtest(df, strat, env)
    metrics = report.summary()

    # 추가 통계
    trades = report.trades
    n_trades = len(trades)
    n_buys = len(trades[trades["side"] == "buy"]) if n_trades > 0 else 0
    n_sells = len(trades[trades["side"] == "sell"]) if n_trades > 0 else 0
    total_fees = trades["fee"].sum() if n_trades > 0 else 0

    # 타임프레임 추출 (파일명에서)
    tf = data_path.split("_")[-1].replace(".parquet", "")

    return {
        "strategy": strategy_name,
        "data_file": data_path,
        "timeframe": tf,
        "candles": len(df),
        "period": f"{df.index[0]} ~ {df.index[-1]}",
        "initial_cash": env.cash,
        "final_equity": float(equity.iloc[-1]),
        "cum_return": metrics["cum_return"],
        "ann_return": metrics["ann_return"],
        "ann_vol": metrics["ann_vol"],
        "sharpe": metrics["sharpe"],
        "mdd": metrics["mdd"],
        "total_trades": n_trades,
        "buys": n_buys,
        "sells": n_sells,
        "total_fees": float(total_fees),
    }


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    strategies = list_strategies()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    all_results = []

    print("=" * 70)
    print(f"전체 백테스트 실행 -{timestamp}")
    print(f"전략: {strategies}")
    print(f"데이터: {DATA_FILES}")
    print("=" * 70)

    for data_path in DATA_FILES:
        if not Path(data_path).exists():
            print(f"\n[스킵] 데이터 없음: {data_path}")
            continue

        for strat_name in strategies:
            print(f"\n--- {strat_name} / {data_path} ---")
            try:
                result = run_single_backtest(strat_name, data_path, DEFAULT_ENV)
                all_results.append(result)

                # 콘솔 출력
                print(f"  누적 수익률:  {result['cum_return']:>8.2%}")
                print(f"  연환산 수익률:{result['ann_return']:>8.2%}")
                print(f"  샤프 비율:    {result['sharpe']:>8.2f}")
                print(f"  MDD:          {result['mdd']:>8.2%}")
                print(f"  거래 횟수:    {result['total_trades']:>8d}")

            except Exception as e:
                print(f"  [에러] {e}")
                all_results.append({
                    "strategy": strat_name,
                    "data_file": data_path,
                    "timeframe": data_path.split("_")[-1].replace(".parquet", ""),
                    "error": str(e),
                })

    # 결과를 DataFrame으로 정리
    df_results = pd.DataFrame(all_results)

    # 파일 저장 (CSV + JSON)
    csv_path = REPORT_DIR / f"backtest_results_{timestamp}.csv"
    json_path = REPORT_DIR / f"backtest_results_{timestamp}.json"

    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 70)
    print("결과 저장 완료")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 70)

    # 요약 테이블 출력
    print("\n[ 전략별 성과 요약 ]")
    print("-" * 90)
    print(f"{'전략':<22} {'TF':<5} {'누적수익률':>10} {'연수익률':>10} {'샤프':>7} {'MDD':>9} {'거래수':>7}")
    print("-" * 90)

    for r in all_results:
        if "error" in r:
            print(f"{r['strategy']:<22} {r['timeframe']:<5} {'에러':>10}")
            continue
        print(
            f"{r['strategy']:<22} {r['timeframe']:<5} "
            f"{r['cum_return']:>9.2%} "
            f"{r['ann_return']:>9.2%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['mdd']:>8.2%} "
            f"{r['total_trades']:>7d}"
        )

    print("-" * 90)


if __name__ == "__main__":
    main()
