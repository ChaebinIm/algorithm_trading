"""
전체 전략 일괄 백테스트 실행 + 결과 히스토리 저장.

모든 전략 × 사용 가능한 데이터 조합을 백테스트하고,
결과를 reports/ 폴더에 날짜별로 저장한다.

성과 지표:
- cum_return: 누적 수익률
- ann_return: 연환산 수익률
- sharpe: 샤프 비율
- calmar: Calmar 비율 (연수익률 / MDD)
- mdd: 최대 낙폭
- win_rate: 승률
- profit_factor: Profit Factor
- total_roundtrips: 라운드트립 거래 수
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import (
    BacktestEnv, BasicSlippage, FixedRateCommission,
    PercentSizer, run_backtest,
)
from strategies import get_strategy, list_strategies


# =============================
# 설정
# =============================

# 존재하는 데이터 파일을 자동 탐색
DATA_DIR = Path("data")

def find_data_files(timeframes=("1h", "4h", "1d")):
    """data/ 폴더에서 지정된 타임프레임의 parquet 파일 목록 반환."""
    files = []
    for tf in timeframes:
        for f in sorted(DATA_DIR.glob(f"*_{tf}.parquet")):
            files.append(str(f))
    return files


# 백테스트 환경: 자본 비례 사이징 (20%)
# 수수료는 바이낸스 기본 0.1% = 10bps (보수적 가정)
DEFAULT_ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=10),   # 0.1% (바이낸스 기본)
    sizer=PercentSizer(percent=0.20),         # 가용 현금의 20% 투입
)

REPORT_DIR = Path("reports")


def load_data(path: str) -> pd.DataFrame:
    """Parquet 데이터 로드."""
    df = pd.read_parquet(path)
    # 인덱스 컬럼명 확인 후 처리
    if "time" in df.columns:
        df = df.set_index("time")
    elif df.index.name != "time":
        df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]]


def run_single_backtest(strategy_name: str, data_path: str, env: BacktestEnv) -> dict:
    """단일 전략/데이터 조합 백테스트 실행."""
    df = load_data(data_path)
    strat = get_strategy(strategy_name)

    equity, report = run_backtest(df, strat, env)
    metrics = report.summary()

    trades = report.trades
    n_trades = len(trades)
    n_buys = len(trades[trades["side"] == "buy"]) if n_trades > 0 else 0
    n_sells = len(trades[trades["side"] == "sell"]) if n_trades > 0 else 0
    total_fees = trades["fee"].sum() if n_trades > 0 else 0

    # 파일명에서 타임프레임 추출
    tf = Path(data_path).stem.split("_")[-1]
    coin = Path(data_path).stem.replace("ohlcv_full_", "").replace(f"_{tf}", "")

    return {
        "strategy": strategy_name,
        "data_file": data_path,
        "coin": coin,
        "timeframe": tf,
        "candles": len(df),
        "period": f"{df.index[0]} ~ {df.index[-1]}",
        "initial_cash": env.cash,
        "final_equity": float(equity.iloc[-1]),
        "cum_return": metrics["cum_return"],
        "ann_return": metrics["ann_return"],
        "ann_vol": metrics["ann_vol"],
        "sharpe": metrics["sharpe"],
        "calmar": metrics.get("calmar", float("nan")),
        "mdd": metrics["mdd"],
        "win_rate": metrics.get("win_rate", float("nan")),
        "profit_factor": metrics.get("profit_factor", float("nan")),
        "avg_win": metrics.get("avg_win", float("nan")),
        "avg_loss": metrics.get("avg_loss", float("nan")),
        "total_roundtrips": metrics.get("total_roundtrips", 0),
        "total_trades": n_trades,
        "buys": n_buys,
        "sells": n_sells,
        "total_fees": float(total_fees),
    }


def main():
    REPORT_DIR.mkdir(exist_ok=True)

    strategies = list_strategies()
    data_files = find_data_files(timeframes=("1h", "4h", "1d"))

    if not data_files:
        print("데이터 파일을 찾을 수 없습니다. collect_data.py를 먼저 실행해주세요.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    all_results = []

    print("=" * 80)
    print(f"전체 백테스트 실행 - {timestamp}")
    print(f"전략: {strategies}")
    print(f"데이터 파일 {len(data_files)}개")
    print("=" * 80)

    for data_path in data_files:
        print(f"\n{'='*40}")
        print(f"데이터: {data_path}")
        print(f"{'='*40}")

        for strat_name in strategies:
            print(f"  [{strat_name}]", end=" ")
            try:
                result = run_single_backtest(strat_name, data_path, DEFAULT_ENV)
                all_results.append(result)

                cum = result["cum_return"]
                sharpe = result["sharpe"]
                mdd = result["mdd"]
                wr = result["win_rate"]
                pf = result["profit_factor"]

                status = "✓" if cum > 0 else "✗"
                pf_str = f"{pf:.2f}" if pf != float("inf") and pf == pf else ("inf" if pf == float("inf") else "N/A")
                wr_str = f"{wr:.1%}" if wr == wr else "N/A"

                print(f"{status} 수익률:{cum:+.1%} 샤프:{sharpe:.2f} MDD:{mdd:.1%} 승률:{wr_str} PF:{pf_str}")

            except Exception as e:
                print(f"[에러] {e}")
                all_results.append({
                    "strategy": strat_name,
                    "data_file": data_path,
                    "timeframe": Path(data_path).stem.split("_")[-1],
                    "error": str(e),
                })

    # 결과 저장
    df_results = pd.DataFrame(all_results)

    csv_path = REPORT_DIR / f"backtest_results_{timestamp}.csv"
    json_path = REPORT_DIR / f"backtest_results_{timestamp}.json"

    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 80)
    print("결과 저장 완료")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print("=" * 80)

    # 성과 요약 — 수익률 양수인 것만 상위 정렬
    valid = [r for r in all_results if "error" not in r]
    valid.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    print("\n[ 전략 성과 랭킹 — 샤프 비율 기준 ]\n")
    header = (
        f"{'순위':>4} {'전략':<22} {'코인':<12} {'TF':<5} "
        f"{'수익률':>9} {'샤프':>7} {'Calmar':>7} {'MDD':>8} "
        f"{'승률':>7} {'PF':>6} {'거래수':>7}"
    )
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(valid[:30], 1):
        wr = r.get("win_rate", float("nan"))
        pf = r.get("profit_factor", float("nan"))
        calmar = r.get("calmar", float("nan"))
        wr_str = f"{wr:.1%}" if wr == wr else "N/A"
        pf_str = f"{min(pf, 99.9):.2f}" if pf == pf else "N/A"
        calmar_str = f"{calmar:.2f}" if calmar == calmar else "N/A"

        flag = " <<" if r.get("cum_return", 0) > 0.3 else ""
        print(
            f"{rank:>4} {r['strategy']:<22} {r.get('coin','?'):<12} {r['timeframe']:<5} "
            f"{r['cum_return']:>+8.1%} {r['sharpe']:>7.2f} {calmar_str:>7} {r['mdd']:>7.1%} "
            f"{wr_str:>7} {pf_str:>6} {r.get('total_roundtrips', 0):>7}{flag}"
        )

    print("-" * len(header))
    print("\n<< 표시: 누적 수익률 +30% 이상 전략")


if __name__ == "__main__":
    main()
