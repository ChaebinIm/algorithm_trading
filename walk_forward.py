"""
Walk-Forward 검증 스크립트 — 과최적화(Overfitting) 방지.

핵심 아이디어:
백테스트 결과가 실제 수익과 다른 가장 큰 이유는 '과최적화'다.
전체 데이터로 파라미터를 최적화하면, 그 파라미터는 과거에만 맞는 값일 수 있다.

Walk-Forward 방식:
  - 데이터를 시간 순서대로 Train / Test로 분할
  - Train 구간: 파라미터 결정 / Test 구간: 실제 성과 측정
  - 복수의 윈도우로 반복하여 강건성 확인

분할 방식:
  [========Train========][===Test===]   윈도우 1
          [========Train========][===Test===]  윈도우 2 (롤링)

설정:
  - 기본 Train/Test 비율: 70% / 30%
  - 고정 분할(최신 30%만 테스트) + 롤링 분할(2개 윈도우) 모두 실행

테스트 대상 전략:
  - supertrend (st_multiplier=2.5)
  - supertrend_ensemble
  - ensemble

실행:
  python walk_forward.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import BacktestEnv, BasicSlippage, FixedRateCommission, PercentSizer, run_backtest
from strategies import get_strategy


# =============================
# 설정
# =============================

ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=5),
    sizer=PercentSizer(percent=0.20),
)

TEST_STRATEGIES = [
    ("supertrend", {}),
    ("supertrend_ensemble", {}),
    ("ensemble", {}),
    ("adaptive_regime", {}),
    ("macd_volume", {}),
    ("champion", {}),
    ("champion_v2", {}),
]

TEST_DATA = [
    # Binance USDT (메인)
    "data/ohlcv_full_BTCUSDT_4h.parquet",
    "data/ohlcv_full_BTCUSDT_1d.parquet",
    "data/ohlcv_full_ETHUSDT_4h.parquet",
    "data/ohlcv_full_SOLUSDT_4h.parquet",
    "data/ohlcv_full_SOLUSDT_1d.parquet",
    "data/ohlcv_full_XRPUSDT_4h.parquet",
    "data/ohlcv_full_XRPUSDT_1d.parquet",
    "data/ohlcv_full_BNBUSDT_4h.parquet",
    # Upbit KRW (기존)
    "data/ohlcv_full_BTCKRW_4h.parquet",
    "data/ohlcv_full_BTCKRW_1d.parquet",
]

TRAIN_RATIO = 0.70   # 70% train, 30% test


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df = df.set_index("time")
    return df[["open", "high", "low", "close", "volume"]]


def run_split(df: pd.DataFrame, strat_name: str, strat_kwargs: dict, label: str) -> dict:
    """단일 데이터셋으로 백테스트 실행 후 결과 반환."""
    try:
        strat = get_strategy(strat_name, **strat_kwargs)
        eq, report = run_backtest(df, strat, ENV)
        m = report.summary()
        return {
            "label": label,
            "n_candles": len(df),
            "period": f"{df.index[0].date()} ~ {df.index[-1].date()}",
            "cum_return": m["cum_return"],
            "ann_return": m["ann_return"],
            "sharpe": m["sharpe"],
            "calmar": m.get("calmar", float("nan")),
            "mdd": m["mdd"],
            "win_rate": m.get("win_rate", float("nan")),
            "profit_factor": m.get("profit_factor", float("nan")),
            "total_roundtrips": m.get("total_roundtrips", 0),
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def walk_forward_fixed(df: pd.DataFrame, strat_name: str, strat_kwargs: dict) -> dict:
    """고정 분할: 마지막 30%를 Out-of-Sample(OOS) 테스트."""
    n = len(df)
    split = int(n * TRAIN_RATIO)

    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    train_result = run_split(train_df, strat_name, strat_kwargs, "train (in-sample)")
    test_result = run_split(test_df, strat_name, strat_kwargs, "test (out-of-sample)")

    return {"train": train_result, "test": test_result}


def walk_forward_rolling(df: pd.DataFrame, strat_name: str, strat_kwargs: dict) -> list:
    """롤링 분할: 2개 윈도우로 검증."""
    n = len(df)
    results = []

    # 윈도우 1: 앞 50% train, 다음 25% test
    s1 = int(n * 0.50)
    e1 = int(n * 0.75)
    r1_train = run_split(df.iloc[:s1], strat_name, strat_kwargs, "window1_train")
    r1_test = run_split(df.iloc[s1:e1], strat_name, strat_kwargs, "window1_test")
    results.append({"window": 1, "train": r1_train, "test": r1_test})

    # 윈도우 2: 앞 65% train, 다음 25% test
    s2 = int(n * 0.65)
    e2 = int(n * 0.90)
    r2_train = run_split(df.iloc[:s2], strat_name, strat_kwargs, "window2_train")
    r2_test = run_split(df.iloc[s2:e2], strat_name, strat_kwargs, "window2_test")
    results.append({"window": 2, "train": r2_train, "test": r2_test})

    return results


def _fmt(v, pct=False):
    if v != v or v is None:
        return "N/A"
    if pct:
        return f"{v:+.1%}"
    return f"{v:.2f}"


def main():
    print("=" * 75)
    print("Walk-Forward 검증 — In-Sample vs Out-of-Sample")
    print("=" * 75)

    all_results = []

    for data_path in TEST_DATA:
        if not Path(data_path).exists():
            print(f"\n[스킵] {data_path} 없음")
            continue

        df = load_data(data_path)
        coin_tf = Path(data_path).stem.replace("ohlcv_full_", "")
        n = len(df)
        split_n = int(n * TRAIN_RATIO)

        print(f"\n{'='*75}")
        print(f"데이터: {coin_tf}  ({n}개 캔들)")
        print(f"  Train: {df.index[0].date()} ~ {df.index[split_n].date()} ({split_n}개)")
        print(f"  Test:  {df.index[split_n].date()} ~ {df.index[-1].date()} ({n-split_n}개)")
        print(f"{'='*75}")

        print(f"\n  {'전략':<24} {'구분':<22} {'수익률':>9} {'샤프':>7} {'MDD':>8} {'승률':>7} {'PF':>6}")
        print(f"  {'-'*80}")

        for strat_name, strat_kwargs in TEST_STRATEGIES:
            wf = walk_forward_fixed(df, strat_name, strat_kwargs)

            for split_label, result in [("in-sample", wf["train"]), ("out-of-sample", wf["test"])]:
                if "error" in result:
                    print(f"  {strat_name:<24} {split_label:<22} 에러: {result['error']}")
                    continue
                cr = result["cum_return"]
                sh = result["sharpe"]
                mdd = result["mdd"]
                wr = result.get("win_rate", float("nan"))
                pf = result.get("profit_factor", float("nan"))
                flag = " <OOS>" if split_label == "out-of-sample" else ""
                wr_s = f"{wr:.1%}" if wr == wr else "N/A"
                pf_s = f"{min(pf, 99):.2f}" if pf == pf else "N/A"
                print(f"  {strat_name:<24} {split_label:<22} {cr:>+8.1%} {sh:>7.2f} {mdd:>7.1%} {wr_s:>7} {pf_s:>6}{flag}")

            all_results.append({
                "data": coin_tf,
                "strategy": strat_name,
                "fixed_split": wf,
            })
            print()

    # 롤링 검증 요약
    print("\n" + "=" * 75)
    print("롤링 윈도우 검증 — BTC 4h + ETH 1h (핵심 데이터셋)")
    print("=" * 75)

    for data_path in [
        "data/ohlcv_full_BTCUSDT_4h.parquet",
        "data/ohlcv_full_SOLUSDT_4h.parquet",
        "data/ohlcv_full_BTCKRW_4h.parquet",
    ]:
        if not Path(data_path).exists():
            continue
        df = load_data(data_path)
        coin_tf = Path(data_path).stem.replace("ohlcv_full_", "")

        print(f"\n[ {coin_tf} ]")
        for strat_name, strat_kwargs in [("supertrend", {}), ("ensemble", {}), ("supertrend_ensemble", {}), ("champion", {}), ("champion_v2", {})]:
            windows = walk_forward_rolling(df, strat_name, strat_kwargs)
            oos_sharpes = []
            for w in windows:
                ts = w["test"]
                if "sharpe" in ts:
                    oos_sharpes.append(ts["sharpe"])
            avg_oos = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else float("nan")
            oos_s = f"{avg_oos:.2f}" if avg_oos == avg_oos else "N/A"

            detail = []
            for w in windows:
                tr = w["train"]
                ts = w["test"]
                if "sharpe" not in tr or "sharpe" not in ts:
                    continue
                detail.append(f"W{w['window']}: IS={tr['sharpe']:.2f}/OOS={ts['sharpe']:.2f}")
            detail_str = " | ".join(detail)
            print(f"  {strat_name:<24} OOS 평균샤프: {oos_s}   ({detail_str})")

    # 결과 저장
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path("reports") / f"walk_forward_{timestamp}.json"
    Path("reports").mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n결과 저장: {out_path}")
    print("\n" + "=" * 75)
    print("검증 완료 — OOS 수익률 양수 전략만 실전 사용 권장")
    print("=" * 75)


if __name__ == "__main__":
    main()
