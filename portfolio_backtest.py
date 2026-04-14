"""
포트폴리오 백테스트 — 최강 전략 조합 시뮬레이션.

Walk-Forward OOS 검증 통과 전략:
  1. ensemble   / BTC 4h  → OOS Sharpe 2.20  (rolling avg 2.36)
  2. champion   / BTC 4h  → OOS Sharpe 1.07  (rolling avg 2.19, IS<OOS)
  3. champion_v2/ BTC 1d  → OOS Sharpe 1.51
  4. champion   / XRP 1d  → OOS Sharpe 1.48
  5. macd_volume/ BTC 1d  → OOS Sharpe 1.31
  6. supertrend_ensemble / BTC 1d → OOS Sharpe 1.39

포트폴리오 A (3전략 분산):
  - 40% ensemble/BTC 4h
  - 30% champion_v2/BTC 1d
  - 30% champion/XRP 1d

포트폴리오 B (2전략 집중):
  - 60% ensemble/BTC 4h
  - 40% champion_v2/BTC 1d

실행:
  python portfolio_backtest.py
"""
from __future__ import annotations

from pathlib import Path
from datetime import timezone

import numpy as np
import pandas as pd

from core import BacktestEnv, BasicSlippage, FixedRateCommission, PercentSizer, run_backtest
from strategies import get_strategy


ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=5),
    sizer=PercentSizer(percent=0.20),
)


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df[["open", "high", "low", "close", "volume"]]


def run_strategy(path: str, strat_name: str, weight: float, total_capital: float) -> pd.DataFrame:
    """단일 전략 백테스트 → 일별 자본 변화 DataFrame 반환."""
    df = load_df(path)
    allocated = total_capital * weight

    env = BacktestEnv(
        cash=allocated,
        slippage=BasicSlippage(bps=5),
        commission=FixedRateCommission(bps=5),
        sizer=PercentSizer(percent=0.20),
    )

    strat = get_strategy(strat_name)
    eq_series, report = run_backtest(df, strat, env)

    # 일별로 리샘플링
    eq_daily = eq_series.resample("1D").last().ffill()
    return eq_daily, report


def combine_portfolios(components: list[tuple[str, str, float]], total_capital: float = 100_000) -> pd.DataFrame:
    """여러 전략을 병합하여 포트폴리오 자본 곡선 반환.

    components: [(data_path, strat_name, weight), ...]
    """
    eq_curves = []
    reports = []

    for path, strat_name, weight in components:
        if not Path(path).exists():
            print(f"  [경고] {path} 없음 — 스킵")
            continue
        eq, report = run_strategy(path, strat_name, weight, total_capital)
        eq_curves.append(eq)
        reports.append((strat_name, weight, report))
        allocated = total_capital * weight
        m = report.summary()
        print(f"    {strat_name:<16} ({weight:.0%}) → 수익률: {m['cum_return']:+.1%}  샤프: {m['sharpe']:.2f}  MDD: {m['mdd']:.1%}")

    if not eq_curves:
        return None, None

    # 각 전략의 자본 변화를 합산
    combined = pd.concat(eq_curves, axis=1).ffill()
    combined.columns = [f"strat_{i}" for i in range(len(combined.columns))]
    portfolio_equity = combined.sum(axis=1)

    return portfolio_equity, reports


def portfolio_metrics(equity: pd.Series, total_capital: float) -> dict:
    """포트폴리오 종합 성과 지표."""
    returns = equity.pct_change().dropna()

    cum_return = (equity.iloc[-1] - total_capital) / total_capital

    # 연간화 (거래일 기준 365일)
    n_days = (equity.index[-1] - equity.index[0]).days
    ann_return = (equity.iloc[-1] / total_capital) ** (365 / n_days) - 1

    # Sharpe (연간)
    daily_std = returns.std()
    sharpe = (returns.mean() / (daily_std + 1e-10)) * np.sqrt(365)

    # MDD
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    mdd = drawdown.min()

    # Calmar
    calmar = ann_return / abs(mdd) if mdd != 0 else float("nan")

    return {
        "cum_return": cum_return,
        "ann_return": ann_return,
        "sharpe": sharpe,
        "mdd": mdd,
        "calmar": calmar,
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
    }


def main():
    TOTAL = 100_000  # 총 자본 (100K KRW equivalent units)

    print("=" * 70)
    print("포트폴리오 백테스트 — Walk-Forward 검증 통과 전략 조합")
    print("=" * 70)

    # ─── 포트폴리오 A: 3전략 분산 ───────────────────────────────────────
    print("\n[ 포트폴리오 A ] 3전략 분산 (BTC 4h + BTC 1d + XRP 1d)")
    print("-" * 70)
    portfolio_a = [
        ("data/ohlcv_full_BTCKRW_4h.parquet",  "ensemble",    0.40),
        ("data/ohlcv_full_BTCKRW_1d.parquet",  "champion_v2", 0.30),
        ("data/ohlcv_full_XRPKRW_1d.parquet",  "champion",    0.30),
    ]
    eq_a, reps_a = combine_portfolios(portfolio_a, TOTAL)

    if eq_a is not None:
        m_a = portfolio_metrics(eq_a, TOTAL)
        print(f"\n  포트폴리오 A 합산:")
        print(f"    기간: {m_a['start']} ~ {m_a['end']}")
        print(f"    누적수익: {m_a['cum_return']:+.1%}")
        print(f"    연수익:   {m_a['ann_return']:+.1%}")
        print(f"    샤프:     {m_a['sharpe']:.2f}")
        print(f"    MDD:      {m_a['mdd']:.1%}")
        print(f"    Calmar:   {m_a['calmar']:.2f}")

    # ─── 포트폴리오 B: BTC 집중 ───────────────────────────────────────────
    print("\n[ 포트폴리오 B ] 2전략 집중 (BTC 4h + BTC 1d)")
    print("-" * 70)
    portfolio_b = [
        ("data/ohlcv_full_BTCKRW_4h.parquet",  "ensemble",    0.60),
        ("data/ohlcv_full_BTCKRW_1d.parquet",  "champion_v2", 0.40),
    ]
    eq_b, reps_b = combine_portfolios(portfolio_b, TOTAL)

    if eq_b is not None:
        m_b = portfolio_metrics(eq_b, TOTAL)
        print(f"\n  포트폴리오 B 합산:")
        print(f"    기간: {m_b['start']} ~ {m_b['end']}")
        print(f"    누적수익: {m_b['cum_return']:+.1%}")
        print(f"    연수익:   {m_b['ann_return']:+.1%}")
        print(f"    샤프:     {m_b['sharpe']:.2f}")
        print(f"    MDD:      {m_b['mdd']:.1%}")
        print(f"    Calmar:   {m_b['calmar']:.2f}")

    # ─── 포트폴리오 C: 최강 단일 (참조) ──────────────────────────────────
    print("\n[ 참조 ] 단일 최강: ensemble/BTC 4h (100%)")
    print("-" * 70)
    portfolio_c = [
        ("data/ohlcv_full_BTCKRW_4h.parquet",  "ensemble",    1.00),
    ]
    eq_c, reps_c = combine_portfolios(portfolio_c, TOTAL)

    if eq_c is not None:
        m_c = portfolio_metrics(eq_c, TOTAL)
        print(f"\n  단일 전략:")
        print(f"    기간: {m_c['start']} ~ {m_c['end']}")
        print(f"    누적수익: {m_c['cum_return']:+.1%}")
        print(f"    연수익:   {m_c['ann_return']:+.1%}")
        print(f"    샤프:     {m_c['sharpe']:.2f}")
        print(f"    MDD:      {m_c['mdd']:.1%}")
        print(f"    Calmar:   {m_c['calmar']:.2f}")

    # ─── OOS 기간만 별도 검증 ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OOS 기간 성과 (2024-02 이후 — 실제 예측력 검증)")
    print("=" * 70)

    oos_start = "2024-02-09"

    oos_tests = [
        ("data/ohlcv_full_BTCKRW_4h.parquet",  "ensemble",    "BTC 4h"),
        ("data/ohlcv_full_BTCKRW_4h.parquet",  "champion",    "BTC 4h"),
        ("data/ohlcv_full_BTCKRW_1d.parquet",  "champion_v2", "BTC 1d"),
        ("data/ohlcv_full_XRPKRW_1d.parquet",  "champion",    "XRP 1d"),
        ("data/ohlcv_full_XRPKRW_1d.parquet",  "champion_v2", "XRP 1d"),
    ]

    print(f"\n  {'전략':<16} {'데이터':<8} {'OOS 수익률':>10} {'OOS 샤프':>9} {'OOS MDD':>8}")
    print(f"  {'-'*60}")

    for path, strat_name, label in oos_tests:
        if not Path(path).exists():
            continue
        df = load_df(path)
        oos_df = df[df.index >= pd.Timestamp(oos_start, tz="UTC")]
        if len(oos_df) < 10:
            continue

        try:
            strat = get_strategy(strat_name)
            eq, report = run_backtest(oos_df, strat, ENV)
            m = report.summary()
            print(f"  {strat_name:<16} {label:<8} {m['cum_return']:>+9.1%} {m['sharpe']:>9.2f} {m['mdd']:>8.1%}")
        except Exception as e:
            print(f"  {strat_name:<16} {label:<8} ERROR: {e}")

    print("\n" + "=" * 70)
    print("결론:")
    print("  1. ensemble/BTC 4h: 최강 단일 전략 (OOS Sharpe 2.20)")
    print("  2. 포트폴리오 분산으로 MDD 개선 가능")
    print("  3. XRP 1d: BTC와 낮은 상관관계로 분산 효과")
    print("=" * 70)


if __name__ == "__main__":
    main()
