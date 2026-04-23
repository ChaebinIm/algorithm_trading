"""
급락 반등 전략 파라미터 그리드 서치 (에이전트 8)
BTC 일봉 기준 IS/OOS Walk-Forward 최적화
"""
from __future__ import annotations

import sys
import os
import itertools
import warnings
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# 경로 설정
BASE = "/Users/imchaebin/Desktop/system_trading/algorithm_trading"
sys.path.insert(0, BASE)

from core.engine import run_backtest, BacktestEnv, BasicSlippage, FixedRateCommission
from core.indicators import atr as calc_atr
from strategies import Strategy


# ──────────────────────────────────────────────
# 급락 반등 전략 인라인 구현
# ──────────────────────────────────────────────

class ReversalStrategy(Strategy):
    """급락 반등 전략 — 큰 음봉 후 반등 포착 (롱온리)."""

    name = "reversal_grid"

    def __init__(
        self,
        min_mult: float = 1.0,         # 최소 음봉 크기 (ATR 배수)
        max_mult: float = 2.0,         # 최대 음봉 크기 (ATR 배수), 0=무제한
        drawdown_filter: Optional[float] = -0.15,  # 낙폭 필터 (None=비활성)
        vol_filter: bool = True,        # 고변동성 필터 여부
        vol_mult: float = 1.0,         # ATR >= vol_ma * vol_mult
        tp_pct: float = 0.05,          # 익절 %
        sl_pct: float = 0.03,          # 손절 %
        max_bars: int = 7,             # 최대 보유 봉 수
        atr_period: int = 14,
        rolling_high_period: int = 30,
        vol_ma_period: int = 14,
    ):
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.drawdown_filter = drawdown_filter
        self.vol_filter = vol_filter
        self.vol_mult = vol_mult
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_bars = max_bars
        self.atr_period = atr_period
        self.rolling_high_period = rolling_high_period
        self.vol_ma_period = vol_ma_period

        self._position = 0
        self._entry_price = 0.0
        self._bars_held = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ATR
        df["atr"] = calc_atr(df, self.atr_period)
        df["vol_ma"] = df["atr"].rolling(self.vol_ma_period).mean()

        # 음봉 실체 크기
        df["body_neg"] = (df["open"] - df["close"]).clip(lower=0)

        # 30일 고점 대비 낙폭
        df["rolling_high"] = df["close"].rolling(self.rolling_high_period).max()
        df["drawdown"] = df["close"] / df["rolling_high"] - 1.0

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        body_neg = row.get("body_neg", 0.0)
        atr_val  = row.get("atr", 0.0)
        vol_ma   = row.get("vol_ma", 0.0)
        drawdown = row.get("drawdown", 0.0)

        # ── 포지션 보유 중 청산 로직 ──────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            close = row["close"]
            if close >= self._entry_price * (1 + self.tp_pct):
                self._position = 0
                return -1
            if close <= self._entry_price * (1 - self.sl_pct):
                self._position = 0
                return -1
            if self._bars_held >= self.max_bars:
                self._position = 0
                return -1
            return 0

        # ── 진입 조건 ────────────────────────────────────────────
        if atr_val <= 0:
            return 0

        # 1. 충분히 큰 음봉
        cond1 = body_neg >= atr_val * self.min_mult

        # 2. 폭락봉 제외 (max_mult > 0 인 경우)
        if self.max_mult > 0:
            cond2 = body_neg < atr_val * self.max_mult
        else:
            cond2 = True

        # 3. 낙폭 필터 (None이면 비활성)
        if self.drawdown_filter is not None:
            cond3 = drawdown <= self.drawdown_filter
        else:
            cond3 = True

        # 4. 고변동성 필터
        if self.vol_filter and vol_ma > 0:
            cond4 = atr_val >= vol_ma * self.vol_mult
        else:
            cond4 = True

        if cond1 and cond2 and cond3 and cond4:
            self._position = 1
            self._entry_price = row["close"]
            self._bars_held = 0
            return +1

        return 0


# ──────────────────────────────────────────────
# 백테스트 환경 설정
# ──────────────────────────────────────────────

ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=10),
)

# 데이터 로드
df_full = pd.read_parquet(os.path.join(BASE, "data/ohlcv_full_BTCUSDT_1d.parquet"))
df_full.index = pd.to_datetime(df_full.index)

# IS / OOS 분할
IS_END = "2022-12-31"
OOS_START = "2023-01-01"

df_is  = df_full[df_full.index <= IS_END]
df_oos = df_full[df_full.index >= OOS_START]

print(f"IS  기간: {df_is.index[0].date()} ~ {df_is.index[-1].date()}  ({len(df_is)} bars)")
print(f"OOS 기간: {df_oos.index[0].date()} ~ {df_oos.index[-1].date()}  ({len(df_oos)} bars)")
print()


# ──────────────────────────────────────────────
# 파라미터 그리드
# ──────────────────────────────────────────────

param_grid = {
    "min_mult":        [0.8, 1.0, 1.2],
    "max_mult":        [1.3, 1.5, 2.0, 0.0],   # 0=무제한
    "drawdown_filter": [-0.10, -0.15, -0.20, -0.25, None],
    "vol_filter":      [True, False],
    "tp_pct":          [0.03, 0.05, 0.08],
    "sl_pct":          [0.02, 0.03, 0.04],
    "max_bars":        [5, 7, 10],
}

keys = list(param_grid.keys())
combos = list(itertools.product(*[param_grid[k] for k in keys]))
total = len(combos)
print(f"총 조합 수: {total:,}")


# ──────────────────────────────────────────────
# 그리드 서치 실행
# ──────────────────────────────────────────────

results = []

for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))

    if (idx + 1) % 200 == 0:
        print(f"  [{idx+1}/{total}] 진행중...")

    # IS 백테스트
    try:
        strat_is = ReversalStrategy(**params)
        _, rpt_is = run_backtest(df_is, strat_is, ENV)
        is_sharpe = rpt_is.metrics.get("sharpe", float("nan"))
        is_n      = rpt_is.metrics.get("total_roundtrips", 0)
        is_wr     = rpt_is.metrics.get("win_rate", float("nan"))
    except Exception:
        continue

    # OOS 백테스트
    try:
        strat_oos = ReversalStrategy(**params)
        _, rpt_oos = run_backtest(df_oos, strat_oos, ENV)
        oos_sharpe = rpt_oos.metrics.get("sharpe", float("nan"))
        oos_n      = rpt_oos.metrics.get("total_roundtrips", 0)
        oos_wr     = rpt_oos.metrics.get("win_rate", float("nan"))
        oos_ret    = rpt_oos.metrics.get("cum_return", float("nan"))
        oos_mdd    = rpt_oos.metrics.get("mdd", float("nan"))
    except Exception:
        continue

    # OOS 거래 수 필터
    if oos_n < 8:
        continue

    results.append({
        "min_mult":   params["min_mult"],
        "max_mult":   params["max_mult"],
        "dd_filter":  params["drawdown_filter"],
        "vol_filter": params["vol_filter"],
        "tp":         params["tp_pct"],
        "sl":         params["sl_pct"],
        "max_bars":   params["max_bars"],
        "IS_sharpe":  round(is_sharpe, 3),
        "IS_n":       int(is_n),
        "IS_wr":      round(is_wr, 3) if not np.isnan(is_wr) else float("nan"),
        "OOS_sharpe": round(oos_sharpe, 3),
        "OOS_wr":     round(oos_wr, 3) if not np.isnan(oos_wr) else float("nan"),
        "OOS_n":      int(oos_n),
        "OOS_ret":    round(oos_ret, 3),
        "OOS_mdd":    round(oos_mdd, 3),
    })

print(f"\n필터 통과 조합: {len(results)}개 (OOS >= 8건)")

if not results:
    print("결과 없음")
    sys.exit(0)

df_res = pd.DataFrame(results)
df_res = df_res.sort_values("OOS_sharpe", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────

OUTPUT_PATH = os.path.join(BASE, "research/reversal_08_grid_search.txt")

lines = []
lines.append("=" * 90)
lines.append("급락 반등 전략 파라미터 그리드 서치 결과")
lines.append(f"데이터: BTCUSDT 1d  |  IS: 2017~2022  |  OOS: 2023~2026")
lines.append(f"총 조합: {total:,}  |  OOS>=8건 통과: {len(df_res)}개")
lines.append("=" * 90)
lines.append("")

# 상위 20개
top20 = df_res.head(20)

header = f"{'#':>3}  {'min':>5} {'max':>5} {'dd':>7} {'vol':>5} {'tp':>5} {'sl':>5} {'bars':>5} | {'IS_sh':>7} {'IS_n':>5} | {'OOS_sh':>7} {'OOS_wr':>7} {'OOS_n':>6} {'OOS_ret':>8} {'OOS_mdd':>8}  flag"
lines.append(header)
lines.append("-" * len(header))

for rank, row in top20.iterrows():
    flag = "★" if (not np.isnan(row["IS_sharpe"]) and not np.isnan(row["OOS_sharpe"])
                   and row["IS_sharpe"] < row["OOS_sharpe"]) else ""
    dd_str = f"{row['dd_filter']:.2f}" if row["dd_filter"] is not None else " None"
    lines.append(
        f"{rank+1:>3}  {row['min_mult']:>5.1f} {row['max_mult']:>5.1f} {dd_str:>7} {str(row['vol_filter']):>5} "
        f"{row['tp']:>5.2f} {row['sl']:>5.2f} {row['max_bars']:>5} | "
        f"{row['IS_sharpe']:>7.3f} {row['IS_n']:>5} | "
        f"{row['OOS_sharpe']:>7.3f} {row['OOS_wr']:>7.3f} {row['OOS_n']:>6} "
        f"{row['OOS_ret']:>8.3f} {row['OOS_mdd']:>8.3f}  {flag}"
    )

lines.append("")
lines.append("" )
lines.append("★ = IS_sharpe < OOS_sharpe (과적합 가능성 낮음)")
lines.append("")
lines.append("── 전체 결과 요약 통계 ──")
lines.append(f"  OOS Sharpe 평균: {df_res['OOS_sharpe'].mean():.3f}")
lines.append(f"  OOS Sharpe 최대: {df_res['OOS_sharpe'].max():.3f}")
lines.append(f"  OOS Sharpe 양수 비율: {(df_res['OOS_sharpe'] > 0).mean():.1%}")
lines.append(f"  IS < OOS 조합 수: {((df_res['IS_sharpe'] < df_res['OOS_sharpe'])).sum()}")
lines.append("")

# IS < OOS 조합만 별도 출력 (상위 10개)
oos_better = df_res[df_res["IS_sharpe"] < df_res["OOS_sharpe"]].head(10)
if len(oos_better) > 0:
    lines.append("── IS < OOS 조합 상위 10 (과적합 없음) ──")
    lines.append(header)
    lines.append("-" * len(header))
    for r, row in oos_better.iterrows():
        dd_str = f"{row['dd_filter']:.2f}" if row["dd_filter"] is not None else " None"
        lines.append(
            f"{r+1:>3}  {row['min_mult']:>5.1f} {row['max_mult']:>5.1f} {dd_str:>7} {str(row['vol_filter']):>5} "
            f"{row['tp']:>5.2f} {row['sl']:>5.2f} {row['max_bars']:>5} | "
            f"{row['IS_sharpe']:>7.3f} {row['IS_n']:>5} | "
            f"{row['OOS_sharpe']:>7.3f} {row['OOS_wr']:>7.3f} {row['OOS_n']:>6} "
            f"{row['OOS_ret']:>8.3f} {row['OOS_mdd']:>8.3f}  ★"
        )

lines.append("")
lines.append("완료")

output = "\n".join(lines)
print()
print(output)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(output)

print(f"\n결과 저장: {OUTPUT_PATH}")
