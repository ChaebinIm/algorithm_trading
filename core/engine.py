"""
백테스트 엔진 — 핵심 시뮬레이션 루프와 성과 지표 계산.

엔진은 전략의 구현 세부를 모른다.
Strategy 인터페이스(prepare, on_bar)만 호출하여 신호를 받고,
주문 체결/포지션 관리/수익률 계산을 처리한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np

from core.types import Fill, TradeLog, BacktestReport
from core.models import (
    SlippageModel, BasicSlippage,
    CommissionModel, FixedRateCommission,
    Sizer, FixedUSDTSizer,
)


# =============================
# 성과 지표 함수
# =============================

def compute_drawdown(equity: pd.Series) -> pd.Series:
    """누적 자본곡선으로부터 드로우다운 계산."""
    peak = equity.cummax()
    return (equity / peak) - 1.0


def _compute_trade_stats(trades: pd.DataFrame) -> Dict[str, float]:
    """매수-매도 쌍을 매칭하여 라운드트립 손익 통계를 계산.

    Returns dict with:
        win_rate, profit_factor, avg_win, avg_loss, total_roundtrips
    """
    if trades.empty:
        return {
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "total_roundtrips": 0,
        }

    buys = trades[trades["side"] == "buy"].reset_index(drop=True)
    sells = trades[trades["side"] == "sell"].reset_index(drop=True)

    n = min(len(buys), len(sells))
    if n == 0:
        return {
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_win": float("nan"),
            "avg_loss": float("nan"),
            "total_roundtrips": 0,
        }

    pnls = []
    for i in range(n):
        buy_price = buys.iloc[i]["price"]
        sell_price = sells.iloc[i]["price"]
        buy_fee = buys.iloc[i]["fee"]
        sell_fee = sells.iloc[i]["fee"]
        qty = buys.iloc[i]["qty"]
        gross = (sell_price - buy_price) * qty
        net = gross - buy_fee - sell_fee
        pnls.append(net)

    pnls = np.array(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / n if n > 0 else float("nan")
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 1e-9 else float("inf")
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0

    return {
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "total_roundtrips": n,
    }


def perf_summary(
    equity: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    rf: float = 0.0,
) -> Dict[str, float]:
    """종합 성과 요약."""
    ann_fac = 365
    daily_ret = returns.resample("1D").sum(min_count=1)
    ann_ret = daily_ret.mean() * ann_fac
    ann_vol = daily_ret.std(ddof=1) * math.sqrt(ann_fac)
    sharpe = (ann_ret - rf) / (ann_vol + 1e-12)
    mdd = compute_drawdown(equity).min()
    calmar = ann_ret / (abs(mdd) + 1e-12)

    trade_stats = _compute_trade_stats(trades)

    result = {
        "cum_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
        "calmar": float(calmar),
    }
    result.update(trade_stats)
    return result


# =============================
# 백테스트 환경 설정
# =============================

@dataclass
class BacktestEnv:
    """브로커/환경 설정."""
    cash: float = 10_000.0
    slippage: SlippageModel = None
    commission: CommissionModel = None
    sizer: Sizer = None
    allow_short: bool = False
    max_position_qty: float = 10.0

    def __post_init__(self):
        if self.slippage is None:
            self.slippage = BasicSlippage(5)
        if self.commission is None:
            self.commission = FixedRateCommission(10)
        if self.sizer is None:
            self.sizer = FixedUSDTSizer(100)


# =============================
# 백테스트 엔진
# =============================

def run_backtest(
    df_raw: pd.DataFrame,
    strategy,
    env: BacktestEnv,
) -> Tuple[pd.Series, BacktestReport]:
    """백테스트 핵심 루프.

    Parameters
    ----------
    df_raw : pd.DataFrame
        OHLCV 데이터 (columns: open, high, low, close, volume / index: DatetimeIndex)
    strategy : Strategy
        prepare()와 on_bar()를 구현한 전략 인스턴스
    env : BacktestEnv
        브로커/환경 설정

    Returns
    -------
    equity_series : pd.Series
    report : BacktestReport

    동작 방식:
    - 전략의 prepare()로 지표/신호 칼럼을 미리 계산 (벡터화)
    - 각 바에서 on_bar()로 신호(+1/0/-1) 생성
    - 체결은 '다음 바의 시가'에 슬리피지를 적용해 실행 (현실성↑)
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    assert required_cols.issubset(df_raw.columns), f"입력 데이터에 {required_cols} 컬럼 필요"

    df = strategy.prepare(df_raw.copy())
    df = df.dropna().copy()
    index = df.index

    cash = env.cash
    pos_qty = 0.0
    entry_price = None

    eq: List[float] = []
    rets: List[float] = []
    last_equity = cash

    tlog = TradeLog(fills=[])

    for i in range(len(df) - 1):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]
        ts = index[i + 1]

        mtm_price = cur["close"]
        signal = strategy.on_bar(i, cur)

        if signal != 0:
            raw_qty = env.sizer.size(
                price=mtm_price, cash=cash, pos_qty=pos_qty, signal=signal
            )

            if not env.allow_short and (pos_qty + raw_qty) < 0:
                raw_qty = -pos_qty

            if abs(raw_qty) > 0:
                side = "buy" if raw_qty > 0 else "sell"
                px = env.slippage.slip_price(side, float(nxt["open"]))
                qty = abs(raw_qty)
                notional = px * qty
                fee = env.commission.cost(notional)

                if side == "buy":
                    if notional + fee > cash + 1e-9:
                        qty = max((cash - fee) / px, 0.0)
                        notional = px * qty
                    cash -= (notional + fee)
                    pos_qty += qty
                    entry_price = px if entry_price is None else entry_price
                else:
                    qty = min(qty, pos_qty) if pos_qty > 0 else qty
                    notional = px * qty
                    fee = env.commission.cost(notional)
                    cash += (notional - fee)
                    pos_qty -= qty
                    if pos_qty <= 1e-12:
                        pos_qty = 0.0
                        entry_price = None

                tlog.fills.append(Fill(
                    ts=ts, side=side, price=float(px),
                    qty=float(qty), fee=float(fee), notional=float(notional),
                ))

        equity_end = cash + pos_qty * cur["close"]
        eq.append(equity_end)
        rets.append((equity_end / last_equity) - 1.0 if last_equity > 0 else 0.0)
        last_equity = equity_end

    equity_series = pd.Series(eq, index=df.index[:-1], name="equity")
    ret_series = pd.Series(rets, index=df.index[:-1], name="returns")

    trades_df = tlog.to_frame()
    metrics = perf_summary(equity_series, ret_series, trades_df)

    report = BacktestReport(
        equity=equity_series,
        returns=ret_series,
        trades=trades_df,
        metrics=metrics,
    )
    return equity_series, report
