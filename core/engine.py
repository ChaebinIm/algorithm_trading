"""
백테스트 엔진 — 핵심 시뮬레이션 루프와 성과 지표 계산.

엔진은 전략의 구현 세부를 모른다.
Strategy 인터페이스(prepare, on_bar)만 호출하여 신호를 받고,
주문 체결/포지션 관리/수익률 계산을 처리한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict

import pandas as pd

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
    """누적 자본곡선(equity)로부터 드로우다운(고점 대비 낙폭)을 계산.
    dd[t] = equity[t] / cummax(equity[:t]) - 1
    값은 0(손실 없음)에서 음수(손실) 사이.
    """
    peak = equity.cummax()
    return (equity / peak) - 1.0


def perf_summary(equity: pd.Series, returns: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """성과 요약치 계산.
    - 누적수익률, 연환산 수익률/변동성, 샤프비율, MDD.
    - 분봉 수익률을 일 단위로 리샘플링(sum)해 연율화.
    """
    ann_fac = 365  # 크립토는 365일 24시간 거래
    daily_ret = returns.resample("1D").sum(min_count=1)
    ann_ret = daily_ret.mean() * ann_fac
    ann_vol = daily_ret.std(ddof=1) * math.sqrt(ann_fac)
    sharpe = (ann_ret - rf) / (ann_vol + 1e-12)
    mdd = compute_drawdown(equity).min()
    return {
        "cum_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "mdd": float(mdd),
    }


# =============================
# 백테스트 환경 설정
# =============================

@dataclass
class BacktestEnv:
    """브로커/환경 설정.
    - cash: 초기 현금 (USDT)
    - slippage: 슬리피지 모델
    - commission: 수수료 모델
    - sizer: 포지션 사이즈 결정자
    - allow_short: 공매도 허용 여부
    - max_position_qty: 최대 보유 수량 (절대값, 과도한 사이징 방지용)
    """
    cash: float = 10_000.0
    slippage: SlippageModel = None
    commission: CommissionModel = None
    sizer: Sizer = None
    allow_short: bool = False
    max_position_qty: float = 10.0

    def __post_init__(self):
        """기본 모델 인스턴스 생성 (dataclass 기본값으로 mutable 객체 불가하므로)."""
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
        바별 자본곡선
    report : BacktestReport
        성과 리포트 (자본곡선, 수익률, 체결로그, 성과지표)

    동작 방식:
    - 전략의 prepare()로 지표/신호 칼럼을 미리 계산 (벡터화)
    - 각 바에서 on_bar()로 신호(+1/0/-1) 생성
    - 체결은 '다음 바의 시가'에 슬리피지를 적용해 실행 (현실성↑)
    - 수수료 반영, 현금/보유수량 업데이트
    """
    # 0) 입력 검증 및 사전 계산
    required_cols = {"open", "high", "low", "close", "volume"}
    assert required_cols.issubset(df_raw.columns), f"입력 데이터에 {required_cols} 컬럼 필요"

    df = strategy.prepare(df_raw.copy())  # 지표/신호 선계산
    df = df.dropna().copy()               # NaN 제거 (롤링 초기 구간 등)
    index = df.index                      # 시각 인덱스

    # 1) 초기 상태
    cash = env.cash
    pos_qty = 0.0           # 보유 수량
    entry_price = None      # 진입가 트래킹

    eq = []                 # 바별 자본값
    rets = []               # 바별 수익률
    last_equity = cash

    tlog = TradeLog(fills=[])

    # 2) 바 단위 시뮬레이션 루프 (다음 바 시가에서 체결하므로 range(len-1))
    for i in range(len(df) - 1):
        cur = df.iloc[i]        # 현재 바
        nxt = df.iloc[i + 1]    # 다음 바 (체결가 참고)
        ts = index[i + 1]       # 체결 타임스탬프

        # 2-1) 현재 바 종가 기준 시가평가
        mtm_price = cur["close"]

        # 2-2) 전략 신호 계산
        signal = strategy.on_bar(i, cur)

        if signal != 0:
            # 2-3) 사이징
            raw_qty = env.sizer.size(
                price=mtm_price, cash=cash, pos_qty=pos_qty, signal=signal
            )

            # 2-4) 공매도 금지 시 음수 포지션 방지
            if not env.allow_short and (pos_qty + raw_qty) < 0:
                raw_qty = -pos_qty

            if abs(raw_qty) > 0:
                side = "buy" if raw_qty > 0 else "sell"

                # 2-5) 체결 가격: 다음 바 시가 + 슬리피지
                px = env.slippage.slip_price(side, float(nxt["open"]))

                # 2-6) 수량/체결금액/수수료
                qty = abs(raw_qty)
                notional = px * qty
                fee = env.commission.cost(notional)

                if side == "buy":
                    # 현금 부족 시 수량 축소
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

                # 2-7) 체결 로그
                tlog.fills.append(Fill(
                    ts=ts, side=side, price=float(px),
                    qty=float(qty), fee=float(fee), notional=float(notional),
                ))

        # 2-8) 바 종료 자본과 수익률 기록
        equity_end = cash + pos_qty * cur["close"]
        eq.append(equity_end)
        rets.append((equity_end / last_equity) - 1.0 if last_equity > 0 else 0.0)
        last_equity = equity_end

    # 3) 결과 구성
    equity_series = pd.Series(eq, index=df.index[:-1], name="equity")
    ret_series = pd.Series(rets, index=df.index[:-1], name="returns")

    metrics = perf_summary(equity_series, ret_series)
    trades_df = tlog.to_frame()

    report = BacktestReport(
        equity=equity_series,
        returns=ret_series,
        trades=trades_df,
        metrics=metrics,
    )
    return equity_series, report
