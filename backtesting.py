"""
Backtesting framework for minute OHLCV (Upbit BTC/KRW etc.)
- Hybrid vector/event design:
  * Vectorized indicators & fast PnL math
  * Event loop for order/position lifecycle & risk rules
- Single-file MVP; clean API to extend Strategy, Slippage, Commission, Sizer

Assumptions
- Input: pandas.DataFrame with columns: ["open","high","low","close","volume"]
  index is tz-aware pandas.DatetimeIndex (e.g., Asia/Seoul)
- All prices are in KRW; base asset is BTC
- Orders execute on next bar's open with slippage; fills can be partial if liquidity model is added (simple here)
- Long-only by default; short can be enabled

Usage
-----
from backtest import *

# 1) 데이터 적재 (Parquet에서 불러오거나 CCXT 수집본 사용)
# df = pd.read_parquet("data/ohlcv_full.parquet").set_index("time")[
#     ["open","high","low","close","volume"]
# ]

# 2) 백테스트 환경/전략 구성 후 실행
# env = BacktestEnv(
#     cash=1_000_000,
#     slippage=BasicSlippage(bps=5),     # 매수는 +, 매도는 - 방향으로 5bp 가격 미끄러짐
#     commission=FixedRateCommission(bps=5),  # 왕복 수수료를 보수적으로 bps로 모델링
#     sizer=FixedKRWSizer(krw_per_trade=100_000),  # 1회 10만원 고정 매수
# )
# strat = MovingAverageCross(short=5, long=20)
# equity_curve, report = run_backtest(df, strat, env)
# print(report.summary())
# print(report.trades.tail())

"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

# =============================
# Utility & Metrics
# =============================

def to_series(x):
    """입력 x를 pandas.Series로 강제 변환 (필요시). 간단 유틸."""
    return x if isinstance(x, pd.Series) else pd.Series(x)


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """누적 자본곡선(equity)로부터 드로우다운(고점 대비 낙폭)을 계산.
    dd[t] = equity[t]/cummax(equity[:t]) - 1
    값은 0(손실 없음)에서 음수(손실) 사이.
    """
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return dd


def annualize_factor(freq: str) -> float:
    """연율화(수익/변동성) 계수를 반환.
    - 분봉 데이터라도 성과 요약은 일 단위로 보는 게 일반적이라 252 거래일 기준 사용.
    - 샤프비율 계산시 변동성 스케일링에 sqrt(252), 수익률엔 *252.
    """
    if freq.lower() in {"d","1d","day","daily"}:
        return math.sqrt(252), 252
    # 기본값도 252 거래일 가정으로 반환
    return math.sqrt(252), 252


def perf_summary(equity: pd.Series, returns: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """성과 요약치 계산:
    - 누적수익률, 연환산 수익률/변동성, 샤프비율, MDD.
    - 분봉 수익률을 일 단위로 리샘플링(sum)해 연율화.
    """
    sharpe_fac, ann_fac = annualize_factor("d")
    daily_ret = returns.resample("1D").sum(min_count=1)  # 분봉 누적 → 일 수익률
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
# Slippage & Commission models
# =============================

class SlippageModel:
    """슬리피지(체결가 미끄러짐) 모델 인터페이스.
    실제 체결가 = 가정된 체결가(open 등)에 슬리피지 조정 적용 결과.
    """
    def slip_price(self, side: str, price: float) -> float:
        raise NotImplementedError

class BasicSlippage(SlippageModel):
    """고정 bps(베이시스 포인트) 슬리피지 모델.
    - 매수: 가격을 +bps만큼 불리하게
    - 매도: 가격을 -bps만큼 불리하게
    """
    def __init__(self, bps: float = 5.0):
        self.bps = bps
    def slip_price(self, side: str, price: float) -> float:
        adj = price * (self.bps / 10_000)
        return price + adj if side == "buy" else price - adj

class CommissionModel:
    """수수료 모델 인터페이스. 공통 입력은 체결 금액(notional)."""
    def cost(self, notional: float) -> float:
        raise NotImplementedError

class FixedRateCommission(CommissionModel):
    """고정 비율 수수료 모델 (bps 단위)."""
    def __init__(self, bps: float = 5.0):
        self.bps = bps
    def cost(self, notional: float) -> float:
        return abs(notional) * (self.bps / 10_000)

# =============================
# Sizer (position sizing)
# =============================

class Sizer:
    """포지션 사이징 인터페이스.
    signal(+1/-1)에 따라 얼마를 살지/팔지(수량) 결정.
    """
    def size(self, price: float, cash: float, pos_qty: float, signal: int) -> float:
        raise NotImplementedError

class FixedKRWSizer(Sizer):
    """1회 매매 금액을 KRW 기준으로 고정하는 사이저.
    - 매수: KRW 금액/현재가 → 수량
    - 매도: 보유 수량 전량 매도 (단순화)
    """
    def __init__(self, krw_per_trade: float = 100_000, min_qty: float = 0.0001):
        self.krw = krw_per_trade
        self.min_qty = min_qty
    def size(self, price: float, cash: float, pos_qty: float, signal: int) -> float:
        if signal > 0:  # buy
            qty = self.krw / price
            return max(qty, self.min_qty)
        elif signal < 0:  # sell → 전량 처분
            return -pos_qty
        return 0.0

# =============================
# Strategy base & example
# =============================

class Strategy:
    """전략 베이스 클래스.
    - prepare(df): 백테스트 시작 전, 인디케이터/전처리 칼럼을 추가
    - on_bar(i, row): i번째 바에서 트레이드 신호 반환(+1 매수, -1 매도, 0 유지)
    """
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    def on_bar(self, i: int, row: pd.Series) -> int:
        return 0

class MovingAverageCross(Strategy):
    """대표 예시 전략: 단기/장기 이동평균 교차.
    - ma_s > ma_l 영역에서 1, 그렇지 않으면 0
    - 신호 변화(sig_change)가 +이면 매수, -이면 매도
    """
    def __init__(self, short: int = 5, long: int = 20):
        self.short = short
        self.long = long
        self._sig = None
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f"ma_s"] = df["close"].rolling(self.short).mean()
        df[f"ma_l"] = df["close"].rolling(self.long).mean()
        df["sig"] = 0
        df.loc[df["ma_s"] > df["ma_l"], "sig"] = 1
        df["sig"] = df["sig"].fillna(0).astype(int)
        df["sig_change"] = df["sig"].diff().fillna(0)
        self._sig = df["sig"]
        return df
    def on_bar(self, i: int, row: pd.Series) -> int:
        ch = int(row.get("sig_change", 0))
        if ch > 0:
            return +1
        if ch < 0:
            return -1
        return 0

# =============================
# Trade & Report containers
# =============================

@dataclass
class Fill:
    """단일 체결 레코드.
    - ts: 체결 시각(다음 바 시가 기준 체결 가정)
    - side: 'buy' 또는 'sell'
    - price: 체결 단가 (슬리피지 적용 후)
    - qty: 체결 수량
    - fee: 수수료 금액
    - notional: 체결 금액(단가*수량)
    """
    ts: pd.Timestamp
    side: str
    price: float
    qty: float
    fee: float
    notional: float

@dataclass
class TradeLog:
    """체결 목록을 보관하고 DataFrame으로 변환하는 헬퍼."""
    fills: List[Fill]
    def to_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["ts","side","price","qty","fee","notional"])
        d = [asdict(f) for f in self.fills]
        return pd.DataFrame(d)

@dataclass
class BacktestReport:
    """백테스트 결과 컨테이너.
    - equity: 바별 자본곡선
    - returns: 바별 수익률(자본곡선 기준)
    - trades: 체결 로그(데이터프레임)
    - metrics: 성과 요약치(dict)
    """
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    def summary(self) -> Dict[str, float]:
        return self.metrics

# =============================
# Environment / Broker sim
# =============================

@dataclass
class BacktestEnv:
    """브로커/환경 설정.
    - cash: 초기 현금(원)
    - slippage: 슬리피지 모델
    - commission: 수수료 모델
    - sizer: 포지션 사이즈 결정자
    - allow_short: 공매도 허용 여부
    - max_position_qty: 최대 보유 수량(절대값, 과도한 사이징 방지용)
    """
    cash: float = 1_000_000.0
    slippage: SlippageModel = BasicSlippage(5)
    commission: CommissionModel = FixedRateCommission(5)
    sizer: Sizer = FixedKRWSizer(100_000)
    allow_short: bool = False
    max_position_qty: float = 10.0

# =============================
# Core backtest engine
# =============================

def run_backtest(df_raw: pd.DataFrame, strategy: Strategy, env: BacktestEnv) -> Tuple[pd.Series, BacktestReport]:
    """백테스트 핵심 루프.
    - 전략의 prepare로 지표/신호 칼럼을 미리 계산(벡터화)
    - 각 바에서 on_bar로 신호(+1/0/-1) 생성
    - 체결은 '다음 바의 시가'에 슬리피지를 적용해 실행 (현실성↑)
    - 수수료 반영, 현금/보유수량 업데이트
    - 자본곡선/수익률/체결로그/성과요약 반환
    """
    # 0) 입력 검증 및 사전 계산
    required_cols = {"open","high","low","close","volume"}
    assert required_cols.issubset(df_raw.columns), f"Input must contain {required_cols}"

    df = strategy.prepare(df_raw.copy())   # 지표/신호 선계산
    df = df.dropna().copy()                # NaN 제거(롤링 초기 구간 등)
    index = df.index                       # 시각 인덱스 (체결/로그용)

    # 1) 초기 상태
    cash = env.cash                        # 현금
    pos_qty = 0.0                          # 보유 수량 (BTC)
    entry_price = None                     # (옵션) 진입가 트래킹

    eq = []       # 바별 자본값 저장
    rets = []     # 바별 자본 수익률 저장
    last_equity = cash

    tlog = TradeLog(fills=[])

    # 2) 바 단위 시뮬레이션 루프
    #    다음 바 시가에서 체결하므로 range(len(df)-1)
    for i in range(len(df) - 1):
        cur = df.iloc[i]           # 현재 바 데이터
        nxt = df.iloc[i + 1]       # 다음 바 데이터 (체결가 참고)
        ts = index[i + 1]          # 체결 타임스탬프는 다음 바 시각으로 기록

        # 2-1) MTM (시가/종가 중 무엇을 쓸지는 설계 선택; 여기선 현재 바 종가)
        mtm_price = cur["close"]
        equity_now = cash + pos_qty * mtm_price

        # 2-2) 전략 신호 계산
        signal = strategy.on_bar(i, cur)   # +1/-1/0

        if signal != 0:
            # 2-3) 사이징: 신호와 현재 현금/포지션에 기반해 수량 결정
            raw_qty = env.sizer.size(price=mtm_price, cash=cash, pos_qty=pos_qty, signal=signal)

            # 2-4) 공매도 금지 시 음수 포지션 방지
            if not env.allow_short and (pos_qty + raw_qty) < 0:
                raw_qty = -pos_qty  # 보유분까지만 청산

            if abs(raw_qty) > 0:
                side = "buy" if raw_qty > 0 else "sell"

                # 2-5) 체결 가격: 다음 바 시가에 슬리피지 적용
                px = env.slippage.slip_price(side, float(nxt["open"]))

                # 2-6) 수량/체결금액/수수료 계산
                qty = abs(raw_qty)
                notional = px * qty
                fee = env.commission.cost(notional)

                if side == "buy":
                    # 현금 부족 시 수량 축소 (체결 가능 범위 내에서 최대치)
                    if notional + fee > cash + 1e-9:
                        qty = max((cash - fee) / px, 0.0)
                        notional = px * qty
                    cash -= (notional + fee)
                    pos_qty += qty
                    entry_price = px if entry_price is None else entry_price
                else:  # sell
                    # 현재 보유보다 많이 팔지 않도록 가드(단순 롱 전제)
                    qty = min(qty, pos_qty) if pos_qty > 0 else qty
                    notional = px * qty
                    fee = env.commission.cost(notional)
                    cash += (notional - fee)
                    pos_qty -= qty
                    if pos_qty <= 1e-12:
                        pos_qty = 0.0
                        entry_price = None

                # 2-7) 체결 로그 기록
                tlog.fills.append(
                    Fill(ts=ts, side=side, price=float(px), qty=float(qty), fee=float(fee), notional=float(notional))
                )

        # 2-8) 바 종료 자본(현재 바 종가 기준)과 수익률 기록
        equity_end = cash + pos_qty * cur["close"]
        eq.append(equity_end)
        if last_equity > 0:
            rets.append((equity_end / last_equity) - 1.0)
        else:
            rets.append(0.0)
        last_equity = equity_end

    # 3) 결과 시리즈/리포트 구성
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

# =============================
# Example Quick Start when run as a script
# =============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1) 데이터 로드: 실데이터가 없으면 더미 데이터 생성
    try:
        raw = pd.read_parquet("data/ohlcv_full.parquet").set_index("time")[
            ["open","high","low","close","volume"]
        ]
    except Exception:
        # 더미 분봉 시계열(테스트용)
        idx = pd.date_range("2024-01-01", periods=2000, freq="1min", tz="Asia/Seoul")
        prices = np.cumsum(np.random.randn(len(idx))) + 100
        raw = pd.DataFrame({
            "open": prices + np.random.randn(len(idx))*0.2,
            "high": prices + np.random.rand(len(idx))*0.5,
            "low":  prices - np.random.rand(len(idx))*0.5,
            "close": prices,
            "volume": np.random.rand(len(idx))*5,
        }, index=idx)

    # 2) 환경/전략 구성
    env = BacktestEnv(
        cash=1_000_000,
        slippage=BasicSlippage(bps=5),
        commission=FixedRateCommission(bps=5),
        sizer=FixedKRWSizer(krw_per_trade=100_000),
    )

    strat = MovingAverageCross(short=5, long=20)

    # 3) 실행 및 결과 확인
    equity, rep = run_backtest(raw, strat, env)

    print("Summary:", rep.summary())
    print("Trades tail:
", rep.trades.tail())

    equity.plot(title="Equity Curve")
    plt.show()
