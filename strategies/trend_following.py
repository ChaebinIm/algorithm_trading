"""
다중 필터 추세 추종 전략 (Trend Following).

설계 철학: 보수적 진입, 높은 승률과 손익비.
- 여러 필터를 중첩하여 확실한 추세에서만 진입
- 추세 내 눌림목 후 반등 시점에서 진입
- ATR 기반 손절 + 트레일링 스탑으로 추세 끝까지 보유

손절/익절:
- 손절: 진입가 - 2.0 x ATR
- 익절: 고정 TP 없음, 트레일링 스탑으로 수익 보호
- 트레일링: 보유 중 고점 - 2.5 x ATR
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, rsi, atr, adx


class TrendFollowingStrategy(Strategy):
    """다중 필터 추세 추종 전략.

    진입 조건 (매수, 모든 조건 동시 충족):
    1. EMA 정배열: EMA(20) > EMA(50) > EMA(200)
    2. ADX > 20: 추세 존재
    3. RSI 반등: 직전 바에서 RSI < 45였다가 현재 바에서 RSI > 45로 회복
       (눌림목 후 반등 시점을 포착)
    4. 종가 > EMA(50): 중기 추세선 위에서 거래

    청산 조건 (하나라도 충족 시):
    - 종가 < 진입가 - 2.0xATR: 손절
    - 종가 < 보유 중 고점 - 2.5xATR: 트레일링 스탑
    - 종가 < EMA(50): 추세 이탈
    """

    name = "trend_following"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 20.0,        # ADX 문턱값 (25 -> 20으로 완화)
        rsi_pullback: float = 45.0,          # RSI 눌림 판단 기준
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 2.5,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_pullback = rsi_pullback
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        # 내부 상태
        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0
        self._prev_rsi = 50.0   # 이전 바 RSI (반등 감지용)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 선계산 (벡터화)."""
        df = df.copy()
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["adx"] = adx(df, self.adx_period)
        df["atr"] = atr(df, 14)

        # EMA 정배열 여부
        df["ema_aligned"] = (
            (df["ema_fast"] > df["ema_mid"]) &
            (df["ema_mid"] > df["ema_slow"])
        ).astype(int)

        # 이전 바 RSI (반등 감지용)
        df["rsi_prev"] = df["rsi"].shift(1)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성."""
        close = row["close"]
        atr_val = row.get("atr", 0)

        # --- 포지션 보유 중: 청산 조건 체크 ---
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # 손절
            sl_price = self._entry_price - self.sl_atr_mult * atr_val
            if close < sl_price:
                self._in_position = False
                return -1

            # 트레일링 스탑
            trailing_stop = self._highest - self.trailing_atr_mult * atr_val
            if close < trailing_stop:
                self._in_position = False
                return -1

            # 추세 이탈
            ema_mid_val = row.get("ema_mid", close)
            if close < ema_mid_val:
                self._in_position = False
                return -1

            return 0

        # --- 미보유: 매수 조건 ---
        ema_aligned = bool(row.get("ema_aligned", 0))
        adx_val = row.get("adx", 0)
        rsi_val = row.get("rsi", 50)
        rsi_prev = row.get("rsi_prev", 50)
        ema_mid_val = row.get("ema_mid", close)

        # RSI 반등 감지: 직전에 눌림 기준 아래 -> 현재 위로 회복
        rsi_bounced = (rsi_prev < self.rsi_pullback) and (rsi_val >= self.rsi_pullback)

        if (
            ema_aligned                          # EMA 정배열
            and adx_val > self.adx_threshold     # 추세 존재
            and rsi_bounced                      # RSI 눌림 후 반등
            and close > ema_mid_val              # 중기 추세선 위
        ):
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
