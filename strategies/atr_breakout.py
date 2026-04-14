"""
ATR 브레이크아웃 전략 (ATR Breakout).

설계 철학: 확실한 움직임에서만 진입, 명확한 손익비.
- 돈키안 채널(N기간 고가) 돌파를 ATR 필터로 검증
- 노이즈성 돌파를 걸러내고 진짜 브레이크아웃만 포착
- 고정 손익비 1:2 (SL 1.5 ATR, TP 3.0 ATR)

손절/익절:
- 손절: 진입가 - 1.5 × ATR (타이트하게, 가짜 돌파 시 빠른 손절)
- 익절: 진입가 + 3.0 × ATR (손익비 1:2 보장)
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, atr, adx, donchian_channel


class ATRBreakoutStrategy(Strategy):
    """ATR 브레이크아웃 전략.

    진입 조건 (모든 조건 동시 충족):
    1. 종가 > N기간 최고가 (돈키안 채널 상단 돌파)
    2. (종가 - 전일 종가) > K × ATR: 움직임이 ATR 대비 충분히 큰지 확인
    3. 종가 > EMA(200): 장기 상승 추세 필터
    4. ADX > 20: 최소한의 추세 존재 확인

    청산 조건:
    - 종가 < 진입가 - 1.5×ATR → 손절
    - 종가 > 진입가 + 3.0×ATR → 익절 (손익비 1:2)
    """

    name = "atr_breakout"

    def __init__(
        self,
        channel_period: int = 20,      # 돈키안 채널 기간
        atr_period: int = 14,
        breakout_atr_mult: float = 0.5, # 돌파 확인 최소 ATR 배수
        ema_trend: int = 200,           # 장기 추세 필터
        adx_period: int = 14,
        adx_threshold: float = 20.0,    # 최소 추세 강도
        sl_atr_mult: float = 1.5,       # 손절 ATR 배수
        tp_atr_mult: float = 3.0,       # 익절 ATR 배수 (손익비 1:2)
    ):
        self.channel_period = channel_period
        self.atr_period = atr_period
        self.breakout_atr_mult = breakout_atr_mult
        self.ema_trend = ema_trend
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # 내부 상태
        self._in_position = False
        self._entry_price = 0.0
        self._entry_atr = 0.0   # 진입 시점의 ATR (SL/TP 고정용)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 선계산."""
        df = df.copy()

        # 돈키안 채널 (전일까지의 고가/저가 사용 → shift(1))
        dc_upper, dc_lower = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)  # 전일까지의 N기간 고가
        df["dc_lower"] = dc_lower.shift(1)

        # ATR
        df["atr"] = atr(df, self.atr_period)

        # 장기 추세 필터
        df["ema_trend"] = ema(df["close"], self.ema_trend)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # 일간 변화폭 (돌파 강도 확인용)
        df["price_change"] = df["close"] - df["close"].shift(1)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성."""
        close = row["close"]

        # --- 포지션 보유 중: SL/TP 체크 ---
        if self._in_position:
            # 손절 (진입 시점 ATR 기준으로 고정)
            sl_price = self._entry_price - self.sl_atr_mult * self._entry_atr
            if close < sl_price:
                self._in_position = False
                return -1

            # 익절
            tp_price = self._entry_price + self.tp_atr_mult * self._entry_atr
            if close > tp_price:
                self._in_position = False
                return -1

            return 0

        # --- 미보유: 돌파 매수 조건 ---
        atr_val = row.get("atr", 0)
        dc_upper = row.get("dc_upper", 0)
        ema_trend_val = row.get("ema_trend", close)
        adx_val = row.get("adx", 0)
        price_change = row.get("price_change", 0)

        if (
            close > dc_upper                              # 채널 상단 돌파
            and price_change > self.breakout_atr_mult * atr_val  # 충분한 돌파 강도
            and close > ema_trend_val                      # 장기 상승 추세
            and adx_val > self.adx_threshold               # 추세 존재
        ):
            self._in_position = True
            self._entry_price = close
            self._entry_atr = atr_val  # 진입 시점 ATR 고정 (SL/TP 일관성)
            return +1

        return 0
