"""
앙상블 전략 (Ensemble / Multi-Signal).

여러 전략의 신호를 종합하여 다수결로 진입/청산을 결정.
- 개별 전략의 노이즈를 서로 상쇄
- 2개 이상 전략이 동시에 매수 신호일 때만 진입
- 과반수 매도 시 청산
- 가짜 신호 필터링 효과

구성 전략:
1. 추세 추종 (EMA 정배열 + RSI 반등)
2. 모멘텀 (절대 모멘텀 + 가속)
3. 브레이크아웃 (돈키안 채널 + 거래량)
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import (
    ema, rsi, atr, adx, momentum,
    donchian_channel, sma,
)


class EnsembleStrategy(Strategy):
    """앙상블 전략.

    3개 서브 신호를 독립적으로 계산하고 다수결로 최종 신호 결정.

    서브 신호 1 (추세): EMA 정배열 + RSI > 50
    서브 신호 2 (모멘텀): 20기간 모멘텀 > 0 + 단기 > 장기 모멘텀
    서브 신호 3 (브레이크아웃): 종가 > 돈키안 상단 + 거래량 확인

    진입: 서브 신호 min_agreement개 이상 매수일 때 (기본 2/3)
    청산: ATR 손절 + 트레일링 스탑 + 서브 신호 과반 매도

    세션 필터 없음 (서브 전략들이 이미 노이즈 필터 역할)
    """

    name = "ensemble"

    def __init__(
        self,
        # 추세 서브 신호 파라미터
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        # 모멘텀 서브 신호 파라미터
        mom_long: int = 90,
        mom_short: int = 20,
        # 브레이크아웃 서브 신호 파라미터
        channel_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        # 공통
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        min_agreement: int = 2,        # 최소 동의 수 (3개 중 N개)
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 2.5,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.mom_long = mom_long
        self.mom_short = mom_short
        self.channel_period = channel_period
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.min_agreement = min_agreement
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- 추세 서브 신호 지표 ---
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["rsi"] = rsi(df["close"], self.rsi_period)

        # --- 모멘텀 서브 신호 지표 ---
        df["mom_long"] = momentum(df["close"], self.mom_long)
        df["mom_short"] = momentum(df["close"], self.mom_short)

        # --- 브레이크아웃 서브 신호 지표 ---
        dc_upper, _ = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)
        df["vol_ma"] = sma(df["volume"], self.vol_period)

        # --- 공통 ---
        df["adx"] = adx(df, self.adx_period)
        df["atr"] = atr(df, 14)

        return df

    def _count_buy_signals(self, row: pd.Series) -> int:
        """서브 전략별 매수 신호 개수를 세어 반환."""
        close = row["close"]
        count = 0

        # 서브 신호 1: 추세
        ema_f = row.get("ema_fast", close)
        ema_m = row.get("ema_mid", close)
        ema_s = row.get("ema_slow", close)
        rsi_val = row.get("rsi", 50)
        if ema_f > ema_m > ema_s and rsi_val > 50:
            count += 1

        # 서브 신호 2: 모멘텀
        mom_l = row.get("mom_long", 0)
        mom_s = row.get("mom_short", 0)
        if mom_l > 0 and mom_s > mom_l:
            count += 1

        # 서브 신호 3: 브레이크아웃
        dc_upper = row.get("dc_upper", 0)
        volume = row.get("volume", 0)
        vol_ma = row.get("vol_ma", 0)
        if close > dc_upper and volume > vol_ma * self.vol_mult:
            count += 1

        return count

    def _count_sell_signals(self, row: pd.Series) -> int:
        """서브 전략별 매도 신호 개수를 세어 반환."""
        close = row["close"]
        count = 0

        # 추세 역전
        if row.get("ema_fast", close) < row.get("ema_mid", close):
            count += 1

        # 모멘텀 음수
        if row.get("mom_long", 0) < 0:
            count += 1

        # RSI 과매수 후 하락
        if row.get("rsi", 50) > 75:
            count += 1

        return count

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 0)
        adx_val = row.get("adx", 0)

        # --- 포지션 보유 중: 청산 ---
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # 손절
            if close < self._entry_price - self.sl_atr_mult * atr_val:
                self._in_position = False
                return -1

            # 트레일링 스탑
            if close < self._highest - self.trailing_atr_mult * atr_val:
                self._in_position = False
                return -1

            # 앙상블 매도: 2개 이상 매도 신호
            if self._count_sell_signals(row) >= 2:
                self._in_position = False
                return -1

            return 0

        # --- 미보유: 앙상블 매수 ---
        if adx_val < self.adx_threshold:
            return 0  # 최소 추세 없으면 스킵

        buy_count = self._count_buy_signals(row)
        if buy_count >= self.min_agreement:
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
