"""
앙상블 숏 전략 (Ensemble Long/Short).

기존 앙상블 전략의 롱 전용 한계 극복:
- 2022년 하락장: ensemble -6.8% (BTC -65%)
- 하락장에서도 숏으로 수익 창출

전략 구성:
  롱 (기존 앙상블):
    - EMA 정배열 + RSI > 50
    - 모멘텀 양수 + 가속
    - 돈키안 상단 돌파 + 거래량

  숏 (반전):
    - EMA 역배열 (단기 < 장기) + RSI < 50
    - 모멘텀 음수
    - 종가 < SMA200 (구조적 하락장)
    - SuperTrend 방향 -1

  추가 안전장치:
    - 롱/숏 동시 보유 불가
    - ADX 필터 (추세 없는 횡보장 제외)
    - ATR 기반 손절 + 트레일링 스탑
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import (
    ema, rsi, atr, adx, momentum,
    donchian_channel, sma, supertrend,
)


class EnsembleShortStrategy(Strategy):
    """앙상블 롱/숏 전략.

    BTC의 강한 추세 특성을 이용하여 상승장에는 롱, 하락장에는 숏.
    """

    name = "ensemble_short"

    def __init__(
        self,
        # 추세 서브 신호
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        # 모멘텀
        mom_long: int = 90,
        mom_short: int = 20,
        # 브레이크아웃
        channel_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        # SuperTrend (숏 필터)
        st_period: int = 10,
        st_multiplier: float = 2.5,
        # 공통
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        min_long_signals: int = 2,   # 롱: 3개 중 2개
        min_short_signals: int = 2,  # 숏: 3개 중 2개
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
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.min_long_signals = min_long_signals
        self.min_short_signals = min_short_signals
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._position = 0       # +1 롱, -1 숏, 0 없음
        self._entry_price = 0.0
        self._extreme = 0.0      # 롱: 고가, 숏: 저가

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"]  = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["rsi"]      = rsi(df["close"], self.rsi_period)
        df["mom_long"] = momentum(df["close"], self.mom_long)
        df["mom_short"]= momentum(df["close"], self.mom_short)

        dc_upper, dc_lower = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)
        df["dc_lower"] = dc_lower.shift(1)
        df["vol_ma"]   = sma(df["volume"], self.vol_period)

        _, st_dir = supertrend(df, self.st_period, self.st_multiplier)
        df["st_dir"]   = st_dir

        df["sma200"]   = sma(df["close"], self.ema_slow)
        df["adx"]      = adx(df, self.adx_period)
        df["atr"]      = atr(df, 14)

        return df

    def _long_signals(self, row: pd.Series) -> int:
        close = row["close"]
        count = 0
        ef = row.get("ema_fast", close)
        em = row.get("ema_mid",  close)
        es = row.get("ema_slow", close)
        rsi_v = row.get("rsi", 50)
        if ef > em > es and rsi_v > 50:
            count += 1
        ml = row.get("mom_long", 0)
        ms = row.get("mom_short", 0)
        if ml > 0 and ms > ml:
            count += 1
        dc_up = row.get("dc_upper", 0)
        vol   = row.get("volume", 0)
        vol_ma= row.get("vol_ma", 0)
        if close > dc_up and vol > vol_ma * self.vol_mult:
            count += 1
        return count

    def _short_signals(self, row: pd.Series) -> int:
        close = row["close"]
        count = 0
        ef = row.get("ema_fast", close)
        em = row.get("ema_mid",  close)
        es = row.get("ema_slow", close)
        rsi_v = row.get("rsi", 50)
        # 역배열 + RSI < 50
        if ef < em < es and rsi_v < 50:
            count += 1
        ml = row.get("mom_long", 0)
        ms = row.get("mom_short", 0)
        # 모멘텀 음수 + 가속 하락
        if ml < 0 and ms < ml:
            count += 1
        dc_low = row.get("dc_lower", float("inf"))
        vol    = row.get("volume", 0)
        vol_ma = row.get("vol_ma", 0)
        # 하단 채널 하향 이탈 + 거래량
        if close < dc_low and vol > vol_ma * self.vol_mult:
            count += 1
        return count

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_v  = row.get("atr", 1.0)
        adx_v  = row.get("adx", 0.0)
        st_dir = int(row.get("st_dir", 0))
        sma200 = row.get("sma200", 0.0)

        # ── 롱 포지션 청산 ──────────────────────────────────────────────
        if self._position == 1:
            if close > self._extreme:
                self._extreme = close
            if (
                close < self._entry_price - self.sl_atr_mult * atr_v
                or close < self._extreme - self.trailing_atr_mult * atr_v
                or self._long_signals(row) == 0
            ):
                self._position = 0
                return -1
            return 0

        # ── 숏 포지션 청산 ──────────────────────────────────────────────
        if self._position == -1:
            if close < self._extreme:
                self._extreme = close
            if (
                close > self._entry_price + self.sl_atr_mult * atr_v
                or close > self._extreme + self.trailing_atr_mult * atr_v
                or self._short_signals(row) == 0
            ):
                self._position = 0
                return +1   # 숏 청산 = 매수 신호
            return 0

        # ── 신규 진입 ───────────────────────────────────────────────────
        if adx_v < self.adx_threshold:
            return 0

        long_cnt  = self._long_signals(row)
        short_cnt = self._short_signals(row)

        # 롱 진입: 신호 충분 + SuperTrend 양수 + SMA200 위
        if long_cnt >= self.min_long_signals and st_dir == 1 and close > sma200:
            self._position = 1
            self._entry_price = close
            self._extreme = close
            return +1

        # 숏 진입: 신호 충분 + SuperTrend 음수 + SMA200 아래
        if short_cnt >= self.min_short_signals and st_dir == -1 and close < sma200:
            self._position = -1
            self._entry_price = close
            self._extreme = close
            return -1

        return 0
