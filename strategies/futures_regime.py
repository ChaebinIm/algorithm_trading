"""
국면 기반 롱/숏 전략 (Futures Regime Strategy).

핵심 아이디어:
  시장 국면(Bull / Bear / Neutral)을 먼저 판단하고,
  국면에 맞는 방향으로만 진입한다.
  강세장 → 롱만 / 약세장 → 숏만 / 불명확 → 현금

국면 판단:
  Bull  : close > SMA200 AND SuperTrend = +1
  Bear  : close < SMA200 AND SuperTrend = -1
  Neutral: 그 외 → 포지션 없음

롱 진입 (Bull 국면):
  앙상블 3개 서브 신호 중 2개 이상 동의
    S1 (추세): EMA20 > EMA50 > EMA200 AND RSI > 50
    S2 (모멘텀): 20기간 모멘텀 > 0 AND 단기 모멘텀 > 장기 모멘텀
    S3 (브레이크아웃): 종가 > 돈키안 상단 AND 거래량 > 평균

숏 진입 (Bear 국면):
  앙상블 3개 서브 신호 중 2개 이상 동의
    S1 (역추세): EMA20 < EMA50 < EMA200 AND RSI < 50
    S2 (역모멘텀): 20기간 모멘텀 < 0 AND 단기 모멘텀 < 장기 모멘텀
    S3 (브레이크다운): 종가 < 돈키안 하단 AND 거래량 > 평균

청산:
  롱: 국면 비Bull 전환 OR 2개 이상 청산 신호 OR 트레일링 스탑
  숏: 국면 비Bear 전환 OR 2개 이상 커버 신호 OR 트레일링 스탑
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, rsi, atr, adx, momentum, donchian_channel, sma
try:
    from core.indicators import supertrend
except ImportError:
    supertrend = None


class FuturesRegimeStrategy(Strategy):
    """국면 기반 롱/숏 앙상블 전략.

    on_bar 반환값:
      +1 : 롱 진입 (or 롱 유지)
      -1 : 숏 진입 (or 숏 유지)
       0 : 플랫 (포지션 없음 / 청산)
    """

    name = "futures_regime"

    def __init__(
        self,
        # 국면 판단
        sma_period: int = 200,
        st_period: int = 10,
        st_multiplier: float = 2.5,
        # 앙상블 파라미터
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        mom_long: int = 90,
        mom_short: int = 20,
        channel_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        min_agreement: int = 2,
        # 리스크 관리
        trailing_atr_mult: float = 2.5,
        sl_atr_mult: float = 2.0,
    ):
        self.sma_period = sma_period
        self.st_period = st_period
        self.st_multiplier = st_multiplier
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
        self.trailing_atr_mult = trailing_atr_mult
        self.sl_atr_mult = sl_atr_mult

        # 포지션 상태
        self._position = 0        # +1=롱, -1=숏, 0=없음
        self._entry_price = 0.0
        self._extreme_price = 0.0  # 롱: 고점 추적 / 숏: 저점 추적

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 국면 판단 지표
        df["sma200"] = sma(df["close"], self.sma_period)

        if supertrend is not None:
            st_line, st_dir = supertrend(df, self.st_period, self.st_multiplier)
            df["st_dir"] = st_dir
        else:
            # supertrend 없으면 SMA200만으로 국면 판단
            df["st_dir"] = (df["close"] > df["sma200"]).astype(int) * 2 - 1

        # 앙상블 지표
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["mom_long"] = momentum(df["close"], self.mom_long)
        df["mom_short"] = momentum(df["close"], self.mom_short)

        dc_upper, dc_lower = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)
        df["dc_lower"] = dc_lower.shift(1)
        df["vol_ma"] = sma(df["volume"], self.vol_period)

        df["adx"] = adx(df, self.adx_period)
        df["atr"] = atr(df, 14)

        return df

    # ── 국면 판단 ──────────────────────────────────────────────────────

    def _get_regime(self, row: pd.Series) -> str:
        """Bull / Bear / Neutral 반환."""
        close = row["close"]
        sma200 = row.get("sma200", close)
        st_dir = row.get("st_dir", 0)

        if close > sma200 and st_dir == 1:
            return "bull"
        if close < sma200 and st_dir == -1:
            return "bear"
        return "neutral"

    # ── 롱 신호 ───────────────────────────────────────────────────────

    def _long_signals(self, row: pd.Series) -> int:
        """롱 진입 서브 신호 개수 (0~3)."""
        close = row["close"]
        count = 0

        # S1: EMA 정배열 + RSI > 50
        if (row.get("ema_fast", close) > row.get("ema_mid", close) >
                row.get("ema_slow", close) and row.get("rsi", 50) > 50):
            count += 1

        # S2: 모멘텀 양수 + 단기 > 장기
        mom_l = row.get("mom_long", 0)
        mom_s = row.get("mom_short", 0)
        if mom_l > 0 and mom_s > mom_l:
            count += 1

        # S3: 돈키안 상단 돌파 + 거래량 확인
        if (close > row.get("dc_upper", close) and
                row.get("volume", 0) > row.get("vol_ma", 0) * self.vol_mult):
            count += 1

        return count

    def _long_exit_signals(self, row: pd.Series) -> int:
        """롱 청산 서브 신호 개수."""
        close = row["close"]
        count = 0

        if row.get("ema_fast", close) < row.get("ema_mid", close):
            count += 1
        if row.get("mom_long", 0) < 0:
            count += 1
        if row.get("rsi", 50) > 75:
            count += 1

        return count

    # ── 숏 신호 ───────────────────────────────────────────────────────

    def _short_signals(self, row: pd.Series) -> int:
        """숏 진입 서브 신호 개수 (0~3)."""
        close = row["close"]
        count = 0

        # S1: EMA 역배열 + RSI < 50
        if (row.get("ema_fast", close) < row.get("ema_mid", close) <
                row.get("ema_slow", close) and row.get("rsi", 50) < 50):
            count += 1

        # S2: 모멘텀 음수 + 단기 < 장기 (하락 가속)
        mom_l = row.get("mom_long", 0)
        mom_s = row.get("mom_short", 0)
        if mom_l < 0 and mom_s < mom_l:
            count += 1

        # S3: 돈키안 하단 이탈 + 거래량 확인
        if (close < row.get("dc_lower", close) and
                row.get("volume", 0) > row.get("vol_ma", 0) * self.vol_mult):
            count += 1

        return count

    def _short_exit_signals(self, row: pd.Series) -> int:
        """숏 청산(커버) 서브 신호 개수."""
        close = row["close"]
        count = 0

        if row.get("ema_fast", close) > row.get("ema_mid", close):
            count += 1
        if row.get("mom_long", 0) > 0:
            count += 1
        if row.get("rsi", 50) < 25:  # 과매도 → 반등 가능
            count += 1

        return count

    # ── 메인 루프 ──────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", close * 0.02)
        adx_val = row.get("adx", 0)
        regime = self._get_regime(row)

        # ── 롱 포지션 보유 중 ─────────────────────────────────────────
        if self._position == 1:
            # 고점 갱신
            if close > self._extreme_price:
                self._extreme_price = close

            # 초기 손절
            if close < self._entry_price - self.sl_atr_mult * atr_val:
                self._position = 0
                return 0

            # 트레일링 스탑
            if close < self._extreme_price - self.trailing_atr_mult * atr_val:
                self._position = 0
                return 0

            # 국면 이탈 → 청산
            if regime != "bull":
                self._position = 0
                return 0

            # 앙상블 청산 신호
            if self._long_exit_signals(row) >= 2:
                self._position = 0
                return 0

            return +1  # 롱 유지

        # ── 숏 포지션 보유 중 ─────────────────────────────────────────
        if self._position == -1:
            # 저점 갱신
            if close < self._extreme_price:
                self._extreme_price = close

            # 초기 손절 (숏: 가격이 올라가면 손실)
            if close > self._entry_price + self.sl_atr_mult * atr_val:
                self._position = 0
                return 0

            # 트레일링 스탑 (숏: 저점 대비 반등 시 청산)
            if close > self._extreme_price + self.trailing_atr_mult * atr_val:
                self._position = 0
                return 0

            # 국면 이탈 → 청산
            if regime != "bear":
                self._position = 0
                return 0

            # 앙상블 커버 신호
            if self._short_exit_signals(row) >= 2:
                self._position = 0
                return 0

            return -1  # 숏 유지

        # ── 미보유: 신규 진입 ─────────────────────────────────────────
        if adx_val < self.adx_threshold:
            return 0  # 추세 없는 횡보장 → 스킵

        if regime == "bull" and self._long_signals(row) >= self.min_agreement:
            self._position = 1
            self._entry_price = close
            self._extreme_price = close
            return +1

        if regime == "bear" and self._short_signals(row) >= self.min_agreement:
            self._position = -1
            self._entry_price = close
            self._extreme_price = close
            return -1

        return 0
