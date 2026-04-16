"""
국면 기반 롱/숏 전략 v3 (Futures Regime Strategy).

설계 철학:
  롱과 숏은 완전히 다른 로직이 필요하다.
  - 롱: 추세 추종 (검증된 ensemble 로직 그대로 사용)
  - 숏: 바운스 페이드 (하락 추세 내 단기 반등에서 진입)

  크립토 하락장은 급격한 반등이 빈번하다.
  새로운 저점 (breakdown)에서 숏 진입 시 스퀴즈 위험이 크다.
  대신 반등 고점에서 숏 진입하면 리스크/리워드가 유리하다.

롱 진입 (Bull 국면: close > SMA200 AND ST=+1):
  ensemble과 동일한 3개 서브 신호 중 2개 이상 동의
  S1: EMA20 > EMA50 > EMA200 AND RSI > 50
  S2: 모멘텀 양수 + 단기 > 장기
  S3: 돈키안 상단 + 거래량 확인

숏 진입 (Bear 국면: close < SMA200 AND ST=-1, 10바 이상 유지):
  "바운스 페이드" — 하락 추세 내 단기 반등을 노림
  조건:
    1. Bear 국면 10바 이상 안정적으로 유지 (찐 하락장)
    2. RSI > 45 (단기 반등이 있었음)
    3. EMA20 < EMA50 (중기 추세는 여전히 하락)
    4. 단기 모멘텀 < 0 (반등이 가속되지 않음)
    5. ADX > 20 (추세 존재)

  이 방식의 장점:
    - 저점이 아닌 반등 고점 근처에서 진입 → 더 좋은 진입가
    - 이미 하락 추세가 충분히 확인된 후 진입
    - 스퀴즈 위험 감소

청산:
  롱: 국면 전환 OR ensemble 청산 신호 OR 트레일링 스탑 (2.5×ATR)
  숏: 국면 전환 OR RSI > 60 (반등 강해짐) OR 트레일링 스탑 (2.0×ATR)
      OR Bear 국면 진입 후 최대 30바 (시간 기반 청산)
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
    """국면 기반 롱/숏 전략 v3 — 바운스 페이드 숏.

    on_bar 반환값:
      +1 : 롱 (진입 or 유지)
      -1 : 숏 (진입 or 유지)
       0 : 플랫 (포지션 없음 / 청산)
    """

    name = "futures_regime"

    def __init__(
        self,
        # 국면 판단
        sma_period: int = 200,
        st_period: int = 10,
        st_multiplier: float = 2.5,
        # Bear 국면 최소 안정 바 수
        bear_stability: int = 10,
        # 롱 파라미터 (ensemble 동일)
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
        long_min_agreement: int = 2,
        long_sl_atr: float = 2.0,
        long_trailing_atr: float = 2.5,
        # 숏 파라미터 (바운스 페이드)
        short_rsi_min: float = 45.0,   # RSI 이 이상이어야 반등으로 간주
        short_rsi_max: float = 65.0,   # RSI 이 이상이면 반등 너무 강함 → 스킵
        short_adx_min: float = 20.0,
        short_sl_atr: float = 1.5,     # 타이트한 손절
        short_trailing_atr: float = 2.0,
        short_max_bars: int = 30,      # 최대 보유 기간 (바 수)
        # 숏 청산 RSI
        short_exit_rsi: float = 60.0,
    ):
        self.sma_period = sma_period
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.bear_stability = bear_stability
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
        self.long_min_agreement = long_min_agreement
        self.long_sl_atr = long_sl_atr
        self.long_trailing_atr = long_trailing_atr
        self.short_rsi_min = short_rsi_min
        self.short_rsi_max = short_rsi_max
        self.short_adx_min = short_adx_min
        self.short_sl_atr = short_sl_atr
        self.short_trailing_atr = short_trailing_atr
        self.short_max_bars = short_max_bars
        self.short_exit_rsi = short_exit_rsi

        self._position = 0
        self._entry_price = 0.0
        self._extreme_price = 0.0
        self._short_bars = 0         # 숏 보유 바 카운터

        self._current_regime = "neutral"
        self._regime_bars = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["sma200"] = sma(df["close"], self.sma_period)

        if supertrend is not None:
            _, st_dir = supertrend(df, self.st_period, self.st_multiplier)
            df["st_dir"] = st_dir
        else:
            df["st_dir"] = (df["close"] > df["sma200"]).astype(int) * 2 - 1

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
        df["adx"]      = adx(df, self.adx_period)
        df["atr"]      = atr(df, 14)
        return df

    # ── 국면 판단 ──────────────────────────────────────────────────────

    def _update_regime(self, row: pd.Series) -> tuple[str, int]:
        close  = row["close"]
        sma200 = row.get("sma200", close)
        st_dir = row.get("st_dir", 0)

        if close > sma200 and st_dir == 1:
            regime = "bull"
        elif close < sma200 and st_dir == -1:
            regime = "bear"
        else:
            regime = "neutral"

        if regime == self._current_regime:
            self._regime_bars += 1
        else:
            self._current_regime = regime
            self._regime_bars = 1

        return regime, self._regime_bars

    # ── 롱 신호 (ensemble 동일) ────────────────────────────────────────

    def _long_signals(self, row: pd.Series) -> int:
        close = row["close"]
        count = 0
        if (row.get("ema_fast", close) > row.get("ema_mid", close) >
                row.get("ema_slow", close) and row.get("rsi", 50) > 50):
            count += 1
        if row.get("mom_long", 0) > 0 and row.get("mom_short", 0) > row.get("mom_long", 0):
            count += 1
        if (close > row.get("dc_upper", close) and
                row.get("volume", 0) > row.get("vol_ma", 0) * self.vol_mult):
            count += 1
        return count

    def _long_exit_signals(self, row: pd.Series) -> int:
        close = row["close"]
        count = 0
        if row.get("ema_fast", close) < row.get("ema_mid", close): count += 1
        if row.get("mom_long", 0) < 0: count += 1
        if row.get("rsi", 50) > 75:    count += 1
        return count

    # ── 숏 진입: 바운스 페이드 ────────────────────────────────────────

    def _can_enter_short(self, row: pd.Series, adx_val: float) -> bool:
        """Bear 추세 내 단기 반등을 페이드하는 숏 진입 조건."""
        rsi_val    = row.get("rsi", 50)
        ema_f      = row.get("ema_fast", 0)
        ema_m      = row.get("ema_mid", 0)
        mom_s      = row.get("mom_short", 0)

        return (
            self.short_rsi_min <= rsi_val <= self.short_rsi_max  # 단기 반등 확인
            and ema_f < ema_m        # 중기 추세는 여전히 하락
            and mom_s < 0            # 단기 모멘텀 음수 (반등 미약)
            and adx_val >= self.short_adx_min
        )

    # ── 메인 루프 ──────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close   = row["close"]
        atr_val = row.get("atr", close * 0.02)
        adx_val = row.get("adx", 0)
        regime, regime_bars = self._update_regime(row)

        # ── 롱 포지션 ─────────────────────────────────────────────────
        if self._position == 1:
            if close > self._extreme_price:
                self._extreme_price = close
            if close < self._entry_price - self.long_sl_atr * atr_val:
                self._position = 0; return 0
            if close < self._extreme_price - self.long_trailing_atr * atr_val:
                self._position = 0; return 0
            if regime != "bull":
                self._position = 0; return 0
            if self._long_exit_signals(row) >= 2:
                self._position = 0; return 0
            return +1

        # ── 숏 포지션 ─────────────────────────────────────────────────
        if self._position == -1:
            self._short_bars += 1
            if close < self._extreme_price:
                self._extreme_price = close

            # 손절 (타이트)
            if close > self._entry_price + self.short_sl_atr * atr_val:
                self._position = 0; return 0
            # 트레일링 (타이트)
            if close > self._extreme_price + self.short_trailing_atr * atr_val:
                self._position = 0; return 0
            # 국면 전환 → 즉시 청산
            if regime != "bear":
                self._position = 0; return 0
            # RSI 반등 강해짐
            if row.get("rsi", 50) > self.short_exit_rsi:
                self._position = 0; return 0
            # 시간 기반 청산 (최대 30바)
            if self._short_bars >= self.short_max_bars:
                self._position = 0; return 0
            return -1

        # ── 신규 진입 ─────────────────────────────────────────────────

        # 롱: Bull 국면 + ensemble 신호
        if (regime == "bull"
                and adx_val >= self.adx_threshold
                and self._long_signals(row) >= self.long_min_agreement):
            self._position = 1
            self._entry_price = close
            self._extreme_price = close
            return +1

        # 숏: Bear 국면 10바 이상 안정 + 바운스 페이드
        if (regime == "bear"
                and regime_bars >= self.bear_stability
                and self._can_enter_short(row, adx_val)):
            self._position = -1
            self._entry_price = close
            self._extreme_price = close
            self._short_bars = 0
            return -1

        return 0
