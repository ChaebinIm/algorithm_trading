"""
일봉 장대양봉 모멘텀 전략 — 롱온리.

설계 철학:
  일봉 장대양봉(강한 매수세) 이후 하위 TF(1h/4h)에서 빠르게 편승.
  눌림목을 기다리지 않고 다음날 첫 양봉에 즉시 진입.

근거 (BTC 1h 백테스트):
  - daily_mult=1.5: IS 29건 83% win, OOS 31건 74% win (TP=1.5%/SL=0.6%)
  - daily_mult=1.0: IS 106건 75% win, OOS 79건 73% win
  - 모두 양의 EV, IS/OOS 일관성 확인됨

진입:
  - 일봉 장대양봉 다음날에 한해 신호 활성화
  - 하위 TF에서 첫 양봉(close > open) 봉에 진입
  - 하루(entry_window_bars) 내 진입 없으면 신호 소멸

청산:
  1. close >= entry * (1 + tp_pct)     (익절)
  2. close <= entry * (1 - sl_pct)     (손절)
  3. 트레일링 스탑
  4. max_bars 초과                      (시간 청산)

핵심 구현 주의사항:
  - daily signal을 1h에 매핑할 때 limit=24 (하루치만 ffill)
  - 청산 후 당일 재활성화 금지 (_last_signal_date)
  - 장대양봉 당일 데이터 사용 금지 (lookahead 없음, shift(1))
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import atr as calc_atr, sma


class JangdaeMomentumStrategy(Strategy):
    """일봉 장대양봉 다음날 1h/4h 첫 양봉 진입 모멘텀 전략."""

    name = "jangdae_momentum"

    def __init__(
        self,
        # 일봉 장대양봉 기준
        daily_mult: float = 1.0,       # body >= 일봉 ATR × 이 값 (1.0=완화, 1.2=엄격)
        daily_atr_period: int = 14,
        # 신호 활성화 창 (하위 TF 봉 수)
        entry_window_bars: int = 5,    # 일봉 기준 5봉 = 5일 이내 첫 양봉에 진입
        # 손익 (일봉 변동성에 맞게)
        tp_pct: float = 0.060,         # 익절 6.0%
        sl_pct: float = 0.030,         # 손절 3.0%
        # 트레일링
        trailing_pct: float = 0.040,   # 고점 대비 4.0% 하락 시 청산
        # 기타
        atr_period: int = 14,
        max_bars: int = 5,             # 최대 보유 5일
        # 추세 필터
        sma_period: int = 200,         # SMA200 필터 (상승추세에서만 진입)
        # 진입 봉 최소 크기 (노이즈 필터)
        min_body_atr: float = 0.3,     # 진입 봉 body >= ATR × 이 값 (0=비활성)
        # 진입 봉 거래대금 조건
        entry_vol_mult: float = 0.0,   # 진입 봉 volume >= 이동평균 × 이 값 (0=비활성)
        vol_ma_period: int = 20,       # 거래대금 이동평균 기간
    ):
        self.daily_mult = daily_mult
        self.daily_atr_period = daily_atr_period
        self.entry_window_bars = entry_window_bars
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_pct = trailing_pct
        self.atr_period = atr_period
        self.max_bars = max_bars
        self.sma_period = sma_period
        self.min_body_atr = min_body_atr
        self.entry_vol_mult = entry_vol_mult
        self.vol_ma_period = vol_ma_period

        # 포지션
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._bars_held = 0

        # 신호 추적
        self._signal_active = False
        self._bars_since_signal = 0
        self._last_signal_date = None   # 당일 재활성화 방지

    # ── 준비 ─────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 하위 TF 지표
        df["atr"] = calc_atr(df, self.atr_period)
        df["body"] = df["close"] - df["open"]
        df["vol_ma"] = df["volume"].rolling(self.vol_ma_period).mean()
        if self.sma_period > 0:
            df["sma200"] = sma(df["close"], self.sma_period)
        else:
            df["sma200"] = 0.0

        # 일봉 리샘플
        daily = df.resample("1D").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["close"])

        daily_atr = calc_atr(daily, self.daily_atr_period)
        daily["daily_atr"] = daily_atr
        daily["daily_body"] = daily["close"] - daily["open"]

        # 장대양봉 감지 (당일 종가 확정 후 다음날부터 유효 → shift(1))
        daily["is_jangdae"] = (
            (daily["daily_body"] >= self.daily_mult * daily["daily_atr"]) &
            (daily["daily_body"] > 0)
        )
        daily["prev_jangdae"] = daily["is_jangdae"].shift(1).fillna(False)

        # 1h에 매핑: limit=24로 하루치 봉만 True (재활성화 방지)
        # daily index는 하루 단위이므로 1h에 ffill하면 다음 daily 값이 올 때까지 전파
        # 그러나 daily가 매일 있으므로 자동으로 하루 단위로 끊김
        jd = daily["prev_jangdae"].reindex(df.index, method="ffill").fillna(False)
        df["daily_jangdae"] = jd.astype(bool)

        return df

    # ── 청산 ─────────────────────────────────────────────────────────────

    def _should_exit(self, row: pd.Series) -> bool:
        close = row["close"]
        entry = self._entry_price

        if close >= entry * (1 + self.tp_pct):
            return True
        if close <= entry * (1 - self.sl_pct):
            return True

        trail = self._peak_price * (1 - self.trailing_pct)
        if close < trail:
            return True

        if self._bars_held >= self.max_bars:
            return True

        return False

    # ── 메인 ─────────────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close  = row["close"]
        open_  = row["open"]
        body   = row.get("body", close - open_)
        atr    = row.get("atr", 0.0)
        volume = row.get("volume", 0.0)
        vol_ma = row.get("vol_ma", 0.0)
        is_jangdae = bool(row.get("daily_jangdae", False))
        sma200 = row.get("sma200", 0.0)
        min_body = self.min_body_atr * atr
        vol_ok = (self.entry_vol_mult == 0.0) or (vol_ma <= 0) or (volume >= self.entry_vol_mult * vol_ma)

        # 날짜 추출 (타임스탬프 → 날짜)
        try:
            current_date = row.name.date()
        except AttributeError:
            current_date = None

        # 추세 필터
        in_uptrend = (self.sma_period == 0) or (close > sma200)

        # ── 포지션 보유 중 ───────────────────────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            if close > self._peak_price:
                self._peak_price = close
            if self._should_exit(row):
                self._position = 0
                self._signal_active = False
                # 청산 시 당일 재활성화 금지
                self._last_signal_date = current_date
                return -1
            return 0

        # ── 신호 활성화 ──────────────────────────────────────────────────
        # 당일 이미 사용된 신호는 재활성화 금지
        if is_jangdae and in_uptrend and not self._signal_active:
            if current_date != self._last_signal_date:
                self._signal_active = True
                self._bars_since_signal = 0

        # ── 신호 추적 ────────────────────────────────────────────────────
        if self._signal_active:
            self._bars_since_signal += 1

            # 신호 만료
            if self._bars_since_signal > self.entry_window_bars:
                self._signal_active = False
                self._last_signal_date = current_date
                return 0

            # 진입: 양봉 + 최소 body 크기 + 거래대금 조건
            if body > min_body and vol_ok:
                self._position = 1
                self._entry_price = close
                self._peak_price = close
                self._bars_held = 0
                self._signal_active = False
                self._last_signal_date = current_date
                return +1

        return 0
