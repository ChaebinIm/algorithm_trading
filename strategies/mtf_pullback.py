"""
MTF 눌림목 전략 (Multi-Timeframe Pullback) — 롱온리.

설계 철학:
  일봉 장대양봉 = "오늘/어제 강한 매수세" 확인.
  그 안에서 하위 타임프레임(1h/5m 등)으로 눌림목을 잡아 단기 매매.

패턴 흐름:
  [상위 TF — 일봉]
  1. 일봉 body >= daily_atr × daily_mult  (장대양봉)
  2. 신호 활성화: 이후 entry_window_bars 봉 동안 진입 허용

  [하위 TF — 현재 봉 주기]
  3. 신호 활성화 구간 내에서:
     - 최저가가 신호 발생 이후 최고 종가 대비 pullback_pct 이상 하락 (눌림)
     - 눌림 후 첫 양봉 → 진입

청산:
  1. close >= entry × (1 + tp_pct)     (익절)
  2. close <= entry × (1 - sl_pct)     (손절)
  3. 트레일링 스탑
  4. max_bars 초과                      (시간 청산)

사용 방법:
  - df: 1h 또는 5m OHLCV 데이터
  - prepare() 내부에서 일봉 리샘플링 후 신호 매핑
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import atr as calc_atr


class MTFPullbackStrategy(Strategy):
    """일봉 장대양봉 후 하위 TF 눌림목 진입 전략."""

    name = "mtf_pullback"

    def __init__(
        self,
        # 일봉 장대양봉 기준
        daily_mult: float = 1.5,      # 일봉 body >= 일봉 ATR × 이 값
        daily_atr_period: int = 14,   # 일봉 ATR 기간
        # 신호 유효 기간 (하위 TF 봉 수)
        entry_window_bars: int = 24,  # 장대양봉 이후 진입 허용 창 (1h 기준 24봉 = 1일)
        # 눌림 기준
        pullback_pct: float = 0.005,  # 최고 종가 대비 이 비율 이상 하락 시 눌림 인정 (0.5%)
        # 손익 (하위 TF 기준으로 타이트하게)
        tp_pct: float = 0.012,        # 익절 1.2%
        sl_pct: float = 0.006,        # 손절 0.6%
        # 트레일링
        trailing_pct: float = 0.010,  # 고점 대비 1.0% 하락 시 트레일 청산
        # 기타
        atr_period: int = 14,
        max_bars: int = 12,           # 최대 보유 기간 (1h 기준 12봉 = 반일)
    ):
        self.daily_mult = daily_mult
        self.daily_atr_period = daily_atr_period
        self.entry_window_bars = entry_window_bars
        self.pullback_pct = pullback_pct
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_pct = trailing_pct
        self.atr_period = atr_period
        self.max_bars = max_bars

        # 포지션
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._bars_held = 0

        # 패턴 추적
        self._signal_active = False    # 일봉 장대양봉 신호 활성화 여부
        self._bars_since_signal = 0   # 신호 이후 경과 봉 수
        self._ref_high = 0.0          # 신호 이후 최고 종가 (눌림 기준점)
        self._pullback_done = False    # 눌림 조건 충족 여부

    # ── 준비 ─────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 하위 TF ATR
        df["atr"] = calc_atr(df, self.atr_period)
        df["body"] = df["close"] - df["open"]

        # 인덱스가 DatetimeIndex가 아니면 변환
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 일봉 리샘플링 → 장대양봉 감지
        daily = df.resample("1D").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).dropna(subset=["close"])

        # 일봉 ATR
        daily_atr = calc_atr(daily, self.daily_atr_period)
        daily["daily_atr"] = daily_atr
        daily["daily_body"] = daily["close"] - daily["open"]

        # 장대양봉 플래그 (당일 종가 확정 후 → 다음날부터 유효)
        daily["is_jangdae"] = (
            (daily["daily_body"] >= self.daily_mult * daily["daily_atr"]) &
            (daily["daily_body"] > 0)
        )
        # 전일 장대양봉 여부를 하위 TF에 매핑 (lookahead 없음)
        # shift(1) → 오늘은 어제 장대양봉인지 확인
        daily["prev_jangdae"] = daily["is_jangdae"].shift(1).fillna(False)

        # 하위 TF에 일봉 신호 매핑 (forward fill)
        jangdae_series = daily["prev_jangdae"].reindex(df.index, method="ffill").fillna(False)
        df["daily_jangdae"] = jangdae_series.astype(bool)

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
        close = row["close"]
        open_ = row["open"]
        body  = row.get("body", close - open_)
        is_jangdae = bool(row.get("daily_jangdae", False))

        # ── 포지션 보유 중 ───────────────────────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            if close > self._peak_price:
                self._peak_price = close
            if self._should_exit(row):
                self._position = 0
                self._signal_active = False
                self._pullback_done = False
                return -1
            return 0

        # ── 신호 활성화 (일봉 장대양봉 감지) ────────────────────────────
        if is_jangdae and not self._signal_active:
            self._signal_active = True
            self._bars_since_signal = 0
            self._ref_high = close
            self._pullback_done = False

        # ── 신호 추적 ────────────────────────────────────────────────────
        if self._signal_active:
            self._bars_since_signal += 1

            # 기준 고점 갱신
            if close > self._ref_high:
                self._ref_high = close

            # 신호 만료
            if self._bars_since_signal > self.entry_window_bars:
                self._signal_active = False
                self._pullback_done = False
                return 0

            # 눌림 조건: 기준 고점 대비 pullback_pct 이상 하락
            if close <= self._ref_high * (1 - self.pullback_pct):
                self._pullback_done = True

            # 진입: 눌림 후 첫 양봉
            if self._pullback_done and body > 0:
                self._position = 1
                self._entry_price = close
                self._peak_price = close
                self._bars_held = 0
                self._signal_active = False
                self._pullback_done = False
                return +1

        return 0
