"""
모멘텀 눌림목 전략 (Momentum Pullback) — 롱온리.

설계 철학:
  '아무 장대양봉'이 아닌 '브레이크아웃 장대양봉'만 선별.
  - N봉 최고가 돌파 + 큰 몸통 = 저항선 돌파 증명
  - 이후 눌림(약손 청산·차익실현) → 해당 레벨 지지 확인
  - 지지 확인 후 첫 양봉 = 강한 손 재진입 시그널 → 편승

패턴 흐름:
  1. 브레이크아웃 감지 : close >= 직전 N봉 최고 close (breakout_window)
                        AND body >= ATR × body_atr_mult
  2. 눌림 확인        : 장대양봉 종가 대비 pullback_ratio 이상 하락
  3. 진입 시그널      : 눌림 후 첫 번째 양봉 (Option A)
  4. 패턴 만료        : expire_bars 초과 시 무효

청산 조건:
  1. close >= entry × (1 + tp_pct)     (익절)
  2. close <= entry × (1 - sl_pct)     (손절)
  3. 트레일링 스탑 — max(ATR기반, 퍼센트 상한)
  4. max_bars 초과                      (시간 청산)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import atr, sma


class MomentumPullbackStrategy(Strategy):
    """장대양봉 눌림목 후 회복 시 롱 진입."""

    name = "momentum_pullback"

    def __init__(
        self,
        # 브레이크아웃 기준
        breakout_window: int = 20,     # 직전 N봉 최고 close를 돌파해야 함
        body_atr_mult: float = 1.5,    # body >= ATR × 이 값
        # 눌림 기준
        pullback_ratio: float = 0.30,  # body의 이 비율만큼 하락해야 눌림 인정
        # 손익
        tp_pct: float = 0.025,         # 익절 2.5%
        sl_pct: float = 0.012,         # 손절 1.2%
        # 트레일링
        trailing_atr: float = 1.5,
        max_trail_pct: float = 0.025,
        # 기타
        atr_period: int = 14,
        sma_period: int = 200,         # 거시 추세 필터 (0 = 비활성)
        expire_bars: int = 10,         # 패턴 유효 기간 (봉 수)
        max_bars: int = 20,            # 최대 보유 기간
    ):
        self.breakout_window = breakout_window
        self.body_atr_mult = body_atr_mult
        self.pullback_ratio = pullback_ratio
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_atr = trailing_atr
        self.max_trail_pct = max_trail_pct
        self.atr_period = atr_period
        self.sma_period = sma_period
        self.expire_bars = expire_bars
        self.max_bars = max_bars

        # 포지션 상태
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._bars_held = 0

        # 패턴 추적
        self._watching = False          # 장대양봉 이후 눌림 대기 중
        self._signal_close = 0.0       # 장대양봉 종가
        self._signal_body = 0.0        # 장대양봉 body 크기
        self._min_since = float("inf") # 장대양봉 이후 최저 종가
        self._bars_since = 0           # 장대양봉 이후 경과 봉 수
        self._pullback_done = False    # 눌림 조건 충족 여부

    # ── 지표 계산 ─────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"]  = atr(df, self.atr_period)
        df["body"] = df["close"] - df["open"]   # 양수=양봉, 음수=음봉
        # 직전 N봉 최고 종가 (현재 봉 제외)
        df["prev_high_close"] = df["close"].shift(1).rolling(self.breakout_window).max()
        if self.sma_period > 0:
            df["sma200"] = sma(df["close"], self.sma_period)
        else:
            df["sma200"] = 0.0
        return df

    # ── 청산 ─────────────────────────────────────────────────────────────

    def _should_exit(self, row: pd.Series) -> bool:
        close = row["close"]
        atr_v = row.get("atr", close * 0.01)
        entry = self._entry_price

        # 익절
        if close >= entry * (1 + self.tp_pct):
            return True

        # 손절
        if close <= entry * (1 - self.sl_pct):
            return True

        # 트레일링 스탑 — 고점에서 ATR 또는 퍼센트 중 타이트한 쪽
        trail = max(
            self._peak_price - self.trailing_atr * atr_v,
            self._peak_price * (1 - self.max_trail_pct),
        )
        if close < trail:
            return True

        # 최대 보유 기간
        if self._bars_held >= self.max_bars:
            return True

        return False

    # ── 패턴 리셋 ─────────────────────────────────────────────────────────

    def _reset_pattern(self):
        self._watching = False
        self._signal_close = 0.0
        self._signal_body = 0.0
        self._min_since = float("inf")
        self._bars_since = 0
        self._pullback_done = False

    # ── 메인 ─────────────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close  = row["close"]
        open_  = row["open"]
        atr_v  = row.get("atr", close * 0.01)
        body   = row.get("body", close - open_)

        # ── 포지션 보유 중 ───────────────────────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            if close > self._peak_price:
                self._peak_price = close
            if self._should_exit(row):
                self._position = 0
                return -1   # 청산
            return 0        # 보유 유지

        # ── 패턴 탐색 ───────────────────────────────────────────────────
        sma200 = row.get("sma200", 0.0)
        # SMA200 필터: 하락 추세에서는 진입 안 함
        in_uptrend = (self.sma_period == 0) or (close > sma200)

        prev_high = row.get("prev_high_close", 0.0)
        is_breakout = (close > prev_high) and not pd.isna(prev_high)

        if not self._watching:
            # 브레이크아웃 장대양봉 감지: 상승추세 + N봉 최고가 돌파 + body >= ATR × 배수
            if in_uptrend and is_breakout and body >= self.body_atr_mult * atr_v and body > 0:
                self._watching = True
                self._signal_close = close
                self._signal_body = body
                self._min_since = close
                self._bars_since = 0
                self._pullback_done = False

        else:
            # 장대양봉 이후 추적 중
            self._bars_since += 1

            # 더 강한 브레이크아웃이 오면 기준 갱신
            if in_uptrend and is_breakout and body >= self.body_atr_mult * atr_v and body > 0:
                self._signal_close = close
                self._signal_body = body
                self._min_since = close
                self._bars_since = 0
                self._pullback_done = False
                return 0

            # 유효 기간 초과 → 리셋
            if self._bars_since > self.expire_bars:
                self._reset_pattern()
                return 0

            # 최저가 갱신
            if close < self._min_since:
                self._min_since = close

            # 눌림 조건 확인: 장대 body의 pullback_ratio 이상 하락
            pullback_threshold = self._signal_close - self._signal_body * self.pullback_ratio
            if self._min_since <= pullback_threshold:
                self._pullback_done = True

            # 진입 시그널: 눌림 완료 후 첫 번째 양봉
            if self._pullback_done and body > 0:
                self._position = 1
                self._entry_price = close
                self._peak_price = close
                self._bars_held = 0
                self._reset_pattern()
                return +1

        return 0
