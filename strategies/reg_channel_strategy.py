"""
선형 회귀 채널 평균회귀 전략 (Regression Channel Mean Reversion).

설계 철학:
  선형 회귀 채널 = 추세의 "고속도로 차선"
  - 하단 채널 (지지선): 가격이 눌렸다 → 중심선으로 회귀 → 롱
  - 상단 채널 (저항선): 가격이 올랐다 → 중심선으로 회귀 → 숏
  - 중심선: 익절 목표 (리워드 중간 지점)
  - 채널 이탈: 회귀가 아닌 돌파이므로 즉시 손절

매매 구조:
  롱 진입:
    - close ≤ reg_lower × (1 + threshold)  [하단 채널 근처]
    - close ≥ reg_lower × (1 - sl_buffer)  [채널 완전 이탈 전]
    - 채널 기울기 ≥ slope_min              [급격한 하락 채널 제외]

  롱 청산:
    - close ≥ reg_center                   [중심선 도달 → 익절]
    - close < reg_lower - sl_atr × ATR     [채널 하향 이탈 → 손절]
    - 트레일링 스탑 (trailing_atr × ATR)

  숏 진입:
    - close ≥ reg_upper × (1 - threshold)  [상단 채널 근처]
    - close ≤ reg_upper × (1 + sl_buffer)  [채널 완전 이탈 전]
    - 채널 기울기 ≤ slope_max              [급격한 상승 채널 제외]

  숏 청산:
    - close ≤ reg_center                   [중심선 도달 → 익절]
    - close > reg_upper + sl_atr × ATR     [채널 상향 이탈 → 손절]
    - 트레일링 스탑 (trailing_atr × ATR)

  공통:
    - 최대 보유 기간 (max_bars) 초과 시 청산
    - 롱/숏 동시 포지션 없음 (항상 0 → +1 or -1 → 0)

on_bar 반환값:
  +1 : 롱 유지/진입
  -1 : 숏 유지/진입
   0 : 플랫 (포지션 없음)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import reg_channel, rsi, atr, sma


class RegChannelStrategy(Strategy):
    """선형 회귀 채널 평균회귀 전략 — 롱/숏 양방향."""

    name = "reg_channel"

    def __init__(
        self,
        # 채널 파라미터
        window: int = 100,
        # 진입 기준
        threshold: float = 0.005,    # 채널 근접 기준 (0.5%)
        slope_min: float = -0.002,   # 채널 기울기 하한 (너무 가파른 하락 제외)
        slope_max: float = 0.002,    # 채널 기울기 상한 (너무 가파른 상승 제외)
        # 손절 (ATR 기반 + 퍼센트 상한)
        sl_atr: float = 1.5,         # ATR 배수 손절
        sl_buffer: float = 0.005,    # 채널 이탈 버퍼 (0.5%)
        max_sl_pct: float = 0.02,    # 손절 최대 퍼센트 (2%) — 최적화 결과 반영
        # 트레일링 스탑
        trailing_atr: float = 2.0,
        max_trail_pct: float = 0.06, # 트레일링 최대 퍼센트 (6%)
        # 익절 목표 (채널 기반 + 퍼센트 상한)
        tp_at_center: bool = True,   # 중심선 도달 시 익절
        max_tp_pct: float = 0.06,    # 익절 최대 퍼센트 (6%) — 최적화 결과 반영
        # 보조 지표
        atr_period: int = 14,
        rsi_period: int = 14,
        sma_period: int = 200,
        # 최대 보유 기간
        max_bars: int = 40,
        # 방향 필터 (선택)
        use_trend_filter: bool = False,  # True면 SMA200 기준 방향 제한
    ):
        self.window = window
        self.threshold = threshold
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.sl_atr = sl_atr
        self.sl_buffer = sl_buffer
        self.max_sl_pct = max_sl_pct
        self.trailing_atr = trailing_atr
        self.max_trail_pct = max_trail_pct
        self.tp_at_center = tp_at_center
        self.max_tp_pct = max_tp_pct
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.max_bars = max_bars
        self.use_trend_filter = use_trend_filter

        self._position = 0       # 0=flat, +1=long, -1=short
        self._entry_price = 0.0
        self._best_price = 0.0   # 롱: 최고가 / 숏: 최저가 (트레일링용)
        self._bars_held = 0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        center, upper, lower = reg_channel(df, self.window)
        df["reg_center"] = center
        df["reg_upper"]  = upper
        df["reg_lower"]  = lower

        # 채널 기울기: 5봉 전 중심선 대비 변화율
        df["reg_slope"] = df["reg_center"].diff(5) / (df["reg_center"].shift(5).abs() + 1e-12)

        df["atr"]    = atr(df, self.atr_period)
        df["rsi"]    = rsi(df["close"], self.rsi_period)
        df["sma200"] = sma(df["close"], self.sma_period)

        return df

    # ── 롱 포지션 청산 판단 ──────────────────────────────────────────

    def _long_should_exit(self, row: pd.Series) -> bool:
        close  = row["close"]
        center = row.get("reg_center", close)
        lower  = row.get("reg_lower", 0.0)
        atr_v  = row.get("atr", close * 0.02)
        entry  = self._entry_price

        # ── 익절 ──────────────────────────────────────────────────
        # 중심선 도달 또는 최대 익절 퍼센트 중 먼저 오는 쪽
        tp_channel = center
        tp_pct     = entry * (1 + self.max_tp_pct)
        tp_price   = min(tp_channel, tp_pct)   # 더 가까운(낮은) 목표를 먼저 챙김
        if close >= tp_price:
            return True

        # ── 손절 ──────────────────────────────────────────────────
        # ATR 기반 손절 vs 퍼센트 상한 → 더 가까운(높은) 선에서 끊음
        sl_atr_price = lower - self.sl_atr * atr_v
        sl_pct_price = entry * (1 - self.max_sl_pct)
        sl_price     = max(sl_atr_price, sl_pct_price)   # 더 위에 있는(타이트한) 쪽
        if close < sl_price:
            return True

        # ── 트레일링 스탑 ─────────────────────────────────────────
        trail_atr_price = self._best_price - self.trailing_atr * atr_v
        trail_pct_price = self._best_price * (1 - self.max_trail_pct)
        trail_price     = max(trail_atr_price, trail_pct_price)
        if close < trail_price:
            return True

        # 최대 보유 기간
        if self._bars_held >= self.max_bars:
            return True
        return False

    # ── 숏 포지션 청산 판단 ──────────────────────────────────────────

    def _short_should_exit(self, row: pd.Series) -> bool:
        close  = row["close"]
        center = row.get("reg_center", close)
        upper  = row.get("reg_upper", float("inf"))
        atr_v  = row.get("atr", close * 0.02)
        entry  = self._entry_price

        # ── 익절 ──────────────────────────────────────────────────
        # 중심선 도달 또는 최대 익절 퍼센트 중 먼저 오는 쪽
        tp_channel = center
        tp_pct     = entry * (1 - self.max_tp_pct)
        tp_price   = max(tp_channel, tp_pct)   # 더 가까운(높은) 목표를 먼저 챙김
        if close <= tp_price:
            return True

        # ── 손절 ──────────────────────────────────────────────────
        # ATR 기반 손절 vs 퍼센트 상한 → 더 가까운(낮은) 선에서 끊음
        sl_atr_price = upper + self.sl_atr * atr_v
        sl_pct_price = entry * (1 + self.max_sl_pct)
        sl_price     = min(sl_atr_price, sl_pct_price)   # 더 아래에 있는(타이트한) 쪽
        if close > sl_price:
            return True

        # ── 트레일링 스탑 ─────────────────────────────────────────
        trail_atr_price = self._best_price + self.trailing_atr * atr_v
        trail_pct_price = self._best_price * (1 + self.max_trail_pct)
        trail_price     = min(trail_atr_price, trail_pct_price)
        if close > trail_price:
            return True

        # 최대 보유 기간
        if self._bars_held >= self.max_bars:
            return True
        return False

    # ── 롱 진입 조건 ─────────────────────────────────────────────────

    def _can_enter_long(self, row: pd.Series) -> bool:
        close  = row["close"]
        lower  = row.get("reg_lower", 0.0)
        slope  = row.get("reg_slope", 0.0)
        sma200 = row.get("sma200", 0.0)

        if pd.isna(lower) or pd.isna(slope):
            return False

        # 거시 추세 필터 (선택)
        if self.use_trend_filter and close < sma200:
            return False

        # 기울기: 너무 가파른 하락 채널은 제외
        if slope < self.slope_min:
            return False

        # 하단 채널 근처 (아직 이탈 전)
        near_lower  = close <= lower * (1 + self.threshold)
        above_lower = close >= lower * (1 - self.sl_buffer)
        return near_lower and above_lower

    # ── 숏 진입 조건 ─────────────────────────────────────────────────

    def _can_enter_short(self, row: pd.Series) -> bool:
        close  = row["close"]
        upper  = row.get("reg_upper", float("inf"))
        slope  = row.get("reg_slope", 0.0)
        sma200 = row.get("sma200", 0.0)

        if pd.isna(upper) or pd.isna(slope):
            return False

        # 거시 추세 필터 (선택)
        if self.use_trend_filter and close > sma200:
            return False

        # 기울기: 너무 가파른 상승 채널은 제외
        if slope > self.slope_max:
            return False

        # 상단 채널 근처 (아직 이탈 전)
        near_upper  = close >= upper * (1 - self.threshold)
        below_upper = close <= upper * (1 + self.sl_buffer)
        return near_upper and below_upper

    # ── 메인 루프 ─────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]

        # ── 롱 포지션 유지/청산 ──────────────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            if close > self._best_price:
                self._best_price = close
            if self._long_should_exit(row):
                self._position = 0
                return 0
            return +1

        # ── 숏 포지션 유지/청산 ──────────────────────────────────────
        if self._position == -1:
            self._bars_held += 1
            if close < self._best_price:
                self._best_price = close
            if self._short_should_exit(row):
                self._position = 0
                return 0
            return -1

        # ── 신규 진입 ────────────────────────────────────────────────
        if self._can_enter_long(row):
            self._position    = 1
            self._entry_price = close
            self._best_price  = close
            self._bars_held   = 0
            return +1

        if self._can_enter_short(row):
            self._position    = -1
            self._entry_price = close
            self._best_price  = close
            self._bars_held   = 0
            return -1

        return 0
