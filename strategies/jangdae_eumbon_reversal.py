"""
일봉 장대음봉 반등 전략 — 롱온리.

설계 철학:
  일봉 장대음봉(강한 매도세) 당일 종가에 즉시 진입하여 반등을 포착.
  SMA200 필터 없음 — 하락장에서 오히려 반등이 강하다는 리서치 결과 반영.
  폭락봉(ATR×1.5 이상)은 제외하여 패닉셀 구간 회피.
  깊은 조정 구간(30일 고점 대비 -20% 이하)에서만 진입.

근거 (리서치 에이전트 9종 분석, 그리드 서치 3240개 조합):
  - T0 종가 즉시 진입이 최강 (T+1 진입 대비 수익률 우수)
  - ATR×1.0~2.0 음봉이 최적 (1.5 상한 제거, 2.0으로 완화)
  - 30일 고점 대비 낙폭 -15% 이상에서 반등 확률 최고 (-20%는 신호 너무 적음)
  - 고변동성 필터(vol_filter)는 오히려 성과 저하 → 기본값 False
  - TP=8%/SL=3%/max_bars=7이 OOS Sharpe 1.441로 최강
  - SMA200 이하(하락장)에서도 반등 엣지 존재 — 오히려 더 강함

진입:
  - 일봉 장대음봉 당일 종가에 즉시 진입 (T0 종가)
  - 음봉 크기: ATR×min_mult 이상 AND ATR×max_mult 미만
  - 30일 고점 대비 낙폭 >= drawdown_filter (예: -20%)
  - 고변동성 필터: 현재 ATR >= vol_period 평균 ATR × vol_mult

청산:
  1. close >= entry * (1 + tp_pct)     (익절)
  2. close <= entry * (1 - sl_pct)     (손절)
  3. 트레일링 스탑 (trailing_pct > 0)
  4. max_bars 초과                      (시간 청산)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import atr as calc_atr


class JangdaeEumbonReversalStrategy(Strategy):
    """일봉 장대음봉 당일 즉시 진입 반등 전략."""

    name = "jangdae_eumbon_reversal"

    def __init__(
        self,
        atr_period: int = 14,
        min_mult: float = 1.0,       # 음봉 최소 크기 (ATR 배수)
        max_mult: float = 2.0,       # 음봉 최대 크기 (0=무제한, 그리드서치 최적: 2.0)
        drawdown_period: int = 30,   # 고점 대비 낙폭 계산 기간
        drawdown_filter: float = -0.15,  # 이 값 이하 낙폭에서만 진입 (그리드서치 최적: -0.15)
        vol_filter: bool = False,    # 고변동성 필터 (그리드서치 결과: False가 더 좋음)
        vol_period: int = 14,        # 변동성 비교 기간
        vol_mult: float = 0.8,       # ATR >= vol_period 평균 ATR × 이 값
        tp_pct: float = 0.08,        # 익절 8% (그리드서치 최적)
        sl_pct: float = 0.03,        # 손절 3%
        trailing_pct: float = 0.0,   # 0=비활성
        max_bars: int = 7,           # 최대 보유 7일 (그리드서치 최적)
    ):
        self.atr_period = atr_period
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.drawdown_period = drawdown_period
        self.drawdown_filter = drawdown_filter
        self.vol_filter = vol_filter
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_pct = trailing_pct
        self.max_bars = max_bars

        # 포지션 상태
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._bars_held = 0

    # ── 준비 ─────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # ATR 계산
        df["atr"] = calc_atr(df, self.atr_period)

        # 음봉 body: open - close (양수면 음봉)
        df["eumbon_body"] = df["open"] - df["close"]

        # 30일 rolling high (현재 바 제외 → shift(1) 불필요, 현재 봉 포함)
        # 낙폭 계산: (현재 close - rolling_high) / rolling_high
        df["rolling_high"] = df["high"].rolling(self.drawdown_period).max()
        df["drawdown_from_high"] = (df["close"] - df["rolling_high"]) / (df["rolling_high"] + 1e-12)

        # ATR rolling mean (고변동성 필터)
        df["atr_mean"] = df["atr"].rolling(self.vol_period).mean()

        # 음봉 조건 (현재 봉이 음봉인지)
        df["is_bearish"] = df["close"] < df["open"]

        # 장대음봉 조건
        df["is_jangdae_eumbon"] = (
            df["is_bearish"] &
            (df["eumbon_body"] >= self.min_mult * df["atr"])
        )

        # 폭락봉 제외 (max_mult > 0인 경우)
        if self.max_mult > 0:
            df["is_jangdae_eumbon"] = df["is_jangdae_eumbon"] & (
                df["eumbon_body"] < self.max_mult * df["atr"]
            )

        # 낙폭 필터 조건
        df["dd_filter_ok"] = df["drawdown_from_high"] <= self.drawdown_filter

        # 고변동성 필터 조건
        df["vol_filter_ok"] = True
        if self.vol_filter:
            df["vol_filter_ok"] = df["atr"] >= self.vol_mult * df["atr_mean"]

        return df

    # ── 청산 ─────────────────────────────────────────────────────────────

    def _should_exit(self, row: pd.Series) -> bool:
        close = row["close"]
        entry = self._entry_price

        # 익절
        if close >= entry * (1 + self.tp_pct):
            return True
        # 손절
        if close <= entry * (1 - self.sl_pct):
            return True
        # 트레일링 스탑
        if self.trailing_pct > 0:
            trail = self._peak_price * (1 - self.trailing_pct)
            if close < trail:
                return True
        # 시간 청산
        if self._bars_held >= self.max_bars:
            return True

        return False

    # ── 메인 ─────────────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]

        # ── 포지션 보유 중 ───────────────────────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            if close > self._peak_price:
                self._peak_price = close
            if self._should_exit(row):
                self._position = 0
                self._entry_price = 0.0
                self._peak_price = 0.0
                self._bars_held = 0
                return -1
            return 0

        # ── 진입 조건 확인 ───────────────────────────────────────────────
        is_jangdae_eumbon = bool(row.get("is_jangdae_eumbon", False))
        dd_filter_ok = bool(row.get("dd_filter_ok", False))
        vol_filter_ok = bool(row.get("vol_filter_ok", True))

        if is_jangdae_eumbon and dd_filter_ok and vol_filter_ok:
            self._position = 1
            self._entry_price = close
            self._peak_price = close
            self._bars_held = 0
            return +1

        return 0
