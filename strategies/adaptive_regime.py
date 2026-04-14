"""
Adaptive Regime 전략 — 시장 레짐을 자동 감지하고 전략 모드를 전환.

핵심 아이디어:
크립토 시장은 세 가지 레짐을 반복한다:
  1. 불장 (Bull): BTC가 200 SMA 위, ADX > 25 → 추세 추종
  2. 횡보 (Range): ADX < 20 → 평균 회귀 (RSI 오실레이터)
  3. 약세장 (Bear): BTC가 200 SMA 아래 → 현금 보유

각 레짐에서 최적화된 다른 전략을 사용:
- Bull: EMA(20/50) 골든크로스 + RSI 확인 → 추세 추종 롱
- Range: RSI < 35 에서 매수, RSI > 65 에서 매도 → 평균 회귀
- Bear: 포지션 전부 현금화, 신규 진입 금지

왜 이 전략인가:
- 2022년 약세장에서 현금 보유로 -75% 하락을 피할 수 있다
- 2020~2021, 2024~2025 불장에서 추세 추종으로 큰 수익
- 횡보 구간(2023년 등)에서도 소폭 이익을 낼 수 있다
- 단일 전략보다 훨씬 시장 적응력이 높다
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, sma, rsi, atr, adx, macd


REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_RANGE = "range"


class AdaptiveRegimeStrategy(Strategy):
    """시장 레짐 적응형 전략.

    레짐 판단 기준:
    - Bull:  종가 > SMA(200) AND SMA(200) 기울기 > 0 AND ADX > 25
    - Bear:  종가 < SMA(200) AND SMA(200) 기울기 < 0
    - Range: 위에 해당하지 않는 나머지

    Bull 진입:
    - EMA(20) > EMA(50) (골든크로스)
    - RSI(14) 40~70 구간 (과매수 아닌 상태)
    - ATR 트레일링 스탑

    Range 진입:
    - RSI(7) < 35 → 매수 (과매도 반등)
    - RSI(7) > 65 → 매도 (과매수 후퇴)

    Bear:
    - 포지션 없으면 진입 금지
    - 포지션 있으면 즉시 청산
    """

    name = "adaptive_regime"

    def __init__(
        self,
        # 레짐 감지 파라미터
        sma_long: int = 200,
        sma_slope_period: int = 20,
        adx_period: int = 14,
        bull_adx_threshold: float = 22.0,
        range_adx_threshold: float = 18.0,
        # Bull 모드 파라미터
        ema_fast: int = 20,
        ema_slow: int = 50,
        rsi_period: int = 14,
        bull_rsi_min: float = 38.0,
        bull_rsi_max: float = 72.0,
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 2.5,
        # Range 모드 파라미터
        range_rsi_period: int = 7,
        range_rsi_buy: float = 35.0,
        range_rsi_sell: float = 65.0,
        range_sl_atr_mult: float = 1.5,
    ):
        self.sma_long = sma_long
        self.sma_slope_period = sma_slope_period
        self.adx_period = adx_period
        self.bull_adx_threshold = bull_adx_threshold
        self.range_adx_threshold = range_adx_threshold

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.bull_rsi_min = bull_rsi_min
        self.bull_rsi_max = bull_rsi_max
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self.range_rsi_period = range_rsi_period
        self.range_rsi_buy = range_rsi_buy
        self.range_rsi_sell = range_rsi_sell
        self.range_sl_atr_mult = range_sl_atr_mult

        # 상태
        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0
        self._entry_regime = REGIME_BEAR
        self._sl_price = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 레짐 감지용
        df["sma200"] = sma(df["close"], self.sma_long)
        df["sma200_slope"] = df["sma200"] - df["sma200"].shift(self.sma_slope_period)
        df["adx"] = adx(df, self.adx_period)
        df["atr"] = atr(df, 14)

        # Bull 모드용
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_slow_trend"] = ema(df["close"], self.ema_slow)
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["rsi_prev"] = df["rsi"].shift(1)

        # Range 모드용
        df["rsi_short"] = rsi(df["close"], self.range_rsi_period)
        df["rsi_short_prev"] = df["rsi_short"].shift(1)

        return df

    def _get_regime(self, row: pd.Series) -> str:
        close = row["close"]
        sma200 = row.get("sma200", close)
        slope = row.get("sma200_slope", 0.0)
        adx_val = row.get("adx", 0.0)

        above_sma = close > sma200
        slope_up = slope > 0

        if above_sma and slope_up and adx_val > self.bull_adx_threshold:
            return REGIME_BULL
        elif not above_sma and not slope_up:
            return REGIME_BEAR
        else:
            return REGIME_RANGE

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 1.0)
        regime = self._get_regime(row)

        # 포지션 보유 중
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # 약세장 진입 시 즉시 청산
            if regime == REGIME_BEAR:
                self._in_position = False
                return -1

            # Bull 진입이었던 경우: 트레일링 스탑
            if self._entry_regime == REGIME_BULL:
                # 손절
                if close < self._entry_price - self.sl_atr_mult * atr_val:
                    self._in_position = False
                    return -1
                # 트레일링 스탑
                if close < self._highest - self.trailing_atr_mult * atr_val:
                    self._in_position = False
                    return -1
                # EMA 역배열
                if row.get("ema_fast", close) < row.get("ema_slow_trend", close):
                    self._in_position = False
                    return -1

            # Range 진입이었던 경우: RSI 과매수 또는 손절
            elif self._entry_regime == REGIME_RANGE:
                rsi_short = row.get("rsi_short", 50)
                if rsi_short > self.range_rsi_sell:
                    self._in_position = False
                    return -1
                if close < self._entry_price - self.range_sl_atr_mult * atr_val:
                    self._in_position = False
                    return -1

            return 0

        # 미보유: 레짐별 진입 조건
        if regime == REGIME_BEAR:
            return 0

        if regime == REGIME_BULL:
            ema_f = row.get("ema_fast", close)
            ema_s = row.get("ema_slow_trend", close)
            rsi_val = row.get("rsi", 50)

            # EMA 정배열 + RSI 조건
            if (ema_f > ema_s
                    and self.bull_rsi_min < rsi_val < self.bull_rsi_max):
                self._in_position = True
                self._entry_price = close
                self._highest = close
                self._entry_regime = REGIME_BULL
                return +1

        if regime == REGIME_RANGE:
            rsi_s = row.get("rsi_short", 50)
            rsi_s_prev = row.get("rsi_short_prev", 50)

            # RSI 과매도 구간에서 반등 신호
            if rsi_s_prev < self.range_rsi_buy and rsi_s >= self.range_rsi_buy:
                self._in_position = True
                self._entry_price = close
                self._highest = close
                self._entry_regime = REGIME_RANGE
                return +1

        return 0
