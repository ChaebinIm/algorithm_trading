"""
Multi-Factor Score 전략 — 다중 요인을 점수화하여 고확률 진입 포착.

핵심 아이디어:
단일 지표는 노이즈에 취약하지만, 여러 독립적 요인을 종합하면
"모든 것이 정렬된" 순간을 포착할 수 있다.

점수 시스템 (0~7점):
  1. 추세: 종가 > SMA(200) → +1
  2. 추세 가속: SMA(50) > SMA(100) → +1
  3. 모멘텀: 20기간 수익률 > 0 → +1
  4. 단기 모멘텀: MACD 히스토그램 > 0 → +1
  5. 자금 흐름: CMF > 0 → +1
  6. OBV 상승: OBV > OBV 20기간 평균 → +1
  7. 추세 강도: ADX > 25 → +1

진입: 점수 >= 5 (7개 중 5개 이상 충족)
청산: 점수 <= 2 또는 ATR 트레일링 스탑

왜 이 방식인가:
- 각 요인은 독립적으로는 약하지만, 함께 충족되면 강한 신호
- 점수 방식으로 "어느 정도 충족"하는 유연한 진입 가능
- 특정 지표에 과적합되지 않음
- 퀀트 연구에서 "factor investing"의 핵심 아이디어를 차용
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import sma, ema, atr, adx, macd, cmf, obv, momentum


class MultiFactorStrategy(Strategy):
    """다중 요인 점수 기반 전략.

    진입: buy_threshold 이상 점수
    청산: sell_threshold 이하 점수 또는 ATR 트레일링 스탑
    """

    name = "multi_factor"

    def __init__(
        self,
        sma_long: int = 200,
        sma_mid: int = 100,
        sma_short: int = 50,
        mom_period: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cmf_period: int = 20,
        obv_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        buy_threshold: int = 5,     # 7점 중 5점 이상
        sell_threshold: int = 2,    # 7점 중 2점 이하
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 3.0,
    ):
        self.sma_long = sma_long
        self.sma_mid = sma_mid
        self.sma_short = sma_short
        self.mom_period = mom_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cmf_period = cmf_period
        self.obv_period = obv_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 추세 요인
        df["sma200"] = sma(df["close"], self.sma_long)
        df["sma100"] = sma(df["close"], self.sma_mid)
        df["sma50"] = sma(df["close"], self.sma_short)

        # 모멘텀
        df["mom20"] = momentum(df["close"], self.mom_period)

        # MACD
        _, _, hist = macd(df["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        df["macd_hist"] = hist

        # CMF (자금 흐름)
        df["cmf"] = cmf(df, self.cmf_period)

        # OBV
        obv_series = obv(df)
        df["obv"] = obv_series
        df["obv_ma"] = sma(obv_series, self.obv_period)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def _score(self, row: pd.Series) -> int:
        """각 요인을 0/1 점수화하여 합산."""
        close = row["close"]
        score = 0

        # 1) 장기 추세
        if close > row.get("sma200", 0):
            score += 1

        # 2) 중기 추세 가속
        if row.get("sma50", 0) > row.get("sma100", 0):
            score += 1

        # 3) 장기 모멘텀
        if row.get("mom20", -1) > 0:
            score += 1

        # 4) 단기 모멘텀 (MACD)
        if row.get("macd_hist", -1) > 0:
            score += 1

        # 5) 자금 유입
        if row.get("cmf", -1) > 0:
            score += 1

        # 6) OBV 상승
        if row.get("obv", 0) > row.get("obv_ma", 0):
            score += 1

        # 7) 추세 강도
        if row.get("adx", 0) > self.adx_threshold:
            score += 1

        return score

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 1.0)
        score = self._score(row)

        # 포지션 보유 중
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

            # 요인 대다수 역전
            if score <= self.sell_threshold:
                self._in_position = False
                return -1

            return 0

        # 미보유: 매수 조건
        if score >= self.buy_threshold:
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
