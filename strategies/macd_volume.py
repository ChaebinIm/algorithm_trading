"""
MACD + CMF + 200SMA 전략 — 모멘텀 + 자금 흐름 + 추세 필터 결합.

핵심 아이디어:
- MACD: 단기/장기 EMA 차이로 모멘텀 방향 포착
- CMF (Chaikin Money Flow): 거래량 가중 자금 유입/유출 측정
- 200 SMA: 구조적 추세 방향 필터

진입 조건:
1. 200 SMA 위에 종가 (강세 구조)
2. MACD 히스토그램이 음수→양수 전환 (상승 모멘텀 확인)
3. CMF > 0.02 (자금 순유입 상태)
4. ADX > 22 (추세 강도 확인)

청산 조건:
- MACD 히스토그램이 양수→음수 전환
- 또는 ATR 기반 트레일링 스탑

왜 이 조합인가:
- MACD만 사용하면 횡보 구간에서 잦은 가짜 신호 발생
- CMF로 "실제 자금이 들어오는 움직임"만 필터링
- 200 SMA로 구조적 방향 정렬
- 세 조건이 모두 충족되는 순간은 드물지만, 신뢰도가 매우 높다
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import sma, atr, adx, macd, cmf


class MACDVolumeStrategy(Strategy):
    """MACD + CMF + 200SMA 전략.

    진입: MACD 히스토그램 양전환 + CMF > threshold + 200SMA 위
    청산: MACD 히스토그램 음전환 + ATR 트레일링 스탑
    """

    name = "macd_volume"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cmf_period: int = 20,
        cmf_threshold: float = 0.02,
        sma_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 22.0,
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 3.0,
    ):
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cmf_period = cmf_period
        self.cmf_threshold = cmf_threshold
        self.sma_period = sma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # MACD
        macd_line, signal_line, histogram = macd(
            df["close"], self.macd_fast, self.macd_slow, self.macd_signal
        )
        df["macd_hist"] = histogram
        df["macd_hist_prev"] = histogram.shift(1)

        # CMF
        df["cmf"] = cmf(df, self.cmf_period)

        # 200 SMA
        df["sma200"] = sma(df["close"], self.sma_period)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 1.0)

        hist = row.get("macd_hist", 0.0)
        hist_prev = row.get("macd_hist_prev", 0.0)
        cmf_val = row.get("cmf", 0.0)
        sma200 = row.get("sma200", 0.0)
        adx_val = row.get("adx", 0.0)

        # 포지션 보유 중
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # MACD 히스토그램 음전환: 모멘텀 약화
            if hist_prev > 0 and hist <= 0:
                self._in_position = False
                return -1

            # 손절
            if close < self._entry_price - self.sl_atr_mult * atr_val:
                self._in_position = False
                return -1

            # 트레일링 스탑
            if close < self._highest - self.trailing_atr_mult * atr_val:
                self._in_position = False
                return -1

            return 0

        # 미보유: 매수 조건
        # 1) MACD 히스토그램 양전환 (음→양)
        hist_cross_up = (hist_prev <= 0) and (hist > 0)
        # 2) CMF > threshold (자금 순유입)
        money_flowing_in = cmf_val > self.cmf_threshold
        # 3) 200 SMA 위
        above_sma = close > sma200
        # 4) 추세 존재
        trend_exists = adx_val > self.adx_threshold

        if hist_cross_up and money_flowing_in and above_sma and trend_exists:
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
