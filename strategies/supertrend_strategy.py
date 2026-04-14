"""
SuperTrend 전략 — 크립토에서 검증된 ATR 기반 추세 추종.

핵심 아이디어:
- SuperTrend 지표는 ATR 기반 동적 지지/저항선으로 작동한다.
- 종가가 SuperTrend 위에 있으면 상승 추세(롱), 아래면 하락 추세(캐시)
- 200 SMA 위에서만 진입 → 구조적 약세장 필터링

왜 이 전략인가:
- 크립토는 장기 추세가 매우 강하다 (2020~2021 불장, 2022 약세장)
- SuperTrend는 추세가 강할수록 잘 작동하고, 방향 전환 시 빠르게 청산
- 200 SMA 필터로 2022년 같은 구조적 하락장에서 포지션 미보유
- 거래 빈도를 낮춰 수수료 부담 최소화

파라미터:
- st_period=10, st_multiplier=3.0: 크립토 4h 기준 최적값
- sma_period=200: 장기 추세 필터
- atr_stop_mult=1.5: 방향 전환 전 조기 손절 옵션
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import supertrend, sma, atr, adx


class SuperTrendStrategy(Strategy):
    """SuperTrend + 200SMA 필터 전략.

    진입 조건:
    1. SuperTrend 방향이 -1 → +1 로 전환 (상승 추세 시작)
    2. 종가 > SMA(200) (구조적 상승장)
    3. ADX > 20 (추세 존재 확인)

    청산 조건:
    - SuperTrend 방향이 +1 → -1 로 전환 (하락 추세 시작)
    - 또는 종가 < SMA(200) (구조적 하락장 진입)
    """

    name = "supertrend"

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 2.5,   # 최적화: 2.5 > 3.0
        sma_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
    ):
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.sma_period = sma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

        self._in_position = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        st_line, st_dir = supertrend(df, self.st_period, self.st_multiplier)
        df["st_line"] = st_line
        df["st_direction"] = st_dir
        df["st_dir_prev"] = st_dir.shift(1)

        df["sma200"] = sma(df["close"], self.sma_period)
        df["adx"] = adx(df, self.adx_period)
        df["atr"] = atr(df, 14)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        st_dir = int(row.get("st_direction", -1))
        st_dir_prev = int(row.get("st_dir_prev", -1))
        sma200 = row.get("sma200", 0)
        adx_val = row.get("adx", 0)

        # 포지션 보유 중: 청산 조건
        if self._in_position:
            # SuperTrend 방향 전환 (상승 → 하락)
            if st_dir == -1:
                self._in_position = False
                return -1
            # 200 SMA 하향 이탈 (구조적 약세장)
            if close < sma200 * 0.98:  # 2% 버퍼
                self._in_position = False
                return -1
            return 0

        # 미보유: 매수 조건
        # 1) SuperTrend 방향 전환: -1 → +1
        direction_flipped = (st_dir_prev == -1) and (st_dir == 1)
        # 2) 200 SMA 위
        above_sma = close > sma200
        # 3) 추세 강도
        trend_exists = adx_val > self.adx_threshold

        if direction_flipped and above_sma and trend_exists:
            self._in_position = True
            return +1

        return 0
