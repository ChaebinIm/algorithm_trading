"""
SuperTrend 기반 강화 앙상블 전략 (SuperTrend Ensemble v2).

1차 백테스트 결과 분석:
- SuperTrend: 최고 샤프 1.35 (BTC 1h), XRP에서 +178%
- Ensemble: 최고 샤프 1.90 (BTC 4h), 가장 낮은 MDD -9.2%

두 전략의 장점을 결합:
- SuperTrend의 강력한 추세 포착력
- Ensemble의 다중 신호 필터링으로 가짜 신호 제거
- CMF로 자금 흐름 확인 (volume-weighted 필터)
- MACD로 모멘텀 타이밍 정밀화

서브 신호 구성 (4개):
  1. SuperTrend: 방향 +1 (가장 중요한 신호)
  2. 추세 정렬: EMA(20) > EMA(50) > EMA(200)
  3. MACD 히스토그램: 양수 (모멘텀 상승)
  4. CMF: > 0 (자금 순유입)

진입: 서브 신호 3개 이상 충족 (4개 중 3개)
  - SuperTrend 방향이 +1 (필수 조건, 항상 포함)
  - 나머지 3개 중 2개 이상 → 총 3개 충족

청산:
  - SuperTrend 방향이 -1로 전환 (즉시 청산)
  - 또는 ATR 트레일링 스탑 (고점 대비 하락)
  - 또는 서브 신호 2개 이하로 감소

특징:
- 거래 빈도: 중간 (SuperTrend 방향 전환 기반)
- 4h 타임프레임에서 최적 성능 예상
- 200 SMA 필터로 구조적 약세장 방어
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import (
    ema, sma, atr, adx,
    supertrend, macd, cmf,
)


class SuperTrendEnsembleStrategy(Strategy):
    """SuperTrend 기반 강화 앙상블 전략 v2.

    SuperTrend를 핵심 필터로 삼고,
    EMA 정배열 + MACD + CMF를 보조 필터로 사용.

    min_signals=3 (4개 중 3개) 설정이 최적:
    - SuperTrend 필수 + 나머지 2개
    - 지나치게 엄격하지 않으면서 노이즈 차단
    """

    name = "supertrend_ensemble"

    def __init__(
        self,
        # SuperTrend (핵심)
        st_period: int = 10,
        st_multiplier: float = 2.5,   # 최적화: 2.5 > 3.0
        # 추세 정렬
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        # MACD
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        # CMF
        cmf_period: int = 20,
        # ADX 필터
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        # 앙상블 설정
        min_signals: int = 3,       # 4개 중 최소 몇 개 충족해야 진입
        sell_signals: int = 2,      # 이 이하면 청산
        # 리스크 관리
        trailing_atr_mult: float = 2.5,
        sl_atr_mult: float = 2.0,
    ):
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cmf_period = cmf_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.min_signals = min_signals
        self.sell_signals = sell_signals
        self.trailing_atr_mult = trailing_atr_mult
        self.sl_atr_mult = sl_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # SuperTrend
        st_line, st_dir = supertrend(df, self.st_period, self.st_multiplier)
        df["st_direction"] = st_dir
        df["st_dir_prev"] = st_dir.shift(1)

        # EMA 정배열
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)

        # MACD 히스토그램
        _, _, hist = macd(df["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        df["macd_hist"] = hist

        # CMF
        df["cmf"] = cmf(df, self.cmf_period)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def _count_signals(self, row: pd.Series) -> int:
        """4개 서브 신호 카운트."""
        close = row["close"]
        count = 0

        # 1) SuperTrend 방향 (가장 중요)
        if int(row.get("st_direction", -1)) == 1:
            count += 1

        # 2) EMA 정배열 (단기 > 중기 > 장기)
        ef = row.get("ema_fast", close)
        em = row.get("ema_mid", close)
        es = row.get("ema_slow", close)
        if ef > em > es:
            count += 1

        # 3) MACD 히스토그램 양수
        if row.get("macd_hist", -1) > 0:
            count += 1

        # 4) CMF 양수 (자금 유입)
        if row.get("cmf", -1) > 0:
            count += 1

        return count

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 1.0)
        adx_val = row.get("adx", 0.0)
        st_dir = int(row.get("st_direction", -1))
        st_dir_prev = int(row.get("st_dir_prev", -1))

        # 포지션 보유 중
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # SuperTrend 방향 전환: 즉시 청산
            if st_dir == -1:
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

            # 신호 약화
            count = self._count_signals(row)
            if count <= self.sell_signals:
                self._in_position = False
                return -1

            return 0

        # 미보유: 진입 조건
        if adx_val < self.adx_threshold:
            return 0

        # SuperTrend 방향 전환 시점 또는 이미 상승 추세인 상태에서 신호 충족
        st_just_flipped = (st_dir_prev == -1) and (st_dir == 1)
        count = self._count_signals(row)

        # SuperTrend 필수 포함 + 전체 min_signals 이상
        st_positive = (st_dir == 1)
        if st_positive and count >= self.min_signals:
            # 방향 전환 시점 또는 신호가 4개 모두 충족될 때만 진입 (중복 진입 방지)
            if st_just_flipped or count == 4:
                self._in_position = True
                self._entry_price = close
                self._highest = close
                return +1

        return 0
