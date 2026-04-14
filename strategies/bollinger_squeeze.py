"""
볼린저 밴드 스퀴즈 전략 (Bollinger Squeeze).

설계 철학: 변동성 수축 후 폭발을 포착.
- 크립토는 변동성 수축→확장 패턴이 빈번하게 발생
- 스퀴즈(밴드폭 최저) 상태에서 상단 돌파 시 진입
- 거래량 증가로 돌파 유효성 확인
- 고정 손익비 1:2

손절/익절:
- 손절: 진입가 - 1.5 × ATR
- 익절: 진입가 + 3.0 × ATR (손익비 1:2)
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, atr, bollinger_bands, sma


class BollingerSqueezeStrategy(Strategy):
    """볼린저 밴드 스퀴즈 전략.

    스퀴즈 감지:
    - 밴드폭(bandwidth)이 최근 lookback 바 중 최저일 때 "스퀴즈 상태"
    - 변동성이 극도로 수축된 상태 → 곧 큰 움직임이 예상됨

    진입 조건 (모든 조건 동시 충족):
    1. 스퀴즈 상태 (또는 직전 squeeze_grace 바 이내에 스퀴즈 발생)
    2. 종가 > 볼린저 상단 밴드: 상방 돌파
    3. 거래량 > 거래량 이동평균 × vol_mult: 거래량 뒷받침 확인
    4. 종가 > EMA(200): 장기 상승 추세

    청산 조건:
    - 종가 < 진입가 - 1.5×ATR → 손절
    - 종가 > 진입가 + 3.0×ATR → 익절
    """

    name = "bollinger_squeeze"

    def __init__(
        self,
        bb_period: int = 20,          # 볼린저 밴드 기간
        bb_std: float = 2.0,          # 표준편차 배수
        squeeze_lookback: int = 120,  # 스퀴즈 판단 구간 (밴드폭 최저 비교 기간)
        squeeze_grace: int = 3,       # 스퀴즈 후 N바 이내까지 진입 허용
        vol_period: int = 20,         # 거래량 이동평균 기간
        vol_mult: float = 1.2,        # 거래량 필터 배수 (평균 대비)
        ema_trend: int = 200,         # 장기 추세 필터
        sl_atr_mult: float = 1.5,     # 손절 ATR 배수
        tp_atr_mult: float = 3.0,     # 익절 ATR 배수
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_grace = squeeze_grace
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.ema_trend = ema_trend
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # 내부 상태
        self._in_position = False
        self._entry_price = 0.0
        self._entry_atr = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 선계산."""
        df = df.copy()

        # 볼린저 밴드
        upper, middle, lower, bandwidth = bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["bb_bandwidth"] = bandwidth

        # 스퀴즈 감지: 밴드폭이 최근 lookback 바 중 최저인지
        df["bw_min"] = df["bb_bandwidth"].rolling(self.squeeze_lookback).min()
        df["is_squeeze"] = (df["bb_bandwidth"] == df["bw_min"]).astype(int)

        # 스퀴즈 후 grace 바 이내인지 (최근 N바 내에 스퀴즈 발생)
        df["squeeze_recent"] = (
            df["is_squeeze"].rolling(self.squeeze_grace + 1).max()
        ).fillna(0).astype(int)

        # 거래량 이동평균
        df["vol_ma"] = sma(df["volume"], self.vol_period)

        # 장기 추세
        df["ema_trend"] = ema(df["close"], self.ema_trend)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성."""
        close = row["close"]

        # --- 포지션 보유 중: SL/TP 체크 ---
        if self._in_position:
            sl_price = self._entry_price - self.sl_atr_mult * self._entry_atr
            if close < sl_price:
                self._in_position = False
                return -1

            tp_price = self._entry_price + self.tp_atr_mult * self._entry_atr
            if close > tp_price:
                self._in_position = False
                return -1

            return 0

        # --- 미보유: 스퀴즈 돌파 매수 ---
        atr_val = row.get("atr", 0)
        bb_upper = row.get("bb_upper", 0)
        squeeze_recent = bool(row.get("squeeze_recent", 0))
        volume = row.get("volume", 0)
        vol_ma = row.get("vol_ma", 0)
        ema_trend_val = row.get("ema_trend", close)

        if (
            squeeze_recent                          # 스퀴즈 상태 (또는 직전 발생)
            and close > bb_upper                    # 상단 밴드 돌파
            and volume > vol_ma * self.vol_mult     # 거래량 뒷받침
            and close > ema_trend_val               # 장기 상승 추세
        ):
            self._in_position = True
            self._entry_price = close
            self._entry_atr = atr_val
            return +1

        return 0
