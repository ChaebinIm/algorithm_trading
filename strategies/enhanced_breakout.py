"""
강화 브레이크아웃 전략 (Enhanced Breakout).

기존 atr_breakout의 업그레이드 버전:
- 세션 필터: 야간(21:00-00:00 UTC)에만 진입 (세션 분석 결과 최적)
- 볼린저 스퀴즈 선행 조건: 변동성 수축 후 돌파만 진입
- 거래량 확인: 평균 대비 거래량 증가 필수
- 다중 타임프레임: EMA(200) 기울기로 상위 추세 확인
- 고정 손익비 1:2.5 (SL 1.5 ATR, TP 3.75 ATR)

세션 분석 결과 atr_breakout은 야간 세션이 최적이었음.
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, atr, adx, donchian_channel, bollinger_bands, sma


class EnhancedBreakoutStrategy(Strategy):
    """강화 브레이크아웃 전략.

    진입 조건 (모든 조건 동시 충족):
    1. 돈키안 채널 상단 돌파: 종가 > N기간 최고가
    2. 볼린저 스퀴즈 해소: 최근 밴드폭이 낮았다가 확장 시작
    3. 거래량 > 이동평균 x 1.3: 의미 있는 볼륨 동반
    4. EMA(200) 기울기 양수: 장기 추세 상승
    5. 세션 필터: UTC 21:00~24:00 (야간)

    청산:
    - 손절: 진입가 - 1.5 x ATR
    - 익절: 진입가 + 3.75 x ATR (손익비 1:2.5)
    """

    name = "enhanced_breakout"

    def __init__(
        self,
        channel_period: int = 20,
        bb_period: int = 20,
        bb_std: float = 2.0,
        squeeze_lookback: int = 60,    # 스퀴즈 판단 기간
        vol_period: int = 20,
        vol_mult: float = 1.3,
        ema_slow: int = 200,
        ema_slope_period: int = 10,
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        session_start: int = 21,       # 야간 세션
        session_end: int = 24,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 3.75,     # 손익비 1:2.5
    ):
        self.channel_period = channel_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.ema_slow = ema_slow
        self.ema_slope_period = ema_slope_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.session_start = session_start
        self.session_end = session_end
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._entry_atr = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 돈키안 채널 (전일 기준)
        dc_upper, dc_lower = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)

        # 볼린저 밴드 (스퀴즈 감지용)
        _, _, _, bandwidth = bollinger_bands(df["close"], self.bb_period, self.bb_std)
        df["bb_bandwidth"] = bandwidth
        df["bw_min"] = df["bb_bandwidth"].rolling(self.squeeze_lookback).min()
        # 스퀴즈 해소: 밴드폭이 최근 최저 대비 20% 이상 확장
        df["squeeze_releasing"] = (
            df["bb_bandwidth"] > df["bw_min"] * 1.2
        ).astype(int)

        # 거래량
        df["vol_ma"] = sma(df["volume"], self.vol_period)

        # 장기 추세 + 기울기
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["ema_slow_slope"] = df["ema_slow"] - df["ema_slow"].shift(self.ema_slope_period)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        # 세션
        if hasattr(df.index, 'hour'):
            df["_hour"] = df.index.hour
        else:
            df["_hour"] = 22

        return df

    def _in_session(self, hour: int) -> bool:
        if self.session_start < self.session_end:
            return self.session_start <= hour < self.session_end
        else:
            return hour >= self.session_start or hour < self.session_end

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]

        # --- 포지션 보유 중: SL/TP ---
        if self._in_position:
            if close < self._entry_price - self.sl_atr_mult * self._entry_atr:
                self._in_position = False
                return -1
            if close > self._entry_price + self.tp_atr_mult * self._entry_atr:
                self._in_position = False
                return -1
            return 0

        # --- 미보유: 매수 조건 ---
        hour = int(row.get("_hour", 22))
        if not self._in_session(hour):
            return 0

        atr_val = row.get("atr", 0)
        dc_upper = row.get("dc_upper", 0)
        squeeze_releasing = bool(row.get("squeeze_releasing", 0))
        volume = row.get("volume", 0)
        vol_ma = row.get("vol_ma", 0)
        ema_slow_slope = row.get("ema_slow_slope", 0)
        adx_val = row.get("adx", 0)

        if (
            close > dc_upper                       # 채널 돌파
            and squeeze_releasing                   # 스퀴즈 해소 중
            and volume > vol_ma * self.vol_mult     # 거래량 확인
            and ema_slow_slope > 0                  # 장기 추세 상승
            and adx_val > self.adx_threshold        # 추세 존재
        ):
            self._in_position = True
            self._entry_price = close
            self._entry_atr = atr_val
            return +1

        return 0
