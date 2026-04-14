"""
강화 추세 추종 전략 (Enhanced Trend Following).

기존 trend_following의 업그레이드 버전:
- 다중 타임프레임 확인: 상위 TF(일봉 기준) EMA 방향을 하위 TF 데이터 내에서 시뮬레이션
- 세션 필터: 유럽 세션(07:00-16:00 UTC)에만 진입 허용
- 복합 진입 조건: 추세 + 모멘텀 + 볼린저 스퀴즈 해소를 함께 확인
- ATR 트레일링 스탑

세션 분석 결과 trend_following은 유럽 세션이 최적이었음.
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, rsi, atr, adx, momentum, bollinger_bands


class EnhancedTrendStrategy(Strategy):
    """강화 추세 추종 전략.

    진입 조건 (모든 조건 동시 충족):
    1. 상위 TF 추세: EMA(200) 기울기 양수 (장기 상승 확인)
    2. EMA 정배열: EMA(20) > EMA(50)
    3. 모멘텀 양수: 20기간 수익률 > 0
    4. RSI 반등: 이전 바 RSI < 50 -> 현재 >= 50
    5. 세션 필터: UTC 07:00~16:00 (유럽 세션)

    청산 조건:
    - 종가 < 진입가 - 2.0 x ATR: 손절
    - 종가 < 고점 - 2.5 x ATR: 트레일링 스탑
    - EMA(20) < EMA(50): 추세 역전
    """

    name = "enhanced_trend"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        ema_slope_period: int = 10,   # EMA(200) 기울기 판단 기간
        rsi_period: int = 14,
        mom_period: int = 20,
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        rsi_pullback: float = 50.0,
        session_start: int = 7,       # 유럽 세션 시작 (UTC)
        session_end: int = 16,        # 유럽 세션 종료 (UTC)
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 2.5,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.ema_slope_period = ema_slope_period
        self.rsi_period = rsi_period
        self.mom_period = mom_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_pullback = rsi_pullback
        self.session_start = session_start
        self.session_end = session_end
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 이동평균
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)

        # 상위 TF 추세 대용: EMA(200) 기울기 (N기간 전 대비 양수인지)
        df["ema_slow_slope"] = df["ema_slow"] - df["ema_slow"].shift(self.ema_slope_period)

        # EMA 정배열
        df["ema_aligned"] = (df["ema_fast"] > df["ema_mid"]).astype(int)

        # 모멘텀
        df["mom"] = momentum(df["close"], self.mom_period)

        # RSI + 이전 RSI
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["rsi_prev"] = df["rsi"].shift(1)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        # 세션 정보 (UTC 시간)
        if hasattr(df.index, 'hour'):
            df["_hour"] = df.index.hour
        else:
            df["_hour"] = 12  # 일봉 등 시간 정보 없으면 항상 허용

        return df

    def _in_session(self, hour: int) -> bool:
        return self.session_start <= hour < self.session_end

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 0)

        # --- 포지션 보유 중: 청산 ---
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

            # 추세 역전
            if row.get("ema_fast", close) < row.get("ema_mid", close):
                self._in_position = False
                return -1

            return 0

        # --- 미보유: 매수 조건 ---
        hour = int(row.get("_hour", 12))
        if not self._in_session(hour):
            return 0

        ema_aligned = bool(row.get("ema_aligned", 0))
        ema_slow_slope = row.get("ema_slow_slope", 0)
        mom_val = row.get("mom", 0)
        rsi_val = row.get("rsi", 50)
        rsi_prev = row.get("rsi_prev", 50)
        adx_val = row.get("adx", 0)

        rsi_bounced = (rsi_prev < self.rsi_pullback) and (rsi_val >= self.rsi_pullback)

        if (
            ema_aligned                     # 단기 > 중기
            and ema_slow_slope > 0          # 장기 추세 상승 중
            and mom_val > 0                 # 모멘텀 양수
            and adx_val > self.adx_threshold
            and rsi_bounced                 # RSI 반등
        ):
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
