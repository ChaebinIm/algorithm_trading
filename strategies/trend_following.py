"""
다중 필터 추세 추종 전략 (Trend Following).

설계 철학: 보수적 진입, 높은 승률과 손익비.
- 여러 필터를 중첩하여 확실한 추세에서만 진입
- 추세 내 눌림목(조정)에서 진입하여 유리한 가격 확보
- 추세 이탈 시 빠른 청산
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, rsi, atr, adx


class TrendFollowingStrategy(Strategy):
    """다중 필터 추세 추종 전략.

    진입 조건 (매수, 모든 조건 동시 충족):
    1. EMA 정배열: EMA(20) > EMA(50) > EMA(200) → 상승 추세 확인
    2. ADX > 25: 충분한 추세 강도 존재
    3. RSI < 40: 추세 내 눌림목(조정) 구간에서 진입
    4. 종가가 EMA(20) 위에 복귀: 눌림 후 반등 확인

    청산 조건 (매도, 하나라도 충족 시):
    - RSI > 70: 과매수 구간 도달 → 수익 실현
    - 종가가 EMA(50) 아래로 하락 → 추세 이탈 판단
    """

    name = "trend_following"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 25.0,    # ADX가 이 값 이상이어야 추세로 판단
        rsi_entry: float = 40.0,         # RSI가 이 값 이하일 때 진입 허용
        rsi_exit: float = 70.0,          # RSI가 이 값 이상이면 익절
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit

        # 내부 상태: 진입 여부 추적 (중복 진입 방지)
        self._in_position = False

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 선계산 (벡터화)."""
        df = df.copy()

        # EMA 3중 이동평균
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)

        # RSI
        df["rsi"] = rsi(df["close"], self.rsi_period)

        # ADX (추세 강도)
        df["adx"] = adx(df, self.adx_period)

        # ATR (변동성 — 참고용)
        df["atr"] = atr(df, 14)

        # EMA 정배열 여부: fast > mid > slow
        df["ema_aligned"] = (
            (df["ema_fast"] > df["ema_mid"]) &
            (df["ema_mid"] > df["ema_slow"])
        ).astype(int)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성."""
        ema_aligned = bool(row.get("ema_aligned", 0))
        adx_val = row.get("adx", 0)
        rsi_val = row.get("rsi", 50)
        close = row["close"]
        ema_fast_val = row.get("ema_fast", close)
        ema_mid_val = row.get("ema_mid", close)

        # --- 매도 조건 (포지션 보유 중) ---
        if self._in_position:
            if rsi_val > self.rsi_exit:          # 과매수 → 익절
                self._in_position = False
                return -1
            if close < ema_mid_val:              # 추세 이탈 → 손절
                self._in_position = False
                return -1
            return 0  # 계속 보유

        # --- 매수 조건 (모든 필터 충족 시) ---
        if (
            ema_aligned
            and adx_val > self.adx_threshold
            and rsi_val < self.rsi_entry
            and close > ema_fast_val
        ):
            self._in_position = True
            return +1

        return 0
