"""
다중 필터 추세 추종 전략 (Trend Following).

설계 철학: 보수적 진입, 높은 승률과 손익비.
- 여러 필터를 중첩하여 확실한 추세에서만 진입
- 추세 내 눌림목(조정)에서 진입하여 유리한 가격 확보
- ATR 기반 손절 + 트레일링 스탑으로 추세 끝까지 보유

손절/익절:
- 손절: 진입가 - 2.0 × ATR (노이즈 회피)
- 익절: 고정 TP 없음, 트레일링 스탑으로 수익 보호
- 트레일링: 보유 중 고점 - 2.5 × ATR (추세 끝까지 추종)
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

    청산 조건 (하나라도 충족 시):
    - 종가 < 진입가 - 2.0×ATR → 손절
    - 종가 < 보유 중 고점 - 2.5×ATR → 트레일링 스탑
    - 종가 < EMA(50) → 추세 이탈
    """

    name = "trend_following"

    def __init__(
        self,
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_entry: float = 40.0,
        sl_atr_mult: float = 2.0,        # 손절: 진입가 - N × ATR
        trailing_atr_mult: float = 2.5,   # 트레일링: 고점 - N × ATR
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.rsi_entry = rsi_entry
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        # 내부 상태
        self._in_position = False
        self._entry_price = 0.0    # 진입가
        self._highest = 0.0        # 보유 중 최고가 (트레일링용)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 선계산 (벡터화)."""
        df = df.copy()
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow"] = ema(df["close"], self.ema_slow)
        df["rsi"] = rsi(df["close"], self.rsi_period)
        df["adx"] = adx(df, self.adx_period)
        df["atr"] = atr(df, 14)
        df["ema_aligned"] = (
            (df["ema_fast"] > df["ema_mid"]) &
            (df["ema_mid"] > df["ema_slow"])
        ).astype(int)
        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성."""
        close = row["close"]
        atr_val = row.get("atr", 0)

        # --- 포지션 보유 중: 청산 조건 체크 ---
        if self._in_position:
            # 고점 갱신 (트레일링용)
            if close > self._highest:
                self._highest = close

            # 손절: 진입가 기준 ATR 손절
            sl_price = self._entry_price - self.sl_atr_mult * atr_val
            if close < sl_price:
                self._in_position = False
                return -1

            # 트레일링 스탑: 고점 기준
            trailing_stop = self._highest - self.trailing_atr_mult * atr_val
            if close < trailing_stop:
                self._in_position = False
                return -1

            # 추세 이탈: EMA 중기선 아래
            ema_mid_val = row.get("ema_mid", close)
            if close < ema_mid_val:
                self._in_position = False
                return -1

            return 0  # 계속 보유

        # --- 미보유: 매수 조건 ---
        ema_aligned = bool(row.get("ema_aligned", 0))
        adx_val = row.get("adx", 0)
        rsi_val = row.get("rsi", 50)
        ema_fast_val = row.get("ema_fast", close)

        if (
            ema_aligned
            and adx_val > self.adx_threshold
            and rsi_val < self.rsi_entry
            and close > ema_fast_val
        ):
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
