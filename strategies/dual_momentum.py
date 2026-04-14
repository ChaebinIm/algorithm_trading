"""
듀얼 모멘텀 전략 (Dual Momentum).

설계 철학: 하락장 회피 + 강한 추세에서만 진입.
- 절대 모멘텀: 최근 N기간 수익률이 양수일 때만 투자 (음수면 현금 보유)
- 장기 추세 필터: EMA(200) 위에 있을 때만 진입
- 낮은 매매 빈도, 큰 추세만 포착

손절/익절:
- 손절: 진입가 - 2.0 × ATR
- 익절: 고정 TP 없음, 트레일링 스탑으로 수익 보호
- 트레일링: 보유 중 고점 - 2.0 × ATR

참고: 단일 코인 백테스트에서는 절대 모멘텀만 적용.
      다중 코인 운용 시 상대 모멘텀(코인 간 순위 비교)을 추가할 수 있음.
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import ema, momentum, atr, rsi


class DualMomentumStrategy(Strategy):
    """듀얼 모멘텀 전략.

    진입 조건 (모든 조건 동시 충족):
    1. 절대 모멘텀 > 0: 최근 N기간 수익률이 양수 (상승 추세 중)
    2. 종가 > EMA(200): 장기 상승 추세 확인
    3. RSI < 65: 과매수 구간이 아닌 상태에서 진입 (약간의 여유)
    4. 모멘텀 가속: 단기 모멘텀 > 장기 모멘텀 (추세 가속 구간)

    청산 조건:
    - 절대 모멘텀 < 0: 추세 반전 → 현금 복귀
    - 종가 < 진입가 - 2.0×ATR → 손절
    - 종가 < 고점 - 2.0×ATR → 트레일링 스탑
    """

    name = "dual_momentum"

    def __init__(
        self,
        mom_long: int = 90,           # 장기 모멘텀 기간 (약 3개월)
        mom_short: int = 20,          # 단기 모멘텀 기간 (약 1개월)
        ema_trend: int = 200,         # 장기 추세 필터
        rsi_period: int = 14,
        rsi_max_entry: float = 65.0,  # 과매수 방지 필터
        sl_atr_mult: float = 2.0,     # 손절 ATR 배수
        trailing_atr_mult: float = 2.0,  # 트레일링 ATR 배수
    ):
        self.mom_long = mom_long
        self.mom_short = mom_short
        self.ema_trend = ema_trend
        self.rsi_period = rsi_period
        self.rsi_max_entry = rsi_max_entry
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        # 내부 상태
        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표 선계산."""
        df = df.copy()

        # 모멘텀 (수익률 기반)
        df["mom_long"] = momentum(df["close"], self.mom_long)
        df["mom_short"] = momentum(df["close"], self.mom_short)

        # 장기 추세 필터
        df["ema_trend"] = ema(df["close"], self.ema_trend)

        # RSI
        df["rsi"] = rsi(df["close"], self.rsi_period)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성."""
        close = row["close"]
        atr_val = row.get("atr", 0)
        mom_long_val = row.get("mom_long", 0)

        # --- 포지션 보유 중: 청산 조건 ---
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # 모멘텀 반전 → 추세 종료
            if mom_long_val < 0:
                self._in_position = False
                return -1

            # 손절
            sl_price = self._entry_price - self.sl_atr_mult * atr_val
            if close < sl_price:
                self._in_position = False
                return -1

            # 트레일링 스탑
            trailing_stop = self._highest - self.trailing_atr_mult * atr_val
            if close < trailing_stop:
                self._in_position = False
                return -1

            return 0

        # --- 미보유: 매수 조건 ---
        mom_short_val = row.get("mom_short", 0)
        ema_trend_val = row.get("ema_trend", close)
        rsi_val = row.get("rsi", 50)

        if (
            mom_long_val > 0                    # 절대 모멘텀 양수
            and close > ema_trend_val            # 장기 추세 위
            and rsi_val < self.rsi_max_entry     # 과매수 아님
            and mom_short_val > mom_long_val     # 모멘텀 가속 중
        ):
            self._in_position = True
            self._entry_price = close
            self._highest = close
            return +1

        return 0
