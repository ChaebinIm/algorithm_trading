"""
이동평균 교차 전략 (Moving Average Crossover).

단기/장기 이동평균의 교차를 이용한 기본 추세 추종 전략.
- 단기 MA > 장기 MA 전환 시 매수
- 단기 MA < 장기 MA 전환 시 매도
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import sma


class MovingAverageCross(Strategy):
    """단기/장기 이동평균 교차 전략.

    신호 로직:
    - ma_short > ma_long 구간에서 sig=1, 아니면 sig=0
    - sig가 0→1로 변하면 매수 (+1)
    - sig가 1→0으로 변하면 매도 (-1)
    """

    name = "ma_cross"

    def __init__(self, short: int = 5, long: int = 20):
        self.short = short
        self.long = long

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ma_s"] = sma(df["close"], self.short)
        df["ma_l"] = sma(df["close"], self.long)
        df["sig"] = 0
        df.loc[df["ma_s"] > df["ma_l"], "sig"] = 1
        df["sig"] = df["sig"].fillna(0).astype(int)
        df["sig_change"] = df["sig"].diff().fillna(0)
        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        ch = int(row.get("sig_change", 0))
        if ch > 0:
            return +1
        if ch < 0:
            return -1
        return 0
