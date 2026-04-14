"""
데이터 타입 정의 — 백테스트 결과를 담는 데이터클래스들.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict

import pandas as pd


@dataclass
class Fill:
    """단일 체결 레코드."""
    ts: pd.Timestamp
    side: str
    price: float
    qty: float
    fee: float
    notional: float


@dataclass
class TradeLog:
    """체결 목록을 보관하고 DataFrame으로 변환하는 헬퍼."""
    fills: List[Fill]

    def to_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["ts", "side", "price", "qty", "fee", "notional"])
        d = [asdict(f) for f in self.fills]
        return pd.DataFrame(d)


@dataclass
class BacktestReport:
    """백테스트 결과 컨테이너.

    metrics에 포함되는 지표:
    - cum_return: 누적 수익률
    - ann_return: 연환산 수익률
    - ann_vol: 연환산 변동성
    - sharpe: 샤프 비율
    - mdd: 최대 낙폭 (음수)
    - calmar: Calmar 비율 = 연수익률 / |MDD|
    - win_rate: 승률 (익절 거래 / 전체 거래)
    - profit_factor: Profit Factor = 총 이익 / 총 손실
    - avg_win: 평균 이기는 거래 수익률
    - avg_loss: 평균 지는 거래 수익률
    - total_trades: 총 거래 수 (라운드트립)
    """
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

    def summary(self) -> Dict[str, float]:
        return self.metrics
