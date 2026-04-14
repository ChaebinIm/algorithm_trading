"""
데이터 타입 정의 — 백테스트 결과를 담는 데이터클래스들.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict

import pandas as pd


@dataclass
class Fill:
    """단일 체결 레코드.
    - ts: 체결 시각 (다음 바 시가 기준 체결 가정)
    - side: 'buy' 또는 'sell'
    - price: 체결 단가 (슬리피지 적용 후, USDT)
    - qty: 체결 수량
    - fee: 수수료 금액 (USDT)
    - notional: 체결 금액 (단가*수량, USDT)
    """
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
    - equity: 바별 자본곡선
    - returns: 바별 수익률(자본곡선 기준)
    - trades: 체결 로그(데이터프레임)
    - metrics: 성과 요약치(dict)
    """
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]

    def summary(self) -> Dict[str, float]:
        return self.metrics
