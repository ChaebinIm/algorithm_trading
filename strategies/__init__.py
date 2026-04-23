"""
strategies 패키지 — 트레이딩 전략 모음.

전략 추가 방법:
1. strategies/ 폴더에 새 .py 파일 생성
2. Strategy 베이스 클래스 상속 후 prepare(df) + on_bar(i, row) 구현
3. _build_registry()에 등록

on_bar 반환값:
  +1  : 매수 (롱 진입)
  -1  : 매도 (포지션 청산)
   0  : 유지 (현재 상태 그대로)
"""
from __future__ import annotations

import pandas as pd


# =============================
# 전략 베이스 클래스
# =============================

class Strategy:
    """전략 베이스 클래스.

    하위 클래스에서 반드시 구현:
    - prepare(df)    : 지표 칼럼을 df에 추가 후 반환 (벡터화 계산)
    - on_bar(i, row) : 현재 바의 신호 반환 (+1 / -1 / 0)
    """

    name: str = "base"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        return 0


# =============================
# 전략 레지스트리
# =============================

def _build_registry() -> dict:
    from strategies.momentum_pullback import MomentumPullbackStrategy
    from strategies.mtf_pullback import MTFPullbackStrategy
    from strategies.jangdae_momentum import JangdaeMomentumStrategy
    from strategies.jangdae_eumbon_reversal import JangdaeEumbonReversalStrategy
    from strategies.intermarket_momentum import IntermarketMomentumStrategy
    return {
        "momentum_pullback": MomentumPullbackStrategy,
        "mtf_pullback": MTFPullbackStrategy,
        "jangdae_momentum": JangdaeMomentumStrategy,
        "jangdae_eumbon_reversal": JangdaeEumbonReversalStrategy,
        "intermarket_momentum": IntermarketMomentumStrategy,
    }


def get_strategy(name: str, **kwargs) -> Strategy:
    registry = _build_registry()
    if name not in registry:
        available = ", ".join(registry.keys()) or "(없음)"
        raise ValueError(f"알 수 없는 전략: '{name}'. 사용 가능: {available}")
    return registry[name](**kwargs)


def list_strategies() -> list:
    return list(_build_registry().keys())
