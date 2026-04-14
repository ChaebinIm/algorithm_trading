"""
strategies 패키지 — 트레이딩 전략 모음.

새 전략 추가 방법:
1. strategies/ 폴더에 새 .py 파일 생성 (예: bollinger_band.py)
2. Strategy를 상속한 클래스 정의
3. 이 파일의 STRATEGY_REGISTRY에 등록

사용 예시:
    from strategies import get_strategy
    strat = get_strategy("trend_following")
"""
from __future__ import annotations

import pandas as pd


# =============================
# 전략 베이스 클래스
# =============================

class Strategy:
    """전략 베이스 클래스.

    모든 전략은 이 클래스를 상속하고 아래 두 메서드를 구현해야 한다:
    - prepare(df): 백테스트 시작 전, 지표/전처리 칼럼을 추가
    - on_bar(i, row): i번째 바에서 트레이드 신호 반환 (+1 매수, -1 매도, 0 유지)
    """

    # 전략 이름 (CLI에서 --strategy로 참조할 때 사용)
    name: str = "base"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """지표/전처리 칼럼 추가. 서브클래스에서 오버라이드."""
        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        """바별 신호 생성. 서브클래스에서 오버라이드.
        Returns: +1 (매수), -1 (매도), 0 (유지)
        """
        return 0


# =============================
# 전략 레지스트리
# =============================

# 전략 이름 → 클래스 매핑 (지연 임포트로 순환참조 방지)
def _build_registry() -> dict:
    """등록된 전략들을 딕셔너리로 반환."""
    from strategies.ma_cross import MovingAverageCross
    from strategies.trend_following import TrendFollowingStrategy
    from strategies.dual_momentum import DualMomentumStrategy
    from strategies.atr_breakout import ATRBreakoutStrategy
    from strategies.bollinger_squeeze import BollingerSqueezeStrategy

    return {
        "ma_cross": MovingAverageCross,
        "trend_following": TrendFollowingStrategy,
        "dual_momentum": DualMomentumStrategy,
        "atr_breakout": ATRBreakoutStrategy,
        "bollinger_squeeze": BollingerSqueezeStrategy,
    }


def get_strategy(name: str, **kwargs) -> Strategy:
    """이름으로 전략 인스턴스를 생성.

    Parameters
    ----------
    name : str
        전략 이름 (예: "trend_following", "ma_cross")
    **kwargs
        전략 생성자에 전달할 파라미터

    Returns
    -------
    Strategy 인스턴스

    Raises
    ------
    ValueError
        등록되지 않은 전략 이름인 경우
    """
    registry = _build_registry()
    if name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"알 수 없는 전략: '{name}'. 사용 가능: {available}")
    return registry[name](**kwargs)


def list_strategies() -> list:
    """등록된 전략 이름 목록 반환."""
    return list(_build_registry().keys())
