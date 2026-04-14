"""
strategies 패키지 — 트레이딩 전략 모음.

새 전략 추가 방법:
1. strategies/ 폴더에 새 .py 파일 생성 (예: bollinger_band.py)
2. Strategy를 상속한 클래스 정의
3. 이 파일의 STRATEGY_REGISTRY에 등록

사용 예시:
    from strategies import get_strategy
    strat = get_strategy("adaptive_regime")
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

    name: str = "base"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        return 0


# =============================
# 전략 레지스트리
# =============================

def _build_registry() -> dict:
    from strategies.ma_cross import MovingAverageCross
    from strategies.trend_following import TrendFollowingStrategy
    from strategies.dual_momentum import DualMomentumStrategy
    from strategies.atr_breakout import ATRBreakoutStrategy
    from strategies.bollinger_squeeze import BollingerSqueezeStrategy
    from strategies.enhanced_trend import EnhancedTrendStrategy
    from strategies.enhanced_breakout import EnhancedBreakoutStrategy
    from strategies.ensemble import EnsembleStrategy
    # 신규 전략
    from strategies.supertrend_strategy import SuperTrendStrategy
    from strategies.adaptive_regime import AdaptiveRegimeStrategy
    from strategies.macd_volume import MACDVolumeStrategy
    from strategies.multi_factor import MultiFactorStrategy
    from strategies.supertrend_ensemble import SuperTrendEnsembleStrategy
    from strategies.champion import ChampionStrategy
    from strategies.champion_v2 import ChampionV2Strategy
    from strategies.ensemble_short import EnsembleShortStrategy

    return {
        # 기존 전략
        "ma_cross": MovingAverageCross,
        "trend_following": TrendFollowingStrategy,
        "dual_momentum": DualMomentumStrategy,
        "atr_breakout": ATRBreakoutStrategy,
        "bollinger_squeeze": BollingerSqueezeStrategy,
        "enhanced_trend": EnhancedTrendStrategy,
        "enhanced_breakout": EnhancedBreakoutStrategy,
        "ensemble": EnsembleStrategy,
        # 신규 전략 (고도화)
        "supertrend": SuperTrendStrategy,
        "adaptive_regime": AdaptiveRegimeStrategy,
        "macd_volume": MACDVolumeStrategy,
        "multi_factor": MultiFactorStrategy,
        "supertrend_ensemble": SuperTrendEnsembleStrategy,
        "champion": ChampionStrategy,
        "champion_v2": ChampionV2Strategy,
        "ensemble_short": EnsembleShortStrategy,
    }


def get_strategy(name: str, **kwargs) -> Strategy:
    registry = _build_registry()
    if name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"알 수 없는 전략: '{name}'. 사용 가능: {available}")
    return registry[name](**kwargs)


def list_strategies() -> list:
    return list(_build_registry().keys())
