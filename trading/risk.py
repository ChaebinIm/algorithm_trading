"""
리스크 관리 설정 및 로직.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskConfig:
    """리스크 관리 파라미터.
    - max_position_usdt: 단일 코인 최대 포지션 크기 (USDT)
    - max_daily_loss_usdt: 일일 최대 허용 손실 (USDT, 초과 시 매매 중단)
    - usdt_per_trade: 1회 매매 금액 (USDT)
    """
    max_position_usdt: float = 1_000.0
    max_daily_loss_usdt: float = 500.0
    usdt_per_trade: float = 100.0
