"""
bot/config.py — 라이브 트레이딩 봇 설정.

수정 방법:
  1. API_KEY, API_SECRET을 발급받은 키로 교체
  2. PORTFOLIO에서 종목/전략/비중 조정
  3. TESTNET=True로 먼저 테스트 후 False로 전환
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PortfolioEntry:
    """포트폴리오 1개 항목 설정."""
    symbol: str          # 예: "BTC/USDT"
    strategy_name: str   # 전략 레지스트리 키 (예: "ensemble")
    timeframe: str       # CCXT 타임프레임 (예: "4h", "1h")
    allocation_pct: float  # 이 종목에 배분할 총 자본 비율 (0.0~1.0)


@dataclass
class BotConfig:
    """전체 봇 설정."""

    # ── 거래소 설정 ─────────────────────────────────────────────────────
    api_key: str = "YOUR_API_KEY_HERE"
    api_secret: str = "YOUR_API_SECRET_HERE"
    testnet: bool = True   # True = 테스트넷, False = 실거래

    # ── 포트폴리오 설정 ─────────────────────────────────────────────────
    # allocation_pct 합계는 반드시 1.0 이하여야 한다.
    # 예: BTC 40%, ETH 35%, BNB 25%
    portfolio: List[PortfolioEntry] = field(default_factory=lambda: [
        PortfolioEntry(
            symbol="BTC/USDT",
            strategy_name="ensemble",
            timeframe="4h",
            allocation_pct=0.40,
        ),
        PortfolioEntry(
            symbol="ETH/USDT",
            strategy_name="ensemble",
            timeframe="4h",
            allocation_pct=0.35,
        ),
        PortfolioEntry(
            symbol="BNB/USDT",
            strategy_name="ensemble",
            timeframe="4h",
            allocation_pct=0.25,
        ),
    ])

    # ── 리스크 설정 ─────────────────────────────────────────────────────
    max_daily_loss_pct: float = 0.05    # 일일 최대 손실 5%
    max_drawdown_pct: float = 0.15      # 최대 낙폭 15%
    max_position_pct: float = 0.25      # 단일 포지션 최대 25%

    # ── 데이터 설정 ─────────────────────────────────────────────────────
    data_lookback: int = 500            # 지표 워밍업용 캔들 수

    # ── 텔레그램 설정 (선택) ───────────────────────────────────────────
    telegram_token: Optional[str] = None    # 없으면 알림 생략
    telegram_chat_id: Optional[str] = None

    # ── 로그 설정 ───────────────────────────────────────────────────────
    log_dir: str = "logs/"

    # ── 페이퍼 트레이딩 ─────────────────────────────────────────────────
    paper_trading: bool = False  # True = 주문 안 냄 (API만 연결)


# ── 싱글턴 기본 설정 ────────────────────────────────────────────────────
# run_bot.py에서 import해서 수정하거나 환경변수로 override할 수 있다.
DEFAULT_CONFIG = BotConfig()
