"""
bot/risk_manager.py — 리스크 관리 레이어.

책임:
- 일일 P&L 추적 (UTC 자정 기준 리셋)
- 포트폴리오 드로우다운 계산
- 주문 허용 여부 판단
- 거래 기록 업데이트
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Tuple

from bot.config import BotConfig

logger = logging.getLogger(__name__)


class RiskManager:
    """리스크 관리 클래스.

    체크 항목:
    1. 일일 손실 한도 초과 여부
    2. 포트폴리오 최대 낙폭 초과 여부
    3. 단일 포지션 비중 초과 여부
    """

    def __init__(self, config: BotConfig, initial_equity: float = 0.0):
        self.config = config

        # 자본 추적
        self._initial_equity = initial_equity   # 봇 시작 시 총 자본
        self._peak_equity = initial_equity      # 사상 최고 자본 (드로우다운 기준)
        self._current_equity = initial_equity   # 현재 추정 자본

        # 일일 P&L 추적
        self._daily_pnl: float = 0.0
        self._daily_start_equity: float = initial_equity
        self._last_reset_date: str = self._today_utc()

        # 종목별 포지션 (qty * avg_price)
        self._positions: Dict[str, float] = {}  # symbol → 평가금액

        logger.info(
            f"[RiskManager] 초기화 완료 | 초기 자본: {initial_equity:.2f} USDT"
        )

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _maybe_reset_daily(self) -> None:
        """UTC 자정이 지나면 일일 P&L을 리셋한다."""
        today = self._today_utc()
        if today != self._last_reset_date:
            logger.info(
                f"[RiskManager] 일일 리셋: {self._last_reset_date} → {today} | "
                f"어제 P&L: {self._daily_pnl:+.2f} USDT"
            )
            self._daily_pnl = 0.0
            self._daily_start_equity = self._current_equity
            self._last_reset_date = today

    def _drawdown(self) -> float:
        """현재 낙폭 비율 (0.0~1.0). 피크 대비 얼마나 빠졌나."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    def _daily_loss_pct(self) -> float:
        """오늘 손실 비율 (음수 = 손실). 일일 시작 자본 대비."""
        if self._daily_start_equity <= 0:
            return 0.0
        return self._daily_pnl / self._daily_start_equity

    def _position_pct(self, symbol: str, additional_value: float) -> float:
        """해당 종목에 추가로 투자했을 때 포트폴리오 대비 비중."""
        if self._current_equity <= 0:
            return 1.0
        existing = self._positions.get(symbol, 0.0)
        return (existing + additional_value) / self._current_equity

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def set_equity(self, equity: float) -> None:
        """현재 총 자본을 외부에서 업데이트한다 (거래소 잔고 조회 후 호출)."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._initial_equity <= 0:
            self._initial_equity = equity
            self._daily_start_equity = equity
            self._peak_equity = equity

    def check_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> Tuple[bool, str]:
        """주문 허용 여부를 판단한다.

        Args:
            symbol: 예) "BTC/USDT"
            side: "buy" 또는 "sell"
            qty: 주문 수량
            price: 현재가

        Returns:
            (allowed: bool, reason: str)
            - allowed=True → 주문 진행 가능
            - allowed=False → reason에 거부 이유
        """
        self._maybe_reset_daily()

        # 매도는 항상 허용 (청산 목적)
        if side == "sell":
            return True, "OK (매도 청산)"

        order_value = qty * price

        # ① 일일 손실 한도 체크
        daily_loss = self._daily_loss_pct()
        if daily_loss < -self.config.max_daily_loss_pct:
            reason = (
                f"일일 손실 한도 초과: {daily_loss:.1%} "
                f"(한도: -{self.config.max_daily_loss_pct:.1%})"
            )
            logger.warning(f"[RiskManager] 주문 거부 — {reason}")
            return False, reason

        # ② 드로우다운 한도 체크
        dd = self._drawdown()
        if dd > self.config.max_drawdown_pct:
            reason = (
                f"최대 낙폭 초과: {dd:.1%} "
                f"(한도: {self.config.max_drawdown_pct:.1%})"
            )
            logger.warning(f"[RiskManager] 주문 거부 — {reason}")
            return False, reason

        # ③ 단일 포지션 비중 체크
        pos_pct = self._position_pct(symbol, order_value)
        if pos_pct > self.config.max_position_pct:
            reason = (
                f"{symbol} 포지션 비중 초과: {pos_pct:.1%} "
                f"(한도: {self.config.max_position_pct:.1%})"
            )
            logger.warning(f"[RiskManager] 주문 거부 — {reason}")
            return False, reason

        logger.debug(
            f"[RiskManager] 주문 허용: {side} {qty:.6f} {symbol} @ {price:.4f} "
            f"| 일손실={daily_loss:.1%}, DD={dd:.1%}, 포지션={pos_pct:.1%}"
        )
        return True, "OK"

    def record_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> None:
        """거래 실행 후 내부 상태를 업데이트한다.

        Args:
            symbol: 예) "BTC/USDT"
            side: "buy" 또는 "sell"
            qty: 실행 수량
            price: 체결가
        """
        self._maybe_reset_daily()
        trade_value = qty * price

        if side == "buy":
            # 포지션 증가
            self._positions[symbol] = self._positions.get(symbol, 0.0) + trade_value
            logger.info(
                f"[RiskManager] 매수 기록: {qty:.6f} {symbol} @ {price:.4f} "
                f"(포지션 가치: {self._positions[symbol]:.2f} USDT)"
            )

        elif side == "sell":
            # 포지션 감소 및 P&L 추산
            entry_value = self._positions.get(symbol, trade_value)
            pnl = trade_value - entry_value
            self._daily_pnl += pnl
            self._current_equity += pnl
            if self._current_equity > self._peak_equity:
                self._peak_equity = self._current_equity
            self._positions[symbol] = 0.0
            logger.info(
                f"[RiskManager] 매도 기록: {qty:.6f} {symbol} @ {price:.4f} "
                f"| P&L: {pnl:+.2f} USDT | 일일 누적: {self._daily_pnl:+.2f}"
            )

    def get_stats(self) -> dict:
        """현재 리스크 통계를 반환한다."""
        self._maybe_reset_daily()
        return {
            "daily_pnl": round(self._daily_pnl, 4),
            "daily_pnl_pct": round(self._daily_loss_pct() * 100, 2),
            "total_equity": round(self._current_equity, 4),
            "peak_equity": round(self._peak_equity, 4),
            "drawdown_pct": round(self._drawdown() * 100, 2),
            "positions": dict(self._positions),
        }
