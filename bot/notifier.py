"""
bot/notifier.py — 텔레그램 알림 모듈.

텔레그램 토큰/채팅 ID가 설정되지 않으면 모든 알림을 조용히 건너뜀.
Bot 토큰 발급: @BotFather, 채팅 ID: @userinfobot
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 텔레그램 라이브러리는 선택적 의존성 — 없어도 동작
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.warning("[Notifier] 'requests' 패키지 없음. 텔레그램 비활성화.")


class Notifier:
    """텔레그램 알림 클래스.

    config.telegram_token이 None이면 모든 메서드가 no-op으로 동작한다.
    """

    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat_id = chat_id
        self._enabled = bool(
            token and chat_id and _REQUESTS_AVAILABLE
        )
        if self._enabled:
            logger.info("[Notifier] 텔레그램 알림 활성화")
        else:
            logger.info("[Notifier] 텔레그램 알림 비활성화 (설정 없음)")

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────

    def _send_raw(self, text: str) -> None:
        """텔레그램 API로 메시지를 전송한다."""
        if not self._enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        try:
            resp = _requests.post(url, json=payload, timeout=10)
            if not resp.ok:
                logger.warning(f"[Notifier] 텔레그램 전송 실패: {resp.text}")
        except Exception as e:
            # 알림 실패가 봇을 멈추면 안 됨 → 로그만
            logger.warning(f"[Notifier] 텔레그램 오류 (무시): {e}")

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def send(self, message: str) -> None:
        """임의 메시지를 전송한다."""
        logger.debug(f"[Notifier] 알림: {message[:80]}")
        self._send_raw(message)

    def notify_entry(
        self,
        symbol: str,
        qty: float,
        price: float,
        strategy: str,
    ) -> None:
        """매수 체결 알림."""
        text = (
            f"<b>매수 체결</b>\n"
            f"심볼: <code>{symbol}</code>\n"
            f"전략: {strategy}\n"
            f"수량: {qty:.6f}\n"
            f"가격: {price:,.4f} USDT\n"
            f"금액: {qty * price:,.2f} USDT"
        )
        logger.info(f"[알림] 매수: {symbol} {qty:.6f} @ {price:.4f}")
        self._send_raw(text)

    def notify_exit(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        """매도 체결 알림."""
        pnl_sign = "+" if pnl_pct >= 0 else ""
        emoji = "▲" if pnl_pct >= 0 else "▼"
        text = (
            f"<b>매도 체결 {emoji}</b>\n"
            f"심볼: <code>{symbol}</code>\n"
            f"수량: {qty:.6f}\n"
            f"진입가: {entry_price:,.4f} USDT\n"
            f"청산가: {exit_price:,.4f} USDT\n"
            f"손익: <b>{pnl_sign}{pnl_pct:.2f}%</b>"
        )
        logger.info(f"[알림] 매도: {symbol} {qty:.6f} @ {exit_price:.4f} ({pnl_sign}{pnl_pct:.2f}%)")
        self._send_raw(text)

    def notify_daily_summary(self, stats: dict) -> None:
        """일일 결산 알림."""
        daily_pnl = stats.get("daily_pnl", 0.0)
        daily_pnl_pct = stats.get("daily_pnl_pct", 0.0)
        equity = stats.get("total_equity", 0.0)
        dd = stats.get("drawdown_pct", 0.0)

        sign = "+" if daily_pnl >= 0 else ""
        text = (
            f"<b>일일 결산 리포트</b>\n"
            f"날짜: {stats.get('date', 'N/A')}\n"
            f"일일 P&L: <b>{sign}{daily_pnl:.2f} USDT ({sign}{daily_pnl_pct:.2f}%)</b>\n"
            f"총 자산: {equity:,.2f} USDT\n"
            f"낙폭: {dd:.2f}%"
        )
        logger.info(f"[알림] 일일 결산: P&L={sign}{daily_pnl:.2f} USDT ({sign}{daily_pnl_pct:.2f}%)")
        self._send_raw(text)

    def notify_error(self, error: str) -> None:
        """에러 알림."""
        text = (
            f"<b>봇 오류 발생</b>\n"
            f"<code>{error[:500]}</code>"
        )
        logger.error(f"[알림] 에러: {error[:200]}")
        self._send_raw(text)
