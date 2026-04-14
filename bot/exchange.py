"""
bot/exchange.py — Binance 거래소 래퍼 (ccxt 기반).

기능:
- OHLCV 데이터 조회
- 잔고/포지션 조회
- 시장가 주문 실행
- 테스트넷 지원
- 자동 재시도 (최대 3회, 지수 백오프)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import ccxt
import pandas as pd

from bot.config import BotConfig

logger = logging.getLogger(__name__)

# 재시도 설정
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0   # 초 (1 → 2 → 4)


def _retry(fn, *args, **kwargs):
    """최대 MAX_RETRIES회 재시도하는 래퍼. 지수 백오프 적용."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            last_exc = e
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                f"[Exchange] 네트워크 오류 (시도 {attempt + 1}/{MAX_RETRIES}): "
                f"{e} — {delay:.1f}초 후 재시도"
            )
            time.sleep(delay)
        except ccxt.RateLimitExceeded as e:
            last_exc = e
            delay = 5.0 * (attempt + 1)
            logger.warning(f"[Exchange] Rate limit 초과 — {delay:.1f}초 대기")
            time.sleep(delay)
        except ccxt.ExchangeError as e:
            # 거래소 레벨 에러는 재시도 없이 바로 raise
            logger.error(f"[Exchange] 거래소 에러: {e}")
            raise
    raise last_exc


class BinanceExchange:
    """Binance ccxt 래퍼 클래스.

    testnet=True 이면 바이낸스 테스트넷에 연결한다.
    paper_trading=True 이면 주문 함수가 실제로 요청을 보내지 않는다.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.paper_trading = config.paper_trading
        self._public = self._build_public_client()   # API 키 불필요 (OHLCV 등)
        self._client = self._build_client()           # API 키 필요 (주문, 잔고)

    # ── 클라이언트 생성 ────────────────────────────────────────────────

    def _build_public_client(self) -> ccxt.binance:
        """공개 API 전용 클라이언트 (API 키 불필요)."""
        client = ccxt.binance({
            "options": {"defaultType": "spot"},
            "enableRateLimit": True,
        })
        # 마켓 정보 미리 로드 (인증 없이 가능)
        client.load_markets()
        return client

    def _build_client(self) -> ccxt.binance:
        """인증 클라이언트 (주문/잔고용). API 키 없으면 None."""
        if not self.config.api_key or not self.config.api_secret:
            return None

        options: dict = {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
        }

        exchange_params = {
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "options": options,
            "enableRateLimit": True,
        }

        if self.config.testnet:
            exchange_params["urls"] = {
                "api": {"rest": "https://testnet.binance.vision/api"}
            }
            logger.info("[Exchange] 테스트넷 모드로 초기화")
        else:
            logger.info("[Exchange] 실거래 모드로 초기화")

        return ccxt.binance(exchange_params)

    # ── 공개 메서드 ────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """OHLCV 캔들 데이터를 가져온다.

        Args:
            symbol: 예) "BTC/USDT"
            timeframe: 예) "4h", "1h", "15m"
            limit: 최대 캔들 수

        Returns:
            columns=[open, high, low, close, volume], index=DatetimeIndex(UTC)
        """
        def _fetch():
            raw = self._public.fetch_ohlcv(symbol, timeframe, limit=limit)
            return raw

        raw = _retry(_fetch)

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.astype(float)
        df = df.sort_index()

        # 마지막 캔들은 현재 진행 중인 미완성 캔들 → 제외
        df = df.iloc[:-1]

        logger.debug(f"[Exchange] {symbol} {timeframe} {len(df)}개 캔들 로드 완료")
        return df

    def get_balance(self) -> Dict[str, float]:
        """보유 잔고를 {asset: free_amount} 형태로 반환한다.

        0보다 큰 잔고만 포함.
        """
        def _fetch():
            return self._client.fetch_balance()

        balance_raw = _retry(_fetch)
        result: Dict[str, float] = {}

        for asset, info in balance_raw.items():
            # ccxt는 asset별로 dict를 반환하고 'free' 키를 포함
            if isinstance(info, dict):
                free = info.get("free", 0.0)
                if free and float(free) > 0:
                    result[asset] = float(free)

        logger.debug(f"[Exchange] 잔고: {result}")
        return result

    def get_position(self, symbol: str) -> Dict:
        """현재 보유 포지션 정보를 반환한다.

        스팟 거래이므로 base currency 보유 수량으로 판단.
        예) symbol="BTC/USDT" → BTC 보유량 확인

        Returns:
            {
                "qty": float,          # 보유 수량 (0이면 포지션 없음)
                "avg_price": float,    # 평균 매수가 (추적 불가 시 0)
                "side": "long" | "none"
            }
        """
        balance = self.get_balance()

        # symbol에서 base 추출 (예: "BTC/USDT" → "BTC")
        base = symbol.split("/")[0]
        qty = balance.get(base, 0.0)

        return {
            "qty": qty,
            "avg_price": 0.0,   # 스팟은 평균가 직접 추적 필요 (positions.json에서 관리)
            "side": "long" if qty > 0 else "none",
        }

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
    ) -> Dict:
        """주문을 실행하고 주문 결과를 반환한다.

        Args:
            symbol: 예) "BTC/USDT"
            side: "buy" 또는 "sell"
            qty: 주문 수량 (base currency)
            order_type: "market" (현재 지원)

        Returns:
            ccxt order dict
        """
        if self.paper_trading:
            ticker = self.get_ticker(symbol)
            fake_order = {
                "id": f"paper_{int(time.time())}",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": qty,
                "price": ticker,
                "status": "closed",
                "filled": qty,
                "average": ticker,
                "timestamp": int(time.time() * 1000),
                "paper": True,
            }
            logger.info(
                f"[Exchange][페이퍼] {side.upper()} {qty:.6f} {symbol} @ {ticker:.4f}"
            )
            return fake_order

        def _place():
            return self._client.create_order(symbol, order_type, side, qty)

        order = _retry(_place)
        logger.info(
            f"[Exchange] {side.upper()} 주문 완료: {qty:.6f} {symbol} "
            f"(id={order.get('id')})"
        )
        return order

    def get_ticker(self, symbol: str) -> float:
        """현재가를 반환한다."""
        def _fetch():
            return self._client.fetch_ticker(symbol)

        ticker = _retry(_fetch)
        price = float(ticker["last"])
        logger.debug(f"[Exchange] {symbol} 현재가: {price}")
        return price

    def get_min_order_qty(self, symbol: str) -> float:
        """심볼의 최소 주문 수량을 반환한다."""
        try:
            markets = self._client.load_markets()
            market = markets.get(symbol, {})
            limits = market.get("limits", {})
            amount_min = limits.get("amount", {}).get("min", 0.0001)
            return float(amount_min) if amount_min else 0.0001
        except Exception as e:
            logger.warning(f"[Exchange] 최소 주문 수량 조회 실패 ({symbol}): {e}")
            return 0.0001
