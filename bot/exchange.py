"""
bot/exchange.py — Binance 거래소 래퍼 (ccxt 기반).

기능:
- OHLCV 데이터 조회 (현물 / 선물)
- 잔고 / 포지션 조회
- 시장가 주문 실행 (롱 / 숏)
- 레버리지 / 포지션 모드 설정
- 펀딩비 조회
- 테스트넷 지원
- 자동 재시도 (최대 3회, 지수 백오프)
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import ccxt
import pandas as pd

from bot.config import BotConfig

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # 초 (1 → 2 → 4)


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
            logger.error(f"[Exchange] 거래소 에러: {e}")
            raise
    raise last_exc


class BinanceExchange:
    """Binance ccxt 래퍼 클래스.

    market_type="spot"    → 현물 거래 (롱 온리)
    market_type="future"  → USDⓈ-M 영구선물 (롱/숏, 레버리지)
    paper_trading=True    → 주문 함수가 실제로 요청을 보내지 않음
    testnet=True          → 바이낸스 테스트넷 연결
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.paper_trading = config.paper_trading
        self.market_type = getattr(config, "market_type", "spot")

        # 공개 클라이언트 (OHLCV 등, API 키 불필요)
        self._public = self._build_public_client()
        # 인증 클라이언트 (주문/잔고/포지션)
        self._client = self._build_client()

        # 선물 모드일 때 초기 설정
        if self._client and self.market_type == "future":
            self._init_futures()

    # ── 클라이언트 생성 ────────────────────────────────────────────────

    def _build_public_client(self) -> ccxt.binance:
        """공개 API 전용 클라이언트."""
        params: dict = {
            "options": {"defaultType": self.market_type},
            "enableRateLimit": True,
        }
        if self.market_type == "future":
            params["urls"] = {
                "api": {"rest": "https://fapi.binance.com"}
            }
        client = ccxt.binance(params)
        client.load_markets()
        return client

    def _build_client(self) -> Optional[ccxt.binance]:
        """인증 클라이언트 (주문/잔고용). API 키 없으면 None."""
        if not self.config.api_key or self.config.api_key == "YOUR_API_KEY_HERE":
            return None

        options: dict = {
            "defaultType": self.market_type,
            "adjustForTimeDifference": True,
        }

        params: dict = {
            "apiKey": self.config.api_key,
            "secret": self.config.api_secret,
            "options": options,
            "enableRateLimit": True,
        }

        if self.market_type == "future":
            params["urls"] = {
                "api": {"rest": "https://fapi.binance.com"}
            }
            logger.info("[Exchange] 선물(USDⓈ-M) 모드로 초기화")
        else:
            logger.info("[Exchange] 현물(Spot) 모드로 초기화")

        if self.config.testnet:
            if self.market_type == "future":
                params["urls"] = {
                    "api": {"rest": "https://testnet.binancefuture.com"}
                }
            else:
                params["urls"] = {
                    "api": {"rest": "https://testnet.binance.vision/api"}
                }
            logger.info("[Exchange] 테스트넷 모드")

        return ccxt.binance(params)

    def _init_futures(self) -> None:
        """선물 계좌 초기 설정 — 포지션 모드, 레버리지."""
        try:
            # 단방향 모드 설정 (One-way: 롱 or 숏 하나씩)
            self._client.set_position_mode(hedged=False)
            logger.info("[Exchange] 포지션 모드: 단방향(One-way)")
        except ccxt.ExchangeError as e:
            # 이미 단방향 모드면 에러 발생 — 무시
            if "No need to change position side" in str(e):
                logger.debug("[Exchange] 이미 단방향 모드")
            else:
                logger.warning(f"[Exchange] 포지션 모드 설정 실패: {e}")

    # ── OHLCV ─────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """OHLCV 캔들 데이터를 가져온다.

        Returns:
            columns=[open, high, low, close, volume], index=DatetimeIndex(UTC)
        """
        def _fetch():
            return self._public.fetch_ohlcv(symbol, timeframe, limit=limit)

        raw = _retry(_fetch)

        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").astype(float).sort_index()
        df = df.iloc[:-1]  # 마지막 미완성 캔들 제거

        logger.debug(f"[Exchange] {symbol} {timeframe} {len(df)}개 캔들 로드")
        return df

    # ── 잔고 / 포지션 ──────────────────────────────────────────────────

    def get_balance(self) -> Dict[str, float]:
        """보유 잔고를 {asset: free_amount} 형태로 반환.

        선물 모드: USDT 증거금 잔고 반환
        현물 모드: 각 자산 잔고 반환
        """
        def _fetch():
            return self._client.fetch_balance()

        balance_raw = _retry(_fetch)
        result: Dict[str, float] = {}

        if self.market_type == "future":
            # 선물은 USDT 증거금만 관리
            usdt = balance_raw.get("USDT", {})
            if isinstance(usdt, dict):
                free = float(usdt.get("free", 0.0) or 0.0)
                total = float(usdt.get("total", 0.0) or 0.0)
                result["USDT"] = free
                result["USDT_total"] = total
        else:
            for asset, info in balance_raw.items():
                if isinstance(info, dict):
                    free = info.get("free", 0.0)
                    if free and float(free) > 0:
                        result[asset] = float(free)

        logger.debug(f"[Exchange] 잔고: {result}")
        return result

    def get_position(self, symbol: str) -> Dict:
        """현재 포지션 정보를 반환.

        선물 모드:
            {
                "qty": float,        # 보유 수량 (양수=롱, 음수=숏, 0=없음)
                "side": "long" | "short" | "none",
                "avg_price": float,  # 평균 진입가
                "unrealized_pnl": float,
                "leverage": int,
            }
        현물 모드:
            {
                "qty": float,
                "side": "long" | "none",
                "avg_price": float,
                "unrealized_pnl": 0.0,
                "leverage": 1,
            }
        """
        if self.market_type == "future":
            return self._get_futures_position(symbol)
        else:
            return self._get_spot_position(symbol)

    def _get_futures_position(self, symbol: str) -> Dict:
        def _fetch():
            return self._client.fetch_positions([symbol])

        positions = _retry(_fetch)

        for pos in positions:
            if pos.get("symbol") == symbol.replace("/", ""):
                qty = float(pos.get("contracts", 0) or 0)
                side_raw = pos.get("side", "")
                if qty == 0 or side_raw == "":
                    break
                return {
                    "qty": qty if side_raw == "long" else -qty,
                    "side": side_raw,
                    "avg_price": float(pos.get("entryPrice", 0) or 0),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0) or 0),
                    "leverage": int(pos.get("leverage", 1) or 1),
                }

        return {"qty": 0.0, "side": "none", "avg_price": 0.0,
                "unrealized_pnl": 0.0, "leverage": 1}

    def _get_spot_position(self, symbol: str) -> Dict:
        balance = self.get_balance()
        base = symbol.split("/")[0]
        qty = balance.get(base, 0.0)
        return {
            "qty": qty,
            "side": "long" if qty > 0 else "none",
            "avg_price": 0.0,
            "unrealized_pnl": 0.0,
            "leverage": 1,
        }

    # ── 레버리지 설정 ──────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> None:
        """선물 레버리지를 설정한다."""
        if self.market_type != "future":
            logger.warning("[Exchange] 현물 모드에서는 레버리지 설정 불가")
            return
        try:
            self._client.set_leverage(leverage, symbol)
            logger.info(f"[Exchange] {symbol} 레버리지 {leverage}x 설정")
        except ccxt.ExchangeError as e:
            logger.warning(f"[Exchange] 레버리지 설정 실패 ({symbol}): {e}")

    # ── 주문 실행 ──────────────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        reduce_only: bool = False,
    ) -> Dict:
        """주문을 실행하고 주문 결과를 반환.

        Args:
            symbol: 예) "BTC/USDT"
            side: "buy" (롱 진입 or 숏 청산) / "sell" (숏 진입 or 롱 청산)
            qty: 주문 수량 (계약 수 또는 base currency)
            order_type: "market"
            reduce_only: True = 포지션 축소만 허용 (청산 시 사용)
        """
        if self.paper_trading:
            ticker = self.get_ticker(symbol)
            action = "숏진입" if side == "sell" and not reduce_only else \
                     "롱진입" if side == "buy" and not reduce_only else \
                     "청산"
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
                "reduceOnly": reduce_only,
            }
            logger.info(
                f"[Exchange][페이퍼] {action} {qty:.6f} {symbol} @ {ticker:.4f}"
            )
            return fake_order

        params = {}
        if self.market_type == "future" and reduce_only:
            params["reduceOnly"] = True

        def _place():
            return self._client.create_order(
                symbol, order_type, side, qty, params=params
            )

        order = _retry(_place)
        logger.info(
            f"[Exchange] {side.upper()} 주문 완료: {qty:.6f} {symbol} "
            f"(id={order.get('id')}, reduceOnly={reduce_only})"
        )
        return order

    def close_position(self, symbol: str) -> Optional[Dict]:
        """현재 포지션 전체를 시장가로 청산한다."""
        pos = self.get_position(symbol)
        if pos["side"] == "none" or pos["qty"] == 0:
            logger.info(f"[Exchange] {symbol} 포지션 없음 — 청산 스킵")
            return None

        qty = abs(pos["qty"])
        # 롱이면 sell로 청산, 숏이면 buy로 청산
        close_side = "sell" if pos["side"] == "long" else "buy"

        logger.info(f"[Exchange] {symbol} {pos['side']} 포지션 청산 ({qty:.6f})")
        return self.place_order(symbol, close_side, qty, reduce_only=True)

    # ── 시세 / 유틸 ────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> float:
        """현재가를 반환한다."""
        def _fetch():
            client = self._client or self._public
            return client.fetch_ticker(symbol)

        ticker = _retry(_fetch)
        price = float(ticker["last"])
        logger.debug(f"[Exchange] {symbol} 현재가: {price}")
        return price

    def get_funding_rate(self, symbol: str) -> float:
        """현재 펀딩비를 반환한다 (선물 전용). 연환산 기준."""
        if self.market_type != "future":
            return 0.0
        try:
            def _fetch():
                return self._public.fetch_funding_rate(symbol)
            fr = _retry(_fetch)
            rate = float(fr.get("fundingRate", 0) or 0)
            # 8시간마다 → 연환산: × 3(하루) × 365
            annualized = rate * 3 * 365
            logger.debug(f"[Exchange] {symbol} 펀딩비: {rate:.4%} (연환산 {annualized:.1%})")
            return rate
        except Exception as e:
            logger.warning(f"[Exchange] 펀딩비 조회 실패 ({symbol}): {e}")
            return 0.0

    def get_min_order_qty(self, symbol: str) -> float:
        """심볼의 최소 주문 수량을 반환한다."""
        try:
            markets = self._public.load_markets()
            market = markets.get(symbol, {})
            limits = market.get("limits", {})
            amount_min = limits.get("amount", {}).get("min", 0.001)
            return float(amount_min) if amount_min else 0.001
        except Exception as e:
            logger.warning(f"[Exchange] 최소 주문 수량 조회 실패 ({symbol}): {e}")
            return 0.001
