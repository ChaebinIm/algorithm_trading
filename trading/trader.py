"""
바이낸스 라이브 트레이더.

- .env에서 API 키를 로딩하여 바이낸스 연결
- 잔고 조회, 시장가 주문 실행, 포지션 관리
- 전략 신호에 따라 자동 매매 루프 실행
- 리스크 관리: 최대 포지션 제한, 일일 손실 제한

사용 예시
--------
# 드라이런 모드 (실제 주문 없이 시뮬레이션)
python run_backtest.py  ← 백테스트 먼저 돌려보기
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

from strategies import Strategy
from strategies.trend_following import TrendFollowingStrategy
from trading.risk import RiskConfig

load_dotenv()


class LiveTrader:
    """바이낸스 라이브 트레이더.

    주요 기능:
    1. 바이낸스 API 연결 및 잔고 조회
    2. 전략 신호에 따른 시장가 주문 실행
    3. 리스크 한도 체크 (포지션 크기, 일일 손실)
    4. 드라이런 모드 지원 (실제 주문 없이 로그만 출력)
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        strategy: Optional[Strategy] = None,
        risk: Optional[RiskConfig] = None,
        dry_run: bool = True,
        lookback: int = 300,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy or TrendFollowingStrategy()
        self.risk = risk or RiskConfig()
        self.dry_run = dry_run
        self.lookback = lookback

        # 일일 손익 추적
        self._daily_pnl = 0.0
        self._daily_reset_date = datetime.now(timezone.utc).date()

        # 바이낸스 인스턴스 생성
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            raise ValueError(
                "API 키가 설정되지 않았습니다. "
                ".env 파일에 BINANCE_API_KEY와 BINANCE_API_SECRET을 입력해주세요."
            )

        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.exchange.load_markets()

    # =============================
    # 잔고 및 포지션 조회
    # =============================

    def get_usdt_balance(self) -> float:
        """USDT 가용 잔고 조회."""
        balance = self.exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0))

    def get_position(self) -> float:
        """현재 보유 중인 베이스 자산 수량 조회."""
        base = self.symbol.split("/")[0]
        balance = self.exchange.fetch_balance()
        return float(balance.get(base, {}).get("free", 0))

    def get_position_value(self, current_price: float) -> float:
        """현재 포지션의 USDT 환산 가치."""
        return self.get_position() * current_price

    # =============================
    # 시장 데이터 조회
    # =============================

    def fetch_recent_candles(self) -> pd.DataFrame:
        """최근 캔들 데이터 조회 (전략 지표 계산용)."""
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol, timeframe=self.timeframe, limit=self.lookback,
        )
        df = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("time")[["open", "high", "low", "close", "volume"]]
        return df

    # =============================
    # 주문 실행
    # =============================

    def place_market_order(self, side: str, usdt_amount: float) -> Optional[dict]:
        """시장가 주문 실행.

        Parameters
        ----------
        side : str
            "buy" 또는 "sell"
        usdt_amount : float
            매매 금액 (USDT). 매도 시 전량 매도.
        """
        ticker = self.exchange.fetch_ticker(self.symbol)
        current_price = ticker["last"]

        if side == "buy":
            qty = usdt_amount / current_price
        else:
            qty = self.get_position()
            if qty <= 0:
                print(f"  [매도 스킵] 보유 수량 없음")
                return None

        # 최소 주문 수량 체크
        market = self.exchange.market(self.symbol)
        min_qty = market.get("limits", {}).get("amount", {}).get("min", 0)
        if qty < min_qty:
            print(f"  [주문 스킵] 수량({qty:.6f})이 최소 주문량({min_qty})보다 작음")
            return None

        if self.dry_run:
            print(f"  [드라이런] {side.upper()} {qty:.6f} {self.symbol} "
                  f"@ ~{current_price:,.2f} USDT (≈{qty * current_price:,.2f} USDT)")
            return None

        # 실제 주문
        try:
            order = self.exchange.create_market_order(
                symbol=self.symbol, side=side, amount=qty,
            )
            filled_price = order.get("average", current_price)
            filled_qty = order.get("filled", qty)
            fee_cost = sum(f.get("cost", 0) for f in order.get("fees", []))
            print(f"  [체결] {side.upper()} {filled_qty:.6f} {self.symbol} "
                  f"@ {filled_price:,.2f} USDT | 수수료: {fee_cost:.4f} USDT")
            return order
        except Exception as e:
            print(f"  [주문 실패] {e}")
            return None

    # =============================
    # 리스크 관리
    # =============================

    def _reset_daily_pnl_if_needed(self):
        """날짜가 바뀌면 일일 손익을 리셋."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            print(f"  [일일 리셋] 전일 손익: {self._daily_pnl:+.2f} USDT")
            self._daily_pnl = 0.0
            self._daily_reset_date = today

    def check_risk(self, side: str, current_price: float) -> bool:
        """리스크 한도 체크. 통과하면 True, 거부하면 False."""
        self._reset_daily_pnl_if_needed()

        # 일일 손실 한도
        if self._daily_pnl < -self.risk.max_daily_loss_usdt:
            print(f"  [리스크] 일일 손실 한도 초과 "
                  f"({self._daily_pnl:+.2f} / -{self.risk.max_daily_loss_usdt} USDT)")
            return False

        if side == "buy":
            # 최대 포지션 크기
            current_pos_value = self.get_position_value(current_price)
            if current_pos_value + self.risk.usdt_per_trade > self.risk.max_position_usdt:
                print(f"  [리스크] 최대 포지션 한도 초과 "
                      f"(현재: {current_pos_value:,.2f} / 한도: {self.risk.max_position_usdt:,.2f} USDT)")
                return False

            # 가용 잔고
            available = self.get_usdt_balance()
            if available < self.risk.usdt_per_trade:
                print(f"  [리스크] 잔고 부족 "
                      f"(가용: {available:,.2f} / 필요: {self.risk.usdt_per_trade:,.2f} USDT)")
                return False

        return True

    # =============================
    # 메인 트레이딩 루프
    # =============================

    def run_once(self) -> int:
        """1회 사이클: 데이터 조회 → 전략 신호 → 리스크 체크 → 주문."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n[{now}] {self.symbol} / {self.timeframe}")

        # 최근 캔들 데이터 조회
        df = self.fetch_recent_candles()
        print(f"  캔들 {len(df)}개 로드 ({df.index[0]} ~ {df.index[-1]})")

        # 전략 지표 계산 및 신호 생성
        df = self.strategy.prepare(df)
        df = df.dropna()

        if len(df) < 2:
            print("  [스킵] 데이터 부족")
            return 0

        # 마지막에서 두 번째 바 사용 (마지막 바는 아직 미완성일 수 있음)
        last_bar = df.iloc[-2]
        signal = self.strategy.on_bar(len(df) - 2, last_bar)

        current_price = float(last_bar["close"])
        pos_qty = self.get_position()
        pos_value = pos_qty * current_price

        print(f"  현재가: {current_price:,.2f} USDT | "
              f"포지션: {pos_qty:.6f} (≈{pos_value:,.2f} USDT) | "
              f"신호: {signal:+d}")

        if signal == +1:
            if self.check_risk("buy", current_price):
                self.place_market_order("buy", self.risk.usdt_per_trade)
        elif signal == -1:
            if pos_qty > 0:
                self.place_market_order("sell", 0)
            else:
                print("  [매도 스킵] 보유 포지션 없음")
        else:
            print("  → 관망 (신호 없음)")

        return signal

    def run_loop(self, interval_seconds: Optional[int] = None):
        """자동 매매 루프 실행."""
        if interval_seconds is None:
            tf_intervals = {
                "1m": 60, "5m": 300, "15m": 900,
                "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
            }
            interval_seconds = tf_intervals.get(self.timeframe, 3600)

        mode = "드라이런" if self.dry_run else "실매매"
        print("=" * 60)
        print(f"라이브 트레이딩 시작 [{mode} 모드]")
        print(f"  심볼: {self.symbol}")
        print(f"  타임프레임: {self.timeframe}")
        print(f"  루프 간격: {interval_seconds}초")
        print(f"  1회 매매: {self.risk.usdt_per_trade} USDT")
        print(f"  최대 포지션: {self.risk.max_position_usdt} USDT")
        print(f"  일일 손실 한도: {self.risk.max_daily_loss_usdt} USDT")
        print("=" * 60)

        try:
            usdt_bal = self.get_usdt_balance()
            print(f"  USDT 가용 잔고: {usdt_bal:,.2f}")
        except Exception as e:
            print(f"  잔고 조회 실패: {e}")

        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                print("\n\n사용자에 의해 중단됨.")
                break
            except Exception as e:
                print(f"  [에러] {e}")

            print(f"  → {interval_seconds}초 후 다음 사이클...")
            try:
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\n\n사용자에 의해 중단됨.")
                break
