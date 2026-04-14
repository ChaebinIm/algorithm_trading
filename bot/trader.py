"""
bot/trader.py — 메인 트레이딩 루프.

흐름:
  run_loop() → (매 캔들 종료 시) run_once()
    → 각 (symbol, strategy, timeframe) 조합에 대해:
       1. OHLCV 조회
       2. 전략 지표 계산 (prepare)
       3. 전략 상태 복원 (on_bar 순차 실행으로 워밍업)
       4. 마지막 바 신호 확인
       5. 리스크 체크
       6. 주문 실행
       7. positions.json 업데이트
    → 거래 발생 시 텔레그램 알림
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

from bot.config import BotConfig
from bot.exchange import BinanceExchange
from bot.notifier import Notifier
from bot.risk_manager import RiskManager
from strategies import get_strategy, Strategy

logger = logging.getLogger(__name__)

# 타임프레임 → 초 변환 테이블
TIMEFRAME_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}

# 캔들 마감 후 대기 버퍼 (초) — 마감 직후 데이터가 확정될 때까지
CANDLE_CLOSE_BUFFER = 30


class Trader:
    """메인 트레이더 클래스."""

    def __init__(self, config: BotConfig):
        self.config = config
        self._setup_logging()

        # 거래소 / 알림
        self.exchange = BinanceExchange(config)
        self.notifier = Notifier(
            token=config.telegram_token,
            chat_id=config.telegram_chat_id,
        )

        # 포지션 파일 경로
        self._positions_file = os.path.join(config.log_dir, "positions.json")
        self._positions: Dict[str, dict] = self._load_positions()

        # 리스크 매니저 (초기 자본은 첫 실행 시 거래소에서 조회)
        self.risk = RiskManager(config, initial_equity=0.0)

        # 전략 인스턴스 캐시 (심볼별)
        self._strategies: Dict[str, Strategy] = {}

        logger.info("[Trader] 초기화 완료")

    # ── 로깅 설정 ──────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        """로그 디렉토리 생성 및 로거 핸들러 설정."""
        os.makedirs(self.config.log_dir, exist_ok=True)

        root_logger = logging.getLogger()
        if root_logger.handlers:
            return  # 이미 설정됨

        root_logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 콘솔 핸들러
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

        # 파일 핸들러 (날짜별)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_path = os.path.join(self.config.log_dir, f"bot_{today}.log")
        try:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(fmt)
            root_logger.addHandler(fh)
        except Exception as e:
            print(f"[경고] 로그 파일 생성 실패: {e}")

    # ── 포지션 영속화 ──────────────────────────────────────────────────

    def _load_positions(self) -> Dict[str, dict]:
        """positions.json을 읽어 포지션 상태를 복원한다."""
        if not os.path.exists(self._positions_file):
            return {}
        try:
            with open(self._positions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"[Trader] 포지션 복원: {list(data.keys())}")
            return data
        except Exception as e:
            logger.warning(f"[Trader] positions.json 읽기 실패: {e}")
            return {}

    def _save_positions(self) -> None:
        """현재 포지션 상태를 positions.json에 저장한다."""
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
            with open(self._positions_file, "w", encoding="utf-8") as f:
                json.dump(self._positions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[Trader] positions.json 저장 실패: {e}")

    # ── 전략 워밍업 ────────────────────────────────────────────────────

    def _get_strategy(self, strategy_name: str, symbol: str) -> Strategy:
        """전략 인스턴스를 가져온다. 캐시에 없으면 새로 생성."""
        key = f"{symbol}_{strategy_name}"
        if key not in self._strategies:
            self._strategies[key] = get_strategy(strategy_name)
            logger.debug(f"[Trader] 전략 생성: {strategy_name} ({symbol})")
        return self._strategies[key]

    def _warmup_strategy(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
    ) -> int:
        """전략의 내부 상태(포지션 여부 등)를 복원하기 위해 전체 바를 순차 실행한다.

        실제 거래는 실행하지 않고, 마지막 바의 신호만 반환한다.
        Returns:
            마지막 바의 신호 (+1, -1, 0)
        """
        signal = 0
        for i in range(len(df)):
            row = df.iloc[i]
            signal = strategy.on_bar(i, row)
        return signal

    # ── 자본 조회 ──────────────────────────────────────────────────────

    def _get_total_equity(self) -> float:
        """USDT 기준 총 자산가치를 계산한다 (현금 + 보유코인 평가액)."""
        try:
            balance = self.exchange.get_balance()
            total = balance.get("USDT", 0.0)

            # 포지션 파일의 코인도 현재가로 평가
            for sym, pos_info in self._positions.items():
                qty = pos_info.get("qty", 0.0)
                if qty > 0:
                    try:
                        price = self.exchange.get_ticker(sym)
                        total += qty * price
                    except Exception:
                        # 가격 조회 실패 시 진입가로 대체
                        total += qty * pos_info.get("entry_price", 0.0)

            return total
        except Exception as e:
            logger.warning(f"[Trader] 총 자산 조회 실패: {e}")
            return self.risk._current_equity  # 이전 값 유지

    # ── 수량 계산 ──────────────────────────────────────────────────────

    def _calc_buy_qty(
        self,
        symbol: str,
        allocation_pct: float,
        price: float,
    ) -> float:
        """매수 수량을 계산한다.

        총 자산 × allocation_pct = 이 종목에 쓸 금액
        수량 = 금액 / 현재가
        """
        equity = self.risk._current_equity
        if equity <= 0:
            equity = self._get_total_equity()

        usdt_to_use = equity * allocation_pct
        qty = usdt_to_use / price

        # 최소 주문 수량 확인
        min_qty = self.exchange.get_min_order_qty(symbol)
        if qty < min_qty:
            logger.warning(
                f"[Trader] {symbol} 계산된 수량({qty:.6f}) < 최소 수량({min_qty}). 주문 건너뜀."
            )
            return 0.0

        return qty

    # ── 단일 종목 처리 ─────────────────────────────────────────────────

    def _process_symbol(
        self,
        symbol: str,
        strategy_name: str,
        timeframe: str,
        allocation_pct: float,
    ) -> Optional[dict]:
        """단일 종목에 대한 신호 확인 → 주문 실행.

        Returns:
            거래가 발생했으면 trade info dict, 아니면 None
        """
        logger.info(f"[Trader] 처리 중: {symbol} / {strategy_name} / {timeframe}")

        # 1. OHLCV 조회
        df = self.exchange.fetch_ohlcv(
            symbol, timeframe, limit=self.config.data_lookback + 1
        )
        if len(df) < 10:
            logger.warning(f"[Trader] {symbol} 데이터 부족 ({len(df)}개). 건너뜀.")
            return None

        # 2. 전략 가져오기 + 지표 계산
        strategy = self._get_strategy(strategy_name, symbol)
        df = strategy.prepare(df)

        # 3. 전략 상태 워밍업 (전체 바 순차 실행)
        #    마지막 바 신호 = 현재 시그널
        signal = self._warmup_strategy(strategy, df)

        last_row = df.iloc[-1]
        current_price = float(last_row["close"])

        logger.info(
            f"[Trader] {symbol} 신호: {signal:+d} | "
            f"현재가: {current_price:.4f} USDT"
        )

        # 4. 현재 포지션 상태
        pos_info = self._positions.get(symbol, {})
        has_position = pos_info.get("qty", 0.0) > 0

        trade_info = None

        # 5. 신호 처리
        if signal == 1 and not has_position:
            # ── 매수 ──
            qty = self._calc_buy_qty(symbol, allocation_pct, current_price)
            if qty <= 0:
                return None

            allowed, reason = self.risk.check_order(symbol, "buy", qty, current_price)
            if not allowed:
                logger.warning(f"[Trader] {symbol} 매수 거부: {reason}")
                self.notifier.send(f"매수 거부 ({symbol}): {reason}")
                return None

            # 주문 실행
            order = self.exchange.place_order(symbol, "buy", qty)
            filled_price = float(order.get("average") or order.get("price") or current_price)
            filled_qty = float(order.get("filled") or qty)

            # 리스크 기록
            self.risk.record_trade(symbol, "buy", filled_qty, filled_price)

            # 포지션 저장
            self._positions[symbol] = {
                "qty": filled_qty,
                "entry_price": filled_price,
                "strategy": strategy_name,
                "timeframe": timeframe,
                "entry_time": datetime.now(timezone.utc).isoformat(),
            }
            self._save_positions()

            # 알림
            self.notifier.notify_entry(symbol, filled_qty, filled_price, strategy_name)

            trade_info = {
                "action": "buy",
                "symbol": symbol,
                "qty": filled_qty,
                "price": filled_price,
            }

        elif signal == -1 and has_position:
            # ── 매도 ──
            sell_qty = pos_info["qty"]
            entry_price = pos_info.get("entry_price", current_price)

            allowed, reason = self.risk.check_order(symbol, "sell", sell_qty, current_price)
            if not allowed:
                logger.warning(f"[Trader] {symbol} 매도 거부: {reason}")
                return None

            # 주문 실행
            order = self.exchange.place_order(symbol, "sell", sell_qty)
            filled_price = float(order.get("average") or order.get("price") or current_price)
            filled_qty = float(order.get("filled") or sell_qty)

            # P&L 계산
            pnl_pct = ((filled_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0

            # 리스크 기록
            self.risk.record_trade(symbol, "sell", filled_qty, filled_price)

            # 포지션 초기화
            self._positions[symbol] = {"qty": 0.0, "entry_price": 0.0}
            self._save_positions()

            # 알림
            self.notifier.notify_exit(
                symbol, filled_qty, entry_price, filled_price, pnl_pct
            )

            trade_info = {
                "action": "sell",
                "symbol": symbol,
                "qty": filled_qty,
                "price": filled_price,
                "pnl_pct": pnl_pct,
            }

        else:
            logger.info(
                f"[Trader] {symbol} 홀드 | "
                f"신호={signal:+d}, 포지션={'있음' if has_position else '없음'}"
            )

        return trade_info

    # ── 메인 실행 ──────────────────────────────────────────────────────

    def run_once(self) -> None:
        """포트폴리오 전체에 대해 한 번 실행한다."""
        logger.info("=" * 60)
        logger.info(f"[Trader] run_once 시작: {datetime.now(timezone.utc).isoformat()}")

        # 총 자산 업데이트
        equity = self._get_total_equity()
        self.risk.set_equity(equity)
        logger.info(f"[Trader] 총 자산: {equity:.2f} USDT")

        trades = []
        for entry in self.config.portfolio:
            try:
                trade = self._process_symbol(
                    entry.symbol,
                    entry.strategy_name,
                    entry.timeframe,
                    entry.allocation_pct,
                )
                if trade:
                    trades.append(trade)
            except Exception as e:
                err_msg = f"{entry.symbol} 처리 중 오류: {e}\n{traceback.format_exc()}"
                logger.error(f"[Trader] {err_msg}")
                self.notifier.notify_error(err_msg[:400])

        # 거래 발생 시 리스크 통계 출력
        if trades:
            stats = self.risk.get_stats()
            logger.info(
                f"[Trader] 거래 완료 {len(trades)}건 | "
                f"일일P&L={stats['daily_pnl']:+.2f} USDT ({stats['daily_pnl_pct']:+.2f}%) | "
                f"DD={stats['drawdown_pct']:.2f}%"
            )

        logger.info("[Trader] run_once 완료")

    def run_loop(self, interval_seconds: Optional[int] = None) -> None:
        """스케줄에 따라 run_once()를 반복 실행한다.

        interval_seconds가 None이면 포트폴리오의 첫 항목 타임프레임 기준으로 자동 계산.
        다음 캔들 마감 시점 + CANDLE_CLOSE_BUFFER 초까지 대기 후 실행.
        예외 발생 시에도 절대 종료하지 않는다.
        """
        # 인터벌 자동 감지
        if interval_seconds is None:
            tf = self.config.portfolio[0].timeframe if self.config.portfolio else "4h"
            interval_seconds = TIMEFRAME_SECONDS.get(tf, 14400)
            logger.info(
                f"[Trader] 타임프레임 '{tf}' → 인터벌 {interval_seconds}초 자동 설정"
            )

        logger.info(
            f"[Trader] 루프 시작 | 인터벌: {interval_seconds}초 | "
            f"버퍼: {CANDLE_CLOSE_BUFFER}초"
        )
        self.notifier.send(
            f"트레이딩 봇 시작\n"
            f"종목: {[e.symbol for e in self.config.portfolio]}\n"
            f"인터벌: {interval_seconds // 3600}시간"
        )

        daily_summary_sent_date: Optional[str] = None

        while True:
            try:
                run_start = time.time()
                self.run_once()
                run_elapsed = time.time() - run_start

                # 일일 결산 알림 (UTC 자정 이후 첫 run)
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if daily_summary_sent_date != today_str:
                    # 자정이 지난 첫 번째 실행
                    if daily_summary_sent_date is not None:
                        stats = self.risk.get_stats()
                        stats["date"] = daily_summary_sent_date  # 어제 날짜
                        self.notifier.notify_daily_summary(stats)
                    daily_summary_sent_date = today_str

                # 다음 캔들 마감까지 대기
                now_ts = time.time()
                # 현재 캔들 시작 시각
                candle_start = (now_ts // interval_seconds) * interval_seconds
                next_candle_close = candle_start + interval_seconds
                sleep_time = max(
                    next_candle_close - now_ts + CANDLE_CLOSE_BUFFER, 10
                )

                next_run_dt = datetime.fromtimestamp(
                    next_candle_close + CANDLE_CLOSE_BUFFER, tz=timezone.utc
                )
                logger.info(
                    f"[Trader] 다음 실행: {next_run_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                    f"({sleep_time:.0f}초 대기)"
                )
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("[Trader] 사용자 중단 (Ctrl+C)")
                self.notifier.send("트레이딩 봇 수동 종료")
                break
            except Exception as e:
                err_msg = f"루프 오류: {e}\n{traceback.format_exc()}"
                logger.error(f"[Trader] {err_msg}")
                self.notifier.notify_error(err_msg[:400])
                # 오류 발생해도 계속 — 60초 후 재시도
                logger.info("[Trader] 60초 후 재시도...")
                time.sleep(60)
