"""
실거래 봇 실행.

설정:
  1. bot/config.py에서 API 키, 종목, 리스크 설정
  2. 테스트넷 먼저 실행: python run_bot.py --testnet
  3. 실거래: python run_bot.py

옵션:
  --testnet   : 바이낸스 테스트넷 사용
  --once      : 1회만 실행 (스케줄 없음, 테스트용)
  --paper     : 페이퍼 트레이딩 (API 연결하되 실제 주문 안 냄)

예시:
  # 테스트넷 1회 실행
  python run_bot.py --testnet --once

  # 테스트넷 연속 실행
  python run_bot.py --testnet

  # 페이퍼 트레이딩 (실 API, 가짜 주문)
  python run_bot.py --paper

  # 실거래 (주의!)
  python run_bot.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# 프로젝트 루트를 sys.path에 추가 (스크립트 직접 실행 시 필요)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()  # .env 파일 자동 로드

from bot.config import BotConfig, PortfolioEntry
from bot.trader import Trader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance 알고리즘 트레이딩 봇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="바이낸스 테스트넷 사용 (기본값은 config.py의 testnet 설정)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="1회만 실행하고 종료 (스케줄 없음, 개발·테스트용)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="페이퍼 트레이딩 모드 (주문을 실제로 제출하지 않음)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Binance API 키 (설정 파일 값을 override)",
    )
    parser.add_argument(
        "--api-secret",
        type=str,
        default=None,
        help="Binance API 시크릿 (설정 파일 값을 override)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BotConfig:
    """CLI 인수를 적용한 BotConfig를 생성한다."""
    config = BotConfig()

    # CLI 플래그 적용
    if args.testnet:
        config.testnet = True

    if args.paper:
        config.paper_trading = True

    if args.api_key:
        config.api_key = args.api_key

    if args.api_secret:
        config.api_secret = args.api_secret

    # 환경변수로도 API 키 설정 가능 (더 안전한 방법)
    env_key = os.environ.get("BINANCE_API_KEY")
    env_secret = os.environ.get("BINANCE_API_SECRET")
    if env_key:
        config.api_key = env_key
    if env_secret:
        config.api_secret = env_secret

    # 환경변수로 텔레그램 설정
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID")
    if tg_token:
        config.telegram_token = tg_token
    if tg_chat:
        config.telegram_chat_id = tg_chat

    return config


def validate_config(config: BotConfig) -> None:
    """설정 유효성을 기본 검증한다."""
    if config.api_key == "YOUR_API_KEY_HERE" and not config.paper_trading:
        print(
            "[경고] API 키가 설정되지 않았습니다.\n"
            "  bot/config.py에서 api_key, api_secret을 설정하거나\n"
            "  환경변수 BINANCE_API_KEY, BINANCE_API_SECRET을 설정하세요.\n"
            "  페이퍼 트레이딩으로 실행하려면 --paper 플래그를 사용하세요."
        )
        sys.exit(1)

    total_allocation = sum(e.allocation_pct for e in config.portfolio)
    if total_allocation > 1.0:
        print(
            f"[오류] 포트폴리오 비중 합계가 100%를 초과합니다: {total_allocation:.0%}\n"
            "  bot/config.py의 allocation_pct 합계가 1.0 이하여야 합니다."
        )
        sys.exit(1)


def print_startup_info(config: BotConfig, run_once: bool) -> None:
    """봇 시작 정보를 출력한다."""
    mode_parts = []
    if config.testnet:
        mode_parts.append("테스트넷")
    else:
        mode_parts.append("실거래")
    if config.paper_trading:
        mode_parts.append("페이퍼 트레이딩")
    mode_str = " + ".join(mode_parts)

    print("=" * 60)
    print(f"  Binance 알고리즘 트레이딩 봇")
    print(f"  모드: {mode_str}")
    print(f"  실행 방식: {'1회' if run_once else '연속 루프'}")
    print(f"  포트폴리오:")
    for e in config.portfolio:
        print(f"    - {e.symbol:12s} | {e.strategy_name:20s} | {e.timeframe} | {e.allocation_pct:.0%}")
    print(f"  리스크:")
    print(f"    - 일일 손실 한도: {config.max_daily_loss_pct:.0%}")
    print(f"    - 최대 낙폭: {config.max_drawdown_pct:.0%}")
    print(f"    - 포지션 최대 비중: {config.max_position_pct:.0%}")
    print(f"  텔레그램: {'활성화' if config.telegram_token else '비활성화'}")
    print("=" * 60)

    if not config.testnet and not config.paper_trading:
        print("\n[주의] 실거래 모드입니다. 실제 자금이 사용됩니다.")
        print("계속하려면 Enter를 누르고, 취소하려면 Ctrl+C를 누르세요.")
        try:
            input()
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            sys.exit(0)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    print_startup_info(config, args.once)

    # 봇 실행
    trader = Trader(config)

    if args.once:
        logging.info("[main] 1회 실행 모드")
        trader.run_once()
    else:
        logging.info("[main] 연속 루프 모드 시작")
        trader.run_loop()


if __name__ == "__main__":
    main()
