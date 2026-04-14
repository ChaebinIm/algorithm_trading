"""
bot 패키지 — 라이브 트레이딩 봇.

주요 컴포넌트:
- BotConfig  : 전체 설정 (bot/config.py)
- BinanceExchange : 거래소 래퍼 (bot/exchange.py)
- RiskManager : 리스크 관리 (bot/risk_manager.py)
- Notifier   : 텔레그램 알림 (bot/notifier.py)
- Trader     : 메인 트레이딩 루프 (bot/trader.py)

빠른 시작:
    from bot.config import BotConfig
    from bot.trader import Trader

    config = BotConfig(api_key="...", api_secret="...", testnet=True)
    trader = Trader(config)
    trader.run_loop()
"""
from bot.config import BotConfig, PortfolioEntry
from bot.exchange import BinanceExchange
from bot.risk_manager import RiskManager
from bot.notifier import Notifier
from bot.trader import Trader

__all__ = [
    "BotConfig",
    "PortfolioEntry",
    "BinanceExchange",
    "RiskManager",
    "Notifier",
    "Trader",
]
