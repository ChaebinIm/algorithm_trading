"""
core 패키지 — 백테스팅 프레임워크의 핵심 모듈들.

주요 클래스/함수를 re-export하여 외부에서 간편하게 임포트 가능.
    from core import BacktestEnv, run_backtest, ...
"""
from core.types import Fill, TradeLog, BacktestReport
from core.models import (
    SlippageModel, BasicSlippage,
    CommissionModel, FixedRateCommission,
    Sizer, FixedUSDTSizer, PercentSizer, FuturesSizer,
)
from core.engine import BacktestEnv, run_backtest, compute_drawdown, perf_summary
from core.indicators import (
    sma, ema, rsi, atr, adx,
    bollinger_bands, donchian_channel, momentum, reg_channel,
)
