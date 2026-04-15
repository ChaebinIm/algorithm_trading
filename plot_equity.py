"""
선택 전략의 자본곡선(equity curve) 비교 그래프.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from core import BacktestEnv, BasicSlippage, FixedRateCommission, PercentSizer, run_backtest
from strategies import get_strategy

# 백테스트 환경 (최신 설정과 동일)
ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=10),  # 바이낸스 기본 0.1%
    sizer=PercentSizer(percent=0.20),
)

# 비교할 전략들
STRATEGIES = ["ensemble", "champion_v2", "ensemble_short"]

# 데이터: 가장 결과가 좋았던 4h 기준, 여러 코인
DATA_FILES = {
    "BTC 4h": "data/ohlcv_full_BTCUSDT_4h.parquet",
    "ETH 4h": "data/ohlcv_full_ETHUSDT_4h.parquet",
}

# KRW 데이터가 있으면 추가
from pathlib import Path
if Path("data/ohlcv_full_BTCKRW_4h.parquet").exists():
    DATA_FILES["BTC KRW 4h"] = "data/ohlcv_full_BTCKRW_4h.parquet"
if Path("data/ohlcv_full_ETHKRW_4h.parquet").exists():
    DATA_FILES["ETH KRW 4h"] = "data/ohlcv_full_ETHKRW_4h.parquet"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df = df.set_index("time")
    return df[["open", "high", "low", "close", "volume"]]


def main():
    n_data = len(DATA_FILES)
    fig, axes = plt.subplots(n_data, 1, figsize=(16, 6 * n_data), sharex=False)
    if n_data == 1:
        axes = [axes]

    colors = {"ensemble": "#2196F3", "champion_v2": "#FF9800", "ensemble_short": "#4CAF50"}

    for ax, (data_label, data_path) in zip(axes, DATA_FILES.items()):
        df = load_data(data_path)

        for strat_name in STRATEGIES:
            strat = get_strategy(strat_name)
            equity, report = run_backtest(df, strat, ENV)
            m = report.summary()

            label = (
                f"{strat_name}  "
                f"(+{m['cum_return']:.0%}, "
                f"Sharpe {m['sharpe']:.2f}, "
                f"MDD {m['mdd']:.1%})"
            )
            ax.plot(equity.index, equity.values, label=label, color=colors.get(strat_name, None), linewidth=1.2)

        # 초기 자본선
        ax.axhline(y=10_000, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

        ax.set_title(f"Equity Curve - {data_label}", fontsize=14, fontweight="bold")
        ax.set_ylabel("USDT")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/equity_comparison.png", dpi=150, bbox_inches="tight")
    print("reports/equity_comparison.png 저장 완료")
    plt.show()


if __name__ == "__main__":
    main()
