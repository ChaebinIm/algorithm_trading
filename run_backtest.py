"""
백테스트 실행 스크립트 (CLI 진입점).

사용 예시
--------
# 추세 추종 전략으로 BTC 1시간봉 백테스트
python run_backtest.py --strategy trend_following --data data/ohlcv_full_BTCUSDT_1h.parquet

# MA 교차 전략
python run_backtest.py --strategy ma_cross --data data/ohlcv_full_ETHUSDT_4h.parquet

# 사용 가능한 전략 목록 확인
python run_backtest.py --list
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import pandas as pd

from core import (
    BacktestEnv, BasicSlippage, FixedRateCommission,
    FixedUSDTSizer, run_backtest, compute_drawdown,
)
from strategies import get_strategy, list_strategies


def main():
    parser = argparse.ArgumentParser(description="백테스트 실행기")
    parser.add_argument(
        "--strategy", default="trend_following",
        help="전략 이름 (기본: trend_following)"
    )
    parser.add_argument(
        "--data", default="data/ohlcv_full_BTCUSDT_1h.parquet",
        help="OHLCV 데이터 파일 경로 (기본: data/ohlcv_full_BTCUSDT_1h.parquet)"
    )
    parser.add_argument(
        "--cash", type=float, default=10_000,
        help="초기 자본 USDT (기본: 10000)"
    )
    parser.add_argument(
        "--trade-size", type=float, default=100,
        help="1회 매매 금액 USDT (기본: 100)"
    )
    parser.add_argument(
        "--slippage", type=float, default=5,
        help="슬리피지 bps (기본: 5)"
    )
    parser.add_argument(
        "--commission", type=float, default=10,
        help="수수료 bps (기본: 10, 바이낸스 0.1%%)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="차트를 표시하지 않음"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="사용 가능한 전략 목록 출력"
    )
    args = parser.parse_args()

    # 전략 목록 출력
    if args.list:
        print("사용 가능한 전략:")
        for name in list_strategies():
            print(f"  - {name}")
        return

    # 1) 데이터 로드
    try:
        raw = pd.read_parquet(args.data).set_index("time")[
            ["open", "high", "low", "close", "volume"]
        ]
        print(f"데이터 로드: {args.data}")
        print(f"  캔들 수: {len(raw):,}개 ({raw.index[0]} ~ {raw.index[-1]})")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {args.data}")
        print("먼저 collect_data.py로 데이터를 수집해주세요.")
        sys.exit(1)

    # 2) 환경 구성
    env = BacktestEnv(
        cash=args.cash,
        slippage=BasicSlippage(bps=args.slippage),
        commission=FixedRateCommission(bps=args.commission),
        sizer=FixedUSDTSizer(usdt_per_trade=args.trade_size),
    )

    # 3) 전략 생성
    try:
        strat = get_strategy(args.strategy)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"전략: {args.strategy}")
    print(f"초기 자본: {args.cash:,.0f} USDT | 1회 매매: {args.trade_size:,.0f} USDT")
    print(f"슬리피지: {args.slippage}bps | 수수료: {args.commission}bps")

    # 4) 백테스트 실행
    print("\n백테스트 실행 중...")
    equity, report = run_backtest(raw, strat, env)

    # 5) 결과 출력
    metrics = report.summary()
    print("\n" + "=" * 50)
    print("백테스트 결과")
    print("=" * 50)
    print(f"  누적 수익률:    {metrics['cum_return']:>10.2%}")
    print(f"  연환산 수익률:  {metrics['ann_return']:>10.2%}")
    print(f"  연환산 변동성:  {metrics['ann_vol']:>10.2%}")
    print(f"  샤프 비율:      {metrics['sharpe']:>10.2f}")
    print(f"  최대 낙폭(MDD): {metrics['mdd']:>10.2%}")
    print(f"  총 거래 횟수:   {len(report.trades):>10d}")

    if not report.trades.empty:
        buys = report.trades[report.trades["side"] == "buy"]
        sells = report.trades[report.trades["side"] == "sell"]
        print(f"  매수 {len(buys)}회 / 매도 {len(sells)}회")
        print(f"\n최근 거래 5건:")
        print(report.trades.tail())

    # 6) 차트
    if not args.no_plot:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(equity.index, equity.values, color="steelblue", linewidth=1)
        axes[0].set_title(f"Equity Curve — {args.strategy} ({args.data})")
        axes[0].set_ylabel("USDT")
        axes[0].grid(True, alpha=0.3)

        dd = compute_drawdown(equity)
        axes[1].fill_between(dd.index, dd.values, 0, color="salmon", alpha=0.5)
        axes[1].set_title("Drawdown")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
