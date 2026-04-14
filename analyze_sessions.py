"""
시간대(세션)별 전략 성과 분석.

크립토 시장의 주요 세션별로 진입을 제한했을 때
각 전략의 성과가 어떻게 달라지는지 검증한다.

세션 정의 (UTC 기준):
- asia:     00:00 ~ 08:00 (도쿄/서울/홍콩)
- europe:   07:00 ~ 16:00 (런던)
- us:       13:00 ~ 21:00 (뉴욕)
- overlap:  13:00 ~ 16:00 (EU/US 겹침, 최대 유동성)
- night:    21:00 ~ 00:00 (야간, 거래량 감소)
- all:      제한 없음 (기준선)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core import (
    BacktestEnv, BasicSlippage, FixedRateCommission,
    FixedUSDTSizer, run_backtest,
)
from strategies import Strategy, get_strategy, list_strategies


# =============================
# 세션 정의 (UTC 시간)
# =============================

SESSIONS = {
    "all":     (0, 24),    # 제한 없음 (기준선)
    "asia":    (0, 8),     # 아시아 세션
    "europe":  (7, 16),    # 유럽 세션
    "us":      (13, 21),   # 미국 세션
    "overlap": (13, 16),   # EU/US 겹침 (최대 유동성)
    "night":   (21, 24),   # 야간
}


# =============================
# 세션 필터 래퍼 전략
# =============================

class SessionFilteredStrategy(Strategy):
    """기존 전략을 감싸서 특정 세션 시간에만 진입을 허용하는 래퍼.

    - 세션 외 시간에는 매수 신호를 무시 (0 반환)
    - 매도 신호는 세션과 무관하게 항상 허용 (손절/익절은 즉시 실행)
    """

    def __init__(self, base_strategy: Strategy, session_start: int, session_end: int):
        self._base = base_strategy
        self._start = session_start
        self._end = session_end

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._base.prepare(df)
        # UTC 시간 정보 추가 (시간대 필터링용)
        df["_hour"] = df.index.hour
        return df

    def on_bar(self, i: int, row: pd.Series) -> int:
        signal = self._base.on_bar(i, row)
        # 매도는 항상 허용, 매수만 세션 필터 적용
        if signal > 0:
            hour = int(row.get("_hour", 0))
            if not self._in_session(hour):
                return 0
        return signal

    def _in_session(self, hour: int) -> bool:
        """해당 시간이 세션 범위 내인지 판단."""
        if self._start < self._end:
            return self._start <= hour < self._end
        else:
            # 자정을 넘는 세션 (예: 21~03)
            return hour >= self._start or hour < self._end


# =============================
# 분석 실행
# =============================

def run_session_analysis(
    data_path: str,
    strategy_name: str,
    env: BacktestEnv,
) -> list[dict]:
    """하나의 전략을 모든 세션에 대해 백테스트."""
    df = pd.read_parquet(data_path).set_index("time")[
        ["open", "high", "low", "close", "volume"]
    ]
    tf = data_path.split("_")[-1].replace(".parquet", "")
    results = []

    for session_name, (s_start, s_end) in SESSIONS.items():
        base_strat = get_strategy(strategy_name)

        if session_name == "all":
            strat = base_strat
        else:
            strat = SessionFilteredStrategy(base_strat, s_start, s_end)

        try:
            equity, report = run_backtest(df, strat, env)
            m = report.summary()
            n_trades = len(report.trades)

            results.append({
                "strategy": strategy_name,
                "timeframe": tf,
                "session": session_name,
                "session_hours": f"{s_start:02d}:00-{s_end:02d}:00 UTC",
                "cum_return": m["cum_return"],
                "ann_return": m["ann_return"],
                "sharpe": m["sharpe"],
                "mdd": m["mdd"],
                "total_trades": n_trades,
            })
        except Exception as e:
            results.append({
                "strategy": strategy_name,
                "timeframe": tf,
                "session": session_name,
                "error": str(e),
            })

    return results


def main():
    # 1h 봉만 시간대 분석 의미 있음 (4h/1d는 세션 구분이 흐려짐)
    DATA_PATH = "data/ohlcv_full_BTCUSDT_1h.parquet"

    if not Path(DATA_PATH).exists():
        print(f"데이터 없음: {DATA_PATH}")
        return

    env = BacktestEnv(
        cash=10_000,
        slippage=BasicSlippage(bps=5),
        commission=FixedRateCommission(bps=10),
        sizer=FixedUSDTSizer(usdt_per_trade=100),
    )

    strategies = list_strategies()
    all_results = []

    print("=" * 80)
    print("시간대(세션)별 전략 성과 분석")
    print(f"데이터: {DATA_PATH}")
    print("=" * 80)

    for strat_name in strategies:
        print(f"\n>>> {strat_name}")
        results = run_session_analysis(DATA_PATH, strat_name, env)
        all_results.extend(results)

        # 콘솔 출력
        print(f"  {'세션':<10} {'시간(UTC)':<18} {'누적수익률':>10} {'샤프':>7} {'MDD':>8} {'거래수':>7}")
        print(f"  {'-'*65}")
        for r in results:
            if "error" in r:
                print(f"  {r['session']:<10} 에러")
                continue
            print(
                f"  {r['session']:<10} {r['session_hours']:<18} "
                f"{r['cum_return']:>9.2%} "
                f"{r['sharpe']:>7.2f} "
                f"{r['mdd']:>7.2%} "
                f"{r['total_trades']:>7d}"
            )

    # 결과 저장
    REPORT_DIR = Path("reports")
    REPORT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    df_results = pd.DataFrame(all_results)
    csv_path = REPORT_DIR / f"session_analysis_{timestamp}.csv"
    json_path = REPORT_DIR / f"session_analysis_{timestamp}.json"

    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\n결과 저장: {csv_path}")

    # 전략별 최적 세션 요약
    print("\n" + "=" * 80)
    print("[ 전략별 최적 세션 ]")
    print("=" * 80)

    for strat_name in strategies:
        strat_results = [r for r in all_results if r["strategy"] == strat_name and "error" not in r]
        if not strat_results:
            continue
        # 샤프 기준 최적 세션 (all 제외)
        filtered = [r for r in strat_results if r["session"] != "all"]
        if not filtered:
            continue
        best = max(filtered, key=lambda x: x["sharpe"])
        all_baseline = next((r for r in strat_results if r["session"] == "all"), None)

        print(f"\n  {strat_name}:")
        if all_baseline:
            print(f"    기준(전체): 수익률 {all_baseline['cum_return']:.2%}, 샤프 {all_baseline['sharpe']:.2f}")
        print(f"    최적 세션:  {best['session']} ({best['session_hours']})")
        print(f"                수익률 {best['cum_return']:.2%}, 샤프 {best['sharpe']:.2f}, MDD {best['mdd']:.2%}")


if __name__ == "__main__":
    main()
