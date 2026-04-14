"""
파라미터 최적화 스크립트 — 그리드 서치로 최적 파라미터 탐색.

대상 전략:
- supertrend  : st_period, st_multiplier, sma_period
- adaptive_regime : sma_long, bull_adx_threshold, ema_fast, ema_slow

최적화 기준: 샤프 비율(sharpe) 최대화 (단, MDD -25% 이하인 조합은 제외)

테스트 데이터:
- data/ohlcv_full_BTCKRW_4h.parquet  (BTC 4h)
- data/ohlcv_full_ETHKRW_1h.parquet  (ETH 1h)

결과 저장: reports/param_optimization_YYYYMMDD_HHMMSS.json
"""
from __future__ import annotations

import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core import (
    BacktestEnv,
    BasicSlippage,
    FixedRateCommission,
    PercentSizer,
    run_backtest,
)
from strategies import get_strategy


# =============================
# 설정
# =============================

REPORT_DIR = Path("reports")

# 백테스트 환경 (고정)
ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),
    commission=FixedRateCommission(bps=5),
    sizer=PercentSizer(percent=0.20),
)

# 테스트 데이터셋
DATASETS = [
    "data/ohlcv_full_BTCKRW_4h.parquet",
    "data/ohlcv_full_ETHKRW_1h.parquet",
]

# MDD 필터 임계값 (이 값보다 더 큰 낙폭은 제외)
MDD_THRESHOLD = -0.25  # -25%

# 상위 결과 출력 개수
TOP_N = 10

# 탐색 파라미터 공간
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "supertrend": {
        "st_period":     [7, 10, 14],
        "st_multiplier": [2.5, 3.0, 3.5],
        "sma_period":    [150, 200, 250],
    },
    "adaptive_regime": {
        "sma_long":           [100, 200],
        "bull_adx_threshold": [18, 22, 25],
        "ema_fast":           [15, 20, 25],
        "ema_slow":           [40, 50, 60],
    },
}


# =============================
# 유틸리티
# =============================

def load_data(path: str) -> pd.DataFrame:
    """Parquet 데이터 로드. 인덱스 컬럼명이 'time'이면 set_index 적용."""
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df = df.set_index("time")
    elif df.index.name != "time":
        df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]]


def _fmt_float(v: float, pct: bool = False, digits: int = 2) -> str:
    """float 포맷 헬퍼."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if math.isinf(v):
        return "inf"
    if pct:
        return f"{v:+.{digits}%}"
    return f"{v:.{digits}f}"


def _safe(v: float) -> Any:
    """JSON 직렬화를 위한 안전 변환 (nan/inf → None)."""
    if v is None:
        return None
    try:
        if math.isnan(v) or math.isinf(v):
            return None
    except TypeError:
        pass
    return v


def grid_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """딕셔너리 형태의 파라미터 그리드를 전체 조합 리스트로 변환."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# =============================
# 단일 백테스트 실행
# =============================

def run_single(
    strategy_name: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    data_label: str,
) -> Dict[str, Any]:
    """단일 파라미터 조합 백테스트 실행 후 결과 딕셔너리 반환.

    반환 딕셔너리의 "filtered" 키가 True이면 MDD 필터로 제외된 것.
    "error" 키가 있으면 실행 오류.
    """
    try:
        strat = get_strategy(strategy_name, **params)
        equity, report = run_backtest(df, strat, ENV)
        metrics = report.summary()

        mdd = metrics.get("mdd", float("nan"))
        sharpe = metrics.get("sharpe", float("nan"))
        cum_return = metrics.get("cum_return", float("nan"))
        win_rate = metrics.get("win_rate", float("nan"))
        profit_factor = metrics.get("profit_factor", float("nan"))
        total_roundtrips = metrics.get("total_roundtrips", 0)

        # MDD 필터 적용
        filtered = (not math.isnan(mdd)) and (mdd < MDD_THRESHOLD)

        return {
            "strategy": strategy_name,
            "data": data_label,
            "params": params,
            "cum_return": _safe(cum_return),
            "sharpe": _safe(sharpe),
            "mdd": _safe(mdd),
            "win_rate": _safe(win_rate),
            "profit_factor": _safe(profit_factor),
            "total_roundtrips": total_roundtrips,
            "filtered": filtered,
        }

    except Exception as exc:
        return {
            "strategy": strategy_name,
            "data": data_label,
            "params": params,
            "error": str(exc),
            "filtered": False,
        }


# =============================
# 전략별 그리드 서치
# =============================

def optimize_strategy(
    strategy_name: str,
    datasets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """주어진 전략의 모든 파라미터 조합을 탐색하고 결과 리스트 반환.

    datasets: [{"label": str, "df": pd.DataFrame}, ...]
    각 (파라미터 조합 × 데이터셋) 쌍을 독립적으로 평가한다.
    """
    param_grid = PARAM_GRIDS[strategy_name]
    combos = grid_combinations(param_grid)
    total = len(combos) * len(datasets)

    print(f"\n{'='*60}")
    print(f"전략: {strategy_name}")
    print(f"파라미터 조합: {len(combos)}개  ×  데이터셋: {len(datasets)}개  =  총 {total}회 실행")
    print(f"{'='*60}")

    results: List[Dict[str, Any]] = []
    run_idx = 0

    for combo in combos:
        param_str = ", ".join(f"{k}={v}" for k, v in combo.items())

        for ds in datasets:
            run_idx += 1
            label = ds["label"]
            df = ds["df"]

            print(f"  [{run_idx:>4}/{total}] {label} | {param_str}", end="  ->  ", flush=True)

            result = run_single(strategy_name, combo, df, label)

            # 오류 케이스
            if "error" in result:
                print(f"[오류] {result['error']}")
                continue

            sharpe = result["sharpe"]
            cum = result["cum_return"]
            mdd = result["mdd"]
            wr = result["win_rate"]
            pf = result["profit_factor"]

            # MDD 필터로 제외된 경우
            if result.get("filtered"):
                print(
                    f"[MDD 필터 제외] 샤프={_fmt_float(sharpe)}  "
                    f"MDD={_fmt_float(mdd, pct=True)}"
                )
                continue

            print(
                f"샤프={_fmt_float(sharpe)}  수익률={_fmt_float(cum, pct=True)}  "
                f"MDD={_fmt_float(mdd, pct=True)}  승률={_fmt_float(wr, pct=True)}  "
                f"PF={_fmt_float(pf)}"
            )

            results.append(result)

    return results


# =============================
# 결과 출력
# =============================

def print_top_results(
    strategy_name: str,
    results: List[Dict[str, Any]],
    top_n: int = TOP_N,
) -> None:
    """전략별 상위 N개 파라미터 조합을 샤프 비율 내림차순으로 출력."""

    # sharpe 기준 정렬 (None/nan은 최하위로)
    def sort_key(r: Dict[str, Any]) -> float:
        s = r.get("sharpe")
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return -9999.0
        return float(s)

    sorted_results = sorted(results, key=sort_key, reverse=True)
    top = sorted_results[:top_n]

    print(f"\n{'='*70}")
    print(f"[ {strategy_name} ] 상위 {min(top_n, len(top))}개 파라미터 조합 (샤프 비율 기준)")
    print(f"{'='*70}")

    if not top:
        print("  유효한 결과가 없습니다.")
        return

    # 헤더
    header = f"{'순위':>4}  {'데이터':<25}  {'샤프':>7}  {'누적수익률':>10}  {'MDD':>8}  {'승률':>7}  {'PF':>6}  파라미터"
    print(header)
    print("-" * max(len(header), 100))

    for rank, r in enumerate(top, 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        sharpe_s = _fmt_float(r["sharpe"])
        cum_s = _fmt_float(r["cum_return"], pct=True)
        mdd_s = _fmt_float(r["mdd"], pct=True)
        wr_s = _fmt_float(r["win_rate"], pct=True)
        pf_s = _fmt_float(r["profit_factor"])

        print(
            f"  {rank:>2}.  {r['data']:<25}  {sharpe_s:>7}  {cum_s:>10}  "
            f"{mdd_s:>8}  {wr_s:>7}  {pf_s:>6}  {param_str}"
        )

    print("-" * max(len(header), 100))


# =============================
# 결과 저장
# =============================

def save_results(all_results: Dict[str, List[Dict[str, Any]]], timestamp: str) -> Path:
    """결과를 JSON 파일로 저장하고 경로 반환."""
    REPORT_DIR.mkdir(exist_ok=True)
    out_path = REPORT_DIR / f"param_optimization_{timestamp}.json"

    # JSON 직렬화를 위해 float nan/inf 정리
    def clean(obj: Any) -> Any:
        if isinstance(obj, float):
            return _safe(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    payload = {
        "generated_at": timestamp,
        "mdd_threshold": MDD_THRESHOLD,
        "env": {
            "cash": ENV.cash,
            "slippage_bps": 5,
            "commission_bps": 5,
            "sizer": "PercentSizer(20%)",
        },
        "datasets": DATASETS,
        "results": clean(all_results),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    return out_path


# =============================
# 메인
# =============================

def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"파라미터 최적화 시작  ({timestamp})")
    print(f"MDD 필터: {MDD_THRESHOLD:.0%} 이하 제외")
    print(f"최적화 기준: 샤프 비율 최대화")
    print("=" * 70)

    # 데이터 로드
    print("\n[데이터 로드]")
    datasets: List[Dict[str, Any]] = []
    for path in DATASETS:
        try:
            df = load_data(path)
            label = Path(path).stem.replace("ohlcv_full_", "")
            print(f"  {label}: {len(df):,}개 캔들  ({df.index[0]} ~ {df.index[-1]})")
            datasets.append({"label": label, "df": df})
        except FileNotFoundError:
            print(f"  [경고] 파일 없음: {path} — 해당 데이터셋 건너뜀")

    if not datasets:
        print("사용 가능한 데이터셋이 없습니다. 종료합니다.")
        return

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    # 전략별 최적화 실행
    for strategy_name in PARAM_GRIDS:
        results = optimize_strategy(strategy_name, datasets)
        all_results[strategy_name] = results
        print_top_results(strategy_name, results)

    # 결과 저장
    out_path = save_results(all_results, timestamp)

    print(f"\n{'='*70}")
    print(f"최적화 완료")
    print(f"결과 저장: {out_path}")
    print(f"{'='*70}")

    # 전체 요약: 전략 × 데이터셋별 최고 샤프
    print("\n[ 전략별 최고 샤프 비율 요약 ]\n")
    summary_header = f"  {'전략':<22}  {'데이터':<25}  {'샤프':>7}  {'누적수익률':>10}  {'MDD':>8}  파라미터"
    print(summary_header)
    print("  " + "-" * (len(summary_header) - 2))

    for strategy_name, results in all_results.items():
        if not results:
            print(f"  {strategy_name:<22}  유효 결과 없음")
            continue

        # 데이터셋별 최고 샤프
        best_by_ds: Dict[str, Dict[str, Any]] = {}
        for r in results:
            ds = r["data"]
            s = r.get("sharpe")
            if s is None:
                continue
            if ds not in best_by_ds or (
                best_by_ds[ds].get("sharpe") is not None and s > best_by_ds[ds]["sharpe"]
            ):
                best_by_ds[ds] = r

        for ds, r in sorted(best_by_ds.items()):
            param_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            print(
                f"  {strategy_name:<22}  {ds:<25}  "
                f"{_fmt_float(r['sharpe']):>7}  "
                f"{_fmt_float(r['cum_return'], pct=True):>10}  "
                f"{_fmt_float(r['mdd'], pct=True):>8}  "
                f"{param_str}"
            )

    print()


if __name__ == "__main__":
    main()
