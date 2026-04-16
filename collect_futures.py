"""
바이낸스 선물(USDⓈ-M) 데이터 수집 스크립트.

수집 대상:
  - BTC/USDT, ETH/USDT, BNB/USDT (선물 종목)
  - 타임프레임: 1h, 4h, 1d
  - 펀딩비 히스토리 (8시간마다)
  - 미결제약정(Open Interest) 히스토리

저장 위치:
  data/futures/ohlcv_futures_{SYMBOL}_{TF}.parquet
  data/futures/funding_{SYMBOL}.parquet
  data/futures/oi_{SYMBOL}.parquet

실행:
  python collect_futures.py
"""
import ccxt
import pandas as pd
import time
from pathlib import Path

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
]

TIMEFRAMES = ["1h", "4h", "1d"]
START = "2020-01-01T00:00:00Z"

DATA_DIR = Path("data/futures")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 선물 전용 클라이언트
exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})
exchange.load_markets()


# ── OHLCV 수집 ────────────────────────────────────────────────────────

def collect_ohlcv(symbol: str, tf: str) -> None:
    coin = symbol.replace("/", "")
    fname = DATA_DIR / f"ohlcv_futures_{coin}_{tf}.parquet"

    if fname.exists():
        existing = pd.read_parquet(fname)
        if "time" in existing.columns:
            existing = existing.set_index("time")
        existing.index = pd.to_datetime(existing.index, utc=True)
        since_ts = int(existing.index[-1].timestamp() * 1000) + 1
        print(f"[OHLCV 업데이트] {fname.name}  마지막: {existing.index[-1].date()}", end="", flush=True)
        all_ohlcv = []
    else:
        since_ts = exchange.parse8601(START)
        print(f"[OHLCV 신규] {fname.name}", end="", flush=True)
        all_ohlcv = []
        existing = None

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, tf, since=since_ts, limit=1000)
        except Exception as e:
            print(f" 오류: {e}")
            return
        if not bars:
            break
        all_ohlcv += bars
        since_ts = bars[-1][0] + 1
        if len(bars) < 1000:
            break
        time.sleep(0.2)

    if not all_ohlcv:
        print(" 신규 없음")
        return

    new_df = pd.DataFrame(
        all_ohlcv, columns=["time", "open", "high", "low", "close", "volume"]
    )
    new_df["time"] = pd.to_datetime(new_df["time"], unit="ms", utc=True)
    new_df = new_df.set_index("time").drop_duplicates()

    if existing is not None:
        combined = pd.concat([existing, new_df]).drop_duplicates()
        combined.sort_index(inplace=True)
    else:
        combined = new_df.sort_index()

    combined.to_parquet(fname)
    print(f" → {len(combined)}개 캔들 ({combined.index[0].date()} ~ {combined.index[-1].date()})")


# ── 펀딩비 히스토리 수집 ───────────────────────────────────────────────

def collect_funding(symbol: str) -> None:
    coin = symbol.replace("/", "")
    fname = DATA_DIR / f"funding_{coin}.parquet"

    if fname.exists():
        existing = pd.read_parquet(fname)
        if "time" in existing.columns:
            existing = existing.set_index("time")
        existing.index = pd.to_datetime(existing.index, utc=True)
        since_ts = int(existing.index[-1].timestamp() * 1000) + 1
        print(f"[펀딩비 업데이트] {fname.name}  마지막: {existing.index[-1].date()}", end="", flush=True)
    else:
        since_ts = exchange.parse8601(START)
        print(f"[펀딩비 신규] {fname.name}", end="", flush=True)
        existing = None

    all_rows = []
    while True:
        try:
            rows = exchange.fetch_funding_rate_history(symbol, since=since_ts, limit=1000)
        except Exception as e:
            print(f" 오류: {e}")
            return
        if not rows:
            break
        all_rows += rows
        since_ts = rows[-1]["timestamp"] + 1
        if len(rows) < 1000:
            break
        time.sleep(0.2)

    if not all_rows:
        print(" 신규 없음")
        return

    new_df = pd.DataFrame([{
        "time": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
        "funding_rate": float(r["fundingRate"]),
    } for r in all_rows]).set_index("time").drop_duplicates()

    if existing is not None:
        combined = pd.concat([existing, new_df]).drop_duplicates()
        combined.sort_index(inplace=True)
    else:
        combined = new_df.sort_index()

    combined.to_parquet(fname)
    print(f" → {len(combined)}개 ({combined.index[0].date()} ~ {combined.index[-1].date()})")


# ── 미결제약정(Open Interest) 히스토리 수집 ────────────────────────────

def collect_open_interest(symbol: str, tf: str = "4h") -> None:
    """OI 히스토리 수집. 바이낸스는 since 파라미터 미지원 → 최근 500개 수집 후 누적."""
    coin = symbol.replace("/", "")
    fname = DATA_DIR / f"oi_{coin}_{tf}.parquet"

    print(f"[OI] {fname.name}", end="", flush=True)

    try:
        rows = exchange.fetch_open_interest_history(symbol, tf, limit=500)
    except Exception as e:
        print(f" 오류: {e}")
        return

    if not rows:
        print(" 데이터 없음")
        return

    new_df = pd.DataFrame([{
        "time": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
        "open_interest": float(r["openInterestAmount"]),
        "open_interest_value": float(r.get("openInterestValue", 0) or 0),
    } for r in rows]).set_index("time").drop_duplicates()

    if fname.exists():
        existing = pd.read_parquet(fname)
        if "time" in existing.columns:
            existing = existing.set_index("time")
        existing.index = pd.to_datetime(existing.index, utc=True)
        combined = pd.concat([existing, new_df]).drop_duplicates()
        combined.sort_index(inplace=True)
    else:
        combined = new_df.sort_index()

    combined.to_parquet(fname)
    print(f" → {len(combined)}개 ({combined.index[0].date()} ~ {combined.index[-1].date()})")


# ── 메인 ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("바이낸스 선물 데이터 수집")
    print("=" * 60)

    # OHLCV
    print("\n▶ OHLCV (선물)")
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            collect_ohlcv(symbol, tf)
            time.sleep(0.3)

    # 펀딩비
    print("\n▶ 펀딩비 히스토리")
    for symbol in SYMBOLS:
        collect_funding(symbol)
        time.sleep(0.3)

    # 미결제약정
    print("\n▶ 미결제약정(Open Interest) — 4h")
    for symbol in SYMBOLS:
        collect_open_interest(symbol, "4h")
        time.sleep(0.5)

    print("\n완료!")


if __name__ == "__main__":
    main()
