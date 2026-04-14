"""
바이낸스 USDT 데이터 수집 스크립트.

수집 대상:
  - BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, BNB/USDT
  - 타임프레임: 1h, 4h, 1d
  - 시작: 2017-01-01 (가능한 최초 시점)

실행:
  python collect_binance.py
"""
import ccxt
import pandas as pd
import time
from pathlib import Path

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "BNB/USDT",
]

TIMEFRAMES = ["1h", "4h", "1d"]

START = "2017-01-01T00:00:00Z"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

exchange = ccxt.binance({"rateLimit": 200})


def collect(symbol: str, tf: str) -> None:
    coin = symbol.replace("/", "")
    fname = DATA_DIR / f"ohlcv_full_{coin}_{tf}.parquet"

    if fname.exists():
        # 기존 파일이 있으면 마지막 시점 이후만 추가 수집
        existing = pd.read_parquet(fname)
        if "time" in existing.columns:
            existing = existing.set_index("time")
        existing.index = pd.to_datetime(existing.index, utc=True)
        since_ts = int(existing.index[-1].timestamp() * 1000) + 1
        print(f"[업데이트] {fname.name}  마지막: {existing.index[-1].date()}", end="", flush=True)
        all_ohlcv = []
    else:
        since_ts = exchange.parse8601(START)
        print(f"[신규수집] {fname.name}", end="", flush=True)
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

    new_df = pd.DataFrame(all_ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    new_df["time"] = pd.to_datetime(new_df["time"], unit="ms", utc=True)
    new_df = new_df.set_index("time").drop_duplicates()

    if existing is not None:
        combined = pd.concat([existing, new_df]).drop_duplicates()
        combined.sort_index(inplace=True)
    else:
        combined = new_df.sort_index()

    combined.to_parquet(fname)
    print(f" → {len(combined)}개 캔들 ({combined.index[0].date()} ~ {combined.index[-1].date()})")


def main():
    print("=" * 60)
    print("바이낸스 USDT 데이터 수집")
    print("=" * 60)

    total = len(SYMBOLS) * len(TIMEFRAMES)
    done = 0
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            collect(symbol, tf)
            done += 1
            time.sleep(0.3)

    print(f"\n완료: {done}개 파일")


if __name__ == "__main__":
    main()
