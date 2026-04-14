"""
업비트 KRW OHLCV 데이터 수집 스크립트 (CCXT 기반).

기존 data/ 폴더의 KRW 데이터를 보완:
- 부족한 타임프레임(4h) 추가 수집
- SOL, XRP 상위 타임프레임 수집

사용 예시:
    python collect_upbit.py
    python collect_upbit.py --symbols BTC/KRW ETH/KRW --timeframes 4h 1d
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import ccxt
import pandas as pd


DEFAULT_SYMBOLS = ["BTC/KRW", "ETH/KRW", "SOL/KRW", "XRP/KRW"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]

CHUNK_LIMIT = 200  # 업비트는 최대 200개
MAX_EMPTY_CHUNKS = 5
END_BUFFER_SEC = 60

DATA_DIR = Path("data")

TIMEFRAME_MS = {
    "1m":  60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# 업비트 지원 타임프레임 매핑 (ccxt 표준 → 업비트)
UPBIT_SUPPORTED = {"1m", "3m", "5m", "10m", "15m", "30m", "1h", "4h", "1d", "1w"}


def make_save_path(symbol: str, timeframe: str) -> Path:
    """BTC/KRW, 4h → data/ohlcv_full_BTCKRW_4h.parquet"""
    tag = symbol.replace("/", "")
    return DATA_DIR / f"ohlcv_full_{tag}_{timeframe}.parquet"


def append_parquet(path: Path, df_new: pd.DataFrame) -> None:
    df_new = df_new.drop_duplicates(subset=["ts"]).sort_values("ts")
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    else:
        df = df_new
    df.to_parquet(path, index=False)


def collect_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_iso: str = "2020-01-01T00:00:00Z",
) -> None:
    if timeframe not in UPBIT_SUPPORTED:
        print(f"  [스킵] 업비트 미지원 타임프레임: {timeframe}")
        return

    save_path = make_save_path(symbol, timeframe)
    tf_ms = TIMEFRAME_MS.get(timeframe, 3_600_000)

    if save_path.exists():
        existing = pd.read_parquet(save_path)
        last_ts = int(existing["ts"].max())
        since = last_ts + 1
        print(f"[이어받기] {symbol} {timeframe}: {len(existing)}개 → "
              f"{pd.to_datetime(last_ts, unit='ms', utc=True)} 이후부터")
    else:
        since = exchange.parse8601(start_iso)
        print(f"[신규 수집] {symbol} {timeframe}: {start_iso}부터")

    end_ms = exchange.milliseconds() - END_BUFFER_SEC * 1000
    empty_streak = 0
    batch_count = 0
    buffer = []

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=CHUNK_LIMIT
            )

            if ohlcv:
                df = pd.DataFrame(
                    ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
                )
                df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                buffer.append(df)

                last_ts = int(df["ts"].iloc[-1])
                since = last_ts + 1
                empty_streak = 0
                batch_count += 1

                if batch_count % 20 == 0:
                    append_parquet(save_path, pd.concat(buffer, ignore_index=True))
                    print(f"    [flush] ~ {df['time'].iloc[-1]}")
                    buffer = []

                print(f"    {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (n={len(df)})")
                time.sleep(exchange.rateLimit / 1000)

            else:
                empty_streak += 1
                if empty_streak > MAX_EMPTY_CHUNKS:
                    print(f"  빈 응답 {empty_streak}회 → 종료")
                    break
                since = min(since + tf_ms * CHUNK_LIMIT, end_ms)
                time.sleep(1)

        except ccxt.RateLimitExceeded:
            print("  Rate limit 초과, 5초 대기...")
            time.sleep(5)
        except Exception as e:
            print(f"  오류: {e}")
            time.sleep(3)

    if buffer:
        append_parquet(save_path, pd.concat(buffer, ignore_index=True))

    if save_path.exists():
        result = pd.read_parquet(save_path)
        print(f"  완료: {len(result)}개 캔들 ({result['time'].min()} ~ {result['time'].max()})")


def main():
    parser = argparse.ArgumentParser(description="업비트 KRW OHLCV 수집기")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES)
    parser.add_argument("--start", default="2020-01-01T00:00:00Z")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True, parents=True)

    exchange = ccxt.upbit({"enableRateLimit": True})
    exchange.load_markets()
    print("업비트 연결 완료")

    for symbol in args.symbols:
        for tf in args.timeframes:
            print(f"\n{'='*50}")
            print(f"수집: {symbol} / {tf}")
            print(f"{'='*50}")
            collect_ohlcv(exchange, symbol, tf, start_iso=args.start)

    print("\n" + "="*50)
    print("수집 완료")


if __name__ == "__main__":
    main()
