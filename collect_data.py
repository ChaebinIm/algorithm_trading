"""
바이낸스 OHLCV 데이터 수집 스크립트 (CCXT 기반)

- 바이낸스 거래소에서 지정된 심볼/타임프레임의 캔들 데이터를 수집
- Parquet 형식으로 저장 (중복 제거, 이어붙이기 지원)
- CLI 인자로 심볼/타임프레임/시작일 지정 가능

사용 예시
--------
# 기본값(모든 심볼, 모든 타임프레임) 수집
python collect_data.py

# 특정 심볼/타임프레임만 수집
python collect_data.py --symbols BTC/USDT ETH/USDT --timeframes 1h 4h

# 시작일 지정
python collect_data.py --start 2022-01-01
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import ccxt
import pandas as pd
from dotenv import load_dotenv

# .env 파일에서 API 키 로딩
load_dotenv()

# =============================
# 기본 설정
# =============================

# 수집 대상 심볼 (유동성 상위 코인)
DEFAULT_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
]

# 수집 타임프레임
DEFAULT_TIMEFRAMES = ["1m", "5m", "1h", "4h", "1d"]

# 바이낸스 OHLCV 호출 시 한 번에 가져올 최대 캔들 수
CHUNK_LIMIT = 1000  # 바이낸스는 최대 1000개까지 지원

# 연속 빈 응답 허용 횟수 (초과 시 해당 심볼/타임프레임 수집 종료)
MAX_EMPTY_CHUNKS = 10

# 최신 캔들 확정 여유 시간(초) — 아직 닫히지 않은 캔들 제외
END_BUFFER_SEC = 60

# 파일 저장 경로 패턴
DATA_DIR = Path("data")

# 타임프레임 → 밀리초 매핑 (빈 응답 시 커서 점프에 사용)
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


# =============================
# 유틸리티 함수
# =============================

def make_save_path(symbol: str, timeframe: str) -> Path:
    """심볼과 타임프레임으로 저장 경로 생성.
    예: BTC/USDT, 1h → data/ohlcv_full_BTCUSDT_1h.parquet
    """
    tag = symbol.replace("/", "")
    return DATA_DIR / f"ohlcv_full_{tag}_{timeframe}.parquet"


def append_parquet(path: Path, df_new: pd.DataFrame) -> None:
    """기존 Parquet 파일에 새 데이터를 이어붙여 저장.
    - 타임스탬프(ts) 기준 중복 제거
    - 시간순 정렬 후 덮어쓰기
    """
    df_new = df_new.drop_duplicates(subset=["ts"]).sort_values("ts")
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    else:
        df = df_new
    df.to_parquet(path, index=False)


# =============================
# 핵심 수집 로직
# =============================

def collect_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_iso: str = "2020-01-01T00:00:00Z",
) -> None:
    """바이낸스에서 특정 심볼/타임프레임의 OHLCV 데이터를 전량 수집.

    Parameters
    ----------
    exchange : ccxt.Exchange
        CCXT 바이낸스 인스턴스 (load_markets 완료 상태)
    symbol : str
        수집 대상 심볼 (예: "BTC/USDT")
    timeframe : str
        캔들 타임프레임 (예: "1h")
    start_iso : str
        수집 시작 시각 (ISO 8601 형식)
    """
    save_path = make_save_path(symbol, timeframe)
    tf_ms = TIMEFRAME_MS[timeframe]

    # 이미 수집된 데이터가 있으면 마지막 타임스탬프 이후부터 이어서 수집
    if save_path.exists():
        existing = pd.read_parquet(save_path)
        last_ts = int(existing["ts"].max())
        since = last_ts + 1  # 마지막 캔들 다음부터
        print(f"[이어받기] {symbol} {timeframe}: 기존 {len(existing)}개, "
              f"마지막 {pd.to_datetime(last_ts, unit='ms', utc=True)} 이후부터 수집")
    else:
        since = exchange.parse8601(start_iso)

    end_ms = exchange.milliseconds() - END_BUFFER_SEC * 1000

    empty_streak = 0
    batch_count = 0
    buffer = []  # 메모리에 쌓아두다가 주기적으로 파일에 flush

    print(f"\n{'='*60}")
    print(f"수집 시작: {symbol} / {timeframe}")
    print(f"{'='*60}")

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=CHUNK_LIMIT
            )

            if ohlcv:
                df = pd.DataFrame(
                    ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
                )
                # UTC 기준 시간 컬럼 추가
                df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                buffer.append(df)

                last_ts = int(df["ts"].iloc[-1])
                since = last_ts + 1  # 정확히 이어붙이기
                empty_streak = 0
                batch_count += 1

                # 20 청크마다 중간 저장 (메모리 절약 & 중간 보존)
                if batch_count % 20 == 0:
                    append_parquet(save_path, pd.concat(buffer, ignore_index=True))
                    print(f"  [flush] {df['time'].iloc[-1]} → {save_path}")
                    buffer = []

                print(f"  {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} (n={len(df)})")

            else:
                # 빈 응답: 커서를 한 덩어리 건너뛰어 진행
                empty_streak += 1
                if empty_streak > MAX_EMPTY_CHUNKS:
                    print(f"  연속 빈 응답 {empty_streak}회 → 수집 종료")
                    break

                jump = tf_ms * CHUNK_LIMIT
                since = min(since + jump, end_ms)
                print(f"  [empty #{empty_streak}] 커서 이동 → "
                      f"{pd.to_datetime(since, unit='ms', utc=True)}")
                time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"  오류: {e}")
            time.sleep(3)  # 에러 시 잠시 대기 후 재시도

    # 남은 버퍼 최종 저장
    if buffer:
        append_parquet(save_path, pd.concat(buffer, ignore_index=True))
        print(f"  [최종 flush] → {save_path}")

    # 수집 결과 요약
    if save_path.exists():
        result = pd.read_parquet(save_path)
        print(f"  완료: 총 {len(result)}개 캔들 저장됨")
    print()


# =============================
# 메인 실행
# =============================

def main():
    parser = argparse.ArgumentParser(description="바이낸스 OHLCV 데이터 수집기")
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS,
        help="수집 대상 심볼 목록 (기본: BTC/USDT ETH/USDT SOL/USDT XRP/USDT)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=DEFAULT_TIMEFRAMES,
        help="수집 타임프레임 목록 (기본: 1m 5m 1h 4h 1d)"
    )
    parser.add_argument(
        "--start", default="2020-01-01T00:00:00Z",
        help="수집 시작일 ISO 8601 형식 (기본: 2020-01-01)"
    )
    args = parser.parse_args()

    # 저장 디렉토리 생성
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    # 바이낸스 인스턴스 생성 (.env에서 API 키 로딩)
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
    })
    exchange.load_markets()

    # API 키 연결 검증
    if api_key and api_secret:
        try:
            balance = exchange.fetch_balance()
            usdt = balance.get("USDT", {})
            print(f"API 연결 성공 — USDT 잔고: {usdt.get('free', 0):.2f} USDT")
        except Exception as e:
            print(f"[경고] API 키 검증 실패: {e}")
            print("  → 공개 데이터 수집은 가능하지만, 라이브 트레이딩은 불가합니다.")
    else:
        print("[안내] API 키 미설정 — 공개 데이터 수집만 가능합니다.")

    print(f"대상 심볼: {args.symbols}")
    print(f"타임프레임: {args.timeframes}")
    print(f"시작일: {args.start}")

    # 모든 심볼 × 타임프레임 조합에 대해 수집
    for symbol in args.symbols:
        for tf in args.timeframes:
            if tf not in TIMEFRAME_MS:
                print(f"[경고] 지원하지 않는 타임프레임: {tf}, 건너뜀")
                continue
            collect_ohlcv(exchange, symbol, tf, start_iso=args.start)

    print("=" * 60)
    print("모든 수집 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
