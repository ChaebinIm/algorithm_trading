"""
공용 기술적 지표 함수 모음.

모든 전략에서 임포트하여 사용할 수 있다.
새 지표가 필요하면 이 파일에 추가.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """단순이동평균 (Simple Moving Average)"""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """지수이동평균 (Exponential Moving Average)"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index)
    - 70 이상: 과매수 구간
    - 30 이하: 과매도 구간
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR (Average True Range) — 변동성 측정 지표.
    손절/익절 폭을 동적으로 설정할 때 사용.
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX (Average Directional Index) — 추세 강도 측정.
    - 25 이상: 추세 존재
    - 20 이하: 횡보/무추세 구간
    """
    high = df["high"]
    low = df["low"]

    # +DM, -DM 계산
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # ATR 계산
    atr_val = atr(df, period)

    # Smoothed +DI, -DI
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr_val + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr_val + 1e-12)

    # DX → ADX
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드.

    Returns:
        (upper, middle, lower, bandwidth)
        - upper: 상단 밴드 (middle + num_std * std)
        - middle: 중심선 (SMA)
        - lower: 하단 밴드 (middle - num_std * std)
        - bandwidth: 밴드폭 비율 ((upper - lower) / middle). 스퀴즈 감지에 사용.
    """
    middle = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=1)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / (middle + 1e-12)
    return upper, middle, lower, bandwidth


def donchian_channel(
    df: pd.DataFrame, period: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """돈키안 채널 (N기간 고가/저가).

    Returns:
        (upper, lower)
        - upper: N기간 최고가
        - lower: N기간 최저가
    """
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return upper, lower


def momentum(series: pd.Series, period: int = 20) -> pd.Series:
    """단순 모멘텀 — N기간 수익률.
    momentum[t] = close[t] / close[t - period] - 1
    양수: 상승 모멘텀, 음수: 하락 모멘텀.
    """
    return series / series.shift(period) - 1.0
