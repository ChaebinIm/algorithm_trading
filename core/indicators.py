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

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_val = atr(df, period)

    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr_val + 1e-12)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / (atr_val + 1e-12)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_val = dx.ewm(span=period, adjust=False).mean()
    return adx_val


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """볼린저 밴드.

    Returns:
        (upper, middle, lower, bandwidth)
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
    """
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return upper, lower


def momentum(series: pd.Series, period: int = 20) -> pd.Series:
    """단순 모멘텀 — N기간 수익률.
    momentum[t] = close[t] / close[t - period] - 1
    """
    return series / series.shift(period) - 1.0


# =============================
# 신규 지표
# =============================

def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence).

    Returns:
        (macd_line, signal_line, histogram)
        - macd_line > 0: 상승 모멘텀 우세
        - histogram > 0: 모멘텀 강화 중
        - signal cross (histogram이 음→양): 매수 신호
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """스토캐스틱 %K, %D.

    Returns:
        (k, d)
        - k < 20: 과매도, k > 80: 과매수
        - k가 d를 상향 돌파: 매수 신호
    """
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100.0 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """SuperTrend 지표 — ATR 기반 추세 추종 지표.

    Returns:
        (supertrend_line, direction)
        - direction = +1: 상승 추세 (종가 > SuperTrend)
        - direction = -1: 하락 추세 (종가 < SuperTrend)
        - 방향 전환 시점이 진입/청산 신호

    크립토 최적 파라미터: period=10, multiplier=3.0 (4h 기준)
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    atr_val = atr(df, period)

    raw_upper = hl2 + multiplier * atr_val
    raw_lower = hl2 - multiplier * atr_val

    n = len(df)
    close = df["close"].values
    upper = raw_upper.values.copy()
    lower = raw_lower.values.copy()
    st = np.zeros(n)
    direction = np.ones(n, dtype=int)

    # 초기값
    st[0] = upper[0]
    direction[0] = -1  # 첫 바는 일단 하락으로 시작

    for i in range(1, n):
        # 밴드 조정 (이전 종가가 밴드 안에 있으면 밴드를 유지)
        if close[i - 1] > upper[i - 1]:
            upper[i] = upper[i]
        else:
            upper[i] = min(upper[i], upper[i - 1])

        if close[i - 1] < lower[i - 1]:
            lower[i] = lower[i]
        else:
            lower[i] = max(lower[i], lower[i - 1])

        # 방향 결정
        if direction[i - 1] == -1:
            if close[i] > upper[i]:
                direction[i] = 1
            else:
                direction[i] = -1
        else:
            if close[i] < lower[i]:
                direction[i] = -1
            else:
                direction[i] = 1

        # SuperTrend 값
        st[i] = lower[i] if direction[i] == 1 else upper[i]

    st_series = pd.Series(st, index=df.index, name="supertrend")
    dir_series = pd.Series(direction, index=df.index, name="st_direction")
    return st_series, dir_series


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow (CMF) — 자금 유입/유출 강도.

    Returns:
        cmf_series: -1 ~ +1
        - > 0: 매수 압력 (자금 유입)
        - < 0: 매도 압력 (자금 유출)
        - |CMF| > 0.05: 의미 있는 신호
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low + 1e-12)
    # Money Flow Volume
    mfv = mfm * volume
    return mfv.rolling(period).sum() / (volume.rolling(period).sum() + 1e-12)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV) — 누적 거래량 지표.

    Returns:
        obv_series: 가격과 같이 상승하면 상승 추세 확인
    """
    delta = df["close"].diff()
    direction = np.sign(delta).fillna(0)
    return (direction * df["volume"]).cumsum()


def keltner_channel(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """켈트너 채널 (Keltner Channel).

    Returns:
        (upper, middle, lower)
        - 볼린저 밴드보다 ATR 기반으로 더 안정적
        - BB가 KC 안에 있으면 스퀴즈 상태 (폭발 임박)
    """
    middle = ema(df["close"], ema_period)
    atr_val = atr(df, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


def stoch_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic RSI — RSI에 스토캐스틱을 적용한 지표.

    Returns:
        (k, d)
        - k < 20: 과매도 구역 (반등 가능성)
        - k > 80: 과매수 구역 (조정 가능성)
    """
    rsi_val = rsi(series, rsi_period)
    low_min = rsi_val.rolling(stoch_period).min()
    high_max = rsi_val.rolling(stoch_period).max()
    k = 100.0 * (rsi_val - low_min) / (high_max - low_min + 1e-12)
    d = k.rolling(d_period).mean()
    k_smooth = k.rolling(k_period).mean()
    return k_smooth, d


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R — 과매수/과매도 오실레이터.

    Returns:
        w_r: -100 ~ 0
        - > -20: 과매수
        - < -80: 과매도
    """
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100.0 * (high_max - df["close"]) / (high_max - low_min + 1e-12)


def reg_channel(
    df: pd.DataFrame,
    window: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """선형 회귀 채널 (Linear Regression Channel).

    매 바마다 직전 window개 봉의 종가로 선형 회귀선을 계산하고,
    그 회귀선과 평행하되 window 내 고점/저점에 닿는 상·하단 채널을 반환.

    구조:
        center  : 회귀 중심선 (현재 바의 회귀 예측값)
        upper   : 회귀선 + max(window 내 고점 편차) → 저항 채널
        lower   : 회귀선 + min(window 내 저점 편차) → 지지 채널

    Parameters
    ----------
    df     : OHLCV DataFrame (high, low, close 필요)
    window : 회귀 계산 구간 (기본 100봉)

    Returns
    -------
    (center, upper, lower) : 각각 pd.Series
    """
    n = len(df)
    center_arr = np.full(n, np.nan)
    upper_arr  = np.full(n, np.nan)
    lower_arr  = np.full(n, np.nan)

    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    x      = np.arange(window, dtype=float)

    for i in range(window - 1, n):
        y = close[i - window + 1 : i + 1]

        # 선형 회귀: y = slope*x + intercept
        slope, intercept = np.polyfit(x, y, 1)

        # window 내 각 점의 회귀 예측값
        reg_vals = slope * x + intercept

        # 현재 바(window 마지막 점)의 회귀 예측값
        cur_reg = reg_vals[-1]

        # 고점/저점의 회귀선 대비 편차
        high_devs = high[i - window + 1 : i + 1] - reg_vals
        low_devs  = low[i - window + 1 : i + 1]  - reg_vals

        center_arr[i] = cur_reg
        upper_arr[i]  = cur_reg + float(np.max(high_devs))
        lower_arr[i]  = cur_reg + float(np.min(low_devs))

    idx = df.index
    return (
        pd.Series(center_arr, index=idx, name="reg_center"),
        pd.Series(upper_arr,  index=idx, name="reg_upper"),
        pd.Series(lower_arr,  index=idx, name="reg_lower"),
    )
