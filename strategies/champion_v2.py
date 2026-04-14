"""
Champion v2 — Walk-Forward 결과를 반영한 2세대 전략.

Champion v1 분석 결과:
  BTC 4h 롤링 OOS: Sharpe 2.19 (IS=0.54~0.82 vs OOS=2.05~2.32)
  → IS << OOS: 과적합 없음, 진짜 알파 확인
  → 약점: ST 방향 전환 시점에만 진입 → 진입 기회 너무 적음 (IS 샤프 낮음)

Champion v2 개선사항:
  1. 진입 기회 확대: ST 방향 전환 + '지속 추세 재진입' 허용
     - ST가 +1 상태로 N개 이상 바 유지 AND 모든 신호 최강
     - 마지막 청산이 트레일링 스탑(추세 유지)이면 재진입 허용
  2. 신호 품질 강화: RSI 50+ 추가 (상승 모멘텀 확인)
  3. 추세 강도: ST +1 지속 기간으로 추세 품질 필터
  4. 포지션 관리: 트레일링 스탑을 더 타이트하게 (2.0x ATR)

검증 대상:
  - BTC 4h, XRP 1d (v1에서 강한 OOS 성과)
  - ETH 4h, ETH 1h

신호 구성 (5개):
  A. SuperTrend 방향 +1 (필수)
  B. Ensemble: EMA 정배열 + 모멘텀 OR 돈키안 돌파
  C. MACD hist > 0 AND CMF > 0
  D. 200 SMA 위 (구조적 상승)
  E. RSI > 50 (상승 모멘텀)
진입: A(필수) + B/C/D/E 중 2개 이상 (총 3개 이상)
청산: ST 전환 OR ATR 트레일링
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import (
    ema, sma, atr, adx, momentum,
    supertrend, macd, cmf, donchian_channel,
)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


class ChampionV2Strategy(Strategy):
    """Champion v2 — 진입 기회 확대 + RSI 필터 추가.

    v1 대비 변경사항:
    - ST 방향 전환 뿐만 아니라 '지속 추세 재진입'도 허용
    - RSI 50+ 필터로 상승 모멘텀 확인
    - min_signals를 3으로 상향 (5개 중 3개)
    - 트레일링 스탑 2.0x로 타이트하게
    """

    name = "champion_v2"

    def __init__(
        self,
        # SuperTrend
        st_period: int = 10,
        st_multiplier: float = 2.5,
        # 장기 추세
        sma_period: int = 200,
        # Ensemble 서브 신호
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow_trend: int = 120,
        mom_period: int = 20,
        channel_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        # MACD + CMF
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cmf_period: int = 20,
        # RSI 필터
        rsi_period: int = 14,
        rsi_threshold: float = 50.0,
        # ADX 필터
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        # 진입 기준
        min_signals: int = 3,         # 5개 중 최소 3개 (ST 필수 포함)
        st_sustained_bars: int = 5,   # 지속 추세 재진입: ST+1 유지 최소 바 수
        # 리스크 관리
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 2.0,   # v1보다 타이트
    ):
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.sma_period = sma_period
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow_trend = ema_slow_trend
        self.mom_period = mom_period
        self.channel_period = channel_period
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.cmf_period = cmf_period
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.min_signals = min_signals
        self.st_sustained_bars = st_sustained_bars
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0
        self._last_exit_trailing = False  # 마지막 청산이 트레일링이었는지

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # SuperTrend
        st_line, st_dir = supertrend(df, self.st_period, self.st_multiplier)
        df["st_direction"] = st_dir
        df["st_dir_prev"] = st_dir.shift(1)

        # ST 연속 양수 바 카운트
        st_positive = (st_dir == 1).astype(int)
        # Rolling count of consecutive +1 bars
        streak = []
        cnt = 0
        for v in st_positive:
            if v == 1:
                cnt += 1
            else:
                cnt = 0
            streak.append(cnt)
        df["st_streak"] = streak

        # SMA 필터
        df["sma200"] = sma(df["close"], self.sma_period)

        # EMA 정배열
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow_t"] = ema(df["close"], self.ema_slow_trend)

        # 모멘텀
        df["mom"] = momentum(df["close"], self.mom_period)

        # 돈키안 채널
        dc_upper, _ = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)
        df["vol_ma"] = sma(df["volume"], self.vol_period)

        # MACD + CMF
        _, _, hist = macd(df["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        df["macd_hist"] = hist
        df["cmf_val"] = cmf(df, self.cmf_period)

        # RSI
        df["rsi"] = _rsi(df["close"], self.rsi_period)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def _count_signals(self, row: pd.Series) -> int:
        """5개 서브 신호 카운트 (ST 포함)."""
        close = row["close"]
        count = 0

        # A. SuperTrend 방향 (필수)
        if int(row.get("st_direction", -1)) == 1:
            count += 1

        # B. Ensemble: EMA 정배열 + 모멘텀 OR 돈키안 돌파
        ema_f = row.get("ema_fast", close)
        ema_m = row.get("ema_mid", close)
        ema_s = row.get("ema_slow_t", close)
        mom_val = row.get("mom", 0)
        dc_upper = row.get("dc_upper", 0)
        volume = row.get("volume", 0)
        vol_ma = row.get("vol_ma", 0)

        ensemble_buy = (
            (ema_f > ema_m > ema_s and mom_val > 0)
            or (close > dc_upper and volume > vol_ma * self.vol_mult)
        )
        if ensemble_buy:
            count += 1

        # C. MACD 양수 + CMF 양수
        if row.get("macd_hist", -1) > 0 and row.get("cmf_val", -1) > 0:
            count += 1

        # D. 200 SMA 위
        if close > row.get("sma200", 0):
            count += 1

        # E. RSI > 50 (상승 모멘텀)
        if row.get("rsi", 0) > self.rsi_threshold:
            count += 1

        return count

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 1.0)
        adx_val = row.get("adx", 0.0)
        st_dir = int(row.get("st_direction", -1))
        st_dir_prev = int(row.get("st_dir_prev", -1))
        st_streak = int(row.get("st_streak", 0))

        # 포지션 보유 중
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # SuperTrend 방향 전환 → 즉시 청산
            if st_dir == -1:
                self._in_position = False
                self._last_exit_trailing = False
                return -1

            # 손절
            if close < self._entry_price - self.sl_atr_mult * atr_val:
                self._in_position = False
                self._last_exit_trailing = True
                return -1

            # 트레일링 스탑
            if close < self._highest - self.trailing_atr_mult * atr_val:
                self._in_position = False
                self._last_exit_trailing = True
                return -1

            return 0

        # 미보유: ADX 필터
        if adx_val < self.adx_threshold:
            return 0

        count = self._count_signals(row)
        st_positive = (st_dir == 1)
        st_flipped = (st_dir_prev == -1) and (st_dir == 1)

        if not st_positive or count < self.min_signals:
            return 0

        # 진입 조건 1: ST 방향 전환 시점
        if st_flipped:
            self._in_position = True
            self._entry_price = close
            self._highest = close
            self._last_exit_trailing = False
            return +1

        # 진입 조건 2: 지속 추세 재진입 (트레일링 스탑 후)
        # ST가 충분히 오랫동안 양수 상태 + 신호 최강 + 마지막 청산이 트레일링
        if (
            self._last_exit_trailing
            and st_streak >= self.st_sustained_bars
            and count >= self.min_signals + 1   # 더 강한 조건 (4/5 이상)
        ):
            self._in_position = True
            self._entry_price = close
            self._highest = close
            self._last_exit_trailing = False
            return +1

        return 0
