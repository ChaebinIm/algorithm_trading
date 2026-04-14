"""
Champion 전략 — Walk-Forward 검증으로 선발된 최강 조합.

Walk-Forward 검증 결과 (OOS 기준):
  1위: ensemble / BTC 4h      → OOS Sharpe 2.20, +91.1%  (OOS > IS: 과적합 없음)
  2위: supertrend_ensemble / BTC 1d → OOS Sharpe 1.39, +65.8%
  3위: supertrend / BTC 4h    → OOS Sharpe 1.43, +55.1%  (OOS > IS!)
  4위: macd_volume / BTC 1d   → OOS Sharpe 1.31, +32.3%

Champion 전략 설계 원칙:
  1. OOS에서 검증된 신호 우선 사용
  2. 과적합 방지: 파라미터 최소화
  3. 여러 독립 신호로 진입 타이밍 개선
  4. 강력한 리스크 관리 (트레일링 스탑 + 손절)

Champion 전략 구성 (4개 서브 신호):
  A. SuperTrend 방향 (multiplier=2.5, 최적화 확인)
  B. Ensemble 다수결 (EMA 정배열 + 모멘텀 + 돈키안)
  C. MACD + CMF (자금 흐름 필터)
  D. 200 SMA 위 (구조적 추세 방향)

진입: A(필수) + B/C/D 중 1개 이상 (최소 2개)
  → 기존 ensemble(3/3)보다 유연하면서 신호 품질 유지
청산: SuperTrend 방향 전환 OR ATR 트레일링 스탑

타임프레임 권장: 4h (BTC), 1h (ETH)
"""
from __future__ import annotations

import pandas as pd

from strategies import Strategy
from core.indicators import (
    ema, sma, atr, adx, momentum,
    supertrend, macd, cmf, donchian_channel,
)


class ChampionStrategy(Strategy):
    """Walk-Forward 검증 기반 최강 전략.

    설계 철학:
    - SuperTrend를 필수 게이트로 사용 (방향이 틀리면 절대 진입 안 함)
    - 나머지 신호는 노이즈를 줄이는 보조 필터 역할
    - 4h 타임프레임에서 최적 (1h도 가능, 1d는 파라미터 조정 필요)
    - 과적합 방지를 위해 파라미터 수를 최소화
    """

    name = "champion"

    def __init__(
        self,
        # SuperTrend (핵심 게이트)
        st_period: int = 10,
        st_multiplier: float = 2.5,
        # 장기 추세 필터
        sma_period: int = 200,
        # Ensemble 서브 신호 (EMA 정배열 + 모멘텀 + 돈키안)
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow_trend: int = 120,
        mom_period: int = 20,
        channel_period: int = 20,
        vol_period: int = 20,
        vol_mult: float = 1.2,
        # MACD + CMF 서브 신호
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        cmf_period: int = 20,
        # ADX 필터
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        # 진입 기준
        min_signals: int = 2,   # 4개 중 최소 2개 (SuperTrend 필수 포함)
        # 리스크 관리
        sl_atr_mult: float = 2.0,
        trailing_atr_mult: float = 2.5,
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
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.min_signals = min_signals
        self.sl_atr_mult = sl_atr_mult
        self.trailing_atr_mult = trailing_atr_mult

        self._in_position = False
        self._entry_price = 0.0
        self._highest = 0.0

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # SuperTrend
        st_line, st_dir = supertrend(df, self.st_period, self.st_multiplier)
        df["st_direction"] = st_dir
        df["st_dir_prev"] = st_dir.shift(1)

        # SMA 필터
        df["sma200"] = sma(df["close"], self.sma_period)

        # EMA 정배열 (Ensemble 서브)
        df["ema_fast"] = ema(df["close"], self.ema_fast)
        df["ema_mid"] = ema(df["close"], self.ema_mid)
        df["ema_slow_t"] = ema(df["close"], self.ema_slow_trend)

        # 모멘텀 (Ensemble 서브)
        df["mom"] = momentum(df["close"], self.mom_period)

        # 돈키안 채널 (Ensemble 서브)
        dc_upper, _ = donchian_channel(df, self.channel_period)
        df["dc_upper"] = dc_upper.shift(1)
        df["vol_ma"] = sma(df["volume"], self.vol_period)

        # MACD + CMF (자금 흐름 서브)
        _, _, hist = macd(df["close"], self.macd_fast, self.macd_slow, self.macd_signal)
        df["macd_hist"] = hist
        df["cmf_val"] = cmf(df, self.cmf_period)

        # ADX
        df["adx"] = adx(df, self.adx_period)

        # ATR
        df["atr"] = atr(df, 14)

        return df

    def _count_signals(self, row: pd.Series) -> int:
        """4개 서브 신호 카운트 (SuperTrend 포함)."""
        close = row["close"]
        count = 0

        # A. SuperTrend 방향 (필수)
        if int(row.get("st_direction", -1)) == 1:
            count += 1

        # B. Ensemble 다수결: EMA 정배열 + 모멘텀 OR 돈키안 돌파
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

        # C. MACD 양수 + CMF 양수 (자금 흐름)
        if row.get("macd_hist", -1) > 0 and row.get("cmf_val", -1) > 0:
            count += 1

        # D. 200 SMA 위 (구조적 상승)
        if close > row.get("sma200", 0):
            count += 1

        return count

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]
        atr_val = row.get("atr", 1.0)
        adx_val = row.get("adx", 0.0)
        st_dir = int(row.get("st_direction", -1))
        st_dir_prev = int(row.get("st_dir_prev", -1))

        # 포지션 보유 중
        if self._in_position:
            if close > self._highest:
                self._highest = close

            # SuperTrend 방향 전환 → 즉시 청산
            if st_dir == -1:
                self._in_position = False
                return -1

            # 손절
            if close < self._entry_price - self.sl_atr_mult * atr_val:
                self._in_position = False
                return -1

            # 트레일링 스탑
            if close < self._highest - self.trailing_atr_mult * atr_val:
                self._in_position = False
                return -1

            return 0

        # 미보유: 진입 조건
        if adx_val < self.adx_threshold:
            return 0

        count = self._count_signals(row)
        st_positive = (st_dir == 1)
        st_flipped = (st_dir_prev == -1) and (st_dir == 1)

        # SuperTrend 필수 + 전체 min_signals 이상
        if st_positive and count >= self.min_signals:
            # 방향 전환 시점에 진입 (중복 진입 방지)
            if st_flipped:
                self._in_position = True
                self._entry_price = close
                self._highest = close
                return +1

        return 0
