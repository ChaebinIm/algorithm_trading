"""
인터마켓 모멘텀 전략 — BTC 신호 기반 ALT 진입.

설계 철학:
  BTC 강한 장대양봉(ATR×1.5+) 당일, ETH/BNB도 양봉으로 확인되는 경우
  시장 전체가 강세임을 동시에 확인하고 ALT에 진입.
  BTC가 ETH/BNB에 '선행'하지 않으므로 다음날 진입은 엣지 없음.
  '동시 확인'이 핵심.

리서치 근거:
  - BTC lag=1 상관계수 -0.06 → 다음날 진입 근거 없음
  - BTC ATR×1.5 강한장대양봉 + SMA200 + ALT 당일 양봉: 5d +4.16%, 승률 65%
  - btc_mult=1.5가 그리드 서치 상위 10개 전부 선점
  - SMA200 필터: 필수 (하락장 신호 완전 차단)

진입:
  - BTC 당일 body >= ATR × btc_mult (장대양봉)
  - BTC close > SMA200 (상승 추세)
  - ALT 당일 body >= atr × alt_min_mult (ALT도 양봉 확인)
  - 위 조건 모두 만족 시 ALT 당일 종가에 진입

청산:
  1. close >= entry × (1 + tp_pct)  (익절)
  2. close <= entry × (1 - sl_pct)  (손절)
  3. max_bars 초과                   (시간 청산)
  4. 트레일링 스탑 (trailing_pct > 0)
"""
from __future__ import annotations

import os

import pandas as pd
import numpy as np

from strategies import Strategy
from core.indicators import atr as calc_atr, sma


class IntermarketMomentumStrategy(Strategy):
    """BTC 강세 동시 확인 후 ALT 당일 종가 진입 인터마켓 모멘텀 전략."""

    name = "intermarket_momentum"

    def __init__(
        self,
        btc_data_path: str = "data/ohlcv_full_BTCUSDT_1d.parquet",
        btc_mult: float = 1.5,        # BTC 장대양봉 기준 (그리드서치 상위 10개 전부 1.5)
        alt_min_mult: float = 0.0,    # ALT 양봉 최소 크기 (0=단순 양봉이면 OK)
        atr_period: int = 14,
        tp_pct: float = 0.04,         # 익절 4% (ETH 최적, BNB는 8% 권장)
        sl_pct: float = 0.03,         # 손절 3% (BNB는 4% 권장)
        trailing_pct: float = 0.0,    # 트레일링 (기본 비활성)
        max_bars: int = 5,            # 최대 보유일 (BNB는 7 권장)
        sma_period: int = 200,        # BTC SMA200 필터 (필수 — 하락장 완전 차단)
    ):
        # 검증된 파라미터 (백테스트 결과):
        # ETH: btc_mult=1.5, tp=4%, sl=3%, max_bars=5  → OOS Sharpe 0.88, 승률 73%
        # BNB: btc_mult=1.5, tp=8%, sl=4%, max_bars=7  → OOS Sharpe 0.84, 승률 61%
        self.btc_data_path = btc_data_path
        self.btc_mult = btc_mult
        self.alt_min_mult = alt_min_mult
        self.atr_period = atr_period
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.trailing_pct = trailing_pct
        self.max_bars = max_bars
        self.sma_period = sma_period

        # 포지션 상태
        self._position = 0
        self._entry_price = 0.0
        self._peak_price = 0.0
        self._bars_held = 0

    # ── 준비 ─────────────────────────────────────────────────────────────

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """df는 ETH 또는 BNB 일봉 데이터"""
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # BTC 데이터 경로 처리 (절대 경로 또는 상대 경로 fallback)
        btc_path = self.btc_data_path
        if not os.path.isabs(btc_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            btc_path = os.path.join(base_dir, btc_path)

        # BTC 일봉 로드
        btc = pd.read_parquet(btc_path)
        if not isinstance(btc.index, pd.DatetimeIndex):
            btc.index = pd.to_datetime(btc.index)

        # 만약 BTC가 4h 데이터라면 1d로 리샘플
        if len(btc) > 0:
            # 인덱스 빈도 추정: 평균 간격이 6시간 미만이면 intraday
            time_diffs = btc.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            if median_diff < pd.Timedelta("12h"):
                btc = btc.resample("1D").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna(subset=["close"])

        # BTC 지표 계산
        btc["atr"] = calc_atr(btc, self.atr_period)
        btc["body"] = btc["close"] - btc["open"]
        if self.sma_period > 0:
            btc["sma200"] = btc["close"].rolling(self.sma_period).mean()
        else:
            btc["sma200"] = 0.0

        # BTC 신호: 강한 장대양봉 + SMA200 위
        btc["btc_signal"] = (
            (btc["body"] >= self.btc_mult * btc["atr"]) &
            (btc["body"] > 0) &
            (btc["close"] > btc["sma200"])
        )

        # ALT 데이터에 BTC 신호 매핑 (같은 날짜 — 동시 확인)
        # reindex로 날짜 매핑 후 ffill 방지: 정확히 같은 날짜만 매핑
        df["btc_signal"] = btc["btc_signal"].reindex(df.index).fillna(False)

        # ALT 지표
        df["atr"] = calc_atr(df, self.atr_period)
        df["body"] = df["close"] - df["open"]

        # ALT 진입 확인 조건 (당일 양봉)
        if self.alt_min_mult > 0:
            df["alt_confirm"] = (df["body"] > 0) & (df["body"] >= self.alt_min_mult * df["atr"])
        else:
            # alt_min_mult=0: 단순 양봉이면 OK (close > open)
            df["alt_confirm"] = df["body"] > 0

        return df

    # ── 청산 ─────────────────────────────────────────────────────────────

    def _should_exit(self, row: pd.Series) -> bool:
        close = row["close"]
        entry = self._entry_price

        # 익절
        if close >= entry * (1 + self.tp_pct):
            return True
        # 손절
        if close <= entry * (1 - self.sl_pct):
            return True
        # 트레일링 스탑
        if self.trailing_pct > 0:
            trail = self._peak_price * (1 - self.trailing_pct)
            if close < trail:
                return True
        # 시간 청산
        if self._bars_held >= self.max_bars:
            return True

        return False

    # ── 메인 ─────────────────────────────────────────────────────────────

    def on_bar(self, i: int, row: pd.Series) -> int:
        close = row["close"]

        # ── 포지션 보유 중: 청산 조건 확인 ─────────────────────────────
        if self._position == 1:
            self._bars_held += 1
            if close > self._peak_price:
                self._peak_price = close
            if self._should_exit(row):
                self._position = 0
                self._entry_price = 0.0
                self._peak_price = 0.0
                self._bars_held = 0
                return -1
            return 0

        # ── 진입: BTC 신호 + ALT 확인 ───────────────────────────────────
        btc_ok = bool(row.get("btc_signal", False))
        alt_ok = bool(row.get("alt_confirm", False))

        if btc_ok and alt_ok:
            self._position = 1
            self._entry_price = close
            self._peak_price = close
            self._bars_held = 0
            return +1

        return 0
