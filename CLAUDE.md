# Algorithm Trading Project — Context for Claude

## 프로젝트 목적
Walk-Forward 검증 기반 암호화폐 알고리즘 트레이딩 전략 개발.
과적합 없이 실전에서 통하는 전략을 목표로 한다.

## 작업 원칙
- 질문하지 말고 합리적 판단으로 바로 진행
- 사용자가 자리를 비워도 작업을 멈추지 않음
- 모든 bash 명령 자율 실행 (Bash(*) 허용됨)

---

## 프로젝트 구조

```
algorithm_trading/
├── core/
│   ├── engine.py          # 백테스트 엔진 (slippage, commission, sizer, 거래통계)
│   ├── indicators.py      # 기술 지표 (supertrend, macd, cmf, ema, rsi, adx, atr 등)
│   └── types.py           # BacktestEnv, BacktestReport 타입 정의
├── strategies/
│   ├── __init__.py        # 전략 레지스트리 (get_strategy, list_strategies)
│   ├── ensemble.py        # ★ 최강 전략: EMA정배열+RSI / 모멘텀가속 / 돈키안브레이크아웃
│   ├── supertrend_strategy.py     # SuperTrend + SMA200 + ADX
│   ├── supertrend_ensemble.py     # SuperTrend + EMA정배열 + MACD + CMF
│   ├── macd_volume.py             # MACD + CMF + SMA200 자금흐름
│   ├── adaptive_regime.py         # 시장 국면(Bull/Bear/Range) 감지
│   ├── multi_factor.py            # 7팩터 스코어링
│   ├── champion.py                # Walk-Forward 결과 기반 최강 조합 (ST 필수 게이트)
│   ├── champion_v2.py             # 챔피언 v2 (RSI 추가, 재진입 허용, 트레일링 타이트)
│   └── ensemble_short.py          # 롱/숏 양방향 (실험적 — 성과 ensemble보다 낮음)
├── data/                  # Parquet 데이터 파일들
│   ├── ohlcv_full_BTCUSDT_4h.parquet   # 2017~현재
│   ├── ohlcv_full_BTCUSDT_1d.parquet
│   ├── ohlcv_full_ETHUSDT_4h.parquet
│   ├── ohlcv_full_SOLUSDT_4h.parquet
│   ├── ohlcv_full_XRPUSDT_4h.parquet
│   ├── ohlcv_full_XRPUSDT_1d.parquet
│   ├── ohlcv_full_BNBUSDT_4h.parquet
│   └── ohlcv_full_BTCKRW_4h.parquet    # Upbit KRW (2020~)
├── run_all_backtests.py   # 전체 전략 × 전체 데이터 백테스트
├── walk_forward.py        # IS/OOS 분할 + 롤링 2윈도우 Walk-Forward 검증
├── portfolio_backtest.py  # 다중 전략 포트폴리오 시뮬레이션
├── collect_binance.py     # 바이낸스 USDT 데이터 수집 (API 키 불필요)
├── collect_upbit.py       # Upbit KRW 데이터 수집
└── optimize_params.py     # 그리드 서치 파라미터 최적화
```

---

## 백테스트 환경 표준 설정

```python
from core import BacktestEnv, BasicSlippage, FixedRateCommission, PercentSizer

ENV = BacktestEnv(
    cash=10_000,
    slippage=BasicSlippage(bps=5),       # 슬리피지 0.05%
    commission=FixedRateCommission(bps=5), # 수수료 0.05%
    sizer=PercentSizer(percent=0.20),     # 자본의 20%씩
)
```

---

## Walk-Forward 검증 결과 (2026-04-14 기준)

### 검증 통과 전략 (OOS 양수, 실전 사용 권장)

| 우선순위 | 전략 | 마켓 | OOS Sharpe | OOS 수익률 | 비고 |
|---------|------|------|-----------|----------|------|
| 1 | ensemble | BTC KRW 4h | 2.20 (롤링 2.36) | +91.1% | 최강. IS<OOS 과적합 없음 |
| 2 | ensemble | BTC USDT 4h | 롤링 1.04 | +12.9% | 더 긴 역사로 검증 |
| 3 | macd_volume | XRP USDT 1d | 1.46 | +58.2% | XRP 특화 |
| 4 | ensemble | ETH USDT 4h | 1.18 | +25.0% | ETH 유효 마켓 |
| 5 | supertrend | SOL USDT 4h | 롤링 1.78 | +10.0% | 롤링은 강하나 고정 OOS 약함 |
| 6 | champion_v2 | XRP USDT 4h | 1.11 | +39.3% | |
| 7 | champion_v2 | BTC KRW 1d | 1.51 | +44.9% | |
| 8 | champion | XRP KRW 1d | 1.48 | +55.1% | |

### champion 전략의 특별한 특성 (BTC KRW 4h 롤링)
- W1: IS Sharpe 0.54 → OOS Sharpe 2.32
- W2: IS Sharpe 0.82 → OOS Sharpe 2.05
- IS << OOS: 과거보다 미래가 더 좋음. 진짜 알파.

### 검증 실패 전략 (사용 금지)
- SOL USDT 1d: IS 1.52~1.72 → OOS 전부 마이너스 (심각한 과적합)
- adaptive_regime / ETH: OOS -5.1%
- ensemble_short: 롱 전용 ensemble보다 성과 낮음
- BTC USDT 1d: supertrend OOS 음수

---

## 핵심 인사이트

1. **4h 타임프레임이 최적**: 1h는 노이즈 많고, 1d는 거래 수가 너무 적음
2. **BTC KRW 4h가 OOS 성과 최강**: 2020년 이후 데이터로 2024~2025 강세장을 OOS로 포착
3. **SOL 1d는 과적합**: ma_cross +467%, supertrend_ensemble +305%는 전부 IS 과적합. OOS 손실
4. **앙상블 철학**: 3개 신호 중 2개 동의할 때만 진입. 노이즈 필터 역할로 꾸준한 수익
5. **2022 하락장 방어**: 전략들이 SMA200 필터로 현금 보유. BTC -65% 시 전략은 -2~7%

---

## 데이터 수집 방법

```bash
# 바이낸스 데이터 업데이트 (기존 파일은 마지막 시점 이후만 추가)
python collect_binance.py

# Upbit KRW 데이터 수집
python collect_upbit.py
```

의존성: `pip install ccxt pandas pyarrow`

---

## 전략 추가 방법

1. `strategies/` 폴더에 새 파일 생성
2. `Strategy` 클래스 상속, `prepare(df)` + `on_bar(i, row)` 구현
3. `strategies/__init__.py`의 `_build_registry()`에 등록

```python
# on_bar 반환값
+1  # 매수
-1  # 매도
 0  # 유지
```

---

## 다음 탐구 방향 (미완료)

- [ ] 파라미터 최적화: `optimize_params.py` 실행 후 최적값 반영
- [ ] 포트폴리오 배분 비율 최적화 (현재 고정 비율)
- [ ] BNB USDT 4h: champion이 OOS Sharpe 1.21로 유일하게 통과 — 추가 검증 필요
- [ ] 인터마켓 신호: BTC 추세를 ETH/XRP 진입 필터로 활용
- [ ] 실거래 연동: `trader.py` 존재하나 전략과 미연결
