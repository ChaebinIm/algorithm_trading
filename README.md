# Crypto Algorithm Trading

바이낸스 기반 암호화폐 알고리즘 트레이딩 프로젝트.

데이터 수집 -> 백테스팅 -> Walk-Forward 검증 -> 라이브 트레이딩까지의 파이프라인.

## 프로젝트 구조

```
algorithm_trading/
├── core/                          # 백테스팅 프레임워크
│   ├── engine.py                  # 백테스트 엔진 (슬리피지, 수수료, 거래통계)
│   ├── indicators.py              # 기술 지표 (SuperTrend, MACD, CMF, EMA, RSI 등)
│   ├── models.py                  # 슬리피지, 수수료, 사이저 모델
│   └── types.py                   # Fill, TradeLog, BacktestReport
│
├── strategies/                    # 전략 모듈 (16종)
│   ├── __init__.py                # Strategy 베이스 + 레지스트리
│   ├── ensemble.py                # 앙상블 (3신호 다수결) — 종합 1위
│   ├── champion_v2.py             # 챔피언 v2 (OOS 검증 최강)
│   ├── supertrend_strategy.py     # SuperTrend 추세 추종
│   ├── adaptive_regime.py         # 시장 국면 자동 감지
│   ├── macd_volume.py             # MACD + 자금흐름
│   ├── multi_factor.py            # 7팩터 스코어링
│   ├── supertrend_ensemble.py     # SuperTrend 앙상블
│   ├── champion.py                # 챔피언 v1
│   ├── ensemble_short.py          # 롱/숏 양방향
│   ├── ma_cross.py                # 이동평균 교차 (벤치마크)
│   ├── trend_following.py         # 추세 추종
│   ├── dual_momentum.py           # 듀얼 모멘텀
│   ├── atr_breakout.py            # ATR 브레이크아웃
│   ├── bollinger_squeeze.py       # 볼린저 스퀴즈
│   ├── enhanced_trend.py          # 강화 추세 (세션 필터)
│   └── enhanced_breakout.py       # 강화 브레이크아웃 (야간 세션)
│
├── bot/                           # 실거래 봇
│   ├── config.py                  # 포트폴리오/리스크 설정
│   ├── exchange.py                # 바이낸스 API 래퍼
│   ├── risk_manager.py            # 리스크 관리
│   ├── notifier.py                # 텔레그램 알림
│   └── trader.py                  # 메인 트레이딩 루프
│
├── trading/                       # 라이브 트레이딩 (구버전)
│
├── collect_binance.py             # 바이낸스 USDT 데이터 수집
├── collect_upbit.py               # 업비트 KRW 데이터 수집
├── collect_data.py                # 데이터 수집 (구버전)
├── run_all_backtests.py           # 전체 전략 일괄 백테스트
├── run_backtest.py                # 단일 전략 백테스트 CLI
├── run_bot.py                     # 봇 실행 CLI
├── walk_forward.py                # Walk-Forward 검증
├── optimize_params.py             # 파라미터 최적화
├── portfolio_backtest.py          # 포트폴리오 시뮬레이션
├── analyze_sessions.py            # 시간대별 성과 분석
├── plot_equity.py                 # 자본곡선 비교 그래프
├── data/                          # 수집된 데이터 (git 미포함)
├── reports/                       # 백테스트 결과 히스토리
├── .env                           # API 키 (git 미포함)
└── .gitignore
```

## 시작하기

### 1. 환경 설정

```bash
pip install ccxt pandas numpy matplotlib python-dotenv pyarrow
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일 생성:
```
BINANCE_API_KEY="발급받은_API_KEY"
BINANCE_API_SECRET="발급받은_SECRET_KEY"
```

### 3. 데이터 수집

```bash
# 바이낸스 데이터
python collect_binance.py

# 업비트 KRW 데이터
python collect_upbit.py
```

### 4. 백테스트

```bash
# 전체 전략 일괄 실행
python run_all_backtests.py

# 단일 전략
python run_backtest.py --strategy ensemble --data data/ohlcv_full_BTCUSDT_4h.parquet

# 사용 가능한 전략 목록
python run_backtest.py --list

# Walk-Forward 검증
python walk_forward.py
```

### 5. 봇 실행

```bash
# 페이퍼 트레이딩 (실제 주문 없음)
python run_bot.py --paper --once

# 페이퍼 연속 실행
python run_bot.py --paper
```

## 백테스트 환경 (보수적 가정)

| 항목 | 값 | 비고 |
|------|-----|------|
| 초기 자본 | 10,000 USDT | |
| 슬리피지 | 5bps (0.05%) | 매수/매도 각각 불리하게 |
| 수수료 | 10bps (0.1%) | 바이낸스 기본 수수료 |
| 포지션 사이징 | 가용 현금의 20% | PercentSizer |
| 체결 | 다음 바 시가 | 미래 정보 사용 없음 |
| 연율화 | 365일 | 크립토 24/7 거래 |

## 전략 성과 TOP 5 (4h, 수수료 10bps 기준)

| 순위 | 전략 | 누적수익률 | 샤프 | MDD | 특징 |
|:---:|------|:---------:|:----:|:---:|------|
| 1 | ensemble | +50~184% | 1.0~1.7 | -7~10% | 종합 1위, 가장 안정적 |
| 2 | champion_v2 | +49~122% | 0.9~1.2 | -10~13% | OOS 검증 최강 |
| 3 | supertrend | +104% (평균) | 1.02 | -18% | 적은 거래, 높은 효율 |
| 4 | adaptive_regime | +135% (평균) | 0.99 | -18% | 하락장 자동 방어 |
| 5 | macd_volume | +56% (평균) | 0.87 | -11% | MDD 최저 |

## 핵심 인사이트

1. **4h 타임프레임이 최적** — 1h는 수수료에 갉아먹히고, 1d는 거래 기회 부족
2. **앙상블 철학이 효과적** — 3개 신호 중 2개 동의 시 진입, 노이즈 필터 역할
3. **2022 하락장 방어** — SMA(200) 필터로 현금 보유, BTC -65% 시 전략은 -2~10%
4. **수수료 민감도** — 거래 빈도 낮은 전략(ensemble, supertrend)이 수수료 변화에 강건

## 대상 코인

유동성 상위 대형 코인: BTC, ETH, SOL, XRP, BNB

## 새 전략 추가 방법

1. `strategies/` 폴더에 새 파일 생성
2. `Strategy` 상속, `prepare()` + `on_bar()` 구현
3. `strategies/__init__.py`의 `_build_registry()`에 등록
