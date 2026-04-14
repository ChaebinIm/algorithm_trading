# Crypto Algorithm Trading

바이낸스 기반 암호화폐 알고리즘 트레이딩 프로젝트.

데이터 수집 → 백테스팅 → 전략 검증 → 라이브 트레이딩까지의 파이프라인을 구축한다.

## 프로젝트 구조

```
algorithm_trading/
├── core/                        # 백테스팅 프레임워크 핵심
│   ├── types.py                 # Fill, TradeLog, BacktestReport
│   ├── indicators.py            # SMA, EMA, RSI, ATR, ADX
│   ├── models.py                # 슬리피지, 수수료, 사이저
│   └── engine.py                # run_backtest, BacktestEnv
│
├── strategies/                  # 전략 모듈 (파일 1개 = 전략 1개)
│   ├── __init__.py              # Strategy 베이스 클래스 + 레지스트리
│   ├── ma_cross.py              # 이동평균 교차 전략
│   └── trend_following.py       # 다중 필터 추세 추종 전략
│
├── trading/                     # 라이브 트레이딩
│   ├── risk.py                  # 리스크 관리 설정
│   └── trader.py                # LiveTrader (드라이런/실매매)
│
├── collect_data.py              # 바이낸스 OHLCV 데이터 수집
├── run_backtest.py              # 백테스트 CLI 진입점
├── data/                        # 수집된 데이터 (git 미포함)
├── .env                         # API 키 (git 미포함)
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
# 전체 수집 (BTC, ETH, SOL, XRP × 1m, 5m, 1h, 4h, 1d)
python collect_data.py

# 특정 심볼/타임프레임만
python collect_data.py --symbols BTC/USDT --timeframes 1h 4h
```

### 4. 백테스트 실행

```bash
# 추세 추종 전략
python run_backtest.py --strategy trend_following

# MA 교차 전략, 다른 데이터
python run_backtest.py --strategy ma_cross --data data/ohlcv_full_ETHUSDT_4h.parquet

# 사용 가능한 전략 목록
python run_backtest.py --list
```

## 새 전략 추가 방법

1. `strategies/` 폴더에 새 파일 생성 (예: `bollinger_band.py`)
2. `Strategy`를 상속한 클래스를 정의하고 `prepare()`, `on_bar()` 구현
3. `strategies/__init__.py`의 `_build_registry()`에 등록

## 대상 코인

유동성 상위 코인만 다룬다: BTC, ETH, SOL, XRP

## 작업 규칙

- `main` 브랜치는 항상 안정 상태 유지
- 작업은 `feature/이름` 브랜치에서 진행 후 PR로 병합
- 커밋은 자주, 의미 단위로
