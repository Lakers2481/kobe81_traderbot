# Data Pipeline Documentation

This document describes the data ingestion, processing, and feature engineering flow in the Kobe trading system.

## Overview

The data pipeline consists of three main stages:

1. **Data Ingestion** - Fetching and caching market data from external providers
2. **Data Lake** - Immutable storage with cryptographic integrity verification
3. **Feature Engineering** - Computing technical indicators and ML features

```
                              ┌─────────────────────────────────┐
                              │       External Providers         │
                              │  (Polygon, Stooq, Binance, etc) │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Ingestion Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ polygon_eod.py  │  │  stooq_eod.py   │  │ binance_klines  │         │
│  │  (EOD Equities) │  │ (Free Equities) │  │    (Crypto)     │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                │                                        │
│                                ▼                                        │
│                      ┌─────────────────┐                                │
│                      │   CSV Cache     │ (data/cache/)                  │
│                      └────────┬────────┘                                │
└───────────────────────────────┼─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Frozen Data Lake                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  manifest.py    │  │      io.py      │  │   SHA256 Hash   │         │
│  │ (Dataset IDs)   │  │ (Lake Read/Write│  │  Verification   │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           └────────────────────┼────────────────────┘                   │
│                                │                                        │
│                          data/lake/ (Parquet/CSV)                       │
└───────────────────────────────┼─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Feature Engineering                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  technical_     │  │    anomaly_     │  │   feature_      │         │
│  │  features.py    │  │  detection.py   │  │   pipeline.py   │         │
│  │ (150+ Indicators)│ │ (Stumpy/MatrixP)│  │ (Normalization) │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           └────────────────────┼────────────────────┘                   │
│                                │                                        │
│                       Feature DataFrame                                 │
└───────────────────────────────┼─────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Strategy/ML Models  │
                    │ (signals, backtesting)│
                    └───────────────────────┘
```

---

## 1. Data Ingestion Layer

### Module: `data/providers/`

The data ingestion layer provides unified access to multiple data sources.

#### Polygon EOD (`polygon_eod.py`)

**Primary data source** for production use. Requires `POLYGON_API_KEY` environment variable.

```python
from data.providers.polygon_eod import fetch_daily_bars_polygon

# Fetch AAPL daily bars with caching
df = fetch_daily_bars_polygon(
    symbol='AAPL',
    start='2020-01-01',
    end='2024-12-31',
    cache_dir=Path('data/cache'),
)
```

**Features:**
- 24-hour cache TTL to minimize API calls
- Rate limiting (0.3s between requests)
- Automatic CSV caching
- Returns standardized OHLCV DataFrame

**Output Schema:**
| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Bar timestamp (UTC) |
| symbol | str | Ticker symbol |
| open | float | Open price |
| high | float | High price |
| low | float | Low price |
| close | float | Close price (adjusted) |
| volume | int | Trading volume |

#### Alternative Providers (No API Key Required)

| Provider | Module | Asset Class | Use Case |
|----------|--------|-------------|----------|
| Stooq | `stooq_eod.py` | US Equities | Free backtesting |
| Yahoo Finance | `yfinance_eod.py` | US Equities | Fallback |
| Binance | `binance_klines.py` | Crypto | USDT pairs |

### Module: `data/universe/loader.py`

Loads trading universes (symbol lists) from CSV files.

```python
from data.universe.loader import load_universe

# Load 900-stock optionable universe
symbols = load_universe('data/universe/optionable_liquid_900.csv', cap=100)
```

**Features:**
- Automatic uppercase normalization
- Deduplication while preserving order
- Optional cap on number of symbols
- Handles missing 'symbol' column gracefully

---

## 2. Frozen Data Lake

### Module: `data/lake/`

The data lake provides **immutable datasets** with cryptographic integrity verification.

#### Key Principles

1. **Immutability**: Once a `dataset_id` exists, the data NEVER changes
2. **Deterministic IDs**: Computed from provider, timeframe, date range, universe hash
3. **Integrity**: SHA256 hashes verify data hasn't drifted
4. **Reproducibility**: Same dataset_id always returns identical data

#### Dataset Manifest (`manifest.py`)

```python
from data.lake.manifest import DatasetManifest

# Create a frozen dataset manifest
manifest = DatasetManifest.create(
    provider='stooq',
    timeframe='1d',
    start_date='2015-01-01',
    end_date='2024-12-31',
    universe_path='data/universe/optionable_liquid_900.csv',
    schema_version='v1.0',
)
# manifest.dataset_id = 'stooq_1d_2015_2024_a1b2c3...'
```

#### Lake I/O (`io.py`)

```python
from data.lake.io import LakeWriter, LakeReader

# Write data to lake
writer = LakeWriter(base_dir='data/lake')
writer.write(df, manifest)

# Read data from lake with integrity check
reader = LakeReader(base_dir='data/lake')
df = reader.read(dataset_id='stooq_1d_2015_2024_a1b2c3')
```

#### Directory Structure

```
data/lake/
├── stooq_1d_2015_2024_abc123/
│   ├── manifest.json          # Dataset metadata + file hashes
│   ├── AAPL.parquet           # Symbol data files
│   ├── MSFT.parquet
│   └── ...
└── binance_1h_2020_2024_def456/
    ├── manifest.json
    ├── BTCUSDT.parquet
    └── ...
```

---

## 3. Feature Engineering

### Module: `ml_features/`

The feature engineering layer computes 150+ technical indicators and ML features.

#### Feature Pipeline (`feature_pipeline.py`)

**Unified interface** for all feature extraction:

```python
from ml_features.feature_pipeline import FeaturePipeline, FeatureConfig, FeatureCategory

# Configure feature extraction
config = FeatureConfig(
    categories=[FeatureCategory.MOMENTUM, FeatureCategory.VOLATILITY],
    scaling_method=ScalingMethod.STANDARD,
    fill_method='ffill',
)

# Extract features
pipeline = FeaturePipeline(config)
features_df = pipeline.extract(price_df)
```

**Feature Categories:**
| Category | Examples | Count |
|----------|----------|-------|
| MOMENTUM | RSI, MACD, Stochastic, Williams %R | ~30 |
| VOLATILITY | ATR, Bollinger Bands, Keltner Channels | ~20 |
| TREND | SMA, EMA, ADX, Supertrend | ~25 |
| VOLUME | OBV, VWAP, MFI, AD | ~15 |
| ANOMALY | Matrix Profile anomaly scores | ~10 |
| PRICE_PATTERN | Candlestick patterns, gaps | ~50 |

#### Technical Features (`technical_features.py`)

Computes 150+ indicators using `pandas-ta`:

```python
from ml_features.technical_features import TechnicalFeatures, TechnicalConfig

config = TechnicalConfig(
    rsi_periods=[2, 7, 14],      # Multiple RSI lookbacks
    atr_periods=[7, 14, 21],     # Multiple ATR lookbacks
    sma_periods=[10, 20, 50, 200],  # Moving averages
)

tech = TechnicalFeatures(config)
features_df = tech.compute(price_df)
```

**Anti-Lookahead**: All indicators are **shifted by 1 bar** to prevent lookahead bias. The signal at time `t` uses only data from `t-1` and earlier.

#### Anomaly Detection (`anomaly_detection.py`)

Uses Matrix Profile (via `stumpy`) for unsupervised anomaly detection:

```python
from ml_features.anomaly_detection import AnomalyDetector, AnomalyConfig

detector = AnomalyDetector(AnomalyConfig(window_size=20))
anomaly_scores = detector.compute(price_df)
```

---

## 4. Data Quality Gate

### Module: `preflight/data_quality.py`

Validates data before backtesting to catch issues early.

**Checks Performed:**
- Minimum history requirement (5 years)
- Gap detection (max 5% missing days)
- OHLC integrity (High >= Low, etc.)
- Staleness check (data age)
- Corporate action verification

```python
from preflight.data_quality import validate_data_quality

report = validate_data_quality(dataset_id='stooq_1d_2015_2024_abc123')
if not report.passed:
    print(f"Data quality issues: {report.issues}")
```

---

## 5. Data Flow for Common Operations

### Backtesting Workflow

```python
# 1. Load universe
symbols = load_universe('data/universe/optionable_liquid_900.csv')

# 2. Fetch/cache price data
for symbol in symbols:
    df = fetch_daily_bars_polygon(symbol, start, end, cache_dir='data/cache')

# 3. Freeze to data lake
manifest = DatasetManifest.create(...)
writer.write(combined_df, manifest)

# 4. Extract features (with anti-lookahead)
features = pipeline.extract(price_df)

# 5. Run strategy
signals = strategy.generate_signals(features)
```

### Live Trading Workflow

```python
# 1. Fetch fresh data (ignore cache TTL)
df = fetch_daily_bars_polygon(symbol, start, end, ignore_cache_ttl=False)

# 2. Validate freshness
if is_data_stale(df):
    raise DataQualityError("Stale data")

# 3. Extract features
features = pipeline.extract(df)

# 4. Generate signals
signals = strategy.generate_signals(features)
```

---

## 6. Configuration

Data pipeline settings are in `config/base.yaml`:

```yaml
data:
  provider: "polygon"
  cache_dir: "data/cache"
  universe_file: "data/universe/optionable_liquid_900.csv"
```

---

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `data/providers/polygon_eod.py` | Polygon EOD data fetching |
| `data/providers/stooq_eod.py` | Free Stooq data (no API key) |
| `data/providers/binance_klines.py` | Binance crypto klines |
| `data/universe/loader.py` | Symbol universe loading |
| `data/lake/manifest.py` | Frozen dataset manifests |
| `data/lake/io.py` | Lake read/write operations |
| `ml_features/technical_features.py` | 150+ technical indicators |
| `ml_features/feature_pipeline.py` | Unified feature extraction |
| `ml_features/anomaly_detection.py` | Matrix Profile anomalies |
| `preflight/data_quality.py` | Data validation checks |

---

## 8. Best Practices

1. **Always use cached data for backtesting** - Set `ignore_cache_ttl=True`
2. **Freeze datasets before major experiments** - Use the data lake
3. **Validate data quality before production runs** - Use preflight checks
4. **Keep universes version-controlled** - Track `optionable_liquid_900.csv`
5. **Monitor API rate limits** - Polygon has 5 calls/minute on free tier
