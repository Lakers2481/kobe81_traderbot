# CHANGELOG.md - Version History

All notable changes to the Kobe Trading Robot are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

### Added
- Comprehensive documentation system (15+ files)
- Living documentation (WORKLOG, CONTRIBUTING)
- **Markov Chain Module** (`ml_advanced/markov_chain/`)
  - `StateClassifier`: Discretize returns into Up/Down/Flat states (threshold, percentile, volatility-adjusted)
  - `TransitionMatrix`: Build P(next|current) matrices with Laplace smoothing
  - `StationaryDistribution`: Compute equilibrium π for asset ranking (eigen, power, linear methods)
  - `HigherOrderMarkov`: 2nd/3rd order chains for multi-day patterns (Down→Down→Up bounces)
  - `MarkovPredictor`: Generate BUY/SELL/HOLD signals with confidence scoring
  - `MarkovAssetScorer`: Rank 800 stocks by π(Up), deviation, persistence scores
  - 38 unit tests (100% passing)
  - Complements existing HMM: HMM=hidden regimes, Markov=observable direction

### Changed
- None

### Fixed
- None

---

## [2.4.0] - 2026-01-04

### Added - Data Pipeline Optimization
- **Scanner Parallelization** (`scripts/scan.py`)
  - ThreadPoolExecutor with 10 workers for I/O-bound symbol fetching
  - ~5-10x speedup for 900-stock scans
- **Walk-Forward Parallelization** (`backtest/walk_forward.py`)
  - ProcessPoolExecutor with 4 workers for CPU-bound backtesting
  - Parallel split execution with proper result aggregation
- **Redis Streams** (`messaging/redis_pubsub.py`, `messaging/quote_broadcaster.py`)
  - Pub/sub messaging for real-time event propagation
  - Quote distribution infrastructure
- **DuckDB Analytics** (`analytics/duckdb_engine.py`)
  - High-performance OLAP queries (10-100x faster than pandas)
  - Direct Parquet/CSV querying with SQL interface
  - Pre-built trade analysis queries

### Added - Codex/Gemini Reliability Improvements (2026-01-04)
- **Strict OHLCV Validation** (`data/validation.py`)
  - 8 validation checks: schema, nulls, types, ranges, OHLC relationships, timestamps, gaps, anomalies
  - Severity levels: INFO, WARNING, ERROR, CRITICAL
  - `DataQualityReport` with hash verification
- **Multi-Source Data Quorum** (`data/quorum.py`)
  - Byzantine fault tolerance for market data
  - Polygon vs Stooq vs Yahoo Finance cross-validation
  - Weighted consensus with discrepancy detection (0.5% minor, 2% major, 10% critical)
  - `QuorumResult` with confidence scoring
- **Circuit Breaker Pattern** (`core/circuit_breaker.py`)
  - CLOSED/OPEN/HALF_OPEN state machine
  - Configurable failure threshold, recovery timeout
  - Pre-configured breakers for Polygon, Alpaca, Stooq, Yahoo Finance
  - `@circuit_protected` decorator for easy integration
- **LLM Output Validator** (`cognitive/llm_validator.py`)
  - Zero-trust validation for LLM-generated content
  - Price claim extraction and verification
  - Hallucination detection (overconfident language, unverifiable sources)
  - Grounding score calculation
- **Decision Reproducibility Packets** (`core/decision_packet.py`)
  - Complete audit trail for every trading decision
  - Captures: market snapshot, indicators, ML inputs/outputs, risk checks, signal, outcome
  - SHA256 hashing for integrity verification
  - JSON serialization for replay and analysis
- **Drift → Kill-Switch Escalation** (`scripts/runner.py`)
  - 3 consecutive CRITICAL drift detections triggers automatic kill-switch
  - Telegram alert before activation
  - Proper counter reset on recovery

### Added - Neural Integration Plan
- **LearningHub** (`integration/learning_hub.py`)
  - Central hub routing trade outcomes to learning pipeline
  - Connects EpisodicMemory, OnlineLearningManager, ReflectionEngine, SemanticMemory
- **24/7 Quant R&D Factory** (`autonomous/` enhancements)
  - Research coordinator for continuous discovery
  - Autonomous experiment execution

### Added - Macro Data Sources
- **FRED API** (`data/providers/fred_macro.py`) - Federal Reserve economic data
- **Treasury Yields** (`data/providers/treasury_yields.py`) - Yield curve data
- **CFTC COT** (`data/providers/cftc_cot.py`) - Commitment of Traders positioning
- **BEA API** (`data/providers/bea_macro.py`) - GDP, PCE data
- **EIA Energy** (`data/providers/eia_energy.py`) - Energy prices
- **Macro Features** (`ml_features/macro_features.py`) - Feature engineering

### Added - Brain Tool Enhancements
- **FAISS Vector Memory** (`cognitive/vector_memory.py`) - Fast similarity search
- **OR-Tools Optimizer** (`risk/advanced/portfolio_optimizer.py`) - Constraint optimization
- **Langfuse Tracer** (`observability/langfuse_tracer.py`) - LLM observability
- **AutoGen Team** (`agents/autogen_team.py`) - Multi-agent orchestration
- **LangGraph Coordinator** (`agents/langgraph_coordinator.py`) - Stateful agent flows

### Changed
- Updated STATUS.md with Section 27: CHANGELOG
- Updated requirements.txt with new dependencies (redis, duckdb, fredapi, faiss-cpu, sentence-transformers, langfuse, ortools)

### Performance
- 900-stock scan: 15-20 min → 2-3 min (parallelization)
- Walk-forward (15 splits): 75 min → 20 min (parallelization)
- Analytical queries: pandas → DuckDB (10-100x faster)

---

## [2.3.0] - 2026-01-03

### Added
- Full repository documentation audit
- `docs/REPO_MAP.md` - Complete directory tree
- `docs/ENTRYPOINTS.md` - All 180+ runnable scripts
- `docs/ARCHITECTURE.md` - Pipeline wiring proof
- `docs/READINESS.md` - Production readiness matrix
- `docs/KNOWN_GAPS.md` - Missing components tracker
- `docs/RISK_REGISTER.md` - Risk assessment (17 risks)
- `docs/START_HERE.md` - Onboarding guide
- `docs/ROBOT_MANUAL.md` - Complete system guide
- `docs/WORKLOG.md` - Work log index
- `docs/CONTRIBUTING.md` - Documentation rules

### Changed
- Updated CLAUDE.md with mandatory reading sections

### Fixed
- Documentation gaps identified and documented

---

## [2.2.1] - 2026-01-02

### Fixed
- **CRITICAL:** Position sizing bypass via manual orders
- Added dual cap enforcement (2% risk + 20% notional)
- See `docs/CRITICAL_FIX_20260102.md`

---

## [2.2.0] - 2026-01-01

### Added
- Bounce analysis database (10Y + 5Y, 1M+ events)
- `bounce/` module for streak analysis
- `tools/build_bounce_db.py` script
- Historical pattern integration

### Changed
- Quality gate threshold lowered to 55 (ML models not trained)

---

## [2.1.0] - 2025-12-31

### Added
- Pre-Game Blueprint system
- `analysis/historical_patterns.py`
- `analysis/options_expected_move.py`
- `explainability/trade_thesis_builder.py`
- `scripts/generate_pregame_blueprint.py`

### Changed
- ONE Scanner System - `scan.py` is the only scanner
- Deleted all other scan scripts

---

## [2.0.0] - 2025-12-29

### Added
- DualStrategyScanner (IBS+RSI + Turtle Soup combined)
- Professional Execution Flow (kill zones, watchlist)
- Walk-forward validation framework
- ML confidence scoring via `ml_meta/model.py`
- 70 skills in `.claude/skills/`

### Changed
- System audit: Grade A+ (100/100)
- 942 tests, 0 critical issues

### Fixed
- Lookahead bias eliminated (all indicators use `.shift(1)`)

---

## [1.5.0] - 2025-12-15

### Added
- Cognitive brain architecture (`cognitive/`)
- LSTM confidence model (`ml_advanced/lstm_confidence/`)
- HMM regime detector (`ml_advanced/hmm_regime_detector.py`)
- Ensemble predictor (`ml_advanced/ensemble/`)

### Changed
- Enhanced risk gates (PolicyGate, KillZoneGate, ExposureGate)

---

## [1.4.0] - 2025-12-01

### Added
- Synthetic options engine (`options/`)
- Frozen data lake (`data/lake/`)
- Multi-source data provider (`data/providers/multi_source.py`)

### Changed
- Improved position sizing with dual caps

---

## [1.3.0] - 2025-11-15

### Added
- ICT Turtle Soup strategy with sweep filter
- Kill zone gate for time-based blocking
- Weekly exposure gate

### Fixed
- Turtle Soup false signals (added 0.3 ATR filter)

---

## [1.2.0] - 2025-11-01

### Added
- IBS+RSI strategy
- Walk-forward backtesting
- Monte Carlo simulation

### Changed
- Refactored strategy interface

---

## [1.1.0] - 2025-10-15

### Added
- Alpaca broker integration
- IOC LIMIT order execution
- Kill switch mechanism
- Idempotency store

---

## [1.0.0] - 2025-10-01

### Added
- Initial release
- Backtesting engine
- Polygon.io data provider
- Basic risk management

---

## Version Format

- MAJOR: Breaking changes or significant architecture updates
- MINOR: New features, non-breaking changes
- PATCH: Bug fixes, documentation updates

---

## Related Documentation

- [STATUS.md](STATUS.md) - Current system status
- [WORKLOG.md](WORKLOG.md) - Detailed work notes
