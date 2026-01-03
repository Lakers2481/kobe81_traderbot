# REPO_MAP.md - Kobe Trading Robot Directory Structure

> **Last Updated:** 2026-01-03
> **Total Python Files:** 563 | **Scripts:** 180+ | **Tests:** 942

---

## Directory Tree (3 Levels Deep)

```
kobe81_traderbot/
|
|-- TIER 1: CORE TRADING SYSTEM
|   |-- strategies/              # Signal generation (7 files)
|   |   |-- dual_strategy/       # DualStrategyScanner (CANONICAL)
|   |   |   |-- combined.py      # IBS+RSI + Turtle Soup combined
|   |   |   |-- __init__.py
|   |   |-- ibs_rsi/             # Mean reversion strategy
|   |   |   |-- strategy.py      # IBS<0.08, RSI(2)<5
|   |   |   |-- __init__.py
|   |   |-- ict/                 # ICT/Smart Money strategies
|   |   |   |-- turtle_soup.py   # Liquidity sweep detection
|   |   |   |-- __init__.py
|   |   |-- adaptive_selector.py # Auto-switching
|   |   |-- registry.py          # Strategy registry (get_production_scanner)
|   |
|   |-- backtest/                # Backtesting engine (16 files)
|   |   |-- engine.py            # Core backtester
|   |   |-- walk_forward.py      # Walk-forward splits
|   |   |-- reproducibility.py   # Deterministic replay
|   |   |-- monte_carlo.py       # Monte Carlo simulation
|   |   |-- vectorized.py        # Vectorized operations
|   |   |-- triple_barrier.py    # Triple barrier exits
|   |   |-- gap_risk_model.py    # Gap detection
|   |
|   |-- risk/                    # Risk management (18 files)
|   |   |-- policy_gate.py       # Budget enforcement ($75/order, $1k/daily)
|   |   |-- kill_zone_gate.py    # ICT time blocks (9:30-10:00)
|   |   |-- weekly_exposure_gate.py # 40% weekly / 20% daily caps
|   |   |-- signal_quality_gate.py  # Score/confidence thresholds
|   |   |-- liquidity_gate.py    # Minimum volume
|   |   |-- equity_sizer.py      # 2% risk per trade
|   |   |-- dynamic_position_sizer.py # Adaptive sizing
|   |   |-- advanced/            # Advanced risk modules
|   |   |   |-- kelly_position_sizer.py  # Kelly Criterion
|   |   |   |-- monte_carlo_var.py       # VaR with stress testing
|   |   |   |-- correlation_limits.py    # Sector/concentration
|   |
|   |-- execution/               # Order execution (15 files)
|   |   |-- broker_alpaca.py     # Alpaca IOC LIMIT orders
|   |   |-- broker_paper.py      # Paper trading simulation
|   |   |-- broker_base.py       # Base broker interface
|   |   |-- order_manager.py     # Order lifecycle
|   |   |-- execution_guard.py   # Execution validation
|   |   |-- intelligent_executor.py # Smart order routing
|   |   |-- tca/                 # Transaction cost analysis
|   |       |-- transaction_cost_analyzer.py
|   |
|   |-- oms/                     # Order Management System (3 files)
|       |-- order_state.py       # Order state machine
|       |-- idempotency_store.py # Duplicate prevention (SQLite)
|
|-- TIER 2: DATA & UNIVERSE
|   |-- data/                    # Data handling (19 files)
|   |   |-- providers/           # Data sources
|   |   |   |-- polygon_eod.py   # Polygon.io EOD (primary)
|   |   |   |-- stooq_eod.py     # Stooq free EOD (fallback)
|   |   |   |-- yfinance_eod.py  # Yahoo Finance (fallback)
|   |   |   |-- multi_source.py  # Multi-source with fallback chain
|   |   |   |-- binance_klines.py # Crypto data
|   |   |   |-- alpaca_live.py   # Live Alpaca data
|   |   |-- lake/                # Frozen data lake
|   |   |   |-- manifest.py      # Dataset manifests (SHA256)
|   |   |   |-- io.py            # LakeReader/LakeWriter
|   |   |-- universe/            # Symbol management
|   |   |   |-- loader.py        # 900-stock universe loader
|   |   |   |-- canonical.py     # Canonical universe definition
|   |   |-- corporate_actions.py # Splits/dividends
|   |   |-- ml/                  # Market simulation
|       |   |-- generative_market_model.py
|
|-- TIER 3: ML & AI SYSTEMS
|   |-- cognitive/               # Brain architecture (24 files)
|   |   |-- cognitive_brain.py   # Main orchestrator
|   |   |-- metacognitive_governor.py # System 1/2 routing
|   |   |-- reflection_engine.py # Learning from outcomes
|   |   |-- episodic_memory.py   # Experience storage
|   |   |-- semantic_memory.py   # Generalized rules
|   |   |-- self_model.py        # Capability tracking
|   |   |-- knowledge_boundary.py # Uncertainty detection
|   |   |-- global_workspace.py  # Attention/workspace
|   |   |-- signal_adjudicator.py # Signal ranking
|   |   |-- curiosity_engine.py  # Hypothesis generation
|   |   |-- llm_trade_analyzer.py # LLM signal analysis
|   |   |-- llm_narrative_analyzer.py # Narrative generation
|   |
|   |-- ml_advanced/             # Advanced ML (12 files)
|   |   |-- ensemble/            # Multi-model ensemble
|   |   |   |-- ensemble_predictor.py
|   |   |   |-- regime_weights.py
|   |   |-- lstm_confidence/     # LSTM confidence
|   |   |   |-- model.py         # Multi-output LSTM
|   |   |-- hmm_regime_detector.py # 3-state HMM
|   |   |-- online_learning.py   # Incremental learning
|   |
|   |-- ml_features/             # Feature engineering (14 files)
|   |   |-- feature_pipeline.py  # Master feature generator (150+)
|   |   |-- technical_features.py # pandas-ta indicators
|   |   |-- signal_confidence.py # Confidence scoring
|   |   |-- pca_reducer.py       # PCA dimensionality reduction
|   |   |-- anomaly_detection.py # Matrix profiles
|   |   |-- regime_ml.py         # KMeans/GMM clustering
|   |   |-- ensemble_brain.py    # Multi-model prediction
|   |
|   |-- ml/                      # Alpha discovery & RL (19 files)
|   |   |-- alpha_discovery/
|   |   |   |-- rl_agent/        # Reinforcement learning
|   |   |   |   |-- agent.py     # PPO/DQN/A2C
|   |   |   |   |-- trading_env.py # Gym environment
|   |   |-- pattern_miner/       # Pattern mining
|   |   |-- feature_discovery/   # Feature importance
|   |
|   |-- ml_meta/                 # Meta-models (2 files)
|       |-- model.py             # GradientBoost per strategy
|       |-- features.py          # Feature computation
|
|-- TIER 4: MONITORING & HEALTH
|   |-- monitor/                 # System health (8 files)
|   |   |-- health_endpoints.py  # Health check API
|   |   |-- circuit_breaker.py   # Fault tolerance
|   |   |-- drift_detector.py    # Model drift detection
|   |   |-- divergence_monitor.py # Signal divergence
|   |   |-- calibration.py       # Model calibration
|   |   |-- heartbeat.py         # System heartbeat
|   |
|   |-- core/                    # Core utilities (21 files)
|       |-- hash_chain.py        # Tamper detection
|       |-- structured_log.py    # JSON event logging
|       |-- kill_switch.py       # Emergency halt
|       |-- clock/               # Market timing
|       |   |-- market_clock.py  # NYSE hours
|       |   |-- equities_calendar.py # NYSE calendar
|       |   |-- macro_events.py  # Economic events
|       |-- earnings_filter.py   # Earnings avoidance
|       |-- regime_filter.py     # Market regime gating
|       |-- rate_limiter.py      # API rate limiting
|       |-- config_pin.py        # Config validation
|
|-- TIER 5: ANALYSIS & EXPLAINABILITY
|   |-- explainability/          # Trade explanation (7 files)
|   |   |-- trade_thesis_builder.py # Full trade thesis
|   |   |-- trade_explainer.py   # Signal explanation
|   |   |-- narrative_generator.py # Narrative generation
|   |   |-- decision_tracker.py  # Decision tracking
|   |
|   |-- analysis/                # Pattern analysis (3 files)
|   |   |-- historical_patterns.py
|   |   |-- options_expected_move.py
|   |
|   |-- options/                 # Synthetic options (11 files)
|       |-- black_scholes.py     # BS pricing
|       |-- volatility.py        # Realized volatility
|       |-- selection.py         # Strike selection
|       |-- backtest.py          # Options backtest
|
|-- TIER 6: PREFLIGHT & VALIDATION
|   |-- preflight/               # Pre-trade validation (4 files)
|       |-- cognitive_preflight.py # AI decision validation
|       |-- data_quality.py      # Data quality checks
|       |-- evidence_gate.py     # Evidence sufficiency
|
|-- TIER 7: CONFIGURATION
|   |-- config/                  # Configuration (8 files)
|       |-- base.yaml            # Base settings
|       |-- base_backtest.yaml   # Backtest settings
|       |-- brokers.yaml         # Broker config
|       |-- trading_policies.yaml # Policy definitions
|       |-- frozen_strategy_params_v2.*.json # Immutable params
|       |-- settings_loader.py   # Config loading
|       |-- env_loader.py        # Environment loading
|
|-- TIER 8: UTILITY PACKAGES
|   |-- altdata/                 # Alternative data (7 files)
|   |   |-- congressional_trades.py
|   |   |-- insider_activity.py
|   |   |-- options_flow.py
|   |   |-- sentiment.py
|   |
|   |-- analytics/               # Trade analytics (4 files)
|   |   |-- edge_decomposition.py
|   |   |-- factor_attribution.py
|   |
|   |-- portfolio/               # Portfolio monitoring (3 files)
|   |   |-- heat_monitor.py
|   |   |-- risk_manager.py
|   |
|   |-- alerts/                  # Alert management
|   |   |-- telegram_alerter.py
|   |   |-- telegram_commander.py
|   |
|   |-- web/                     # Web dashboard (7 files)
|   |   |-- main.py              # FastAPI server
|   |   |-- dashboard.py
|   |   |-- dashboard_pro.py
|   |   |-- api/
|   |
|   |-- tools/                   # Utility tools (4 files)
|   |   |-- bounce_profile.py
|   |   |-- build_bounce_db.py
|   |
|   |-- bounce/                  # Bounce analysis module (8 files)
|       |-- data_loader.py
|       |-- streak_analyzer.py
|       |-- bounce_score.py
|       |-- strategy_integration.py
|
|-- TIER 9: SCRIPTS (180+ files)
|   |-- scripts/
|       |-- scan.py              # Daily scanner (CANONICAL)
|       |-- run_paper_trade.py   # Paper trading
|       |-- run_live_trade_micro.py # Live trading
|       |-- runner.py            # 24/7 scheduler
|       |-- backtest_dual_strategy.py # Strategy verification
|       |-- run_wf_polygon.py    # Walk-forward
|       |-- preflight.py         # Pre-trade checks
|       |-- overnight_watchlist.py # Build watchlist
|       |-- premarket_validator.py # Validate gaps
|       |-- generate_pregame_blueprint.py # Pre-game analysis
|       |-- ... (170+ more)
|
|-- TIER 10: STATE & RUNTIME
|   |-- state/                   # Persistent state
|   |   |-- autonomous/          # Brain state
|   |   |   |-- brain_state.json
|   |   |   |-- heartbeat.json
|   |   |-- watchlist/           # Trading watchlists
|   |   |   |-- next_day.json
|   |   |   |-- today_validated.json
|   |   |-- positions.json       # Current positions
|   |   |-- order_state.json     # Current orders
|   |   |-- hash_chain.jsonl     # Audit chain
|   |   |-- idempotency_store.sqlite # Idempotency
|   |
|   |-- logs/                    # Daily logs
|   |   |-- events.jsonl         # All events
|   |   |-- daily_picks.csv      # Top 3 picks
|   |   |-- trade_of_day.csv     # TOTD signal
|   |   |-- signals.jsonl        # All signals
|   |
|   |-- reports/                 # Analysis output
|   |   |-- pregame_*.json       # Pre-game blueprints
|   |   |-- bounce/              # Bounce analysis
|   |
|   |-- cache/                   # Data cache
|       |-- polygon/             # Polygon EOD cache
|
|-- TIER 11: TESTS (942 tests)
|   |-- tests/
|       |-- cognitive/           # Brain tests (19)
|       |-- execution/           # Execution tests (3)
|       |-- unit/                # Unit tests
|       |-- integration/         # Integration tests
|       |-- conftest.py          # Pytest fixtures
|
|-- TIER 12: DOCUMENTATION
|   |-- docs/                    # Primary reference
|   |   |-- STATUS.md            # CANONICAL SSOT
|   |   |-- PROFESSIONAL_EXECUTION_FLOW.md
|   |   |-- ARCHITECTURE.md
|   |   |-- ... (15+ more)
|   |
|   |-- CLAUDE.md                # Project instructions (36KB)
|   |-- README.md                # Project overview
|
|-- TIER 13: CONFIGURATION FILES
    |-- .env                     # Environment (MASKED)
    |-- .gitignore               # Git ignore
    |-- pytest.ini               # Pytest config
    |-- requirements.txt         # Dependencies
    |-- Dockerfile               # Docker image
    |-- docker-compose.yml       # Docker compose
    |-- .claude/                 # Claude Code config
        |-- skills/              # 70 skills
```

---

## Package Purposes

| Package | Purpose | Key File |
|---------|---------|----------|
| `strategies/` | Signal generation | `dual_strategy/combined.py` |
| `backtest/` | Backtesting engine | `engine.py` |
| `risk/` | Risk management gates | `policy_gate.py` |
| `execution/` | Order execution | `broker_alpaca.py` |
| `oms/` | Order state management | `order_state.py` |
| `data/` | Data fetching/caching | `providers/multi_source.py` |
| `cognitive/` | AI brain architecture | `cognitive_brain.py` |
| `ml_advanced/` | LSTM/HMM/Ensemble | `hmm_regime_detector.py` |
| `ml_features/` | Feature engineering | `feature_pipeline.py` |
| `ml/` | Alpha discovery/RL | `alpha_discovery/rl_agent/` |
| `ml_meta/` | ML meta-models | `model.py` |
| `monitor/` | Health monitoring | `health_endpoints.py` |
| `core/` | Core utilities | `structured_log.py` |
| `explainability/` | Trade explanation | `trade_thesis_builder.py` |
| `analysis/` | Pattern analysis | `historical_patterns.py` |
| `options/` | Synthetic options | `black_scholes.py` |
| `preflight/` | Pre-trade validation | `data_quality.py` |
| `config/` | Configuration | `settings_loader.py` |
| `altdata/` | Alternative data | `congressional_trades.py` |
| `analytics/` | Trade analytics | `edge_decomposition.py` |
| `portfolio/` | Portfolio monitoring | `heat_monitor.py` |
| `alerts/` | Alert management | `telegram_alerter.py` |
| `web/` | Web dashboard | `main.py` |
| `tools/` | Utility tools | `build_bounce_db.py` |
| `bounce/` | Bounce analysis | `streak_analyzer.py` |
| `scripts/` | Runnable scripts | `scan.py` |
| `state/` | Persistent state | JSON/SQLite files |
| `logs/` | Daily logs | JSONL/CSV files |
| `tests/` | Test suite | 942 tests |

---

## Statistics

| Metric | Count |
|--------|-------|
| Python Files | 563 |
| Main Packages | 28 |
| Scripts | 180+ |
| Tests | 942 |
| Skills | 70 |
| Config Files | 8 |
| State Files | 25+ |

---

## Related Documentation

- [ENTRYPOINTS.md](ENTRYPOINTS.md) - All runnable scripts
- [ARCHITECTURE.md](ARCHITECTURE.md) - Pipeline wiring
- [COMPONENT_REGISTRY.md](COMPONENT_REGISTRY.md) - Component details
