# PHASE 1: REPO CENSUS

**Generated:** 2026-01-05 20:20 ET
**Auditor:** Claude SUPER AUDIT
**Status:** COMPLETE

---

## EXECUTIVE SUMMARY

| Metric | Count |
|--------|-------|
| **Total Python Files** | 722 |
| **Top-Level Directories** | 72 |
| **Script Files** | 181 |
| **Test Files** | 119 |
| **Strategy Files** | 20 |
| **Execution Files** | 30 |
| **ML/Cognitive Files** | 111 |
| **Agent Files** | 14 |

---

## DIRECTORY STRUCTURE (72 Top-Level)

```
kobe81_traderbot/
|-- AUDITS/              # Audit artifacts (this folder)
|-- agents/              # ReAct agent framework (14 files)
|-- alerts/              # Telegram & professional alerts
|-- altdata/             # Alternative data (sentiment)
|-- analysis/            # Historical patterns, expected move
|-- analytics/           # Performance analytics
|-- archive/             # Archived code
|-- autonomous/          # 24/7 brain system
|-- backtest/            # Backtesting engine
|-- backups/             # State backups
|-- bounce/              # Bounce analysis
|-- cache/               # Market data cache
|-- cognitive/           # Brain-inspired decision system
|-- compliance/          # Regulatory compliance
|-- config/              # Configuration files
|-- core/                # Core utilities (kill switch, hash chain, alerts)
|-- dashboard/           # Web dashboard
|-- data/                # Data providers, universe, lake
|-- data_exploration/    # Feature discovery
|-- docs/                # Documentation
|-- evolution/           # Genetic optimizer
|-- execution/           # Broker interfaces (30 files)
|-- experiments/         # Experiment registry
|-- explainability/      # Trade thesis builder
|-- guardian/            # Guardian patterns
|-- integration/         # Integration layer
|-- llm/                 # LLM interfaces
|-- logs/                # Log files
|-- messaging/           # Message bus
|-- ml/                  # Alpha discovery, RL agent
|-- ml_advanced/         # LSTM, HMM, ensemble
|-- ml_features/         # Feature pipeline
|-- ml_meta/             # Meta-learning
|-- models/              # Trained models
|-- monitor/             # Health endpoints
|-- news/                # News monitoring
|-- notebooks/           # Jupyter notebooks
|-- observability/       # Observability layer
|-- oms/                 # Order management system
|-- ops/                 # Operations
|-- optimization/        # Bayesian hyperopt
|-- optimize_outputs/    # Optimization results
|-- options/             # Options trading
|-- outputs/             # General outputs
|-- pipelines/           # Quant R&D pipeline
|-- portfolio/           # Portfolio management
|-- preflight/           # Preflight checks
|-- quant_gates/         # Quality gates
|-- reports/             # Report generation
|-- research/            # Research engine
|-- research_os/         # Research OS workflow
|-- risk/                # Risk management (core)
|-- safety/              # Trading mode enforcement
|-- scripts/             # Runnable scripts (181 files)
|-- selfmonitor/         # Self-monitoring
|-- showdown_*/          # Strategy showdown outputs
|-- smoke_*/             # Smoke test outputs
|-- state/               # State files (heartbeat, watchlist, etc.)
|-- stateguardian/       # State guardian
|-- strategies/          # Trading strategies (20 files)
|-- strategy_specs/      # Strategy specifications
|-- tax/                 # Tax lot accounting
|-- testing/             # Additional testing
|-- tests/               # Unit & integration tests (119 files)
|-- tools/               # Utility tools
|-- trade_logging/       # Trade logging & Prometheus
|-- web/                 # Web interfaces
|-- wf_outputs/          # Walk-forward outputs
```

---

## FILE COUNTS BY MODULE

| Module | File Count | Purpose |
|--------|------------|---------|
| scripts/ | 181 | Runnable entry points |
| tests/ | 119 | Unit and integration tests |
| ml_features/ | 42 | Feature engineering |
| cognitive/ | 35 | Brain-inspired system |
| ml_advanced/ | 34 | Advanced ML (LSTM, HMM) |
| execution/ | 30 | Broker interfaces |
| risk/ | 28 | Risk management |
| strategies/ | 20 | Trading strategies |
| autonomous/ | 18 | 24/7 brain |
| data/ | 16 | Data providers |
| agents/ | 14 | ReAct agents |
| core/ | 12 | Core utilities |
| backtest/ | 10 | Backtesting |
| research_os/ | 8 | Research workflow |
| safety/ | 4 | Mode enforcement |

---

## CRITICAL FILE INVENTORY

### Safety Layer (4 files)
- `safety/__init__.py` - Exports safety constants
- `safety/mode.py` - PAPER_ONLY, LIVE_TRADING_ENABLED
- `core/kill_switch.py` - Emergency halt mechanism
- `research_os/approval_gate.py` - APPROVE_LIVE_ACTION

### Strategy Layer (Key Files)
- `strategies/dual_strategy/combined.py` - DualStrategyScanner
- `strategies/dual_strategy/params.py` - Frozen parameters
- `strategies/ibs_rsi/strategy.py` - IBS+RSI strategy
- `strategies/ict/turtle_soup.py` - Turtle Soup strategy
- `strategies/registry.py` - Strategy registry

### Execution Layer (Key Files)
- `execution/broker_alpaca.py` - Alpaca broker interface
- `execution/broker_paper.py` - Paper trading broker
- `execution/broker_base.py` - Base broker class
- `execution/order_manager.py` - Order management
- `execution/intelligent_executor.py` - Smart execution

### Risk Layer (Key Files)
- `risk/policy_gate.py` - Budget enforcement
- `risk/equity_sizer.py` - Position sizing
- `risk/kill_zone_gate.py` - Time-based blocking
- `risk/signal_quality_gate.py` - Signal quality
- `risk/advanced/monte_carlo_var.py` - VaR calculation

### Data Layer (Key Files)
- `data/providers/polygon_eod.py` - Polygon EOD data
- `data/universe/loader.py` - Universe loading
- `data/lake/io.py` - Data lake I/O

### Autonomous Brain (Key Files)
- `autonomous/brain.py` - Main brain
- `autonomous/master_brain_full.py` - Full brain with scheduler
- `autonomous/scheduler_full.py` - 462 scheduled tasks
- `autonomous/awareness.py` - Time/day awareness

---

## MANIFEST FILES CREATED

1. `AUDITS/01_PYTHON_FILES_MANIFEST.txt` - All 722 Python files with sizes

---

## NEXT: PHASE 2 - ENTRYPOINT DISCOVERY

**Signature:** SUPER_AUDIT_PHASE1_2026-01-05_COMPLETE
