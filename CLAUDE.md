# CLAUDE.md

> Alignment Banner (v2.2): docs/STATUS.md is the canonical, single source of truth for active strategies, parameters, and performance metrics (QUANT INTERVIEW READY). Always consult docs/STATUS.md first, then follow its Replication Checklist.



This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. The trading robot is named "Kobe". Do not refer to any prior system name.

## Project Overview

Python quantitative trading system: backtesting, paper trading, live execution for mean-reversion strategies (IBS+RSI and ICT Turtle Soup). Uses Polygon.io for EOD data, Alpaca for execution.

## Requirements

- Python 3.11+
- `pip install -r requirements.txt`
- Environment: `.env` file with `POLYGON_API_KEY`, `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `ALPACA_BASE_URL`

## Common Comms

```bash
# Preflight check (env keys, config pin, broker probe)
python scripts/preflight.py --dotenv ./.env

# Build 900-stock universe (optionable, liquid, â‰¥10 years)
python scripts/build_universe_polygon.py --cidates data/universe/optionable_liquid_cidates.csv --start 2015-01-01 --end 2024-12-31 --min-years 10 --cap 900 --concurrency 3

# Prefetch EOD bars for faster WF
python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31

# Walk-forward + HTML report
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63
python scripts/aggregate_wf_report.py --wfdir wf_outputs

# Paper trade (micro budget)
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50

# Live trade (micro budget; requires live ALPACA_BASE_URL)
python scripts/run_live_trade_micro.py --universe data/universe/optionable_liquid_900.csv --cap 10

# 24/7 runner (paper)
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55

# Verify audit chain
python scripts/verify_hash_chain.py

# Reconcile broker positions
python scripts/reconcile_alpaca.py
```

All scripts accept `--dotenv` to specify env file location.

## Skills (Slash Comms)

**70 skills** organized by category. Definitions in `.claude/skills/*.md`.

### Startup & Shutdown (4 skills)
| Skill | Purpose |
|-------|---------|
| `/start` | Start the Kobe trading system |
| `/stop` | Graceful shutdown |
| `/restart` | Restart cleanly |
| `/runner` | Control 24/7 scheduler |

### Core Operations (6 skills)
| Skill | Purpose |
|-------|---------|
| `/preflight` | Run 10 critical checks before trading |
| `/validate` | Run tests + type checks |
| `/status` | Show system health dashboard |
| `/scan` | Run daily stock scanner |
| `/paper` | Start paper trading session |
| `/live` | Start live trading (REAL MONEY) |

### Emergency Controls (2 skills)
| Skill | Purpose |
|-------|---------|
| `/kill` | Emergency halt - creates KILL_SWITCH file |
| `/resume` | Deactivate kill switch after safe check |

### Position & P&L (3 skills)
| Skill | Purpose |
|-------|---------|
| `/positions` | Show open positions with live P&L |
| `/pnl` | Daily/weekly/total P&L summary |
| `/orders` | Order history  fill details |

### Strategy & Signals (4 skills)
| Skill | Purpose |
|-------|---------|
| `/strategy` | View/compare strategy parameters |
| `/signals` | View raw generated signals |
| `/backtest` | Run simple backtest |
| `/showdown` | Strategy comparison |

### Walk-Forward & Validation (2 skills)
| Skill | Purpose |
|-------|---------|
| `/wf` | Walk-forward backtest |
| `/smoke` | Run smoke tests |

### Data Management (3 skills)
| Skill | Purpose |
|-------|---------|
| `/data` | Data fetch status, cache health |
| `/prefetch` | Prefetch EOD bars for universe |
| `/universe` | Manage 900-stock universe |

### Broker & Execution (3 skills)
| Skill | Purpose |
|-------|---------|
| `/broker` | Broker connection status |
| `/reconcile` | Compare broker vs local positions |
| `/idempotency` | View/clear idempotency store |

### Integrity & Compliance (3 skills)
| Skill | Purpose |
|-------|---------|
| `/audit` | Verify hash chain (tamper detection) |
| `/risk` | Check all risk limits  gates |
| `/config` | View/modify config with signature |

### System Management (4 skills)
| Skill | Purpose |
|-------|---------|
| `/state` | View all state files |
| `/logs` | View recent events (errors, trades, alerts) |
| `/health` | Control health check server |
| `/backup` | Backup state, logs, configs |

### Environment & Secrets (3 skills)
| Skill | Purpose |
|-------|---------|
| `/env` | Environment variable management |
| `/secrets` | Validate/rotate API keys |
| `/calendar` | Market hours, holidays, early closes |

### Monitoring & Alerts (2 skills)
| Skill | Purpose |
|-------|---------|
| `/metrics` | Performance stats (win rate, PF, Sharpe) |
| `/alerts` | Manage alert thresholds  channels |

### Analytics & Reporting (3 skills)
| Skill | Purpose |
|-------|---------|
| `/benchmark` | Compare performance vs SPY |
| `/report` | Generate performance reports |
| `/replay` | Replay historical signals |

### Deployment & Debug (2 skills)
| Skill | Purpose |
|-------|---------|
| `/deploy` | Safe deployment with rollback |
| `/debug` | Toggle debug mode |

### Notifications (1 skill)
| Skill | Purpose |
|-------|---------|
| `/telegram` | Telegram bot alerts & notifications |

### Simulation & Optimization (2 skills)
| Skill | Purpose |
|-------|---------|
| `/simulate` | Monte Carlo simulation for forward testing |
| `/optimize` | Parameter optimization with grid search |

### Portfolio Analysis (2 skills)
| Skill | Purpose |
|-------|---------|
| `/exposure` | Sector/market cap/factor exposure analysis |
| `/watchlist` | Manage custom watchlists |

### Trading Journal (1 skill)
| Skill | Purpose |
|-------|---------|
| `/journal` | Trading notes, lessons, trade reviews |

### Options & Hedging (3 skills)
| Skill | Purpose |
|-------|---------|
| `/options` | Options chain lookup, IV, greeks |
| `/hedge` | Suggest protective puts for positions |
| `/earnings` | Earnings calendar, avoid/target earnings |

### AI Assistant (3 skills)
| Skill | Purpose |
|-------|---------|
| `/explain` | Explain why a signal was generated |
| `/suggest` | AI suggests next actions based on state |
| `/learn` | Show what Kobe learned from recent trades |

### Advanced Analytics (3 skills)
| Skill | Purpose |
|-------|---------|
| `/regime` | Market regime detection (bull/bear/chop) |
| `/correlation` | Position correlation matrix |
| `/drawdown` | Drawdown analysis  recovery stats |

### Data Validation (2 skills)
| Skill | Purpose |
|-------|---------|
| `/polygon` | Validate Polygon data source & 900 coverage |
| `/integrity-check` | Detect lookahead, bias, bugs, fake data, manipulation |

### Dashboard (1 skill)
| Skill | Purpose |
|-------|---------|
| `/dashboard` | Launch/manage web dashboard for trading status |

### Quality & Testing (1 skill)
| Skill | Purpose |
|-------|---------|
| `/quality` | Run code, data, test,  system quality checks with scoring |

### Quant Analysis (1 skill)
| Skill | Purpose |
|-------|---------|
| `/quant` | High-level quant dashboard (Sharpe, alpha, factor exposures) |

### Debugging (1 skill)
| Skill | Purpose |
|-------|---------|
| `/debugger` | Error diagnosis, signal tracing, performance profiling |

### System Maintenance (5 skills)
| Skill | Purpose |
|-------|---------|
| `/version` | Show Kobe version, last update |
| `/cleanup` | Purge old logs, cache, temp files |
| `/snapshot` | Full state snapshot for recovery |
| `/test` | Run unit tests  integration tests |
| `/performance` | Real-time system performance monitoring |

## Architecture

### Layer Structure
| Layer | Module | Purpose |
|-------|--------|---------|
| Data | `data/providers/polygon_eod.py` | EOD OHLCV fetch with CSV caching |
| Universe | `data/universe/loader.py` | Symbol list loading, dedup, cap |
| Strategies | `strategies/dual_strategy/`, `strategies/ibs_rsi/`, `strategies/ict/` | Signal generation with shifted indicators |
| Backtest | `backtest/engine.py`, `backtest/walk_forward.py` | Simulation engine, WF splits |
| Risk | `risk/policy_gate.py` | Per-order ($75)  daily ($1k) budgets |
| Risk Advanced | `risk/advanced/` | VaR, Kelly sizing, correlation limits |
| ML Advanced | `ml_advanced/` | LSTM, HMM regime, ensemble, online learning |
| OMS | `oms/order_state.py`, `oms/idempotency_store.py` | Order records, duplicate prevention |
| Execution | `execution/broker_alpaca.py` | IOC LIMIT orders via Alpaca |
| Core | `core/hash_chain.py`, `core/structured_log.py` | Audit chain, JSON logging |
| Monitor | `monitor/health_endpoints.py` | Health check endpoint |

### Advanced Risk Management (`risk/advanced/`)
| Module | Purpose |
|--------|---------|
| `monte_carlo_var.py` | Portfolio VaR with Cholesky decomposition, stress testing |
| `kelly_position_sizer.py` | Optimal Kelly Criterion position sizing |
| `correlation_limits.py` | Correlation/concentration limits with sector mapping |

### ML/AI Components (`ml_advanced/`)
| Module | Purpose |
|--------|---------|
| `hmm_regime_detector.py` | Hidden Markov Model market regime detection |
| `lstm_confidence/` | Multi-output LSTM for signal confidence (A/B/C grades) |
| `ensemble/` | Multi-model ensemble predictor (XGBoost, LightGBM, LSTM) |
| `online_learning.py` | Incremental learning with concept drift detection |

### Cognitive Architecture (`cognitive/`)
Brain-inspired decision system with self-awareness. **83 unit tests passing.**

| Module | Purpose |
|--------|---------|
| `cognitive_brain.py` | Main orchestrator - deliberation, learning, introspection |
| `metacognitive_governor.py` | System 1/2 routing (fast vs slow thinking) |
| `reflection_engine.py` | Learning from outcomes (Reflexion pattern) |
| `self_model.py` | Capability tracking, calibration, self-awareness |
| `episodic_memory.py` | Experience storage (context → reasoning → outcome) |
| `semantic_memory.py` | Generalized rules and knowledge |
| `knowledge_boundary.py` | Uncertainty detection, stand-down recommendations |
| `curiosity_engine.py` | Hypothesis generation and edge discovery |

**Configuration:** `config/base.yaml` (cognitive section)
**Tests:** `tests/cognitive/`
**State:** `state/cognitive/`

### Strategy Interface
```python
class Strategy:
    def __init__(self, params: Optional[Params] = None)
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame  # last bar only
    def scan_signals_over_time(self, df: pd.DataFrame) -> pd.DataFrame  # all bars (backtest)
```
Output columns: `timestamp, symbol, side, entry_price, stop_loss, take_profit, reason`

### Critical Invariants
- **No lookahead**: Indicators shifted 1 bar (`col_sig = col.shift(1)`)
- **Next-bar fills**: Signals at close(t), fills at open(t+1)
- **Exits**: ATR(14)x2 stop + 5-bar time stop
- **Execution**: IOC LIMIT only (limit = best_ask x 1.001)
- **Kill switch**: Create `state/KILL_SWITCH` to halt submissions

### Evidence Artifacts
- `wf_outputs/wf_summary_compare.csv` - strategy comparison
- `wf_outputs/<strategy>/split_NN/{trade_list.csv, equity_curve.csv, summary.json}`
- `logs/events.jsonl` - structured logs
- `state/hash_chain.jsonl` - audit chain

### Frozen Data Lake (`data/lake/`)
Immutable datasets for reproducible backtesting. Once frozen, data never changes.

| Module | Purpose |
|--------|---------|
| `manifest.py` | Dataset manifests with SHA256 hashes, deterministic IDs |
| `io.py` | LakeWriter/LakeReader for parquet/CSV with integrity verification |

```bash
# Freeze equities from Stooq (free, no API key)
python scripts/freeze_equities_eod.py \
    --universe data/universe/optionable_liquid_900.csv \
    --start 2015-01-01 --end 2025-12-31 \
    --provider stooq

# Freeze crypto from Binance (free, no API key)
python scripts/freeze_crypto_ohlcv.py \
    --symbols BTCUSDT,ETHUSDT,SOLUSDT \
    --start 2020-01-01 --end 2025-12-31 \
    --timeframe 1h

# Validate frozen dataset
python scripts/validate_lake.py --dataset-id YOUR_DATASET_ID
```

### Free Data Providers (`data/providers/`)
No API keys required for backtesting.

| Provider | Asset Class | Source |
|----------|-------------|--------|
| `stooq_eod.py` | Equities (US) | Stooq.com (primary) |
| `yfinance_eod.py` | Equities (US) | Yahoo Finance (fallback) |
| `binance_klines.py` | Crypto (USDT pairs) | Binance public API |

### Synthetic Options Engine (`options/`)
Black-Scholes pricing for options backtesting without real options data.

| Module | Purpose |
|--------|---------|
| `black_scholes.py` | BS pricing, Greeks, implied volatility |
| `volatility.py` | Realized volatility (close-to-close, Parkinson, Yang-Zhang) |
| `selection.py` | Delta-targeted strike selection via binary search |
| `position_sizing.py` | 2% risk enforcement for long/short options |
| `backtest.py` | Daily repricing with transaction costs |

```bash
# Run synthetic options backtest
python scripts/run_backtest_options_synth.py \
    --dataset-id YOUR_DATASET_ID \
    --signals my_signals.csv \
    --equity 100000 --risk-pct 0.02

# Demo mode (synthetic data)
python scripts/run_backtest_options_synth.py --demo
```

### Experiment Registry (`experiments/`)
Track all experiments for reproducibility.

| Module | Purpose |
|--------|---------|
| `registry.py` | Register experiments, record results, verify reproducibility |

```python
from experiments import register_experiment, record_experiment_results

exp_id = register_experiment(
    name="momentum_test",
    dataset_id="stooq_1d_2015_2025_abc123",
    strategy="momentum",
    params={"lookback": 20},
    seed=42,
)
# Run backtest...
record_experiment_results(exp_id, results)
```

### Data Quality Gate (`preflight/data_quality.py`)
Validates data before backtesting.

- Coverage checks (min 5 years history)
- Gap detection (max 5% missing)
- OHLC violation detection
- Staleness checks
- KnowledgeBoundary integration for stand-down decisions

```python
from preflight import validate_data_quality

report = validate_data_quality(dataset_id="stooq_1d_2015_2025_abc123")
if report.passed:
    print("Data quality OK")
```

## Advisory Usage Policy
- Claude acts as advisory-only reviewer; never in hot execution path
- Cannot override PolicyGate budgets or kill switch
- Suggestions must be specific, testable, minimally invasive

## Key Files

### Core Trading
- `backtest/engine.py`: Backtester with equity curve, ATR/time stops, FIFO P&L
- `strategies/dual_strategy/combined.py`: DualStrategyScanner (IBS+RSI + Turtle Soup combined)
- `strategies/ibs_rsi/strategy.py`: IBS<0.15 + RSI(2)<10 entry, SMA(200) filter (62.3% WR)
- `strategies/ict/turtle_soup.py`: ICT Turtle Soup liquidity sweep, SMA(200) filter (61.1% WR)
- `execution/broker_alpaca.py`: `place_ioc_limit()`, `get_best_ask()`, idempotency
- `risk/policy_gate.py`: `PolicyGate.check()` for budget enforcement
- `scripts/runner.py`: 24/7 scheduler with `--scan-times`  state persistence

### Advanced Risk (Quant Interview Ready)
- `risk/advanced/monte_carlo_var.py`: 10K-simulation VaR with stress testing
- `risk/advanced/kelly_position_sizer.py`: Fractional Kelly with volatility adjustment
- `risk/advanced/correlation_limits.py`: Sector exposure, beta, ENP checks

### ML/AI Components (Quant Interview Ready)
- `ml_advanced/hmm_regime_detector.py`: 3-state HMM regime detection (bull/bear/neutral)
- `ml_advanced/lstm_confidence/model.py`: Multi-output LSTM (direction, magnitude, success)
- `ml_advanced/ensemble/ensemble_predictor.py`: Weighted ensemble with confidence scoring
- `ml_advanced/online_learning.py`: Experience replay, concept drift detection

## Quick Reference - Essential Skills

```
/start          Start trading
/stop           Stop trading
/kill           EMERGENCY STOP
/status         System health
/positions      Current holdings
/pnl            Profit & loss
/preflight      Pre-trade checks
/logs           Recent events
/broker         Broker connection
```







