# CLAUDE.md

> Alignment Banner (v2.3): docs/STATUS.md is the canonical, single source of truth for active strategies, parameters, and performance metrics (QUANT INTERVIEW READY). Always consult docs/STATUS.md first, then follow its Replication Checklist.

## CRITICAL: Strategy Verification (READ FIRST)

**Use `backtest_dual_strategy.py` for ALL strategy verification.** This is the canonical test.

| Script | Use For | Strategy Class | Has Sweep Filter |
|--------|---------|----------------|------------------|
| `backtest_dual_strategy.py` | **Verification** | `DualStrategyScanner` | YES (0.3 ATR) |
| `run_wf_polygon.py` | Walk-forward | `DualStrategyScanner` | YES (0.3 ATR) |
| `scan.py` | **Daily Scan** | `DualStrategyScanner` | YES (0.3 ATR) |

## ONE Scanner System (Updated 2025-12-31)

**THE ONLY SCANNER COMMAND:**
```bash
python scripts/scan.py --cap 900 --deterministic --top3
```

**Output Files:**
- `logs/daily_picks.csv` - Top 3 picks (quality gate filtered)
- `logs/trade_of_day.csv` - Single highest-confidence TOTD
- `logs/signals.jsonl` - All signals (append-only)

**Raw Signals (no quality gate):**
```bash
python scripts/scan.py --cap 900 --deterministic --no-quality-gate
```

**Quality Gate Settings (2026-01-02):**
- Threshold: 70 (ML models now trained: HMM, LSTM, XGBoost, LightGBM, RL)
- Max signals per day: 3 (default)

**There is ONLY ONE scanner: `scan.py`. All other scan scripts have been deleted.**

**Verification Command:**
```bash
python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_900.csv --start 2023-01-01 --end 2024-12-31 --cap 150
```
**Expected:** ~64% WR, ~1.60 PF

**System Audit (2025-12-29):** Grade A+ (100/100), 22/22 modules verified, 14/14 AI/LLM/ML verified, 942 tests, 0 critical issues.
n---

## CRITICAL: Position Sizing (2026-01-02 INCIDENT)

> **READ `docs/CRITICAL_FIX_20260102.md` FOR FULL DETAILS**
>
> **NEVER place manual orders. ALWAYS use `run_paper_trade.py` which has dual caps:**
> - 2% risk cap per trade
> - 20% notional cap per position
> - Formula: `final_shares = min(shares_by_risk, shares_by_notional)`

---

## CRITICAL: Professional Execution Flow (v3.0)

> **THIS IS HOW PROFESSIONALS TRADE. FOLLOW THIS EXACTLY.**
>
> Full documentation: `docs/PROFESSIONAL_EXECUTION_FLOW.md`

### Kill Zones (ICT-Style Time-Based Blocking)

**NEVER trade outside valid kill zones. The system enforces this automatically.**

| Time (ET) | Zone | Trading | Reason |
|-----------|------|---------|--------|
| Before 9:30 | `pre_market` | ❌ BLOCKED | Market not open |
| 9:30-10:00 | `opening_range` | ❌ **BLOCKED** | Amateur hour - let volatility settle |
| 10:00-11:30 | `london_close` | ✅ **PRIMARY WINDOW** | Best setups develop here |
| 11:30-14:00 | `lunch_chop` | ❌ BLOCKED | Low volume, fake moves |
| 14:00-14:30 | `lunch_chop` | ❌ BLOCKED | Extended lunch |
| 14:30-15:30 | `power_hour` | ✅ **SECONDARY WINDOW** | Institutional positioning |
| 15:30-16:00 | `close` | ❌ BLOCKED | No new entries, manage only |
| After 16:00 | `after_hours` | ❌ BLOCKED | Market closed |

**Check Kill Zone:**
```python
from risk.kill_zone_gate import can_trade_now, check_trade_allowed, get_current_zone

if can_trade_now():
    execute_trade()
else:
    allowed, reason = check_trade_allowed()
    print(f"BLOCKED: {reason}")
```

### Daily Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PREVIOUS DAY (3:30 PM)                           │
├─────────────────────────────────────────────────────────────────────┤
│  OVERNIGHT_WATCHLIST                                                 │
│  ├── Scan 900 stocks for NEXT DAY setups                            │
│  ├── Generate Top 5 watchlist + TOTD                                │
│  └── Save to state/watchlist/next_day.json                          │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREMARKET (8:00 AM)                              │
├─────────────────────────────────────────────────────────────────────┤
│  PREMARKET_VALIDATOR                                                 │
│  ├── Load overnight watchlist                                        │
│  ├── Check each stock for gaps > 3%, news, corporate actions        │
│  ├── Flag: VALID, GAP_INVALIDATED, NEWS_RISK, IMPROVED, DEGRADED    │
│  └── Save to state/watchlist/today_validated.json                   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OPENING RANGE (9:30-10:00)                       │
├─────────────────────────────────────────────────────────────────────┤
│  OPENING_RANGE_OBSERVER (9:30, 9:45 AM)                             │
│  ├── ⛔ NO TRADES - OBSERVE ONLY                                    │
│  ├── Log opening prices, strength/weakness                          │
│  └── Save to state/watchlist/opening_range.json                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 PRIMARY WINDOW (10:00-11:30)                        │
├─────────────────────────────────────────────────────────────────────┤
│  FIRST_SCAN (10:00 AM)                                              │
│  ├── Trade ONLY from validated watchlist                            │
│  ├── Quality: Score >= 65, Confidence >= 0.60, R:R >= 1.5:1        │
│  └── Max 2 trades from watchlist                                    │
│                                                                      │
│  FALLBACK_SCAN (10:30 AM) - Only if watchlist fails                 │
│  ├── Scan 900 stocks with HIGHER bar                                │
│  ├── Quality: Score >= 75, Confidence >= 0.70, R:R >= 2.0:1        │
│  └── Max 1 trade from fallback                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### State Files

| File | Purpose | Updated By |
|------|---------|------------|
| `state/watchlist/next_day.json` | Tomorrow's Top 5 | overnight_watchlist.py (3:30 PM) |
| `state/watchlist/today_validated.json` | Today's validated watchlist | premarket_validator.py (8:00 AM) |
| `state/watchlist/opening_range.json` | Opening observations | opening_range_observer.py (9:30, 9:45) |

### Professional Execution Scripts

| Script | Time | Purpose |
|--------|------|---------|
| `scripts/overnight_watchlist.py` | 3:30 PM | Build Top 5 for next day |
| `scripts/premarket_validator.py` | 8:00 AM | Validate gaps/news |
| `scripts/opening_range_observer.py` | 9:30, 9:45 | Observe (NO TRADES) |
| `scripts/run_paper_trade.py --watchlist-only` | 10:00 AM | Trade from watchlist |

### Edge Cases Covered

| Scenario | Handling |
|----------|----------|
| All 5 watchlist stocks gap > 3% | Fallback scan with higher quality bar |
| TOTD gaps up 5% | Removed from watchlist, flagged GAP_INVALIDATED |
| News hits premarket | Flagged NEWS_RISK, may remove or downgrade |
| Signal at 9:35 AM | **BLOCKED** - must wait until 10:00 AM |
| No signals all day | Capital preservation - no trades is valid |
| 3 watchlist stocks trigger | Take best 2 only (daily limit) |

### Quality Gates by Source

| Source | Min Score | Min Confidence | Min R:R | Max Trades |
|--------|-----------|----------------|---------|------------|
| Watchlist (TOTD) | 60 | 0.55 | 1.5:1 | Priority |
| Watchlist (Top 5) | 65 | 0.60 | 1.5:1 | Up to 2 |
| Fallback (900 scan) | 75 | 0.70 | 2.0:1 | Max 1 |
| Power Hour | 70 | 0.65 | 1.5:1 | Max 1 |

---

## CRITICAL: ALWAYS Use DualStrategyScanner (NEVER Standalone Strategies)

**THIS IS NON-NEGOTIABLE. ALWAYS USE THE CORRECT STRATEGY CLASS.**

| WRONG (DEPRECATED) | CORRECT |
|-------------------|---------|
| `from strategies.ict.turtle_soup import TurtleSoupStrategy` | `from strategies.registry import get_production_scanner` |
| `from strategies.ibs_rsi.strategy import IbsRsiStrategy` | `from strategies.dual_strategy import DualStrategyScanner` |

**Why This Matters:**
- `DualStrategyScanner` has `ts_min_sweep_strength=0.3` filter = **61% WR, 1.37 PF**
- `TurtleSoupStrategy` (standalone) has NO filter = **~48% WR, 0.85 PF** (FAIL!)
- Using wrong strategy costs you **13% win rate** and turns a profitable system into a losing one

**Canonical Usage:**
```python
# ALWAYS use this for production
from strategies.registry import get_production_scanner
scanner = get_production_scanner()
signals = scanner.scan_signals_over_time(df)

# Or this
from strategies.dual_strategy import DualStrategyScanner, DualStrategyParams
scanner = DualStrategyScanner(DualStrategyParams())
```

**Safeguards Implemented:**
1. Deprecation warnings in `turtle_soup.py` and `ibs_rsi/strategy.py`
2. Strategy registry at `strategies/registry.py` with validation
3. Startup validation in `runner.py` and `scan.py`
4. Frozen parameters at `config/frozen_strategy_params_v2.2.json`

---

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

# Pre-Game Blueprint (comprehensive evidence-backed analysis)
python scripts/generate_pregame_blueprint.py --positions TSLA PLTR  # Analyze existing positions
python scripts/generate_pregame_blueprint.py --cap 900 --top 5 --execute 2  # Scanner mode
```

All scripts accept `--dotenv` to specify env file location.

## Pre-Game Blueprint System (08:15 AM ET)

Generates comprehensive evidence-backed analysis for every trade:

| Component | Description |
|-----------|-------------|
| Historical Patterns | Consecutive days, reversal rates, sample sizes |
| Expected Move | Weekly range using realized volatility |
| Support/Resistance | Pivot points with justification |
| Sector Strength | Performance vs sector ETF, beta |
| AI Confidence | Weighted score breakdown |
| Bull/Bear Cases | Narratives for both scenarios |
| What Could Go Wrong | Risk factors |

**Output:** `reports/pregame_YYYYMMDD.json` + `reports/pregame_YYYYMMDD.md`

**Files:**
- `analysis/historical_patterns.py` - Pattern analysis
- `analysis/options_expected_move.py` - Expected move calculator
- `explainability/trade_thesis_builder.py` - Full trade thesis
- `scripts/generate_pregame_blueprint.py` - Main entry point

---

## CRITICAL: Trade Analysis Standard (2026-01-02)

> **FULL DOCUMENTATION: `docs/TRADE_ANALYSIS_STANDARD.md`**
>
> **THE TOP 2 TRADES EACH DAY MUST HAVE COMPREHENSIVE ANALYSIS. NO SHORTCUTS.**

### Historical Pattern Auto-Pass

Signals with strong historical backing **automatically bypass** the quality gate:

| Criteria | Threshold | Example |
|----------|-----------|---------|
| Sample Size | 20+ instances | PLTR: 23 samples |
| Win Rate | 90%+ | PLTR: 100% (23/23) |
| Streak Length | 5+ consecutive days | PLTR: 5 down days |

**If ALL criteria are met → AUTO-PASS with ELITE tier (score = 95)**

```python
# This pattern auto-passes quality gate
from analysis.historical_patterns import enrich_signal_with_historical_pattern

signal = enrich_signal_with_historical_pattern({'symbol': 'PLTR'})
# If 25+ samples with 90%+ win rate → signal['historical_pattern']['qualifies_for_auto_pass'] = True
```

### Mandatory Analysis Components (Top 2 Trades)

Every trade in the Top 2 **MUST** have:

| Component | Required Data | Source |
|-----------|---------------|--------|
| Price Action | Current price, week open, % move | Polygon EOD |
| Consecutive Pattern | Streak, samples, win rate, bounce stats | `historical_patterns.py` |
| Expected Move | Weekly EM, remaining room up/down | `options_expected_move.py` |
| Support/Resistance | Pivot points, psychological levels | `historical_patterns.py` |
| **News & Headlines** | Last 7 days, sentiment scores | Polygon/Finnhub |
| **Political Activity** | Congressional trades, insider activity | Quiver Quant |
| Sector Context | Relative strength, beta vs sector | Calculated |
| Volume Analysis | ADV, relative volume, trend | Polygon |
| Entry/Stop/Target | Prices with full justification | Calculated |
| R:R Analysis | All scenarios with ratios | Calculated |
| Bull/Bear Cases | Narratives for both directions | AI-generated |
| What Could Go Wrong | Top 5 risk factors | AI-generated |
| AI Confidence | Full breakdown by factor | `signal_quality_gate.py` |

### Position Sizing Formula

```python
# Dual-cap position sizing (MANDATORY)
account_equity = 50000
max_risk_pct = 0.02  # 2%
max_notional_pct = 0.20  # 20%

risk_per_trade = account_equity * max_risk_pct  # $1,000
max_notional = account_equity * max_notional_pct  # $10,000

shares_by_risk = risk_per_trade / (entry - stop)
shares_by_notional = max_notional / entry

final_shares = min(shares_by_risk, shares_by_notional)  # ALWAYS use the smaller
```

### This Is How We Analyze EVERY Trade

1. **Historical Evidence First**: Always check for consecutive day patterns with 25+ samples
2. **No Fake Data**: All statistics come from Polygon EOD - verifiable on Yahoo Finance
3. **Full Reasoning**: Every entry, stop, and target has written justification
4. **Risk-First**: Calculate R:R before considering the trade
5. **Multiple Scenarios**: Day 1 target, total bounce target, max historical case
6. **What Could Go Wrong**: Never trade without identifying risks

**Claude must generate this full analysis for the Top 2 trades. No exceptions.**

---

## Weekend-Safe Scanning

The scanner automatically handles weekends and holidays using NYSE calendar:

```bash
# Weekend (auto-detects and uses Friday's close + preview mode)
python scripts/scan.py --cap 200
# Output: *** WEEKEND: Using 2025-12-26 (Friday) + PREVIEW mode ***

# Weekday (uses fresh data + normal mode)
python scripts/scan.py --cap 200
# Output: *** WEEKDAY: Using 2025-12-29 + NORMAL mode (fresh data) ***

# Force preview mode (see what would trigger if current bar closes now)
python scripts/scan.py --cap 200 --preview
```

**Key Concepts:**
- **Normal Mode** (weekdays): Uses `.shift(1)` for lookahead safety - signals based on PRIOR bar
- **Preview Mode** (weekends): Uses current bar values - shows what WOULD trigger Monday
- Scanner auto-detects weekends/holidays and enables preview mode automatically

**Why This Matters:**
Strategy uses `col_sig = col.shift(1)` to prevent lookahead bias. On Friday close, normal mode sees Thursday's values. Preview mode lets you see signals that will actually trigger Monday.

See `docs/STATUS.md` Section 16 for detailed explanation.

## Skills (Slash Comms)

**70 skills** organized by category. Definitions in `.claude/skills/*.md`.

### Startup & Shutdown (4 skills)
| Skill | Purpose |
|-------|---------|
| `/start` | Start the Kobe trading system |
| `/stop` | Graceful shutdown |
| `/restart` | Restart cleanly |
| `/runner` | Control 24/7 scheduler |

### Autonomous Brain (4 skills)
| Skill | Purpose |
|-------|---------|
| `/brain` | Start/stop/status of autonomous brain |
| `/awareness` | Show current time/day/season awareness |
| `/research` | View research experiments and discoveries |
| `/learning` | View learning progress and daily reflections |

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
| Risk | `risk/policy_gate.py`, `risk/equity_sizer.py` | 2% equity-based sizing, notional caps |
| Risk Advanced | `risk/advanced/` | VaR, Kelly sizing, correlation limits |
| ML Advanced | `ml_advanced/` | LSTM, HMM regime, ensemble, online learning |
| OMS | `oms/order_state.py`, `oms/idempotency_store.py` | Order records, duplicate prevention |
| Execution | `execution/broker_alpaca.py` | IOC LIMIT orders via Alpaca |
| Core | `core/hash_chain.py`, `core/structured_log.py` | Audit chain, JSON logging |
| Monitor | `monitor/health_endpoints.py` | Health check endpoint |

### Professional Execution (`risk/` + `scripts/`)
| Module | Purpose |
|--------|---------|
| `risk/kill_zone_gate.py` | ICT-style time-based trade blocking (9:30-10:00 blocked) |
| `risk/weekly_exposure_gate.py` | 40% weekly / 20% daily exposure caps |
| `risk/dynamic_position_sizer.py` | Adaptive sizing based on signal count |
| `scripts/overnight_watchlist.py` | Build Top 5 watchlist (3:30 PM) |
| `scripts/premarket_validator.py` | Validate gaps/news (8:00 AM) |
| `scripts/opening_range_observer.py` | Observe opening range (9:30-10:00) |

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

### ML Features (`ml_features/`)
| Module | Purpose |
|--------|---------|
| `pca_reducer.py` | **NEW** PCA dimensionality reduction (95% variance retention) |
| `feature_pipeline.py` | 150+ features + **NEW** lag features + time/calendar features |
| `technical_features.py` | pandas-ta indicators (RSI, ATR, MACD, Bollinger, etc.) |
| `anomaly_detection.py` | Matrix profiles for unusual pattern detection |
| `regime_ml.py` | KMeans/GMM regime clustering |
| `ensemble_brain.py` | Multi-model prediction ensemble |

### RL Trading Agent (`ml/alpha_discovery/rl_agent/`)
| Module | Purpose |
|--------|---------|
| `trading_env.py` | Gym-compatible trading environment |
| `agent.py` | PPO/DQN/A2C via stable-baselines3 |

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

### Autonomous Brain (`autonomous/`) - 24/7 Self-Improving System

The autonomous brain runs continuously, always aware of time/day/season, and never stops working.

| Module | Purpose |
|--------|---------|
| `awareness.py` | Time/day/season awareness (market phases, holidays, FOMC, OpEx) |
| `scheduler.py` | Task queue with priority and context-based execution |
| `brain.py` | Main orchestrator - decides what to work on 24/7 |
| `research.py` | Self-improvement: random parameter experiments, strategy discovery |
| `learning.py` | Trade analysis, episodic memory updates, daily reflections |
| `maintenance.py` | Data quality, cleanup, health checks |
| `monitor.py` | Heartbeat monitoring, alerts, status dashboard |

**Start Autonomous Brain:**
```bash
python scripts/run_autonomous.py              # Run forever (60s cycles)
python scripts/run_autonomous.py --status     # Show status
python scripts/run_autonomous.py --awareness  # Show current awareness
python scripts/run_autonomous.py --once       # Single cycle (testing)
```

**Work Modes by Time:**
| Time (ET) | Phase | Work Mode | Activities |
|-----------|-------|-----------|------------|
| 4:00-7:00 AM | Pre-market Early | Research | Backtests, experiments |
| 7:00-9:30 AM | Pre-market Active | Monitoring | Watchlist prep, gap check |
| 9:30-10:00 AM | Market Opening | Monitoring | Observe only, NO trades |
| 10:00-11:30 AM | Market Morning | Active Trading | Scan, trade, monitor |
| 11:30-14:00 PM | Market Lunch | Research | Choppy - run experiments |
| 14:00-15:30 PM | Market Afternoon | Active Trading | Power hour trading |
| 15:30-16:00 PM | Market Close | Monitoring | Manage positions |
| 16:00-20:00 PM | After Hours | Learning | Analyze trades, reflect |
| 20:00-4:00 AM | Night | Optimization | Walk-forward, retrain models |
| Weekends | Weekend | Deep Research | Extended backtests, discovery |

**Auto-Scheduled Tasks:**
- Scan for signals (every 30 min during trading hours)
- Check positions P&L (every 5 min)
- Reconcile broker (hourly)
- Run random parameter experiments (every 2 hours during research)
- Daily reflection (after market close)
- Retrain ML models (nightly)
- Data quality checks (every 3 hours)

**State:** `state/autonomous/`

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
- **Exits**: ATR(14)x2 stop + 7-bar time stop (IBS_RSI)
- **Execution**: IOC LIMIT only (limit = best_ask x 1.001)
- **Kill switch**: Create `state/KILL_SWITCH` to halt submissions
- **Weekend scanning**: Auto-uses Friday's close + preview mode (bypasses shift(1))
- **Kill zones**: NO trades 9:30-10:00 AM (opening range) or 11:30-14:30 (lunch chop)
- **Watchlist-first**: Trade from validated watchlist; fallback requires higher bar
- **Position limits**: 10% per position, 20% daily, 40% weekly exposure caps

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
- `strategies/ibs_rsi/strategy.py`: IBS<0.08 + RSI(2)<5 entry, SMA(200) filter — v2.2 (59.9% WR, 1.46 PF)
- `strategies/ict/turtle_soup.py`: Turtle Soup (sweep≥0.3 ATR) — v2.2 (61.0% WR, 1.37 PF)
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
- `ml_features/pca_reducer.py`: **NEW** PCA dimensionality reduction (95% variance)
- `ml_features/feature_pipeline.py`: **NEW** Lag features (t-1 to t-20) + time/calendar features
- `ml/alpha_discovery/rl_agent/agent.py`: PPO/DQN/A2C RL trading agent

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







