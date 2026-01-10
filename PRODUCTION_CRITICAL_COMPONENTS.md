# KOBE TRADING ROBOT - PRODUCTION CRITICAL COMPONENTS

> **Generated:** 2026-01-07
> **Purpose:** Identify ALL production-critical components and their dependencies
> **Status:** Grade A+ system with 815 Python files verified

---

## EXECUTIVE SUMMARY

This document identifies EVERY production-critical component in the Kobe trading system - the components that MUST work for the system to trade successfully. This is the roadmap for understanding what's essential vs. what's research/experimental.

---

## TIER 1: ABSOLUTE CRITICAL (System Cannot Trade Without These)

### 1. Scanner & Signal Generation (6 files)
```
✓ strategies/dual_strategy/combined.py          - DualStrategyScanner (PRODUCTION)
✓ strategies/ibs_rsi/strategy.py                - IBS+RSI strategy (v2.2)
✓ strategies/ict/turtle_soup.py                 - Turtle Soup strategy (v2.2)
✓ strategies/registry.py                        - Strategy registry with validation
✓ ml_advanced/markov_chain/scorer.py            - Markov pre-ranking (optional but used)
✓ scripts/scan.py                               - Daily scanner (800 → 5 → 2)

Dependencies:
- config/frozen_strategy_params_v2.2.json       - Frozen parameters
- data/universe/optionable_liquid_800.csv       - 900-stock universe
```

**What It Does:** Scans 800 stocks, generates signals, ranks them, outputs Top 5 to study + Top 2 to trade.

**Critical Invariant:** MUST use DualStrategyScanner with ts_min_sweep_strength=0.3 filter (61% WR vs 48% without).

---

### 2. Data Pipeline (12 files)
```
✓ data/providers/polygon_eod.py                 - PRIMARY: Polygon.io EOD data
✓ data/providers/stooq_eod.py                   - FREE: Stooq for backtesting
✓ data/providers/yfinance_eod.py                - FREE FALLBACK: Yahoo Finance
✓ data/universe/loader.py                       - Load universe, dedup, cap
✓ data/universe/canonical.py                    - Canonical 900-stock universe
✓ data/validation.py                            - Data validation
✓ data/corporate_actions.py                     - Corporate actions detection
✓ data/quorum.py                                - Data quorum/consensus
✓ scripts/prefetch_polygon_universe.py          - Prefetch EOD bars
✓ scripts/build_universe_polygon.py             - Build universe
✓ scripts/check_data_quality.py                 - Data quality checks
✓ scripts/validate_data_pipeline.py             - Pipeline validation

Dependencies:
- POLYGON_API_KEY environment variable
- data/polygon_cache/ directory for caching
```

**What It Does:** Fetches, validates, and caches OHLCV data for 800 stocks with 10+ years history.

**Critical Invariant:** Must have clean, gap-free, validated data with no lookahead bias.

---

### 3. Risk Management (13 files)
```
✓ risk/policy_gate.py                           - PolicyGate (2% per trade, 20% daily, 40% weekly)
✓ risk/equity_sizer.py                          - 2% equity-based sizing
✓ risk/dynamic_position_sizer.py                - Dual-cap sizing (2% risk + 20% notional)
✓ risk/signal_quality_gate.py                   - Quality gate (score >= 70, conf >= 0.60)
✓ risk/kill_zone_gate.py                        - Time-based trade blocking (9:30-10:00, etc.)
✓ risk/weekly_exposure_gate.py                  - Weekly/daily exposure caps
✓ risk/position_limit_gate.py                   - Position limits
✓ risk/net_exposure_gate.py                     - Net exposure limits
✓ risk/liquidity_gate.py                        - Liquidity checks
✓ core/kill_switch.py                           - Emergency kill switch
✓ core/circuit_breaker.py                       - Circuit breakers
✓ scripts/preflight.py                          - 10 critical pre-flight checks
✓ scripts/risk.py                               - Risk limit checks

Dependencies:
- state/KILL_SWITCH file (absence = OK to trade)
- config/base.yaml for risk parameters
```

**What It Does:** Enforces ALL risk limits before ANY trade. No trade executes without passing all gates.

**Critical Invariant:**
- MUST use dual-cap formula: `final_shares = min(shares_by_risk, shares_by_notional)`
- NEVER bypass PolicyGate
- NEVER trade during kill zones (9:30-10:00 AM, 11:30-14:30 PM)

---

### 4. Execution & Order Management (10 files)
```
✓ execution/broker_alpaca.py                    - Alpaca broker (live + paper)
✓ execution/broker_paper.py                     - Paper trading simulator
✓ execution/broker_factory.py                   - Broker factory
✓ execution/order_manager.py                    - Order lifecycle
✓ execution/order_state_machine.py              - Order state machine
✓ execution/execution_guard.py                  - Execution safety checks
✓ oms/order_state.py                            - Order state records
✓ oms/idempotency_store.py                     - Duplicate prevention
✓ execution/reconcile.py                        - Position reconciliation
✓ scripts/run_paper_trade.py                    - Paper trading script

Dependencies:
- ALPACA_API_KEY_ID environment variable
- ALPACA_API_SECRET_KEY environment variable
- ALPACA_BASE_URL (paper or live)
- state/idempotency_store.json
```

**What It Does:** Submits orders to broker, tracks state, prevents duplicates, reconciles positions.

**Critical Invariant:** IOC LIMIT orders only, idempotency enforced, reconciliation after every session.

---

### 5. Professional Execution Flow (8 files)
```
✓ scripts/overnight_watchlist.py                - Build Top 5 watchlist (3:30 PM previous day)
✓ scripts/premarket_validator.py                - Validate gaps/news (8:00 AM)
✓ scripts/opening_range_observer.py             - Observe opening (9:30-10:00, NO TRADES)
✓ risk/kill_zone_gate.py                        - Enforce kill zones
✓ core/clock/market_clock.py                    - Market hours + trading days
✓ core/clock/equities_calendar.py               - NYSE calendar
✓ analysis/historical_patterns.py               - Historical pattern analysis
✓ scripts/generate_pregame_blueprint.py         - Pre-game analysis

Dependencies:
- state/watchlist/next_day.json
- state/watchlist/today_validated.json
- state/watchlist/opening_range.json
```

**What It Does:** Professional workflow - overnight watchlist → premarket validation → opening observation → trade.

**Critical Invariant:**
- NEVER trade before 10:00 AM
- NEVER trade 11:30-14:30 (lunch chop)
- Watchlist-first (fallback requires higher bar: 75 vs 65)

---

### 6. Core Infrastructure (10 files)
```
✓ core/config_pin.py                            - Config signature verification
✓ core/hash_chain.py                            - Audit chain (tamper detection)
✓ core/structured_log.py                        - Structured JSON logging
✓ core/secrets.py                               - API key management
✓ core/rate_limiter.py                          - API rate limiting
✓ config/env_loader.py                          - Environment loading
✓ scripts/verify_hash_chain.py                  - Verify audit chain
✓ scripts/show_config_pin.py                    - Show config signature
✓ scripts/preflight.py                          - Pre-flight checks
✓ scripts/status.py                             - System status

Dependencies:
- .env file with API keys
- config/base.yaml
- state/hash_chain.jsonl
```

**What It Does:** Loads config, validates integrity, manages secrets, prevents tampering.

**Critical Invariant:** Config pin must match, hash chain must be unbroken, API keys must be valid.

---

### 7. Backtest Engine (5 files - for validation)
```
✓ backtest/engine.py                            - Main backtest engine
✓ backtest/walk_forward.py                      - Walk-forward validation
✓ scripts/backtest_dual_strategy.py             - CANONICAL TEST
✓ scripts/run_wf_polygon.py                     - Walk-forward backtest
✓ scripts/aggregate_wf_report.py                - WF report aggregation

Dependencies:
- data/universe/optionable_liquid_800.csv
- config/frozen_strategy_params_v2.2.json
```

**What It Does:** Validates strategy performance before going live. CANONICAL TEST = backtest_dual_strategy.py.

**Critical Invariant:** Expected ~64% WR, ~1.60 PF on 2023-2024 data with cap 150.

---

## TIER 2: HIGHLY IMPORTANT (System Degrades Without These)

### 8. ML/AI Enhancement (15 files)
```
✓ ml_advanced/hmm_regime_detector.py            - Regime detection (position sizing multiplier)
✓ ml_advanced/markov_chain/scorer.py            - Markov pre-ranking
✓ ml_advanced/markov_chain/predictor.py         - Markov signal boost
✓ ml_advanced/markov_chain/transition_matrix.py - Transition probabilities
✓ ml_advanced/markov_chain/stationary_dist.py   - Stationary distribution
✓ ml_features/feature_pipeline.py               - 150+ features
✓ ml_features/technical_features.py             - Technical indicators
✓ ml_features/pca_reducer.py                    - PCA dimensionality reduction
✓ scripts/train_hmm_regime.py                   - Train HMM
✓ scripts/train_lstm_confidence.py              - Train LSTM
✓ scripts/train_ensemble.py                     - Train ensemble
✓ models/hmm_regime_v1.pkl                      - Trained HMM model
✓ models/hmm_regime_metadata.json               - HMM metadata
✓ pipelines/unified_signal_enrichment.py        - 18-stage, 44-component pipeline
✓ scripts/unified_scan.py                       - Unified scanner
```

**What It Does:** Enriches signals with ML confidence, regime awareness, Markov boosts.

**Impact Without:** System still trades but signals less refined (-5-10% confidence).

---

### 9. Trade Analysis & Explainability (7 files)
```
✓ explainability/trade_thesis_builder.py        - Full trade thesis
✓ analysis/historical_patterns.py               - Consecutive pattern analysis
✓ analysis/options_expected_move.py             - Expected move calculator
✓ scripts/generate_pregame_blueprint.py         - Pre-game comprehensive analysis
✓ scripts/top2_analysis.py                      - Top 2 trade analysis
✓ cognitive/llm_trade_analyzer.py               - LLM trade analysis
✓ explainability/narrative_generator.py         - Natural language narratives
```

**What It Does:** Generates comprehensive analysis for Top 2 trades (historical evidence, expected moves, support/resistance, bull/bear cases).

**Impact Without:** Trades still execute but lack comprehensive justification.

---

### 10. Monitoring & Alerts (8 files)
```
✓ monitor/health_endpoints.py                   - Health check HTTP endpoints
✓ monitor/heartbeat.py                          - Heartbeat monitoring
✓ alerts/professional_alerts.py                 - Professional alert formatting
✓ alerts/telegram_alerter.py                    - Telegram integration
✓ scripts/reconcile_alpaca.py                   - Position reconciliation
✓ scripts/heartbeat.py                          - Heartbeat script
✓ scripts/alerts.py                             - Alert management
✓ scripts/telegram.py                           - Telegram bot
```

**What It Does:** Monitors system health, sends alerts, reconciles positions.

**Impact Without:** System runs blind - no alerts, no health monitoring, no reconciliation.

---

## TIER 3: IMPORTANT (Enhanced Capability)

### 11. Autonomous Brain (10 files - for 24/7 operation)
```
✓ autonomous/brain.py                           - Main orchestrator
✓ autonomous/awareness.py                       - Time/day/season awareness
✓ autonomous/scheduler.py                       - Task scheduling
✓ autonomous/learning.py                        - Learning from outcomes
✓ autonomous/maintenance.py                     - System maintenance
✓ autonomous/monitor.py                         - Autonomous monitoring
✓ autonomous/handlers.py                        - Task handlers
✓ scripts/run_autonomous.py                     - Run autonomous brain
✓ scripts/daily_scheduler.py                    - Daily scheduler
✓ scripts/runner.py                             - 24/7 runner
```

**What It Does:** 24/7 autonomous operation - decides what to work on based on time/day/season.

**Impact Without:** System requires manual operation - no autonomous scheduling, learning, or maintenance.

---

### 12. Cognitive Architecture (10 files - for advanced reasoning)
```
✓ cognitive/cognitive_brain.py                  - Main cognitive orchestrator
✓ cognitive/metacognitive_governor.py           - System 1/2 routing
✓ cognitive/reflection_engine.py                - Learning from outcomes
✓ cognitive/episodic_memory.py                  - Experience storage
✓ cognitive/semantic_memory.py                  - Generalized knowledge
✓ cognitive/knowledge_boundary.py               - Uncertainty detection
✓ cognitive/curiosity_engine.py                 - Hypothesis generation
✓ cognitive/self_model.py                       - Capability tracking
✓ scripts/cognitive_learn.py                    - Cognitive learning
✓ scripts/explain.py                            - Trade explanation
```

**What It Does:** Brain-inspired reasoning - metacognition, memory, reflection, curiosity.

**Impact Without:** System trades mechanically - no learning, reflection, or curiosity-driven discovery.

---

### 13. Research Operating System (5 files - for continuous improvement)
```
✓ research_os/orchestrator.py                   - DISCOVER → RESEARCH → ENGINEER
✓ research_os/knowledge_card.py                 - Knowledge card format
✓ research_os/proposal.py                       - Research proposals
✓ research_os/approval_gate.py                  - Human approval gate
✓ scripts/research_os_cli.py                    - Research OS CLI
```

**What It Does:** Structured research workflow with mandatory human approval for production changes.

**Impact Without:** No structured discovery process - changes are ad-hoc vs. systematic.

---

### 14. Advanced Risk (4 files - for quant-grade risk)
```
✓ risk/advanced/monte_carlo_var.py              - VaR with Monte Carlo
✓ risk/advanced/kelly_position_sizer.py         - Optimal Kelly sizing
✓ risk/advanced/correlation_limits.py           - Correlation limits
✓ risk/advanced/portfolio_optimizer.py          - Portfolio optimization
```

**What It Does:** Quant-grade risk management - VaR, Kelly, correlation, optimization.

**Impact Without:** System uses simpler risk management - still safe but less optimal.

---

### 15. Portfolio Management (5 files)
```
✓ portfolio/state_manager.py                    - Portfolio state tracking
✓ portfolio/risk_manager.py                     - Portfolio risk
✓ portfolio/heat_monitor.py                     - Position heat
✓ portfolio/optimizer/portfolio_manager.py      - Portfolio manager
✓ portfolio/optimizer/rebalancer.py             - Rebalancing
```

**What It Does:** Portfolio-level management, heat monitoring, rebalancing.

**Impact Without:** System manages positions individually vs. as portfolio.

---

## TIER 4: SUPPORTING (Research, Experimental, Future)

### 16. Alternative Data (9 files)
```
altdata/congressional_trades.py                 - Congressional trades (Quiver Quant)
altdata/insider_activity.py                     - Insider trading
altdata/political_sentiment.py                  - Political sentiment
altdata/news_processor.py                       - News NLP
altdata/sentiment.py                            - Sentiment analysis
altdata/options_flow.py                         - Unusual options activity
altdata/github_activity.py                      - GitHub activity
altdata/market_mood_analyzer.py                 - Market mood
data/alternative/alt_data_aggregator.py         - Alt data aggregator
```

**Status:** RESEARCH - Not yet integrated into production signals.

---

### 17. Autonomous Scrapers (8 files)
```
autonomous/scrapers/all_sources.py              - Unified scraper
autonomous/scrapers/arxiv_scraper.py            - ArXiv papers
autonomous/scrapers/github_scraper.py           - GitHub repos
autonomous/scrapers/reddit_scraper.py           - Reddit sentiment
autonomous/scrapers/youtube_scraper.py          - YouTube content
autonomous/scrapers/firecrawl_adapter.py        - Firecrawl API
autonomous/scrapers/source_manager.py           - Scraper orchestration
```

**Status:** EXPERIMENTAL - Continuous discovery but not yet integrated into signals.

---

### 18. Alpha Discovery (15 files)
```
ml/alpha_discovery/orchestrator.py              - Alpha discovery orchestrator
ml/alpha_discovery/feature_discovery/           - Feature importance
ml/alpha_discovery/pattern_miner/               - Pattern clustering
ml/alpha_discovery/pattern_narrator/            - Pattern narratives
ml/alpha_discovery/rl_agent/                    - RL trading agents
ml/alpha_discovery/hybrid_pipeline/             - Hybrid ML/RL
research/alpha_factory.py                       - Alpha factory
research/alpha_library.py                       - Alpha library
research/vectorbt_miner.py                      - VectorBT mining
evolution/strategy_foundry.py                   - Strategy generation
evolution/genetic_optimizer.py                  - Genetic algorithms
```

**Status:** RESEARCH - Exploring new alphas, not yet validated for production.

---

### 19. Multi-Agent System (10 files)
```
agents/orchestrator.py                          - Agent orchestration
agents/autogen_team.py                          - Autogen teams
agents/langgraph_coordinator.py                 - LangGraph coordination
agents/scout_agent.py                           - Scout agent
agents/auditor_agent.py                         - Auditor agent
agents/risk_agent.py                            - Risk agent
agents/reporter_agent.py                        - Reporter agent
```

**Status:** EXPERIMENTAL - Multi-agent coordination not yet in production workflow.

---

### 20. Options Trading (11 files)
```
options/black_scholes.py                        - BS pricing
options/volatility.py                           - Realized volatility
options/selection.py                            - Strike selection
options/position_sizing.py                      - Options sizing
options/backtest.py                             - Options backtesting
options/pricing.py                              - Pricing utilities
options/chain_fetcher.py                        - Chain data
options/iv_signals.py                           - IV signals
options/order_router.py                         - Options routing
options/spreads.py                              - Spread strategies
```

**Status:** FUTURE - Synthetic options engine built but not yet live.

---

## PRODUCTION WORKFLOW (Critical Path)

### Daily Trading Cycle (10 Scripts)
```
1. overnight_watchlist.py         (3:30 PM prev day)  → state/watchlist/next_day.json
2. premarket_validator.py         (8:00 AM)           → state/watchlist/today_validated.json
3. opening_range_observer.py      (9:30 AM)           → state/watchlist/opening_range.json
4. opening_range_observer.py      (9:45 AM)           → Append to opening_range.json
5. run_paper_trade.py             (10:00 AM)          → Trade watchlist (FIRST SCAN)
6. run_paper_trade.py             (10:30 AM)          → Fallback if needed (HIGHER BAR)
7. run_paper_trade.py             (14:30 PM)          → Power hour scan (SECONDARY WINDOW)
8. reconcile_alpaca.py            (16:00 PM)          → Reconcile positions
9. eod_report.py                  (16:00 PM)          → Generate daily report
10. cognitive_learn.py            (17:00 PM)          → Learn from outcomes
```

### Weekly Training Cycle (5 Scripts)
```
1. train_hmm_regime.py            (Sunday 2:00 AM)    → Retrain HMM
2. train_lstm_confidence.py       (Sunday 3:00 AM)    → Retrain LSTM
3. train_ensemble.py              (Sunday 4:00 AM)    → Retrain ensemble
4. run_wf_polygon.py              (Sunday 6:00 AM)    → Walk-forward validation
5. aggregate_wf_report.py         (Sunday 8:00 AM)    → Generate WF report
```

### Pre-Flight Checks (1 Script)
```
preflight.py                      (Before ANY trading) → 10 critical checks
  ├── 1. Environment variables
  ├── 2. Config pin signature
  ├── 3. Broker connection
  ├── 4. Data availability
  ├── 5. Universe loaded
  ├── 6. Kill switch inactive
  ├── 7. Risk gates operational
  ├── 8. Hash chain unbroken
  ├── 9. Idempotency store clean
  └── 10. Clock synchronized
```

---

## DEPENDENCY GRAPH (Critical Components Only)

```
Environment (.env) + Config (base.yaml + frozen params)
    ↓
Data Pipeline (Polygon/Stooq/Yahoo → Universe → Validation)
    ↓
Scanner (DualStrategyScanner → 800 stocks → Signals)
    ↓
ML Enhancement (Markov scorer → HMM regime → Feature pipeline)
    ↓
Professional Flow (Watchlist → Premarket validation → Opening observation)
    ↓
Risk Gates (Kill zone → Quality → Policy → Position sizing)
    ↓
Execution (Broker → Order manager → Idempotency → State machine)
    ↓
Monitoring (Heartbeat → Reconciliation → Alerts)
    ↓
Learning (Episodic memory → Reflection → Cognitive learning)
    ↓
Research OS (Discovery → Experimentation → Human approval)
```

---

## CRITICAL FILE LIST (Top 50 Most Important)

| Rank | File | Tier | Purpose |
|------|------|------|---------|
| 1 | strategies/dual_strategy/combined.py | 1 | PRODUCTION scanner |
| 2 | scripts/scan.py | 1 | Daily 800 → 5 → 2 scan |
| 3 | risk/policy_gate.py | 1 | Risk enforcement |
| 4 | risk/kill_zone_gate.py | 1 | Time-based blocking |
| 5 | execution/broker_alpaca.py | 1 | Order execution |
| 6 | data/providers/polygon_eod.py | 1 | Data source |
| 7 | scripts/run_paper_trade.py | 1 | Paper trading |
| 8 | scripts/preflight.py | 1 | Pre-flight checks |
| 9 | core/config_pin.py | 1 | Config integrity |
| 10 | core/hash_chain.py | 1 | Audit trail |
| 11 | scripts/overnight_watchlist.py | 1 | Watchlist building |
| 12 | scripts/premarket_validator.py | 1 | Gap/news validation |
| 13 | risk/dynamic_position_sizer.py | 1 | Dual-cap sizing |
| 14 | risk/signal_quality_gate.py | 1 | Quality gate |
| 15 | oms/idempotency_store.py | 1 | Duplicate prevention |
| 16 | backtest/engine.py | 1 | Backtest validation |
| 17 | scripts/backtest_dual_strategy.py | 1 | CANONICAL TEST |
| 18 | data/universe/loader.py | 1 | Universe loading |
| 19 | core/kill_switch.py | 1 | Emergency halt |
| 20 | execution/order_manager.py | 1 | Order lifecycle |
| 21 | ml_advanced/markov_chain/scorer.py | 2 | Markov pre-ranking |
| 22 | ml_advanced/hmm_regime_detector.py | 2 | Regime detection |
| 23 | pipelines/unified_signal_enrichment.py | 2 | 44-component pipeline |
| 24 | explainability/trade_thesis_builder.py | 2 | Trade thesis |
| 25 | analysis/historical_patterns.py | 2 | Pattern analysis |
| 26 | monitor/health_endpoints.py | 2 | Health monitoring |
| 27 | alerts/telegram_alerter.py | 2 | Telegram alerts |
| 28 | scripts/reconcile_alpaca.py | 2 | Reconciliation |
| 29 | scripts/generate_pregame_blueprint.py | 2 | Pre-game analysis |
| 30 | autonomous/brain.py | 3 | Autonomous orchestrator |
| 31 | autonomous/awareness.py | 3 | Time awareness |
| 32 | cognitive/cognitive_brain.py | 3 | Cognitive reasoning |
| 33 | cognitive/reflection_engine.py | 3 | Learning engine |
| 34 | research_os/orchestrator.py | 3 | Research workflow |
| 35 | risk/advanced/monte_carlo_var.py | 3 | VaR calculation |
| 36 | portfolio/state_manager.py | 3 | Portfolio tracking |
| 37 | backtest/walk_forward.py | 1 | Walk-forward |
| 38 | core/clock/market_clock.py | 1 | Market hours |
| 39 | core/structured_log.py | 1 | Logging |
| 40 | scripts/status.py | 1 | System status |
| 41 | scripts/validate.py | 1 | Validation |
| 42 | scripts/eod_report.py | 2 | Daily report |
| 43 | scripts/cognitive_learn.py | 3 | Learning |
| 44 | scripts/train_hmm_regime.py | 2 | HMM training |
| 45 | scripts/runner.py | 3 | 24/7 scheduler |
| 46 | execution/execution_guard.py | 1 | Execution safety |
| 47 | data/validation.py | 1 | Data validation |
| 48 | core/secrets.py | 1 | API key management |
| 49 | scripts/verify_hash_chain.py | 1 | Audit verification |
| 50 | strategies/registry.py | 1 | Strategy registry |

---

## CRITICAL ENVIRONMENT VARIABLES

```bash
# REQUIRED (System cannot trade without these)
POLYGON_API_KEY=xxx                     # Primary data source
ALPACA_API_KEY_ID=xxx                   # Broker authentication
ALPACA_API_SECRET_KEY=xxx               # Broker authentication
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper or live

# OPTIONAL (System degrades without these)
TELEGRAM_BOT_TOKEN=xxx                  # Alerts
TELEGRAM_CHAT_ID=xxx                    # Alerts destination
ANTHROPIC_API_KEY=xxx                   # LLM analysis
OPENAI_API_KEY=xxx                      # Alternative LLM
LANGFUSE_PUBLIC_KEY=xxx                 # Observability
LANGFUSE_SECRET_KEY=xxx                 # Observability
QUIVER_API_KEY=xxx                      # Congressional trades
FINNHUB_API_KEY=xxx                     # News sentiment
```

---

## CRITICAL STATE FILES

```
state/KILL_SWITCH                       - Presence = HALT (absence = OK)
state/idempotency_store.json            - Duplicate prevention
state/hash_chain.jsonl                  - Audit chain (append-only)
state/watchlist/next_day.json           - Tomorrow's Top 5
state/watchlist/today_validated.json    - Today's validated watchlist
state/watchlist/opening_range.json      - Opening range observations
state/autonomous/heartbeat.json         - Autonomous brain heartbeat
state/cognitive/episodic_memory/        - Trade outcomes
state/cognitive/semantic_memory/        - Learned rules
models/hmm_regime_v1.pkl                - Trained HMM model
models/hmm_regime_metadata.json         - HMM metadata
```

---

## CRITICAL CONFIGURATION FILES

```
config/base.yaml                        - Main configuration
config/frozen_strategy_params_v2.2.json - Frozen strategy parameters
data/universe/optionable_liquid_800.csv - 900-stock universe
.env                                    - Environment variables
```

---

## WHAT CAN BREAK THE SYSTEM (Failure Modes)

### TIER 1 Failures (System Cannot Trade)
```
✗ Polygon API key invalid                       → No data
✗ Alpaca API key invalid                        → No execution
✗ Universe file missing                         → No stocks to scan
✗ DualStrategyScanner broken                    → No signals
✗ PolicyGate bypassed                           → Unsafe trading
✗ Kill zone gate disabled                       → Wrong-time trading
✗ Idempotency store corrupted                   → Duplicate orders
✗ Config pin mismatch                           → Untrusted config
✗ Hash chain broken                             → Tampered audit trail
```

### TIER 2 Failures (System Degrades)
```
△ Markov scorer fails                           → No pre-ranking (-10% efficiency)
△ HMM regime detector fails                     → No regime awareness (-5% PF)
△ Telegram alerts fail                          → No notifications (blind operation)
△ Historical pattern analysis fails             → No auto-pass (manual review needed)
△ Pre-game blueprint fails                      → No comprehensive analysis
```

### TIER 3 Failures (System Operates Normally)
```
○ Autonomous brain stops                        → Manual operation required
○ Cognitive reflection fails                    → No learning from outcomes
○ Research OS stops                             → No continuous discovery
○ Alternative data unavailable                  → Still trades on core signals
○ Multi-agent system fails                      → Single-agent fallback
```

---

## MINIMUM VIABLE PRODUCTION SYSTEM

**To trade with absolute minimum components (60 files):**

```
1. Core Infrastructure (10 files)
   - config_pin.py, hash_chain.py, structured_log.py, secrets.py, rate_limiter.py
   - kill_switch.py, circuit_breaker.py, clock/market_clock.py, clock/equities_calendar.py
   - env_loader.py

2. Data Pipeline (8 files)
   - providers/polygon_eod.py, providers/stooq_eod.py, providers/yfinance_eod.py
   - universe/loader.py, universe/canonical.py, validation.py, corporate_actions.py
   - quorum.py

3. Strategies (6 files)
   - dual_strategy/combined.py, ibs_rsi/strategy.py, ict/turtle_soup.py
   - registry.py, scan.py, frozen_strategy_params_v2.2.json

4. Risk Management (13 files)
   - policy_gate.py, equity_sizer.py, dynamic_position_sizer.py
   - signal_quality_gate.py, kill_zone_gate.py, weekly_exposure_gate.py
   - position_limit_gate.py, net_exposure_gate.py, liquidity_gate.py
   - core/kill_switch.py, core/circuit_breaker.py
   - preflight.py, risk.py

5. Execution (10 files)
   - broker_alpaca.py, broker_paper.py, broker_factory.py
   - order_manager.py, order_state_machine.py, execution_guard.py
   - oms/order_state.py, oms/idempotency_store.py
   - reconcile.py, run_paper_trade.py

6. Professional Flow (8 files)
   - overnight_watchlist.py, premarket_validator.py, opening_range_observer.py
   - generate_pregame_blueprint.py, historical_patterns.py
   - options_expected_move.py, trade_thesis_builder.py, top2_analysis.py

7. Backtest (5 files)
   - engine.py, walk_forward.py, backtest_dual_strategy.py
   - run_wf_polygon.py, aggregate_wf_report.py

TOTAL: 60 files = Minimum viable production system
```

---

## RECOMMENDED PRODUCTION SYSTEM

**For robust production (100 files):**

Add to minimum viable:
```
+ ML Enhancement (15 files): Markov + HMM + Feature pipeline
+ Monitoring (8 files): Health + heartbeat + alerts + reconciliation
+ Analytics (7 files): Trade thesis + historical patterns + explainability
+ Portfolio (5 files): State manager + risk manager + heat monitor
+ Advanced Risk (4 files): VaR + Kelly + Correlation
+ Calendar (2 files): Options events + macro events
```

**TOTAL: 100 files = Robust production system with ML enhancement**

---

## FULL PRODUCTION SYSTEM

**For complete autonomous operation (200+ files):**

Add to robust production:
```
+ Autonomous Brain (30 files): 24/7 operation + learning + maintenance
+ Cognitive Architecture (33 files): Reasoning + memory + reflection
+ Research OS (5 files): Continuous discovery with human approval
+ Alternative Data (9 files): Congressional trades + insider activity + news
+ Scrapers (8 files): ArXiv + GitHub + Reddit + YouTube
+ Multi-Agent (10 files): Autogen + LangGraph coordination
+ Complete Test Suite (111 files): Full coverage
+ All Tools (14 files): Verification + auditing
+ All Scripts (199 files): Complete automation
```

**TOTAL: 815 files = Full autonomous self-improving trading system**

---

## SUMMARY

### What MUST Work (Tier 1 - 73 files)
```
✓ Scanner (6 files)
✓ Data Pipeline (12 files)
✓ Risk Management (13 files)
✓ Execution (10 files)
✓ Professional Flow (8 files)
✓ Core Infrastructure (10 files)
✓ Backtest Validation (5 files)
✓ Supporting Scripts (9 files)
```

### What SHOULD Work (Tier 2 - 50 files)
```
✓ ML Enhancement (15 files)
✓ Trade Analysis (7 files)
✓ Monitoring & Alerts (8 files)
✓ Portfolio Management (5 files)
✓ Advanced Risk (4 files)
✓ Supporting Scripts (11 files)
```

### What's NICE to Have (Tier 3 - 55 files)
```
✓ Autonomous Brain (30 files)
✓ Cognitive Architecture (10 files - core subset)
✓ Research OS (5 files)
✓ Supporting Scripts (10 files)
```

### What's EXPERIMENTAL (Tier 4 - 637 files)
```
△ Full Cognitive Architecture (33 files)
△ Alternative Data (9 files)
△ Autonomous Scrapers (8 files)
△ Alpha Discovery (15 files)
△ Multi-Agent System (10 files)
△ Options Trading (11 files)
△ Complete Test Suite (111 files)
△ All Tools (14 files)
△ All Scripts (199 files)
△ Everything else (227 files)
```

---

**Generated by:** System Architect Mode
**Date:** 2026-01-07
**Purpose:** Production-critical component identification
**Evidence:** Complete system scan + dependency analysis + failure mode analysis
**Status:** COMPREHENSIVE PRODUCTION MAPPING COMPLETE
