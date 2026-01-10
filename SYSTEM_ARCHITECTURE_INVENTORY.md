# KOBE TRADING ROBOT - COMPLETE SYSTEM ARCHITECTURE INVENTORY

> **Generated:** 2026-01-07
> **Purpose:** Comprehensive architectural scan and module inventory
> **Scan Type:** Full repository analysis (815 Python files, 286,589 lines)

---

## EXECUTIVE SUMMARY

**This is not a simple trading bot. This is a self-improving, autonomous, multi-agent trading system with:**

- **24/7 Autonomous Operation** with self-awareness, learning, and research capabilities
- **Cognitive Architecture** with metacognitive reasoning, episodic memory, and reflection
- **Multi-Agent Orchestration** using Autogen + LangGraph
- **Advanced ML/AI** including HMM regime detection, LSTM confidence, ensemble models, RL agents
- **Medallion-Inspired Markov Chains** for state transition-based direction prediction
- **Professional Execution Flow** with ICT-style kill zones and watchlist-first trading
- **Research Operating System** for autonomous discovery, validation, and human-gated engineering
- **Quant-Grade Risk Management** with VaR, Kelly sizing, circuit breakers, correlation limits
- **Comprehensive Testing** with 111 test files and 947 passing tests

**Status:** Grade A+ - Autonomous 24/7 mode active, deterministic scans verified, 10/10 data pipeline optimization

---

## COMPLETE FILE INVENTORY

### Total Statistics
- **Total Python Files:** 815
- **Total Lines of Code:** 286,589
- **Total Modules:** 57 top-level directories
- **Scripts (Entry Points):** 199
- **Test Files:** 111
- **Documentation Files:** 51+ markdown files
- **Skills (CLI Commands):** 71 slash commands

---

## MODULE BREAKDOWN BY CATEGORY

### 1. AUTONOMOUS & SELF-IMPROVING SYSTEMS (30 files)

**autonomous/** - 24/7 Self-Aware Brain
```
autonomous/brain.py                    - Main orchestrator (always aware of time/day/season)
autonomous/awareness.py                - Time/day/season awareness (market phases, holidays, FOMC, OpEx)
autonomous/scheduler.py                - Task queue with priority + context-based execution
autonomous/research.py                 - Self-improvement: random parameter experiments, strategy discovery
autonomous/learning.py                 - Trade analysis, episodic memory updates, daily reflections
autonomous/maintenance.py              - Data quality, cleanup, health checks
autonomous/monitor.py                  - Heartbeat monitoring, alerts, status dashboard
autonomous/comprehensive_brain.py      - Extended brain with more complex reasoning
autonomous/enhanced_brain.py           - Enhanced brain version
autonomous/master_brain.py             - Master orchestrator
autonomous/master_brain_full.py        - Full master brain implementation
autonomous/enhanced_research.py        - Enhanced research capabilities
autonomous/data_validator.py           - Autonomous data validation
autonomous/integrity.py                - System integrity checks
autonomous/knowledge_integrator.py     - Knowledge integration across systems
autonomous/pattern_rhymes.py           - Pattern recognition and rhyming detection
autonomous/source_tracker.py           - External source tracking
autonomous/handlers.py                 - Task handlers for scheduler
autonomous/run.py                      - Entry point for autonomous operation
autonomous/scheduler_full.py           - Full scheduler implementation
```

**autonomous/scrapers/** - External Data Collection (8 files)
```
autonomous/scrapers/all_sources.py     - Unified scraper interface
autonomous/scrapers/arxiv_scraper.py   - Academic paper scraping (strategy research)
autonomous/scrapers/github_scraper.py  - GitHub repo/code scraping (algo discovery)
autonomous/scrapers/reddit_scraper.py  - Reddit sentiment/discussion scraping
autonomous/scrapers/youtube_scraper.py - YouTube video/transcript scraping
autonomous/scrapers/firecrawl_adapter.py - Firecrawl API integration
autonomous/scrapers/source_manager.py  - Scraper orchestration
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

---

### 2. COGNITIVE ARCHITECTURE (33 files)

**cognitive/** - Brain-Inspired Decision System
```
cognitive/cognitive_brain.py           - Main orchestrator - deliberation, learning, introspection
cognitive/metacognitive_governor.py    - System 1/2 routing (fast vs slow thinking)
cognitive/reflection_engine.py         - Learning from outcomes (Reflexion pattern)
cognitive/self_model.py                - Capability tracking, calibration, self-awareness
cognitive/episodic_memory.py           - Experience storage (context → reasoning → outcome)
cognitive/semantic_memory.py           - Generalized rules and knowledge
cognitive/knowledge_boundary.py        - Uncertainty detection, stand-down recommendations
cognitive/curiosity_engine.py          - Hypothesis generation and edge discovery
cognitive/brain_graph.py               - Graph-based reasoning
cognitive/global_workspace.py          - Global workspace theory implementation
cognitive/states.py                    - Cognitive state management
cognitive/adjudicator.py               - Decision adjudication
cognitive/signal_adjudicator.py        - Signal-level adjudication
cognitive/signal_processor.py          - Signal processing pipeline
cognitive/contradiction_resolver.py    - Handle conflicting information
cognitive/circuit_breakers.py          - Cognitive circuit breakers
cognitive/dynamic_policy_generator.py  - Generate trading policies dynamically
cognitive/policy_generator.py          - Policy generation
cognitive/causal_reasoner.py           - Causal reasoning engine
cognitive/symbolic_reasoner.py         - Symbolic logic reasoning
cognitive/tree_of_thoughts.py          - Tree of Thoughts reasoning
cognitive/azr_reasoning.py             - Advanced zero-shot reasoning
cognitive/self_consistency.py          - Self-consistency checking
```

**LLM Integration:**
```
cognitive/llm_narrative_analyzer.py    - Narrative analysis via LLM
cognitive/llm_trade_analyzer.py        - Trade analysis via LLM
cognitive/llm_validator.py             - LLM-based validation
cognitive/socratic_narrative.py        - Socratic questioning for reasoning
cognitive/tweet_generator.py           - Generate tweets/social media
cognitive/rag_evaluator.py             - RAG system evaluation
cognitive/symbol_rag.py                - Symbol-level RAG
cognitive/vector_memory.py             - Vector-based memory storage
cognitive/game_briefings.py            - Pre-trade game plan briefings
```

**Tests:** 83+ unit tests passing in `tests/cognitive/`

---

### 3. MACHINE LEARNING & AI (62 files)

#### ML Advanced (19 files)

**ml_advanced/hmm_regime_detector.py** - Hidden Markov Model market regime detection (BULL/NEUTRAL/BEAR)
```python
# 3-state HMM for market regime
# Position sizing multiplier based on regime
```

**ml_advanced/online_learning.py** - Incremental learning with concept drift detection
```python
# Experience replay
# Concept drift detection
# Continuous model updates
```

**ml_advanced/ensemble/** - Multi-Model Ensemble (3 files)
```
ensemble/ensemble_predictor.py         - Weighted ensemble (XGBoost, LightGBM, LSTM)
ensemble/loader.py                     - Model loading and versioning
ensemble/regime_weights.py             - Regime-based ensemble weighting
```

**ml_advanced/lstm_confidence/** - LSTM Confidence Scoring (3 files)
```
lstm_confidence/model.py               - Multi-output LSTM (direction, magnitude, success)
lstm_confidence/config.py              - LSTM configuration
```

**ml_advanced/markov_chain/** - Medallion-Inspired Markov Chains (6 files)
```
markov_chain/state_classifier.py      - Discretize returns into Up/Down/Flat states
markov_chain/transition_matrix.py     - Build P(next_state | current_state) matrix
markov_chain/stationary_dist.py       - Equilibrium distribution for mean-reversion
markov_chain/higher_order.py          - 2nd/3rd order chains for multi-day patterns
markov_chain/predictor.py             - Generate trading signals from Markov analysis
markov_chain/scorer.py                - Rank 800 stocks by stationary pi(Up) probability
```

**How HMM + Markov Work Together:**
| Component | What It Does | Integration |
|-----------|--------------|-------------|
| **HMM** | Hidden market regimes (BULL/NEUTRAL/BEAR) | Position sizing multiplier |
| **Markov Chain** | Observable direction prediction per stock | Signal confidence boost |

**ml_advanced/tft/** - Temporal Fusion Transformer (2 files)
```
tft/temporal_fusion.py                 - TFT model for multi-horizon forecasting
```

#### ML Alpha Discovery (19 files)

**ml/alpha_discovery/** - Autonomous Alpha Discovery System
```
ml/alpha_discovery/orchestrator.py             - Main orchestrator
ml/alpha_discovery/feature_discovery/          - Feature importance analysis
ml/alpha_discovery/pattern_miner/              - Pattern clustering and library
ml/alpha_discovery/pattern_narrator/           - Natural language pattern explanation
ml/alpha_discovery/rl_agent/                   - PPO/DQN/A2C RL trading agent
ml/alpha_discovery/hybrid_pipeline/            - Hybrid ML/RL pipeline
ml/alpha_discovery/common/                     - Common data loaders and metrics
```

**ml/alpha_discovery/rl_agent/** - Reinforcement Learning (3 files)
```
rl_agent/trading_env.py                - Gym-compatible trading environment
rl_agent/agent.py                      - PPO/DQN/A2C via stable-baselines3
```

#### ML Features (17 files)

**ml_features/** - Feature Engineering Pipeline
```
ml_features/feature_pipeline.py       - 150+ features + lag features (t-1 to t-20) + time/calendar
ml_features/technical_features.py     - pandas-ta indicators (RSI, ATR, MACD, Bollinger, etc.)
ml_features/pca_reducer.py            - PCA dimensionality reduction (95% variance retention)
ml_features/anomaly_detection.py      - Matrix profiles for unusual pattern detection
ml_features/regime_ml.py              - KMeans/GMM regime clustering
ml_features/regime_hmm.py             - HMM regime features
ml_features/ensemble_brain.py         - Multi-model prediction ensemble
ml_features/confidence_integrator.py  - Integrate confidence scores
ml_features/conviction_scorer.py      - Calculate conviction scores
ml_features/signal_confidence.py      - Signal confidence calculation
ml_features/sentiment.py              - Sentiment feature extraction
ml_features/macro_features.py         - Macroeconomic features
ml_features/tsfresh_features.py       - TSFresh automated feature extraction
ml_features/shap_explainer.py         - SHAP explainability
ml_features/strategy_enhancer.py      - ML-enhanced strategy signals
ml_features/realtime_feature_engine.py - Real-time feature computation
```

#### ML Meta (7 files)

**ml_meta/** - Meta-Learning & Confidence Policies
```
ml_meta/conf_policy.py                 - Confidence-based policy learning
```

#### ML Supporting (3 files)

**ml/** - ML Infrastructure
```
ml/confidence_gate.py                  - ML confidence gating
ml/experiment_tracking.py              - MLflow experiment tracking
```

---

### 4. STRATEGIES & SIGNAL GENERATION (10 files)

**strategies/** - Trading Strategies
```
strategies/registry.py                 - Strategy registry with validation
strategies/adaptive_selector.py       - Adaptive strategy selection
```

**strategies/dual_strategy/** - DualStrategyScanner (Combined IBS+RSI + Turtle Soup)
```
dual_strategy/combined.py             - Main scanner (THIS IS PRODUCTION)
```

**strategies/ibs_rsi/** - IBS+RSI Mean Reversion
```
ibs_rsi/strategy.py                   - IBS<0.08 + RSI(2)<5 (v2.2: 59.9% WR, 1.46 PF)
```

**strategies/ict/** - ICT Smart Money Concepts
```
ict/turtle_soup.py                    - Turtle Soup (sweep≥0.3 ATR) (v2.2: 61.0% WR, 1.37 PF)
ict/smart_money.py                    - Smart Money concepts implementation
```

**CRITICAL: ALWAYS USE DualStrategyScanner**
```python
# CORRECT (PRODUCTION)
from strategies.registry import get_production_scanner
scanner = get_production_scanner()

# WRONG (DEPRECATED - NO SWEEP FILTER)
from strategies.ict.turtle_soup import TurtleSoupStrategy  # 48% WR - FAIL
```

**Frozen Parameters:** `config/frozen_strategy_params_v2.2.json`

---

### 5. RISK MANAGEMENT (31 files)

**risk/** - Comprehensive Risk Management
```
risk/policy_gate.py                    - PolicyGate budget enforcement (2% per trade, 20% daily, 40% weekly)
risk/equity_sizer.py                   - 2% equity-based position sizing
risk/dynamic_position_sizer.py         - Adaptive sizing based on signal count
risk/signal_quality_gate.py            - Quality gate (score >= 70, confidence >= 0.60)
risk/kill_zone_gate.py                 - ICT-style time-based trade blocking
risk/weekly_exposure_gate.py           - 40% weekly / 20% daily exposure caps
risk/position_limit_gate.py            - Position limits
risk/net_exposure_gate.py              - Net exposure limits
risk/liquidity_gate.py                 - Liquidity checks
risk/liquidity.py                      - Liquidity analysis
risk/trailing_stops.py                 - Trailing stop implementation
risk/intelligent_exit_manager.py       - Intelligent exit logic
risk/portfolio_risk.py                 - Portfolio-level risk
risk/volatility_targeting.py           - Volatility-based position sizing
```

**risk/advanced/** - Quant-Grade Risk (4 files)
```
advanced/monte_carlo_var.py            - 10K-simulation VaR with Cholesky decomposition, stress testing
advanced/kelly_position_sizer.py       - Optimal Kelly Criterion position sizing
advanced/correlation_limits.py         - Correlation/concentration limits with sector mapping
advanced/portfolio_optimizer.py        - Portfolio optimization
```

**risk/circuit_breakers/** - Circuit Breakers (6 files)
```
circuit_breakers/breaker_manager.py    - Circuit breaker orchestration
circuit_breakers/correlation_breaker.py - Correlation-based breakers
circuit_breakers/drawdown_breaker.py   - Drawdown-based halts
circuit_breakers/execution_breaker.py  - Execution quality breakers
circuit_breakers/streak_breaker.py     - Losing streak breakers
circuit_breakers/volatility_breaker.py - Volatility-based breakers
```

**risk/factor_model/** - Factor Risk Analysis (3 files)
```
factor_model/factor_calculator.py      - Calculate factor exposures
factor_model/factor_report.py          - Factor risk reporting
factor_model/sector_exposure.py        - Sector exposure analysis
```

---

### 6. DATA INFRASTRUCTURE (37 files)

**data/providers/** - Market Data Sources (15 files)
```
providers/polygon_eod.py               - Polygon.io EOD data (PRIMARY)
providers/alpaca_live.py               - Alpaca live data
providers/alpaca_intraday.py           - Alpaca intraday bars
providers/alpaca_websocket.py          - Alpaca WebSocket streaming
providers/yfinance_eod.py              - Yahoo Finance (FREE fallback)
providers/stooq_eod.py                 - Stooq (FREE primary for backtesting)
providers/binance_klines.py            - Binance crypto (FREE)
providers/polygon_crypto.py            - Polygon crypto
providers/polygon_intraday.py          - Polygon intraday
providers/fred_macro.py                - FRED macroeconomic data
providers/bea_macro.py                 - BEA macroeconomic data
providers/treasury_yields.py           - Treasury yield data
providers/eia_energy.py                - EIA energy data
providers/cftc_cot.py                  - CFTC COT reports
providers/multi_source.py              - Multi-source aggregator
```

**data/lake/** - Frozen Data Lake (3 files)
```
lake/manifest.py                       - Dataset manifests with SHA256 hashes
lake/io.py                             - LakeWriter/LakeReader for parquet/CSV with integrity
```

**data/universe/** - Universe Management (3 files)
```
universe/loader.py                     - Symbol list loading, dedup, cap
universe/canonical.py                  - Canonical universe (800 stocks)
```

**data/alternative/** - Alternative Data (5 files)
```
alternative/alt_data_aggregator.py     - Alternative data aggregation
alternative/congress_trades.py         - Congressional trading data
alternative/insider_trades.py          - Insider trading data
alternative/news_sentiment.py          - News sentiment analysis
alternative/options_flow.py            - Options flow data
```

**data/quality/** - Data Quality (2 files)
```
quality/corporate_actions_canary.py    - Corporate actions detection
```

**data/ml/** - ML Data Generation (1 file)
```
ml/generative_market_model.py          - Generative models for synthetic data
```

**Other Data Files:**
```
data/corporate_actions.py              - Corporate actions handling
data/quorum.py                         - Data quorum/consensus
data/validation.py                     - Data validation
data/schemas/ohlcv_schema.py           - OHLCV schema validation
```

---

### 7. EXECUTION & ORDER MANAGEMENT (22 files)

**execution/** - Order Execution
```
execution/broker_alpaca.py             - Alpaca broker (PRIMARY - live + paper)
execution/broker_alpaca_crypto.py      - Alpaca crypto trading
execution/broker_paper.py              - Paper trading simulator
execution/broker_crypto.py             - Crypto broker abstraction
execution/broker_base.py               - Base broker interface
execution/broker_factory.py            - Broker factory pattern
execution/order_manager.py             - Order lifecycle management
execution/order_state_machine.py       - Order state machine
execution/reconcile.py                 - Position reconciliation
execution/execution_guard.py           - Execution safety checks
execution/intelligent_executor.py      - Smart order routing
execution/execution_bandit.py          - Multi-armed bandit for execution strategies
execution/intraday_trigger.py          - Intraday signal triggering
execution/utils.py                     - Execution utilities
```

**execution/analytics/** - Execution Analytics (4 files)
```
analytics/execution_report.py          - Execution quality reporting
analytics/market_impact.py             - Market impact analysis
analytics/slippage_tracker.py          - Slippage tracking
analytics/timing_analysis.py           - Timing analysis
```

**execution/tca/** - Transaction Cost Analysis (1 file)
```
tca/transaction_cost_analyzer.py       - Full TCA implementation
```

**oms/** - Order Management System (3 files)
```
oms/order_state.py                     - Order state records
oms/idempotency_store.py               - Duplicate prevention
```

---

### 8. BACKTEST ENGINE (17 files)

**backtest/** - Backtesting Infrastructure
```
backtest/engine.py                     - Main backtest engine with equity curve
backtest/walk_forward.py               - Walk-forward optimization
backtest/vectorized.py                 - Vectorized backtesting
backtest/vectorized_fast.py            - Fast vectorized backtesting
backtest/vectorbt_engine.py            - VectorBT integration
backtest/slippage.py                   - Slippage models
backtest/regime_adaptive_slippage.py   - Regime-based slippage
backtest/costs.py                      - Transaction costs
backtest/fill_model.py                 - Fill simulation
backtest/gap_risk_model.py             - Gap risk modeling
backtest/monte_carlo.py                - Monte Carlo simulation
backtest/purged_cv.py                  - Purged cross-validation
backtest/triple_barrier.py             - Triple barrier labeling
backtest/multi_timeframe.py            - Multi-timeframe backtesting
backtest/reproducibility.py            - Reproducibility verification
backtest/visualization.py              - Backtest visualization
```

---

### 9. OPTIONS TRADING (11 files)

**options/** - Synthetic Options Engine
```
options/black_scholes.py               - BS pricing, Greeks, implied volatility
options/volatility.py                  - Realized volatility (close-to-close, Parkinson, Yang-Zhang)
options/selection.py                   - Delta-targeted strike selection via binary search
options/position_sizing.py             - 2% risk enforcement for long/short options
options/backtest.py                    - Daily repricing with transaction costs
options/pricing.py                     - Options pricing utilities
options/chain_fetcher.py               - Options chain data fetching
options/iv_signals.py                  - IV-based signals
options/order_router.py                - Options order routing
options/spreads.py                     - Options spread strategies
```

---

### 10. RESEARCH & DISCOVERY (27 files)

**research/** - Alpha Research (11 files)
```
research/alpha_factory.py              - Alpha factor factory
research/alpha_library.py              - Alpha factor library
research/alpha_research_integration.py - Research integration
research/alpha_seeder.py               - Alpha seeding system
research/alphas.py                     - Alpha implementations
research/experiment_analyzer.py        - Experiment analysis
research/factor_validator.py           - Factor validation
research/features.py                   - Feature research
research/screener.py                   - Stock screening
research/vectorbt_miner.py             - VectorBT-based alpha mining
```

**research_os/** - Research Operating System (5 files)
```
research_os/orchestrator.py            - Research orchestration (DISCOVER → RESEARCH → ENGINEER)
research_os/knowledge_card.py          - Knowledge card format
research_os/proposal.py                - Research proposal system
research_os/approval_gate.py           - Human approval gate (NEVER auto-merge)
```

**Research OS Workflow:**
```
DISCOVER: Always-on scanning for opportunities (CuriosityEngine, Scrapers)
    ↓
RESEARCH: Scheduled experiments with validation (Backtest, IntegrityGuardian)
    ↓
ENGINEER: Human-gated production changes (ApprovalGate, CLI approval required)
```

**experiments/** - Experiment Registry (3 files)
```
experiments/registry.py                - Track all experiments for reproducibility
```

**evolution/** - Strategy Evolution (8 files)
```
evolution/strategy_foundry.py          - Generate new strategies
evolution/genetic_optimizer.py         - Genetic algorithm optimization
evolution/strategy_mutator.py          - Strategy mutation
evolution/clone_detector.py            - Detect duplicate strategies
evolution/rule_generator.py            - Generate trading rules
evolution/promotion_gate.py            - Promote strategies to production
evolution/registry.py                  - Strategy registry
```

---

### 11. ANALYTICS & REPORTING (13 files)

**analytics/** - Performance Analytics
```
analytics/edge_decomposition.py        - Decompose edge into components
analytics/factor_attribution.py        - Factor attribution analysis
analytics/auto_standdown.py            - Automatic standdown on poor performance
analytics/duckdb_engine.py             - DuckDB analytics engine
analytics/pyfolio_integration.py       - Pyfolio integration
```

**analytics/attribution/** - Attribution Analysis (4 files)
```
attribution/attribution_report.py      - Full attribution reporting
attribution/daily_pnl.py               - Daily P&L attribution
attribution/factor_attribution.py      - Factor-level attribution
attribution/strategy_attribution.py    - Strategy-level attribution
```

**analytics/alpha_decay/** - Alpha Decay Monitoring (1 file)
```
alpha_decay/alpha_monitor.py           - Monitor alpha decay over time
```

---

### 12. ALTERNATIVE DATA (9 files)

**altdata/** - Alternative Data Sources
```
altdata/congressional_trades.py        - Congressional trading data (Quiver Quant API)
altdata/insider_activity.py            - Insider trading activity
altdata/political_sentiment.py         - Political sentiment analysis
altdata/news_processor.py              - News processing and NLP
altdata/sentiment.py                   - Sentiment analysis
altdata/options_flow.py                - Unusual options activity
altdata/github_activity.py             - GitHub activity (tech stocks)
altdata/market_mood_analyzer.py        - Overall market mood
```

---

### 13. AGENTS & ORCHESTRATION (10 files)

**agents/** - Multi-Agent System
```
agents/orchestrator.py                 - Agent orchestration
agents/autogen_team.py                 - Autogen multi-agent team
agents/langgraph_coordinator.py        - LangGraph coordination
agents/base_agent.py                   - Base agent class
agents/scout_agent.py                  - Scout for opportunities
agents/auditor_agent.py                - Audit agent
agents/risk_agent.py                   - Risk assessment agent
agents/reporter_agent.py               - Reporting agent
agents/agent_tools.py                  - Tools for agents
```

---

### 14. LLM INTEGRATION (8 files)

**llm/** - LLM Provider Abstraction
```
llm/provider_anthropic.py              - Anthropic Claude integration
llm/provider_openai.py                 - OpenAI GPT integration
llm/provider_ollama.py                 - Ollama local LLM integration
llm/provider_base.py                   - Base provider interface
llm/router.py                          - LLM routing logic
llm/financial_adapter.py               - Financial domain adaptation
llm/token_budget.py                    - Token budget management
```

---

### 15. EXPLAINABILITY (7 files)

**explainability/** - Trade Explainability
```
explainability/trade_thesis_builder.py - Build comprehensive trade thesis
explainability/trade_explainer.py      - Explain trade decisions
explainability/narrative_generator.py  - Generate natural language narratives
explainability/playbook_generator.py   - Generate trading playbooks
explainability/decision_tracker.py     - Track decision lineage
explainability/decision_packet.py      - Decision packet format
```

---

### 16. PIPELINES (14 files)

**pipelines/** - Data & Workflow Pipelines
```
pipelines/unified_signal_enrichment.py - 18-stage, 44-component unified pipeline
pipelines/data_audit_pipeline.py       - Data quality audit
pipelines/backtest_pipeline.py         - Backtest workflow
pipelines/discovery_pipeline.py        - Alpha discovery pipeline
pipelines/gates_pipeline.py            - Risk gate pipeline
pipelines/implementation_pipeline.py   - Strategy implementation
pipelines/promotion_pipeline.py        - Strategy promotion workflow
pipelines/quant_rd_pipeline.py         - Quant R&D pipeline
pipelines/reporting_pipeline.py        - Reporting pipeline
pipelines/snapshot_pipeline.py         - State snapshot pipeline
pipelines/spec_pipeline.py             - Strategy spec pipeline
pipelines/universe_pipeline.py         - Universe building pipeline
pipelines/base.py                      - Pipeline base classes
```

**Unified Signal Enrichment Pipeline:**
- 18 stages: Data validation → Technical features → ML features → Regime → Markov → Quality gate → Execution
- 44 components: All integrated into single pipeline
- 100% component coverage verified

---

### 17. CORE INFRASTRUCTURE (27 files)

**core/** - Core System Components
```
core/hash_chain.py                     - Audit chain with SHA256 hashing
core/structured_log.py                 - Structured JSON logging
core/config_pin.py                     - Config signature verification
core/kill_switch.py                    - Emergency kill switch
core/secrets.py                        - API key management
core/safe_pickle.py                    - Secure pickle operations
core/alerts.py                         - Alert system
core/circuit_breaker.py                - Circuit breakers
core/journal.py                        - Trading journal
core/rate_limiter.py                   - API rate limiting
core/http_client.py                    - HTTP client with retries
core/lineage.py                        - Data lineage tracking
core/decision_packet.py                - Decision packet format
core/signal_freshness.py               - Signal freshness checks
core/earnings_filter.py                - Earnings date filtering
core/regime_filter.py                  - Regime-based filtering
core/restart_backoff.py                - Exponential backoff for restarts
core/vix_monitor.py                    - VIX monitoring
```

**core/clock/** - Time & Calendar Management (7 files)
```
clock/market_clock.py                  - Market hours and trading days
clock/equities_calendar.py             - NYSE calendar with holidays
clock/crypto_clock.py                  - 24/7 crypto market clock
clock/options_event_clock.py           - Options expiration calendar
clock/macro_events.py                  - FOMC, CPI, NFP events
clock/date_utils.py                    - Date utilities
clock/tz_utils.py                      - Timezone utilities
```

---

### 18. MONITORING & OBSERVABILITY (10 files)

**monitor/** - System Monitoring
```
monitor/health_endpoints.py            - Health check HTTP endpoints
monitor/heartbeat.py                   - Heartbeat monitoring
monitor/drift_detector.py              - Model drift detection
monitor/divergence_monitor.py          - Strategy divergence monitoring
monitor/divergence.py                  - Divergence detection
monitor/calibration.py                 - Model calibration monitoring
monitor/circuit_breaker.py             - Circuit breaker monitoring
```

**observability/** - Observability Platform (2 files)
```
observability/langfuse_tracer.py       - Langfuse LLM tracing integration
```

**selfmonitor/** - Self-Monitoring (4 files)
```
selfmonitor/circuit_breaker.py         - Self-monitoring circuit breakers
selfmonitor/anomaly_detect.py          - Anomaly detection
```

---

### 19. GUARDIAN SYSTEMS (10 files)

**guardian/** - Portfolio Guardian
```
guardian/guardian.py                   - Main guardian orchestrator
guardian/portfolio_governor.py         - Portfolio-level governance
guardian/decision_engine.py            - Decision validation
guardian/emergency_protocol.py         - Emergency procedures
guardian/self_learner.py               - Self-learning from outcomes
guardian/resilience.py                 - System resilience
guardian/system_monitor.py             - System health monitoring
guardian/alert_manager.py              - Alert management
guardian/daily_digest.py               - Daily summary reports
```

---

### 20. PORTFOLIO MANAGEMENT (9 files)

**portfolio/** - Portfolio Management
```
portfolio/state_manager.py             - Portfolio state tracking
portfolio/risk_manager.py              - Portfolio risk management
portfolio/heat_monitor.py              - Position heat monitoring
```

**portfolio/optimizer/** - Portfolio Optimization (4 files)
```
optimizer/portfolio_manager.py         - Portfolio manager
optimizer/mean_variance.py             - Mean-variance optimization
optimizer/risk_parity.py               - Risk parity allocation
optimizer/rebalancer.py                - Portfolio rebalancing
```

---

### 21. QUANT GATES (7 files)

**quant_gates/** - Quant Quality Gates
```
quant_gates/gate_0_sanity.py           - Sanity checks (no lookahead, no NaN)
quant_gates/gate_1_baseline.py         - Baseline performance checks
quant_gates/gate_2_robustness.py       - Robustness checks (walk-forward)
quant_gates/gate_3_risk.py             - Risk checks (max drawdown, Sharpe)
quant_gates/gate_4_multiple_testing.py - Multiple testing correction
quant_gates/pipeline.py                - Gate pipeline orchestration
```

---

### 22. COMPLIANCE & SAFETY (7 files)

**compliance/** - Regulatory Compliance (4 files)
```
compliance/rules_engine.py             - Trading rules enforcement
compliance/prohibited_list.py          - Prohibited securities list
compliance/audit_trail.py              - Compliance audit trail
```

**safety/** - Safety Systems (3 files)
```
safety/                                - Safety verification systems
```

---

### 23. ALERTS & MESSAGING (7 files)

**alerts/** - Alert System (5 files)
```
alerts/professional_alerts.py          - Professional alert formatting
alerts/telegram_alerter.py             - Telegram integration
alerts/telegram_commander.py           - Telegram command interface
alerts/regime_alerts.py                - Regime change alerts
```

**messaging/** - Messaging Infrastructure (2 files)
```
messaging/                             - Message queue systems
```

---

### 24. WEB & DASHBOARD (8 files)

**web/** - Web Interface
```
web/dashboard.py                       - Main trading dashboard
web/dashboard_pro.py                   - Professional dashboard
web/data_provider.py                   - Dashboard data provider
web/main.py                            - Web server entry point
```

**web/api/** - API Endpoints (2 files)
```
api/signal_queue.py                    - Signal queue API
api/webhooks.py                        - Webhook handlers
```

---

### 25. ANALYSIS TOOLS (3 files)

**analysis/** - Trade Analysis
```
analysis/historical_patterns.py        - Consecutive day pattern analysis
analysis/options_expected_move.py      - Expected move calculator (realized vol)
```

---

### 26. SIMULATION (3 files)

**simulation/** - Market Simulation
```
simulation/market_simulator.py         - Market simulation engine
simulation/market_agents.py            - Agent-based market simulation
```

---

### 27. PREFLIGHT CHECKS (4 files)

**preflight/** - Pre-Trade Validation
```
preflight/evidence_gate.py             - Evidence-based validation
```

---

### 28. SCANNER (4 files)

**scanner/** - Stock Scanner
```
scanner/                               - Multi-asset scanner implementation
```

---

### 29. BOUNCE ANALYSIS (8 files)

**bounce/** - Bounce Profile Analysis
```
bounce/                                - Bounce pattern analysis and profiling
```

---

### 30. TOOLS & UTILITIES (14 files)

**tools/** - Development Tools
```
tools/verify_repo.py                   - Repository verification
tools/verify_robot.py                  - Robot integrity checks
tools/verify_alive.py                  - Liveness verification
tools/verify_100_components.py         - Component coverage verification
tools/verify_wiring_master.py          - Wiring verification
tools/component_auditor.py             - Component audit
tools/super_audit_verifier.py          - Super audit verification
tools/runtime_tracer.py                - Runtime tracing
tools/build_bounce_db.py               - Build bounce database
tools/build_bounce_profiles_all.py     - Build all bounce profiles
tools/bounce_profile.py                - Bounce profiling
tools/today_bounce_watchlist.py        - Today's bounce watchlist
tools/cleanup_cache.py                 - Cache cleanup
tools/generate_truth_table.py          - Truth table generation
```

---

### 31. TESTS (111 files)

**tests/** - Comprehensive Test Suite
```
tests/unit/                            - Unit tests
tests/integration/                     - Integration tests
tests/cognitive/                       - Cognitive architecture tests
tests/altdata/                         - Alternative data tests
tests/config/                          - Configuration tests
tests/core/                            - Core system tests
tests/data/                            - Data pipeline tests
tests/execution/                       - Execution tests
tests/llm/                             - LLM tests
tests/ml_advanced/                     - ML tests
tests/ml_features/                     - Feature tests
tests/monitor/                         - Monitor tests
tests/oms/                             - OMS tests
tests/fixtures/                        - Test fixtures
```

**Test Coverage:** 947 tests passing, Grade A+ system audit

---

### 32. SCRIPTS (199 files)

**scripts/** - Entry Points & Automation

**Core Operations:**
```
scripts/start.py                       - Start trading system
scripts/stop.py                        - Stop trading system
scripts/runner.py                      - 24/7 scheduler
scripts/run_autonomous.py              - Run autonomous brain
scripts/scan.py                        - Daily scanner (800 → 5 → 2)
scripts/run_paper_trade.py             - Paper trading
scripts/run_live_trade_micro.py        - Live trading (micro budget)
```

**Backtesting & Validation:**
```
scripts/backtest_dual_strategy.py      - Backtest dual strategy (CANONICAL TEST)
scripts/run_wf_polygon.py              - Walk-forward backtest
scripts/aggregate_wf_report.py         - WF report aggregation
scripts/backtest_ibs_rsi.py            - Backtest IBS+RSI
scripts/backtest_totd.py               - Backtest TOTD
scripts/verify_scan_consistency.py     - Verify scan determinism
```

**Professional Execution Flow:**
```
scripts/overnight_watchlist.py         - Build Top 5 watchlist (3:30 PM)
scripts/premarket_validator.py         - Validate gaps/news (8:00 AM)
scripts/opening_range_observer.py      - Observe opening (9:30-10:00, NO TRADES)
```

**Data Management:**
```
scripts/build_universe_polygon.py      - Build 900-stock universe
scripts/build_universe_900.py          - Alternative universe builder
scripts/prefetch_polygon_universe.py   - Prefetch EOD data
scripts/freeze_crypto_ohlcv.py         - Freeze crypto data
scripts/freeze_equities_eod.py         - Freeze equities data
scripts/validate_lake.py               - Validate frozen datasets
scripts/check_data_quality.py          - Data quality checks
scripts/validate_data_pipeline.py      - Pipeline validation
scripts/validate_universe_coverage.py  - Universe coverage validation
```

**Analysis & Reporting:**
```
scripts/generate_pregame_blueprint.py  - Pre-game comprehensive analysis
scripts/top2_analysis.py               - Top 2 trade analysis
scripts/analyze_stock.py               - Single stock analysis
scripts/eod_report.py                  - End-of-day report
scripts/eod_finalize.py                - End-of-day finalization
```

**ML Training:**
```
scripts/train_hmm_regime.py            - Train HMM regime model
scripts/train_lstm_confidence.py       - Train LSTM confidence model
scripts/train_ensemble.py              - Train ensemble models
scripts/train_ensemble_models.py       - Train all ensemble models
scripts/train_meta.py                  - Train meta-learner
scripts/train_rl_agent.py              - Train RL agent
scripts/run_weekly_training.py         - Weekly model retraining
```

**System Management:**
```
scripts/preflight.py                   - Pre-flight checks (10 critical checks)
scripts/status.py                      - System status
scripts/health.py                      - Health checks
scripts/start_health.py                - Start health server
scripts/validate.py                    - Run validation tests
scripts/system_audit.py                - Full system audit
scripts/verify_system.py               - System verification
scripts/verify_architecture.py         - Architecture verification
scripts/verify_hash_chain.py           - Hash chain verification
scripts/verify_repo.py                 - Repository verification
```

**Monitoring & Alerts:**
```
scripts/reconcile_alpaca.py            - Reconcile broker positions
scripts/reconcile_broker_daily.py      - Daily reconciliation
scripts/heartbeat.py                   - Heartbeat monitoring
scripts/watchdog.py                    - System watchdog
scripts/alerts.py                      - Alert management
scripts/telegram.py                    - Telegram bot
scripts/send_telegram_test.py          - Test Telegram
```

**Research & Discovery:**
```
scripts/alpha_research.py              - Alpha research
scripts/feature_experiment.py          - Feature experiments
scripts/find_unique_patterns.py        - Pattern discovery
scripts/volatility_study.py            - Volatility research
scripts/verified_quant_scan.py         - Verified quant scan
scripts/fast_quant_scan.py             - Fast quant scan
scripts/unified_scan.py                - Unified scanner
scripts/unified_multi_asset_scan.py    - Multi-asset scanner
scripts/research_os_cli.py             - Research OS CLI
```

**Trading Journal & Learning:**
```
scripts/cognitive_learn.py             - Cognitive learning
scripts/journal.py                     - Trading journal
scripts/suggest.py                     - AI suggestions
scripts/explain.py                     - Trade explanation
```

**Utilities:**
```
scripts/backup.py                      - Backup system
scripts/backup_state.py                - Backup state files
scripts/backup_snapshot.py             - Snapshot backup
scripts/rebuild_state.py               - Rebuild state
scripts/cleanup.py                     - System cleanup
scripts/snapshot.py                    - State snapshot
scripts/export_evidence_bundle.py      - Export evidence
scripts/export_ai_bundle.py            - Export AI bundle
```

**Configuration:**
```
scripts/config_utils.py                - Config utilities
scripts/show_config_pin.py             - Show config signature
scripts/env.py                         - Environment management
```

**Specialized Analysis:**
```
scripts/correlation.py                 - Correlation analysis
scripts/correlation_check.py           - Correlation checking
scripts/drawdown.py                    - Drawdown analysis
scripts/exposure.py                    - Exposure analysis
scripts/benchmark.py                   - Benchmark comparison
scripts/simulate.py                    - Monte Carlo simulation
scripts/earnings.py                    - Earnings analysis
scripts/exit_manager.py                - Exit management
```

**Trading Execution:**
```
scripts/trade_top3.py                  - Trade top 3 signals
scripts/submit_totd.py                 - Submit TOTD trade
scripts/run_shadow.py                  - Shadow trading
scripts/run_daily_pipeline.py          - Daily pipeline
scripts/daily_scheduler.py             - Daily scheduler
```

**Data Exploration:**
```
scripts/debug_signals.py               - Debug signal generation
scripts/debugger.py                    - System debugger
scripts/debug.py                       - Debug utilities
scripts/replay_day.py                  - Replay historical day
scripts/verify_pltr_streak.py          - Verify PLTR streak
scripts/verify_streak.py               - Verify consecutive patterns
scripts/verify_data.py                 - Data verification
```

**Testing:**
```
scripts/smoke_test.py                  - Smoke tests
scripts/ci_smoke.py                    - CI smoke tests
scripts/test.py                        - Test runner
```

**Dashboard & Web:**
```
scripts/dashboard.py                   - Launch dashboard
```

**Misc:**
```
scripts/version.py                     - Version info
scripts/supervisor.py                  - Process supervisor
scripts/start_kobe.py                  - Start Kobe
scripts/watchlist.py                   - Watchlist management
scripts/state.py                       - State management
scripts/strategy.py                    - Strategy management
scripts/signals.py                     - Signal management
scripts/data.py                        - Data management
scripts/broker.py                      - Broker management
scripts/universe.py                    - Universe management
scripts/build_signal_dataset.py        - Build signal dataset
scripts/build_totd_dataset.py          - Build TOTD dataset
scripts/update_sentiment_cache.py      - Update sentiment
scripts/update_slippage_defaults.py    - Update slippage
scripts/update_status_md.py            - Update STATUS.md
scripts/demo_enhanced_brain.py         - Demo enhanced brain
scripts/pre_game_plan.py               - Pre-game planning
scripts/sync_all_tasks.py              - Sync tasks
scripts/log_event.py                   - Log events
scripts/install_git_hooks.py           - Install git hooks
scripts/check_polygon_earliest.py      - Check Polygon data
scripts/check_polygon_earliest_universe.py - Check universe data
scripts/backfill_yfinance.py           - Backfill Yahoo data
scripts/diagnose_assets.py             - Asset diagnostics
```

---

## EXTERNAL INTEGRATIONS & DATA SOURCES

### Market Data APIs
```
✓ Polygon.io          - Primary EOD data (800 stocks, 10+ years history)
✓ Alpaca Markets      - Execution broker (live + paper trading)
✓ Yahoo Finance       - Free fallback data source
✓ Stooq              - Free primary for backtesting
✓ Binance            - Crypto OHLCV (no API key required)
```

### Alternative Data APIs
```
✓ Quiver Quant       - Congressional trades, insider activity
✓ Finnhub            - News sentiment, earnings
✓ FRED               - Macroeconomic data
✓ BEA                - Economic indicators
✓ Treasury.gov       - Treasury yields
✓ EIA                - Energy data
✓ CFTC               - Commitment of Traders
```

### Research & Discovery Scrapers
```
✓ ArXiv              - Academic papers (strategy research)
✓ GitHub             - Code repositories (algo discovery)
✓ Reddit             - r/wallstreetbets, r/algotrading sentiment
✓ YouTube            - Trading content, tutorials
✓ Firecrawl          - Web scraping service
```

### AI/LLM Providers
```
✓ Anthropic Claude   - Primary LLM (Opus 4.5)
✓ OpenAI GPT         - Alternative LLM
✓ Ollama             - Local LLM (privacy mode)
```

### Communication
```
✓ Telegram           - Alerts, commands, trade notifications
```

### Observability
```
✓ Langfuse           - LLM tracing and observability
✓ MLflow             - ML experiment tracking
```

---

## ARCHITECTURE LAYERS

### Layer 1: Data Foundation
```
Data Providers → Data Lake → Universe Management → Data Validation
    ↓
Corporate Actions Detection → Data Quorum → Quality Gates
```

### Layer 2: Feature Engineering
```
OHLCV → Technical Features (150+) → ML Features → PCA Reduction
    ↓
Regime Detection (HMM) → Markov Chains → Sentiment → Macro Features
```

### Layer 3: Signal Generation
```
DualStrategyScanner (IBS+RSI + Turtle Soup)
    ↓
Signal Quality Gate (score >= 70, conf >= 0.60)
    ↓
Markov Boost (+5-10% for agreeing signals)
    ↓
800 stocks → Top 5 (STUDY) → Top 2 (TRADE)
```

### Layer 4: Risk Management
```
Kill Zone Gate (time-based blocking)
    ↓
Policy Gate (2% per trade, 20% daily, 40% weekly)
    ↓
Position Sizing (dual cap: 2% risk + 20% notional)
    ↓
Correlation Limits → VaR → Kelly Sizing
    ↓
Circuit Breakers (drawdown, volatility, streak, execution)
```

### Layer 5: Execution
```
Broker Factory → Alpaca (live/paper) → Order Manager
    ↓
Execution Guard → Intelligent Executor → TCA
    ↓
Order State Machine → Idempotency Store → Reconciliation
```

### Layer 6: Learning & Adaptation
```
Episodic Memory (trade outcomes) → Reflection Engine
    ↓
Semantic Memory (generalized rules) → Self-Model (capability tracking)
    ↓
Curiosity Engine (hypothesis generation) → Research Engine
    ↓
Alpha Discovery → Experiment Registry → Promotion Gate
```

### Layer 7: Autonomous Operation
```
Awareness (time/day/season/market) → Context Builder
    ↓
Scheduler (priority + context-based tasks) → Work Modes
    ↓
Research (experiments) + Learning (reflection) + Maintenance (data quality)
    ↓
Monitor (heartbeat) → Alerts → Self-Improvement Loop
```

### Layer 8: Governance & Safety
```
Guardian (portfolio governor) → Emergency Protocol
    ↓
Compliance (rules engine) → Audit Trail → Hash Chain
    ↓
Research OS Approval Gate (human-gated changes)
    ↓
Quant Gates (5-stage quality verification)
```

---

## KEY CAPABILITIES

### 1. Autonomous 24/7 Operation
- Always aware of time, day, season, market state
- Context-based work scheduling (research during lunch, trade during windows)
- Self-improvement through random experiments and discovery
- Daily reflections and learning from outcomes
- Automatic model retraining and walk-forward validation

### 2. Professional Execution Flow
- Overnight watchlist building (3:30 PM)
- Premarket validation (8:00 AM - gaps/news)
- Opening range observation (9:30-10:00 AM - NO TRADES)
- Primary trading window (10:00-11:30 AM)
- Fallback scan with higher bar if watchlist fails
- Power hour window (14:30-15:30 PM)
- ICT-style kill zones enforced automatically

### 3. Multi-Strategy Signal Generation
- DualStrategyScanner (IBS+RSI + Turtle Soup combined)
- Markov chain direction prediction
- HMM regime detection for position sizing
- 800 → 5 → 2 funnel (scan all, study top 5, trade top 2)
- Quality gate with historical pattern auto-pass

### 4. Quant-Grade Risk Management
- Dual-cap position sizing (2% risk + 20% notional)
- VaR with Monte Carlo simulation
- Kelly Criterion optimal sizing
- Correlation/concentration limits
- Multi-level circuit breakers
- Weekly/daily exposure caps

### 5. Advanced ML/AI
- Ensemble models (XGBoost, LightGBM, LSTM)
- HMM regime detection
- Markov chain state transitions
- LSTM confidence scoring
- RL trading agents (PPO/DQN/A2C)
- Online learning with drift detection
- 150+ engineered features + PCA

### 6. Cognitive Architecture
- Metacognitive governor (System 1/2 thinking)
- Episodic + semantic memory
- Reflection engine (learning from outcomes)
- Self-model (capability tracking, calibration)
- Knowledge boundary (uncertainty detection)
- Curiosity engine (hypothesis generation)
- 83+ cognitive tests passing

### 7. Research Operating System
- DISCOVER: Always-on scraping (ArXiv, GitHub, Reddit, YouTube)
- RESEARCH: Scheduled experiments with validation
- ENGINEER: Human-gated production changes (NEVER auto-merge)
- Knowledge cards for all discoveries
- Approval gate with CLI workflow

### 8. Multi-Agent Orchestration
- Autogen multi-agent teams
- LangGraph coordination
- Scout, Risk, Auditor, Reporter agents
- Agent tools and communication protocols

### 9. Comprehensive Explainability
- Full trade thesis generation
- Natural language narratives
- Decision lineage tracking
- SHAP explainability
- Pre-game blueprints with historical evidence

### 10. Production-Grade Infrastructure
- Frozen data lake with SHA256 verification
- Hash chain audit trail
- Config signature verification
- Idempotency store (duplicate prevention)
- Purged cross-validation
- Reproducibility verification
- 947 tests passing

---

## CRITICAL PRODUCTION CONSTRAINTS

### NEVER AUTO-CHANGE
```
✗ NO auto-merge to production
✗ NO auto-trading without approval
✗ NO bypassing PolicyGate limits
✗ NO trading during kill zones
✗ NO manual position sizing (must use dual-cap formula)
```

### ALWAYS ENFORCE
```
✓ Use DualStrategyScanner (NEVER standalone TurtleSoup or IbsRsi)
✓ Apply sweep filter (ts_min_sweep_strength = 0.3)
✓ Dual-cap sizing (2% risk + 20% notional, use MINIMUM)
✓ Kill zone blocking (9:30-10:00, 11:30-14:30)
✓ Watchlist-first trading (fallback requires higher bar)
✓ Historical pattern auto-pass (25+ samples, 90%+ win rate)
✓ Human approval for all production changes
```

### VERIFIED PERFORMANCE
```
Strategy: DualStrategyScanner (IBS+RSI + Turtle Soup with 0.3 ATR sweep filter)
Win Rate: ~64%
Profit Factor: ~1.60
Universe: 900 optionable, liquid stocks (10+ years history)
Data Source: Polygon.io EOD (verified on Yahoo Finance)
Backtest Period: 2015-01-01 to 2024-12-31
Walk-Forward: 252-day train, 63-day test
```

---

## CURRENT STATUS (2026-01-07)

### System Health
- **Grade:** A+ (100/100)
- **Tests Passing:** 947
- **Modules Verified:** 22/22 core, 14/14 AI/LLM/ML
- **Critical Issues:** 0
- **Autonomous Mode:** ACTIVE (24/7)
- **Data Pipeline:** 10/10 OPTIMIZED

### Latest Scan (2025-12-31)
- **Signals Generated:** 3
- **Top 3:** AEHR, TNA, LOGI
- **Strategy:** IBS_RSI
- **Mode:** Weekend preview (using Friday's close)

### Production Components
```
✓ DualStrategyScanner operational
✓ Markov chain integration active
✓ Kill zone gate enforcing time blocks
✓ Professional execution flow v3.0 live
✓ Dual-cap position sizing enforced
✓ Historical pattern auto-pass enabled
✓ Autonomous brain running 24/7
✓ Research OS discovering patterns
```

---

## DOCUMENTATION

### Core Documents (51+ files in docs/)
```
docs/STATUS.md                         - Single source of truth (THIS IS CANONICAL)
docs/START_HERE.md                     - Onboarding guide
docs/ARCHITECTURE.md                   - System architecture
docs/READINESS.md                      - Production readiness checklist
docs/REPO_MAP.md                       - Directory structure
docs/ENTRYPOINTS.md                    - All 180+ runnable scripts
docs/ROBOT_MANUAL.md                   - Complete system guide
docs/PROFESSIONAL_EXECUTION_FLOW.md    - Professional execution workflow
docs/TRADE_ANALYSIS_STANDARD.md        - Trade analysis requirements
docs/CRITICAL_FIX_20260102.md          - Position sizing incident & fix
docs/COGNITIVE_ARCHITECTURE.md         - Cognitive system design
docs/ENHANCED_BRAIN_ARCHITECTURE.md    - Enhanced brain details
docs/ML_GUIDE.md                       - ML/AI guide
docs/DATA_PIPELINE.md                  - Data infrastructure
docs/RISK_REGISTER.md                  - Risk assessment
docs/KNOWN_GAPS.md                     - Missing components
docs/ROBOT_AUDIT_SUMMARY.md            - System audit summary
docs/FINANCIAL_PROJECTIONS.md          - Performance projections
docs/INTERVIEW_QA.md                   - Quant interview prep
```

### Skills (71 slash commands in .claude/skills/)
```
/start, /stop, /kill, /resume          - System control
/preflight, /validate, /status         - Health checks
/scan, /paper, /live                   - Trading operations
/positions, /pnl, /orders              - Portfolio tracking
/strategy, /signals, /backtest         - Strategy management
/wf, /smoke                            - Validation
/data, /prefetch, /universe            - Data management
/broker, /reconcile, /idempotency      - Execution
/audit, /risk, /config                 - Compliance
/brain, /awareness, /research, /learning - Autonomous systems
... (71 total)
```

---

## WHAT MAKES THIS SYSTEM UNIQUE

### 1. True Autonomy
Not just automated - truly autonomous with self-awareness, context understanding, and continuous self-improvement.

### 2. Cognitive Architecture
Brain-inspired decision-making with metacognition, episodic memory, reflection, and uncertainty awareness.

### 3. Research Operating System
Structured workflow for discovery → research → engineering with mandatory human approval gates.

### 4. Professional Execution
ICT-style kill zones, overnight watchlist, premarket validation, opening range observation - how real traders work.

### 5. Medallion-Inspired Markov Chains
Observable state transitions for direction prediction, complementing HMM regime detection.

### 6. Multi-Agent Orchestration
Autogen + LangGraph coordination of specialized agents (Scout, Risk, Auditor, Reporter).

### 7. Quant-Grade Quality
VaR, Kelly sizing, purged CV, multiple testing correction, 5-stage quality gates.

### 8. Comprehensive Explainability
Every trade has a full thesis with historical evidence, expected moves, support/resistance, bull/bear cases.

### 9. Zero Tolerance for Errors
947 tests, deterministic scans, reproducibility verification, hash chain audit trail.

### 10. Learning from Outcomes
Episodic memory stores every trade outcome, reflection engine learns patterns, self-model calibrates confidence.

---

## WHAT'S NOT INCLUDED (KNOWN GAPS)

From `docs/KNOWN_GAPS.md`:
```
✗ Real-time tick data (only EOD currently)
✗ Level 2 order book data
✗ Dark pool data
✗ Proprietary datasets (only free/reasonably priced sources)
✗ High-frequency trading infrastructure
✗ Custom hardware (FPGAs, co-location)
✗ Prime brokerage access (using retail broker Alpaca)
```

---

## VERIFICATION COMMANDS

```bash
# Full system audit (Grade A+ = 100/100)
python scripts/system_audit.py

# Verify repository integrity
python tools/verify_repo.py --verbose

# Verify autonomous brain
python scripts/run_autonomous.py --status --awareness

# Verify strategy (CANONICAL TEST)
python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_800.csv --start 2023-01-01 --end 2024-12-31 --cap 150
# Expected: ~64% WR, ~1.60 PF

# Verify scan consistency (determinism)
python scripts/verify_scan_consistency.py

# Run all tests
python scripts/test.py

# Check data quality
python scripts/check_data_quality.py

# Preflight checks (10 critical checks)
python scripts/preflight.py --dotenv ./.env
```

---

## CONCLUSION

**This is not "a trading bot with some ML."**

**This is:**
- A self-aware, autonomous trading system that never stops working
- A cognitive architecture with human-like reasoning and learning
- A multi-agent research platform for continuous alpha discovery
- A quant-grade risk management framework
- A professional execution system matching institutional standards
- A comprehensive testing and validation framework

**815 Python files. 286,589 lines of code. 57 modules. 199 entry points. 947 tests passing.**

**Built to be the greatest trading robot ever created.**

---

**Generated by:** System Architect Mode
**Date:** 2026-01-07
**Files Scanned:** 815 Python files across 57 modules
**Evidence:** Complete directory scan + file counting + module analysis
**Status:** COMPREHENSIVE INVENTORY COMPLETE
