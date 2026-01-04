# KOBE TRADING SYSTEM - INTERVIEW Q&A

> **Quant Interview Ready** - All answers grounded in real code with file:line references.

---

## SECTION A: SYSTEM OVERVIEW

### Q1: "Explain your trading system architecture"

**Direct Answer:**
Kobe is an autonomous quantitative trading system with a 24/7 brain that orchestrates signal generation, risk management, and execution. It uses a dual mean-reversion strategy (IBS+RSI and ICT Turtle Soup) validated through a 5-gate quant pipeline.

**Deep Answer:**
The system is organized in distinct layers:

1. **Data Layer** (`data/providers/polygon_eod.py`): EOD OHLCV from Polygon.io with CSV caching
2. **Strategy Layer** (`strategies/dual_strategy/combined.py`): DualStrategyScanner combining IBS+RSI (59.9% WR) and Turtle Soup (61.0% WR)
3. **Risk Layer** (`risk/policy_gate.py`): 2% per trade, 20% daily exposure, 40% weekly caps
4. **Execution Layer** (`execution/broker_alpaca.py`): IOC LIMIT orders via Alpaca
5. **Brain Layer** (`autonomous/brain.py`): 24/7 orchestrator with phase-aware scheduling

**Evidence:**
- `autonomous/brain.py:542` - Main brain class
- `strategies/dual_strategy/combined.py:1-50` - Strategy scanner
- `risk/policy_gate.py:check()` - Risk enforcement
- `config/base.yaml` - 696 lines of configuration

---

### Q2: "How does the 24/7 autonomous operation work?"

**Direct Answer:**
The autonomous brain runs in 60-second cycles, checking market phase, selecting appropriate tasks, and executing them based on priority and cooldowns. It's always aware of time, day type, and season.

**Deep Answer:**
```
autonomous/
├── brain.py       # Main orchestrator - think() method called every 60s
├── scheduler.py   # Task queue with 43 registered tasks
├── awareness.py   # 9 market phases, 8 seasons, 7 work modes
├── handlers.py    # 35+ task handlers
├── research.py    # Experiments, PF optimization
└── learning.py    # Trade analysis, reflections
```

**Work Modes by Time (ET):**
| Time | Phase | Mode |
|------|-------|------|
| 4:00-7:00 | Pre-market Early | Research |
| 9:30-10:00 | Market Opening | Monitoring (NO TRADES) |
| 10:00-11:30 | Market Morning | Active Trading |
| 11:30-14:00 | Lunch | Research |
| 14:00-15:30 | Afternoon | Active Trading |
| Weekends | Weekend | Deep Research |

**Evidence:**
- `autonomous/awareness.py:get_context()` - Returns current phase/mode
- `autonomous/scheduler.py:get_eligible_tasks()` - Filters by phase/mode
- `config/autonomous.yaml` - Phase definitions

---

### Q3: "What happens when the system detects an error?"

**Direct Answer:**
Errors trigger a cascading response: immediate logging, task retry with backoff, alert generation, and if critical, automatic activation of the kill switch to halt all trading.

**Deep Answer:**
1. **Logging**: All errors go to `logs/events.jsonl` with structured JSON
2. **Retry**: Tasks retry up to 3 times with exponential backoff (60s, 300s, 900s)
3. **Kill Switch**: Critical errors create `state/KILL_SWITCH` file
4. **Alerting**: Telegram notifications (if configured)
5. **Recovery**: Brain checks kill switch every cycle, halts if present

**Evidence:**
- `core/structured_log.py` - Structured logging
- `safety/mode.py:_check_kill_switch()` - Kill switch check
- `autonomous/scheduler.py:retry_policy` - Retry configuration

---

## SECTION B: STRATEGY & BACKTESTING

### Q4: "Explain your mean-reversion strategy"

**Direct Answer:**
We use two complementary mean-reversion strategies: IBS+RSI (Internal Bar Strength < 0.08 + RSI(2) < 5) and ICT Turtle Soup (sweep of prior lows with 0.3 ATR strength). Both target oversold conditions with SMA(200) trend filter.

**Deep Answer:**

**IBS+RSI Strategy:**
- Entry: IBS < 0.08 AND RSI(2) < 5 AND Close > SMA(200)
- Exit: ATR(14) x 2 stop OR 7-bar time stop
- Win Rate: 59.9%, Profit Factor: 1.46

**Turtle Soup Strategy:**
- Entry: Price sweeps below 20-day low by >= 0.3 ATR, then reverses
- Exit: ATR(14) x 2 stop OR 7-bar time stop
- Win Rate: 61.0%, Profit Factor: 1.37

**Combined:**
- DualStrategyScanner prioritizes Turtle Soup signals
- Deconfliction: Only one signal per symbol per day

**Evidence:**
- `strategies/ibs_rsi/strategy.py:generate_signals()` - IBS+RSI logic
- `strategies/ict/turtle_soup.py:_detect_sweep()` - Sweep detection
- `strategies/dual_strategy/combined.py` - Combined scanner
- `config/frozen_strategy_params_v2.2.json` - Frozen parameters

---

### Q5: "How do you prevent overfitting?"

**Direct Answer:**
We use walk-forward validation with 252-day train / 63-day test splits, 5-gate quant validation, and parameter freezing. No in-sample optimization ever touches test data.

**Deep Answer:**

**Prevention Techniques:**
1. **Walk-Forward**: Rolling train/test splits (80/20)
2. **Quant Gates**: 5-gate validation pipeline
   - Gate 0: Lookahead/leakage detection
   - Gate 1: Baseline (50% WR, 1.0 PF minimum)
   - Gate 2: Train/test correlation
   - Gate 3: Risk (25% max DD, 100+ trades)
   - Gate 4: FDR multiple testing correction
3. **Parameter Freezing**: `config/frozen_strategy_params_v2.2.json`
4. **Reproducibility**: SHA256 experiment hashing

**Evidence:**
- `backtest/walk_forward.py` - WF implementation
- `quant_gates/pipeline.py` - 5-gate orchestrator
- `experiments/registry.py` - SHA256 hashing

---

### Q6: "Walk me through your walk-forward validation"

**Direct Answer:**
We split 10 years of data into rolling windows: 252 trading days for training, 63 days for testing. We train on each window, test on the next, then aggregate out-of-sample results.

**Deep Answer:**

```
2015 |----TRAIN----|TEST|
2016      |----TRAIN----|TEST|
2017           |----TRAIN----|TEST|
...
2024                          |----TRAIN----|TEST|
```

**Process:**
1. Load 10 years of data (2015-2024)
2. Create rolling splits: 252-day train, 63-day test
3. For each split:
   - Optimize parameters on train data
   - Lock parameters
   - Run backtest on test data (out-of-sample)
4. Aggregate all OOS results for final metrics
5. Compare to random baseline (Gate 1)

**Evidence:**
- `backtest/walk_forward.py:create_splits()` - Split creation
- `scripts/run_wf_polygon.py` - WF runner
- `wf_outputs/wf_summary_compare.csv` - Results comparison

---

## SECTION C: RISK MANAGEMENT

### Q7: "How do you size positions?"

**Direct Answer:**
Dual-cap position sizing: 2% risk cap (based on stop distance) AND 20% notional cap (based on entry price). We always take the smaller of the two.

**Deep Answer:**

```python
# From risk/equity_sizer.py
account_equity = 50000
max_risk_pct = 0.02   # 2%
max_notional_pct = 0.20  # 20%

risk_per_trade = account_equity * max_risk_pct  # $1,000
max_notional = account_equity * max_notional_pct  # $10,000

shares_by_risk = risk_per_trade / (entry - stop)
shares_by_notional = max_notional / entry

final_shares = min(shares_by_risk, shares_by_notional)
```

**Evidence:**
- `risk/equity_sizer.py:calculate_shares()` - Position sizing
- `risk/dynamic_position_sizer.py` - Adaptive sizing
- `docs/CRITICAL_FIX_20260102.md` - Incident documentation

---

### Q8: "Explain your risk limits (2%/20%/40%)"

**Direct Answer:**
- 2% per trade: Maximum risk on any single position
- 20% per day: Maximum total exposure added in one day
- 40% per week: Maximum total portfolio exposure

**Deep Answer:**

| Limit | Level | Enforcement | Reason |
|-------|-------|-------------|--------|
| 2% | Trade | `risk/equity_sizer.py` | Survive 10 consecutive losers |
| 20% | Day | `risk/weekly_exposure_gate.py` | Prevent overtrading |
| 40% | Week | `risk/weekly_exposure_gate.py` | Maintain cash buffer |
| $75 | Order | `risk/policy_gate.py` | Micro-budget paper trading |
| $1000 | Daily | `risk/policy_gate.py` | Daily P&L limit |

**Evidence:**
- `risk/policy_gate.py:PolicyGate.check()` - Budget enforcement
- `risk/weekly_exposure_gate.py` - Exposure caps
- `config/base.yaml:risk` section

---

### Q9: "What are kill zones and why?"

**Direct Answer:**
Kill zones are time-based trading restrictions based on ICT (Inner Circle Trader) methodology. We block trading 9:30-10:00 (opening volatility) and 11:30-14:30 (lunch chop) to avoid low-quality setups.

**Deep Answer:**

| Time (ET) | Zone | Trading | Reason |
|-----------|------|---------|--------|
| 9:30-10:00 | Opening Range | BLOCKED | Amateur hour, fake moves |
| 10:00-11:30 | London Close | ALLOWED | Best setups |
| 11:30-14:30 | Lunch Chop | BLOCKED | Low volume, whipsaws |
| 14:30-15:30 | Power Hour | ALLOWED | Institutional positioning |
| 15:30-16:00 | Close | BLOCKED | Manage only |

**Evidence:**
- `risk/kill_zone_gate.py:can_trade_now()` - Zone enforcement
- `autonomous/awareness.py:get_phase()` - Phase detection
- `config/autonomous.yaml:phases` - Phase definitions

---

## SECTION D: ML/AI COMPONENTS

### Q10: "What ML models do you use and why?"

**Direct Answer:**
We use HMM for regime detection (bull/bear/neutral), LSTM for signal confidence scoring, and XGBoost/LightGBM ensemble for prediction. Each model serves a specific purpose in the decision pipeline.

**Deep Answer:**

| Model | Purpose | File |
|-------|---------|------|
| HMM (3-state) | Regime detection | `ml_advanced/hmm_regime_detector.py` |
| LSTM | Signal confidence (A/B/C grades) | `ml_advanced/lstm_confidence/model.py` |
| XGBoost | Ensemble prediction | `ml_advanced/ensemble/ensemble_predictor.py` |
| LightGBM | Ensemble prediction | `ml_advanced/ensemble/ensemble_predictor.py` |
| PPO/DQN | Reinforcement learning agent | `ml/alpha_discovery/rl_agent/agent.py` |

**Training Schedule:**
- HMM: Weekly (Saturday 10:00)
- LSTM: Weekly (Saturday 10:00)
- XGBoost: Weekly (Saturday 10:00)

**Evidence:**
- `ml_advanced/hmm_regime_detector.py:detect_regime()` - HMM
- `ml_advanced/lstm_confidence/` - LSTM directory
- `ml_advanced/ensemble/` - Ensemble directory

---

### Q11: "How does the cognitive architecture work?"

**Direct Answer:**
The cognitive brain is inspired by dual-process theory (System 1/2). Fast decisions go through heuristics, complex decisions use deliberation. It includes episodic memory, self-model tracking, and curiosity-driven exploration.

**Deep Answer:**

```
cognitive/
├── cognitive_brain.py      # Main orchestrator
├── metacognitive_governor.py  # System 1/2 routing
├── reflection_engine.py    # Learning from outcomes
├── self_model.py          # Capability tracking
├── episodic_memory.py     # Experience storage
├── semantic_memory.py     # Generalized rules
├── knowledge_boundary.py  # Uncertainty detection
└── curiosity_engine.py    # Hypothesis generation
```

**Decision Flow:**
1. Signal arrives
2. Metacognitive governor decides: fast (System 1) or slow (System 2)?
3. If complex: deliberation with full context
4. After execution: reflection and memory update
5. Knowledge boundary check: if uncertain, recommend stand-down

**Evidence:**
- `cognitive/cognitive_brain.py` - 83 tests passing
- `tests/cognitive/` - Test suite
- `state/cognitive/` - State persistence

---

### Q12: "Explain your signal confidence scoring"

**Direct Answer:**
Signals receive a confidence score (0-100) based on historical pattern strength, ML model agreement, and current market regime. Scores below 60 are rejected by the quality gate.

**Deep Answer:**

**Confidence Components:**
| Factor | Weight | Source |
|--------|--------|--------|
| Historical Pattern | 30% | `analysis/historical_patterns.py` |
| ML Ensemble | 25% | `ml_advanced/ensemble/` |
| Regime Alignment | 20% | `ml_advanced/hmm_regime_detector.py` |
| Technical Setup | 15% | Strategy-specific |
| Volume Confirmation | 10% | Price/volume analysis |

**Quality Gate Thresholds:**
- Watchlist signals: >= 60 confidence
- Fallback signals: >= 70 confidence
- Auto-pass: >= 90% historical win rate with 20+ samples

**Evidence:**
- `signals/signal_quality_gate.py:evaluate()` - Gate logic
- `analysis/historical_patterns.py:enrich_signal_with_historical_pattern()` - Pattern enrichment

---

## SECTION E: SAFETY & OPERATIONS

### Q13: "How do you ensure paper-only mode?"

**Direct Answer:**
`PAPER_ONLY = True` is a hardcoded constant in `safety/mode.py`. Every execution path calls `assert_paper_only()` before order submission. The kill switch file takes precedence over everything.

**Deep Answer:**

```python
# safety/mode.py
PAPER_ONLY: bool = True  # NEVER change programmatically
LIVE_TRADING_ENABLED: bool = False

def assert_paper_only(context: Optional[str] = None) -> None:
    if _check_kill_switch():
        raise SafetyViolationError("Kill switch active")
    # Additional checks...
```

**Enforcement Points:**
1. `execution/broker_alpaca.py:place_order()` - Checks before API call
2. `autonomous/brain.py:execute_trade()` - Brain-level check
3. `agents/base_agent.py:_safe_execute()` - Agent-level check
4. `scripts/run_paper_trade.py` - Script-level validation

**Evidence:**
- `safety/mode.py` - Central safety module
- `safety/__init__.py` - Public exports
- `state/KILL_SWITCH` - Kill switch file

---

### Q14: "Explain the incident on 2026-01-02"

**Direct Answer:**
On 2026-01-02, a position sizing bug allowed a $10,000+ position to be attempted on a $50,000 account. Root cause: missing notional cap. Fix: dual-cap sizing (risk AND notional limits).

**Deep Answer:**

**Timeline:**
1. Signal generated for PLTR
2. Position sizer calculated shares by risk (2%)
3. Bug: No notional cap check
4. Order would have been 20%+ of account
5. Caught before live trading (paper mode)

**Fix Applied:**
```python
# risk/equity_sizer.py
shares_by_risk = risk / (entry - stop)
shares_by_notional = max_notional / entry
final_shares = min(shares_by_risk, shares_by_notional)  # DUAL CAP
```

**Evidence:**
- `docs/CRITICAL_FIX_20260102.md` - Full incident report
- `risk/equity_sizer.py` - Fixed code
- `tests/test_position_sizing.py` - Regression tests

---

### Q15: "How do you handle broker disconnections?"

**Direct Answer:**
The broker watchdog checks connection every 5 minutes. On disconnect: 3 reconnect attempts with backoff, then kill switch activation if all fail. Positions are preserved in local state.

**Deep Answer:**

**Recovery Process:**
1. Detection: `execution/broker_alpaca.py:health_check()` fails
2. Alert: Warning logged, notification sent
3. Retry: 3 attempts at 60s, 120s, 300s intervals
4. Fallback: If all fail, activate kill switch
5. Recovery: On reconnect, reconcile positions

**Position Tracking:**
- Local: `state/positions.json`
- Broker: Alpaca API
- Reconciliation: `scripts/reconcile_alpaca.py`

**Evidence:**
- `execution/broker_alpaca.py:reconnect()` - Reconnection logic
- `scripts/reconcile_alpaca.py` - Position reconciliation
- `config/autonomous.yaml:broker_watchdog` - Watchdog settings

---

## SECTION F: INTERVIEW TIPS

### How to Demonstrate the System:

1. **Quick Demo (5 min):**
   ```bash
   python -m autonomous.run --demo
   ```

2. **Show Status:**
   ```bash
   python -m autonomous.run --status
   ```

3. **Show Awareness:**
   ```bash
   python -m autonomous.run --awareness
   ```

4. **Run Scan:**
   ```bash
   python scripts/scan.py --cap 50 --deterministic --top3
   ```

### Key Files to Reference:

| Question About | Show This File |
|----------------|----------------|
| Architecture | `autonomous/brain.py` |
| Strategy | `strategies/dual_strategy/combined.py` |
| Risk | `risk/policy_gate.py` |
| Safety | `safety/mode.py` |
| ML | `ml_advanced/hmm_regime_detector.py` |
| Tests | `tests/` (942 tests) |

### Metrics to Quote:

- **Win Rate:** ~60% (combined strategies)
- **Profit Factor:** ~1.40
- **Max Drawdown:** <15%
- **Test Coverage:** 942 tests
- **Universe:** 900 optionable, liquid stocks
- **History:** 10+ years for backtesting

---

*Generated for Kobe Trading System - Quant Interview Ready*
