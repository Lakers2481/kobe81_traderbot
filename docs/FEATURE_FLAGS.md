# Kobe81 Feature Flags - Complete Reference

**Last Updated:** 2025-12-30
**System Version:** 2.3

This document explains every configurable feature in the Kobe81 trading robot - what it does, why it exists, how it works, and when to enable it.

---

## Table of Contents

1. [Features Enabled (No History Needed)](#features-enabled-no-history-needed)
2. [Features Disabled (Need Trading History)](#features-disabled-need-trading-history)
3. [Feature Enablement Timeline](#feature-enablement-timeline)
4. [Quick Reference Table](#quick-reference-table)

---

## Features Enabled (No History Needed)

These features work immediately without any trading history. They use real-time data, external calendars, or pre-computed statistics.

### 1. Intraday Trigger (ENABLED 2025-12-30)

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: execution.intraday_trigger.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Delays order submission until price reclaims VWAP (Volume-Weighted Average Price).

**WHY:**
- Prevents chasing weak openings that immediately reverse
- Confirms momentum before committing capital
- Filters out gap-and-fade scenarios

**HOW:**
1. Scanner identifies setup at previous close
2. Next morning, instead of blindly buying at open...
3. System polls Alpaca quotes every 60 seconds
4. Waits for price > VWAP (shows strength)
5. Only THEN submits the order
6. If no trigger within 120 minutes, skips the trade

**MODES:**
- `vwap_reclaim` (default): Price must be above VWAP
- `first_hour_high`: Price must break first hour's high
- `first_hour_low`: For shorts, price must break first hour's low
- `combined`: Both conditions must be met

---

### 2. Earnings Filter

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: filters.earnings.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Skips signals for stocks with earnings within 2 days before or 1 day after.

**WHY:**
- Earnings announcements cause unpredictable gaps
- Technical setups become invalid around earnings
- Risk/reward is unknowable when fundamental catalyst pending

**HOW:**
- Queries Polygon earnings calendar API
- Caches results in `state/earnings_cache.json`
- Filters signals during screening phase

---

### 3. Rate Limiter

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: execution.rate_limiter.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Limits API requests to 120/minute with exponential backoff on 429 errors.

**WHY:**
- Prevents API throttling from Alpaca/Polygon
- Ensures smooth operation during high activity
- Auto-retries failed requests

**HOW:**
- Token bucket algorithm (120 tokens/minute)
- On 429 error: exponential backoff (500ms base, 3 retries max)
- Logs warnings when near rate limit

---

### 4. Execution Guard

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: execution_guard.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Validates quotes before order submission.

**WHY:**
- Prevents fills at stale prices (quote > 5 seconds old)
- Prevents fills at wide spreads (> 0.5%)
- Critical safety check for live trading

**HOW:**
- Fetches fresh quote from Alpaca
- Checks timestamp freshness
- Calculates bid-ask spread
- Rejects order if either check fails

---

### 5. Regime Filter

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: regime_filter.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Only trades when market regime is favorable (SPY > SMA(200), volatility < 25%).

**WHY:**
- Mean-reversion strategies fail in bear markets
- High volatility invalidates normal stop distances
- Trend alignment improves win rate

**HOW:**
- Fetches SPY daily bars
- Calculates SMA(200) and realized volatility
- Blocks all signals if regime unfavorable

---

### 6. Portfolio Risk Gate

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: portfolio_risk.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Enforces portfolio-level risk limits.

**WHY:**
- Prevents concentration in single names (max 10%)
- Prevents sector concentration (max 30%)
- Limits correlated positions (max 40% in correlated basket)

**HOW:**
- Checks current positions before new entry
- Calculates sector exposure using sector map
- Computes correlation using 20-day returns
- Rejects trade if any limit exceeded

---

### 7. Quality Gate

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: quality_gate.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Scores each signal on multiple factors (min 70/100 to pass).

**WHY:**
- Reduces ~50 signals/week to ~5 high-quality trades
- Ensures only best setups are taken
- Prevents overtrading

**HOW:**
- Conviction score (30 points)
- ML confidence (25 points)
- Strategy raw score (15 points)
- Regime alignment (15 points)
- Liquidity score (15 points)
- Must score >= 70 to pass

---

### 8. Macro Blackout

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: risk.macro_blackout_enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Skips trading on major economic event days (FOMC, NFP, CPI).

**WHY:**
- Macro events cause unpredictable volatility
- Technical setups become invalid
- Risk/reward is unknowable

**HOW:**
- Calendar lookup in `core/clock/macro_events.py`
- Checks before signal generation
- Blocks all new entries on blackout days

---

### 9. Execution Clamp

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: execution.clamp.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Limits order price to 2% from last quote.

**WHY:**
- Prevents runaway limit orders during flash moves
- Protects against fat-finger errors
- Safety net for volatile conditions

**HOW:**
- Gets current quote before submission
- Calculates max allowable price (quote + 2%)
- Clamps limit price if exceeded

---

### 10. Cognitive Brain

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: cognitive.enabled` |
| **Default** | `true` |
| **Needs History** | PARTIAL (learns but works without) |

**WHAT:** AI-powered decision system with System 1/2 routing.

**WHY:**
- Smarter trade decisions via metacognition
- Learns from outcomes over time
- Provides human-like reasoning

**HOW:**
- MetacognitiveGovernor routes decisions (fast vs slow path)
- ReflectionEngine learns from trade outcomes
- SelfModel tracks capability and calibration
- Works immediately, improves with history

---

### 11. LLM Analyzer

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: llm_analyzer.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Claude-powered narrative generation for trades.

**WHY:**
- Human-readable explanations for each trade
- Helps understand AI decision-making
- Useful for trade review and learning

**HOW:**
- Calls Claude API with trade context
- Generates narrative explaining setup
- Caches results for 60 minutes

---

### 12. Supervisor

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: supervisor.enabled` |
| **Default** | `true` |
| **Needs History** | NO |

**WHAT:** Process health monitoring with auto-restart.

**WHY:**
- Ensures system stays running 24/7
- Recovers from crashes automatically
- Alerts on repeated failures

**HOW:**
- Health check every 60 seconds
- Restarts on 5 errors within 15 minutes
- Max 3 restart attempts

---

### 13. Historical Edge Boost

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: historical_edge.enabled` |
| **Default** | `true` |
| **Needs History** | NO (uses backtest history, not live) |

**WHAT:** Boosts confidence for symbols with proven backtest performance.

**WHY:**
- Favors historically profitable setups
- Symbol-specific win rate adjustments
- Based on pre-computed backtest stats

**HOW:**
- Looks up symbol's historical win rate
- Compares to baseline (60%)
- Adjusts confidence by up to +/- 15 percentage points

---

## Features Disabled (Need Trading History)

These features require actual trade outcomes to function. Enable after accumulating sufficient trading history.

### 1. Probability Calibration

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: ml.calibration.enabled` |
| **Default** | `false` |
| **Needs History** | YES (50-100 trades) |

**WHAT:** Converts raw ML scores to true probabilities using isotonic regression.

**WHY:**
- Raw model outputs are often miscalibrated
- Calibration maps scores to actual win rates
- Improves position sizing accuracy

**WHEN TO ENABLE:** After 50+ paper trades with recorded outcomes.

**HOW IT WILL WORK:**
- Collects (prediction, actual_outcome) pairs
- Fits isotonic regression model
- Transforms future predictions

---

### 2. Conformal Prediction

| Attribute | Value |
|-----------|-------|
| **Config Location** | `config/base.yaml: ml.conformal.enabled` |
| **Default** | `false` |
| **Needs History** | YES (50-100 trades) |

**WHAT:** Sizes positions based on prediction uncertainty.

**WHY:**
- Smaller positions when model is uncertain
- Larger positions when model is confident
- Uncertainty-aware risk management

**WHEN TO ENABLE:** After 50+ paper trades with recorded outcomes.

**HOW IT WILL WORK:**
- Computes prediction intervals from historical residuals
- Returns multiplier (0.5 to 1.0) based on interval width
- Adjusts position size accordingly

---

### 3. Execution Bandit

| Attribute | Value |
|-----------|-------|
| **Config Location** | N/A (not yet in config) |
| **Default** | `false` |
| **Needs History** | YES (100+ executions) |

**WHAT:** Learns optimal order routing via multi-armed bandit.

**WHY:**
- Different order types work better in different conditions
- Learns from fill quality over time
- Adaptive order type selection

**WHEN TO ENABLE:** After 100+ paper executions.

---

## Feature Enablement Timeline

| Phase | Trades | Features to Enable |
|-------|--------|-------------------|
| **Day 1** | 0 | All 13 "No History" features (already on) |
| **Week 4** | ~50 | Consider: calibration, conformal |
| **Week 8** | ~100 | Consider: exec_bandit |

---

## Quick Reference Table

| Feature | Enabled | Needs History | Config Path |
|---------|---------|---------------|-------------|
| intraday_trigger | YES | NO | execution.intraday_trigger.enabled |
| earnings_filter | YES | NO | filters.earnings.enabled |
| rate_limiter | YES | NO | execution.rate_limiter.enabled |
| execution_guard | YES | NO | execution_guard.enabled |
| regime_filter | YES | NO | regime_filter.enabled |
| portfolio_risk | YES | NO | portfolio_risk.enabled |
| quality_gate | YES | NO | quality_gate.enabled |
| macro_blackout | YES | NO | risk.macro_blackout_enabled |
| execution_clamp | YES | NO | execution.clamp.enabled |
| cognitive_brain | YES | NO* | cognitive.enabled |
| llm_analyzer | YES | NO | llm_analyzer.enabled |
| supervisor | YES | NO | supervisor.enabled |
| historical_edge | YES | NO | historical_edge.enabled |
| calibration | NO | YES (50+) | ml.calibration.enabled |
| conformal | NO | YES (50+) | ml.conformal.enabled |
| exec_bandit | NO | YES (100+) | N/A |

*Learns from history but works without it

---

## How to Enable a Feature

1. Edit `config/base.yaml`
2. Find the feature section
3. Change `enabled: false` to `enabled: true`
4. Save and restart the system

Example:
```yaml
ml:
  calibration:
    enabled: true  # Changed from false
```

---

*Generated: 2025-12-30 by Claude*
*Kobe81 Trading Robot v2.3*
