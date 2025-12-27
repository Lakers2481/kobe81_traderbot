# KOBE81 System Assumptions

This document lists assumptions made during the Top-1% Gaps implementation.

## Data Assumptions

### A1: Crypto Data Availability
**Assumption:** System has access to crypto price data via existing Polygon or Binance integration.

**Rationale:** The crypto_clock module schedules scans but relies on existing data providers.

**Fallback:** If crypto data unavailable, crypto scanning is skipped with a warning.

---

### A2: Sector Mapping Coverage
**Assumption:** Initial `data/sector_map.csv` covers top 100 stocks by market cap. Other symbols default to sector "Unknown".

**Rationale:** Sector data is not freely available for all symbols. We prioritize major holdings.

**Fallback:** Symbols with unknown sector are treated conservatively (counted toward a generic bucket).

---

### A3: Polygon Split Adjustment
**Assumption:** Polygon EOD data is already split-adjusted. We do not re-adjust.

**Rationale:** Polygon provides adjusted data by default. Re-adjusting could introduce errors.

**Fallback:** Corporate actions module logs warnings when potential splits are detected but does not modify data.

---

### A4: Correlation Window
**Assumption:** Uses 20-day rolling correlation for cluster detection.

**Rationale:** 20 days balances responsiveness with stability; industry standard for short-term correlation.

**Fallback:** Configurable via `portfolio_risk.correlation_window_days` in config.

---

## Execution Assumptions

### A5: Quote Freshness Threshold
**Assumption:** Quotes older than 5 seconds are considered stale and orders are rejected.

**Rationale:** Market conditions can change rapidly; stale quotes may lead to slippage.

**Fallback:** Configurable via `execution.guard.max_quote_age_seconds`.

---

### A6: Alpaca Trading Status
**Assumption:** Best-effort halt detection via `asset.tradable` flag from Alpaca API.

**Rationale:** Alpaca doesn't provide real-time halt notifications. The `tradable` field is our best signal.

**Fallback:** If tradable status cannot be determined, order is rejected (stand-down on uncertainty).

---

### A7: Spread Threshold
**Assumption:** Maximum acceptable spread is 0.50% of mid-price.

**Rationale:** Wide spreads indicate illiquidity and potential for adverse fills.

**Fallback:** Configurable via `execution.guard.max_spread_pct`.

---

## Backtest Assumptions

### A8: Default Slippage Model
**Assumption:** Default slippage is 5 basis points (fixed).

**Rationale:** Conservative estimate for liquid equities; more aggressive for illiquid names.

**Fallback:** Users can select ATR-based, spread-based, or volume-impact models.

---

### A9: Commission Structure
**Assumption:** Alpaca has zero-commission equity trades; we model SEC/FINRA fees only.

**Rationale:** Commission-free trading is standard; regulatory fees still apply.

**Fallback:** Full cost model available for brokers with commissions.

---

### A10: Fill Probability
**Assumption:** Limit orders fill if price touches the limit during the bar (simplified model).

**Rationale:** Without tick data, we cannot precisely model fill probability.

**Fallback:** Conservative partial-fill model available for illiquid names.

---

## Evolution Assumptions

### A11: Clone Detection Threshold
**Assumption:** Strategies with parameter similarity > 95% are considered clones.

**Rationale:** Prevents redundant strategies that would just dilute capital.

**Fallback:** Threshold configurable in evolution config.

---

### A12: Minimum Trades Per Regime
**Assumption:** Requires at least 10 trades per regime (trend, chop, high-vol, low-vol) for promotion.

**Rationale:** Insufficient sample size makes regime-specific performance unreliable.

**Fallback:** Configurable; can be disabled for research.

---

### A13: Multiple-Testing Penalty
**Assumption:** Applies Bonferroni-style adjustment when evaluating many strategy candidates.

**Rationale:** Prevents overfitting by chance when testing many combinations.

**Fallback:** Configurable multiplier based on number of candidates tested.

---

## LLM Playbook Assumptions

### A14: Claude API Availability
**Assumption:** Claude API (Anthropic) is optional. If API key not present, deterministic fallback is used.

**Rationale:** Not all deployments have API access; core functionality must work offline.

**Fallback:** Template-based playbook generation without LLM.

---

### A15: Decision Packet Completeness
**Assumption:** LLM playbooks ONLY reference fields present in the Decision Packet. Missing fields are labeled "Unknown".

**Rationale:** Prevents hallucination and ensures audit trail integrity.

**Fallback:** Strict field validation before playbook generation.

---

## Infrastructure Assumptions

### A16: Live Trading Approval
**Assumption:** Live trading requires BOTH `LIVE_TRADING_APPROVED=YES` environment variable AND `--approve-live` CLI flag.

**Rationale:** Defense in depth against accidental live trading.

**Fallback:** N/A - this is a hard requirement.

---

### A17: Volatility Targeting
**Assumption:** Optional feature, disabled by default.

**Rationale:** Volatility targeting adds complexity and may not suit all strategies.

**Fallback:** Enable via config if desired.

---

### A18: Crypto Universe
**Assumption:** Separate file `data/universe/crypto_top10.csv` if crypto trading enabled.

**Rationale:** Crypto universe is distinct from equity universe.

**Fallback:** If file missing and crypto enabled, log warning and skip crypto scanning.

---

### A19: Supervisor Check Interval
**Assumption:** Supervisor checks system health every 60 seconds.

**Rationale:** Balance between responsiveness and resource usage.

**Fallback:** Configurable interval.

---

### A20: Stand-Down Policy
**Assumption:** Any uncertainty in execution (missing data, API error, validation failure) results in order rejection, NOT execution.

**Rationale:** Safety first. It's better to miss a trade than to place an incorrect one.

**Fallback:** N/A - this is a core safety principle.

---

## Version Control

| Date | Author | Changes |
|------|--------|---------|
| 2025-12-27 | Claude Opus 4.5 | Initial assumptions for Top-1% Gaps implementation |
