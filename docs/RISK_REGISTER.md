# RISK_REGISTER.md - Risk Assessment and Mitigation

> **Last Updated:** 2026-01-03
> **Risk Owner:** System Operator
> **Review Frequency:** Weekly

---

## Risk Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Trading | 0 | 2 | 3 | 2 | 7 |
| Technical | 0 | 1 | 3 | 2 | 6 |
| Operational | 0 | 1 | 2 | 1 | 4 |
| **Total** | **0** | **4** | **8** | **5** | **17** |

---

## Trading Risks

### TR-001: Position Sizing Error

| Attribute | Value |
|-----------|-------|
| **Severity** | HIGH |
| **Likelihood** | LOW |
| **Impact** | Excessive loss on single trade |
| **Control** | Dual cap enforcement (2% risk + 20% notional) |
| **Evidence** | `risk/equity_sizer.py:calculate_position_size()` |
| **Incident** | 2026-01-02 - Manual order bypassed caps (FIXED) |

**Mitigation:**
- NEVER place manual orders - always use `run_paper_trade.py`
- PolicyGate enforces $75/order, $1k/daily caps
- Dual position cap formula: `min(shares_by_risk, shares_by_notional)`

---

### TR-002: Kill Zone Violation

| Attribute | Value |
|-----------|-------|
| **Severity** | HIGH |
| **Likelihood** | LOW |
| **Impact** | Trading during amateur hour (9:30-10:00) |
| **Control** | KillZoneGate automatic blocking |
| **Evidence** | `risk/kill_zone_gate.py:can_trade_now()` |

**Mitigation:**
- System automatically blocks orders during restricted times
- Opening range observer logs but DOES NOT trade
- Decorator enforces check before every order

---

### TR-003: Gap Risk

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | MEDIUM |
| **Impact** | Overnight gap invalidates setup |
| **Control** | Premarket validator flags >3% gaps |
| **Evidence** | `scripts/premarket_validator.py` |

**Mitigation:**
- Premarket validator runs at 8:00 AM
- Stocks with >3% gap flagged as GAP_INVALIDATED
- Removed from watchlist automatically

---

### TR-004: Liquidity Risk

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Slippage on entry/exit |
| **Control** | IOC LIMIT orders only |
| **Evidence** | `execution/broker_alpaca.py:place_ioc_limit()` |

**Mitigation:**
- Universe limited to optionable, liquid stocks
- IOC LIMIT prevents chasing
- No market orders ever

---

### TR-005: Concentration Risk

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Correlated losses |
| **Control** | Weekly exposure gate (40% cap) |
| **Evidence** | `risk/weekly_exposure_gate.py` |

**Mitigation:**
- 10% per position cap
- 20% daily exposure cap
- 40% weekly exposure cap
- Sector exposure tracking available

---

### TR-006: Strategy Degradation

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Likelihood** | MEDIUM |
| **Impact** | Win rate drops below threshold |
| **Control** | Walk-forward validation |
| **Evidence** | `wf_outputs/` split results |

**Mitigation:**
- Walk-forward testing validates out-of-sample
- Monte Carlo simulation for variance
- Performance metrics monitored via `/metrics` skill

---

### TR-007: Wrong Strategy Class

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Likelihood** | LOW (safeguards in place) |
| **Impact** | 13% win rate loss if wrong strategy used |
| **Control** | Registry + deprecation warnings |
| **Evidence** | `strategies/registry.py` |

**Mitigation:**
- Deprecation warnings in standalone strategies
- Strategy registry validates correct class
- CLAUDE.md documents correct usage

---

## Technical Risks

### TE-001: Data Provider Failure

| Attribute | Value |
|-----------|-------|
| **Severity** | HIGH |
| **Likelihood** | LOW |
| **Impact** | No signals generated |
| **Control** | Multi-source fallback chain |
| **Evidence** | `data/providers/multi_source.py` |

**Mitigation:**
- Primary: Polygon.io
- Fallback 1: Stooq
- Fallback 2: Yahoo Finance
- Prefetch caches data locally

---

### TE-002: Broker API Failure

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Orders not submitted |
| **Control** | Health check + Telegram alerts |
| **Evidence** | `monitor/health_endpoints.py` |

**Mitigation:**
- Preflight checks broker connectivity
- Health endpoints enable monitoring
- Kill switch preserves capital

---

### TE-003: State Corruption

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Inconsistent position tracking |
| **Control** | Hash chain audit + reconciliation |
| **Evidence** | `core/hash_chain.py`, `scripts/reconcile_alpaca.py` |

**Mitigation:**
- Hash chain detects tampering
- Weekly reconciliation with broker
- SQLite idempotency store (ACID)

---

### TE-004: Concurrent Write Collision

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Data loss on state files |
| **Control** | Single-instance enforcement |
| **Evidence** | `runner.py` PID file mechanism |

**Mitigation:**
- Only one runner instance allowed
- File locking on critical operations
- Atomic file writes where possible

---

### TE-005: Model Drift

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Likelihood** | MEDIUM |
| **Impact** | ML predictions degrade |
| **Control** | Drift detector + online learning |
| **Evidence** | `monitor/drift_detector.py`, `ml_advanced/online_learning.py` |

**Mitigation:**
- Concept drift detection enabled
- Periodic model retraining scripts
- Quality gate acts as safety net

---

### TE-006: Lookahead Bias

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Likelihood** | LOW (eliminated) |
| **Impact** | Inflated backtest results |
| **Control** | All indicators use `.shift(1)` |
| **Evidence** | `strategies/dual_strategy/combined.py` |

**Mitigation:**
- Code review enforces shift
- Next-bar fills in backtester
- Walk-forward validates OOS

---

## Operational Risks

### OP-001: Operator Error

| Attribute | Value |
|-----------|-------|
| **Severity** | HIGH |
| **Likelihood** | LOW |
| **Impact** | Manual intervention causes loss |
| **Control** | Kill switch + documented procedures |
| **Evidence** | `scripts/kill.py`, `docs/READINESS.md` |

**Mitigation:**
- Kill switch immediately halts all orders
- Emergency procedures documented
- Telegram alerts for all fills

---

### OP-002: Key/Secret Exposure

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Unauthorized access |
| **Control** | .env files gitignored |
| **Evidence** | `.gitignore` |

**Mitigation:**
- API keys in `.env` (not committed)
- Secrets skill validates key health
- Rotate keys periodically

---

### OP-003: System Unavailability

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Likelihood** | LOW |
| **Impact** | Missed trading opportunities |
| **Control** | Health monitoring + scheduler restart |
| **Evidence** | `monitor/health_endpoints.py`, `scripts/scheduler_kobe.py` |

**Mitigation:**
- Health endpoints on port 5000
- Windows Task Scheduler can restart
- Heartbeat monitoring

---

### OP-004: Documentation Drift

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Likelihood** | MEDIUM |
| **Impact** | Incorrect procedures followed |
| **Control** | Living documentation system |
| **Evidence** | `docs/WORKLOG.md`, `docs/CONTRIBUTING.md` |

**Mitigation:**
- Worklog template enforced
- Changelog tracks versions
- PR template requires doc updates

---

## Risk Response Matrix

| Response | When to Use | Example |
|----------|-------------|---------|
| **Avoid** | Critical risks | Never use market orders |
| **Mitigate** | High risks | Dual position caps |
| **Transfer** | External dependency | Use IOC (broker cancels unfilled) |
| **Accept** | Low likelihood | Model drift (monitoring in place) |

---

## Emergency Procedures

### Immediate Halt
```bash
python scripts/kill.py --reason "Emergency - [describe issue]"
```

### View Positions
```bash
python scripts/positions.py
```

### Manual Close (via Alpaca Dashboard)
1. Log into Alpaca dashboard
2. Navigate to Positions
3. Close individual positions manually

### Resume Trading
```bash
python scripts/resume.py --confirm
```

---

## Risk Review Checklist

**Daily:**
- [ ] Check Telegram for overnight alerts
- [ ] Run preflight before market open
- [ ] Verify no KILL_SWITCH file exists

**Weekly:**
- [ ] Run `reconcile_alpaca.py` to sync positions
- [ ] Review `docs/RISK_REGISTER.md` for updates
- [ ] Check walk-forward results for degradation

**Monthly:**
- [ ] Review win rate trend
- [ ] Assess model performance
- [ ] Update risk register if needed

---

## Related Documentation

- [KNOWN_GAPS.md](KNOWN_GAPS.md) - Missing components
- [READINESS.md](READINESS.md) - Production readiness
- [ARCHITECTURE.md](ARCHITECTURE.md) - System wiring
