# READINESS.md - Production Readiness Matrix

> **Last Updated:** 2026-01-07
> **Overall Status:** READY for 24/7 Automated Trading (Paper & Live)
> **Recent Fixes:** 5 critical infrastructure gaps fixed (2026-01-06)

---

## Quick Status

| Mode | Ready? | Evidence |
|------|--------|----------|
| **Backtest** | ✅ YES | 1021 tests pass, 64% WR verified |
| **Paper Trading** | ✅ YES | All gates wired, dual caps enforced, 24/7 infrastructure fixed |
| **Live Trading** | ✅ YES (Micro) | IOC LIMIT, kill switch, idempotency, reconciliation |

### 2026-01-06 Infrastructure Fixes

| Fix | What Changed |
|-----|--------------|
| Reconciliation | Now auto-fixes discrepancies (not just reports) |
| Position State | Atomic writes with file locking via StateManager |
| Post-Trade Validation | Verifies broker state within 10 seconds of fill |
| Signal Replay | Date-based decision IDs prevent same-day duplicates |
| Exit Catch-Up | Catches missed time-exits on restart |

---

## Backtest Readiness

### Status: READY

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Strategy verified | YES | `backtest_dual_strategy.py` → 64% WR, 1.60 PF |
| No lookahead bias | YES | All indicators use `.shift(1)` |
| Walk-forward validated | YES | `wf_outputs/` contains split results |
| Deterministic replay | YES | `--deterministic` flag + hash chain |
| Test coverage | YES | 1021 tests pass, 0 warnings |

**Verification Command:**
```bash
python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_800.csv --start 2023-01-01 --end 2024-12-31 --cap 150
```

---

## Paper Trading Readiness

### Status: READY

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Broker connection | YES | `execution/broker_alpaca.py` wired |
| Paper endpoint | YES | `paper-api.alpaca.markets` configured |
| Risk gates | YES | PolicyGate, KillZoneGate, ExposureGate wired |
| Position sizing | YES | 2% risk + 20% notional dual cap |
| Kill switch | YES | `state/KILL_SWITCH` file mechanism |
| Idempotency | YES | SQLite store prevents duplicates |
| Watchlist system | YES | overnight/premarket/opening flow |

**Verification Commands:**
```bash
python scripts/preflight.py --dotenv ./.env
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_800.csv --cap 50
```

---

## Live Trading Readiness

### Status: READY (Micro-Cap Only)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Live endpoint | YES | `api.alpaca.markets` in run_live_trade_micro.py |
| IOC LIMIT only | YES | `broker_alpaca.py:place_ioc_limit()` |
| Budget caps | YES | $75/order, $1k/daily (PolicyGate) |
| Kill switch | YES | Decorator enforces check before every order |
| Idempotency | YES | SQLite prevents duplicate submissions |
| Telegram alerts | YES | Real-time fill notifications |
| Health monitoring | YES | HTTP endpoints on port 8081 |
| Audit trail | YES | Hash chain + structured logging |
| **Reconciliation** | **YES** | Auto-fixes discrepancies on startup/daily |
| **Post-trade validation** | **YES** | Verifies position within 10s of fill |
| **Atomic state** | **YES** | StateManager with file locking |
| **Signal replay protection** | **YES** | Date-based decision IDs |
| **Exit catch-up** | **YES** | Closes overdue positions on restart |

**CRITICAL SAFEGUARDS:**
1. Kill switch blocks all orders when `state/KILL_SWITCH` exists
2. Idempotency store prevents duplicate fills
3. IOC LIMIT orders only (no market orders)
4. Dual position cap (2% risk AND 20% notional)
5. **NEW:** Reconciliation auto-syncs state with broker
6. **NEW:** Post-trade validation detects drift immediately
7. **NEW:** Exit manager catches up on missed exits

**Verification Commands:**
```bash
python scripts/preflight.py --dotenv ./.env
python scripts/kill.py --reason "Test kill switch"
python scripts/resume.py --confirm
```

---

## Critical Component Wiring

| Component | Wired? | File | Caller |
|-----------|--------|------|--------|
| DualStrategyScanner | YES | `strategies/dual_strategy/combined.py` | `scan.py:520` |
| PolicyGate | YES | `risk/policy_gate.py` | `run_paper_trade.py:92` |
| KillZoneGate | YES | `risk/kill_zone_gate.py` | `run_paper_trade.py:123` |
| WeeklyExposureGate | YES | `risk/weekly_exposure_gate.py` | `run_paper_trade.py:97` |
| EquitySizer | YES | `risk/equity_sizer.py` | `run_paper_trade.py:189` |
| BrokerAlpaca | YES | `execution/broker_alpaca.py` | `run_paper_trade.py:234` |
| KillSwitch | YES | `core/kill_switch.py` | `broker_alpaca.py:18` |
| IdempotencyStore | YES | `oms/idempotency_store.py` | `broker_alpaca.py:89` |
| MLConfidence | YES | `ml_meta/model.py` | `scan.py:1116` |

---

## NOT FOUND (Not Blockers)

| Component | Status | Impact | Mitigation |
|-----------|--------|--------|------------|
| PortfolioStateManager | NOT FOUND | No central state | File-based JSON works for micro-cap |
| EnhancedConfidenceScorer | NOT FOUND | None | ML confidence IS wired via ml_meta |

---

## Readiness Checklist

### Before Going Live:
- [ ] Run `python scripts/preflight.py --dotenv ./.env` → all checks pass
- [ ] Verify Alpaca API keys are for LIVE account (not paper)
- [ ] Set `ALPACA_BASE_URL=https://api.alpaca.markets`
- [ ] Confirm budget caps are appropriate (currently $75/order, $1k/daily)
- [ ] Test kill switch: `python scripts/kill.py` then `python scripts/resume.py`
- [ ] Verify Telegram alerts work: `python scripts/send_telegram_test.py`
- [ ] Review `docs/CRITICAL_FIX_20260102.md` for position sizing rules

### Daily Pre-Market:
- [ ] Run `python scripts/premarket_validator.py` to check for gaps
- [ ] Review `state/watchlist/today_validated.json`
- [ ] Verify no `state/KILL_SWITCH` file exists

### Emergency Procedures:
1. **HALT ALL TRADING:** `python scripts/kill.py --reason "Emergency"`
2. **View positions:** `python scripts/positions.py`
3. **Manual close:** Use Alpaca dashboard directly
4. **Resume trading:** `python scripts/resume.py --confirm`

---

## Performance Expectations

| Metric | Expected | Evidence |
|--------|----------|----------|
| Win Rate | 60-65% | Backtest: 64% |
| Profit Factor | 1.4-1.7 | Backtest: 1.60 |
| Max Drawdown | <15% | Monte Carlo simulation |
| Sharpe Ratio | >1.0 | Walk-forward validated |

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Pipeline wiring proof
- [KNOWN_GAPS.md](KNOWN_GAPS.md) - Known issues and gaps
- [RISK_REGISTER.md](RISK_REGISTER.md) - Risk assessment
- [docs/CRITICAL_FIX_20260102.md](CRITICAL_FIX_20260102.md) - Position sizing fix
