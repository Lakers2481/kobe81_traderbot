# Forward Test Protocol - The Gauntlet

> Last Updated: 2026-01-07
> Status: MANDATORY before any live trading
> Duration: 1-3 months paper trading

---

## Overview

**The Gauntlet is a mandatory 1-3 month live paper trading period with zero human intervention.** This protocol validates that Kobe performs in live market conditions as expected from backtesting before any real capital is risked.

**Philosophy: The backtest shows what COULD happen. The Gauntlet shows what WILL happen.**

---

## Why The Gauntlet Exists

| Risk | Mitigation |
|------|------------|
| Backtest overfitting | Live data has no lookahead |
| Execution slippage | Real fills, real spreads |
| Data quality issues | Live data stream |
| System reliability | 24/7 operation test |
| Edge decay detection | Compare live vs historical |

---

## The Three Phases

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        THE GAUNTLET - 3 PHASES                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: SYSTEM STABILITY (Week 1-2)                                    │
│    Focus: Does everything work?                                          │
│    ├── Scanner runs daily without errors                                │
│    ├── Orders submit and fill correctly                                 │
│    ├── Positions track accurately                                        │
│    ├── Reconciliation passes                                             │
│    └── No unhandled exceptions                                           │
│                                                                          │
│  PHASE 2: PERFORMANCE VALIDATION (Week 3-8)                              │
│    Focus: Does performance match expectations?                           │
│    ├── Win rate within 10% of backtest                                  │
│    ├── Sharpe within 0.5 of backtest                                    │
│    ├── Slippage < 25 BPS average                                        │
│    ├── No SEV-0 incidents                                               │
│    └── Drawdown within expected range                                    │
│                                                                          │
│  PHASE 3: FULL AUTONOMY (Week 9-12)                                      │
│    Focus: Can it run completely unattended?                              │
│    ├── Zero human intervention                                          │
│    ├── Self-healing mechanisms work                                     │
│    ├── All monitoring/alerts function                                   │
│    └── Edge preserved over time                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Entry Criteria (Must Pass Before Starting)

All items must be checked before beginning The Gauntlet:

### Technical Readiness
- [ ] All 7 safety gates verified (`python tools/super_audit_verifier.py`)
- [ ] Walk-forward backtest completed with Sharpe > 1.0
- [ ] All tests passing (`python -m pytest tests/ -v`)
- [ ] No F821/E999 linting errors
- [ ] Data providers healthy (`python scripts/preflight.py`)

### Operational Readiness
- [ ] RUNBOOKS reviewed and understood
- [ ] Telegram alerts configured and tested
- [ ] Kill switch tested manually (activate and deactivate)
- [ ] Reconciliation script tested
- [ ] Health endpoints responding

### Documentation Readiness
- [ ] KILL_SWITCH_POLICY.md read and understood
- [ ] PROMOTION_GATE_WORKFLOW.md read and understood
- [ ] INCIDENT_RESPONSE.md read and understood

### Backtest Baseline Recorded
- [ ] Backtest period: _____________ to _____________
- [ ] Backtest Win Rate: _________%
- [ ] Backtest Sharpe: __________
- [ ] Backtest Profit Factor: __________
- [ ] Backtest Max Drawdown: __________%

---

## Starting The Gauntlet

### Launch Command
```bash
# Start paper trading with full logging
python scripts/runner.py \
    --mode paper \
    --universe data/universe/optionable_liquid_800.csv \
    --cap 50 \
    --scan-times 10:00,14:30 \
    --gauntlet-mode
```

### Record Start Time
```bash
echo '{"gauntlet_start": "'$(date -Iseconds)'", "backtest_wr": 0.64, "backtest_sharpe": 1.2}' > state/gauntlet_metadata.json
```

---

## Monitoring During The Gauntlet

### Daily Checks (5 minutes)
- [ ] Review `state/heartbeat.json` - last update within 30 minutes
- [ ] Check Telegram for any alerts
- [ ] Verify no kill switch file exists

### Weekly Checks (30 minutes)
- [ ] Run execution report: `python scripts/generate_execution_report.py`
- [ ] Review slippage analysis: `python execution/analytics/slippage_tracker.py --report`
- [ ] Check reconciliation: `python scripts/reconcile_alpaca.py`
- [ ] Review P&L: `/pnl --weekly`

### Monthly Checks (2 hours)
- [ ] Full performance attribution
- [ ] Compare live vs backtest metrics
- [ ] Run `scripts/live_vs_backtest_reconcile.py`
- [ ] Document any anomalies

---

## Intervention Rules (CRITICAL)

### ALLOWED Interventions
| Action | When Permitted | Documentation Required |
|--------|----------------|------------------------|
| Kill switch activation | True emergency only (see K1-K10) | Full incident report |
| Restart after crash | System genuinely down | Log restart time |
| Infrastructure fixes | Server/network issues | Technical log only |

### FORBIDDEN Interventions
| Action | Why Forbidden |
|--------|---------------|
| Parameter changes | Invalidates test |
| Manual trades | Corrupts statistics |
| Strategy tweaks | Must restart Gauntlet |
| Cherry-picking signals | Defeats purpose |
| Pausing for "bad markets" | Unrealistic |

**If you intervene in a forbidden way, The Gauntlet must restart from Week 1.**

---

## Exit Criteria (Must Pass to Graduate)

### Phase 1 Exit (Week 2)
- [ ] Zero unhandled exceptions for 14 consecutive days
- [ ] All daily scans completed successfully
- [ ] Reconciliation passing daily
- [ ] No data gaps > 1 day

### Phase 2 Exit (Week 8)
- [ ] Live Win Rate: ___% (must be within 10% of backtest)
- [ ] Live Sharpe: ___ (must be within 0.5 of backtest)
- [ ] Average Slippage: ___ BPS (must be < 25 BPS)
- [ ] No SEV-0 incidents
- [ ] Maximum Drawdown: ___% (must not exceed backtest 2x)

### Phase 3 Exit (Week 12)
- [ ] Zero human interventions for 4 consecutive weeks
- [ ] Self-healing mechanisms triggered and recovered at least once
- [ ] All alerts functioning correctly
- [ ] Edge not degraded (Sharpe stable)

---

## Live vs Backtest Comparison

Run weekly during Phase 2 and 3:

```bash
python scripts/live_vs_backtest_reconcile.py \
    --live-start 2026-01-07 \
    --backtest-file wf_outputs/wf_summary.csv
```

### Acceptable Divergence

| Metric | Max Divergence | Action if Exceeded |
|--------|----------------|-------------------|
| Win Rate | +/- 10% | Investigate, may continue |
| Sharpe | +/- 0.5 | Investigate, may continue |
| Avg P&L/Trade | +/- 25% | Pause, analyze slippage |
| Profit Factor | +/- 0.3 | Pause, full audit |
| Max Drawdown | +100% of backtest | Kill switch, investigate |

---

## Failure Scenarios

### Gauntlet Failed - Must Restart
- Human intervention (forbidden type)
- SEV-0 incident during Phase 2/3
- Metrics exceed acceptable divergence
- Extended system downtime (> 24 hours)

### Gauntlet Paused - Can Resume
- Infrastructure issue (not strategy related)
- Market closure (holiday)
- Data provider temporary outage (< 4 hours)

### Gauntlet Passed
All exit criteria met for all three phases. System approved for live trading consideration.

---

## Graduation Ceremony

When The Gauntlet is passed:

### 1. Generate Final Report
```bash
python scripts/generate_gauntlet_report.py --complete
```

### 2. Document Results
```markdown
## Gauntlet Completion Certificate

Start Date: _____________
End Date: _____________
Duration: _____ weeks

### Performance Summary
- Total Trades: _____
- Win Rate: _____%
- Sharpe Ratio: _____
- Profit Factor: _____
- Max Drawdown: _____%
- Average Slippage: _____ BPS

### Comparison to Backtest
- Win Rate Delta: _____% (backtest: _____%)
- Sharpe Delta: _____ (backtest: _____)

### Incidents
- SEV-0: _____
- SEV-1: _____
- Kill Switch Activations: _____

### Verdict
[ ] PASSED - Approved for live trading consideration
[ ] FAILED - Requires investigation
```

### 3. Archive Gauntlet Data
```bash
python scripts/archive_gauntlet.py --period 2026Q1
```

---

## Post-Gauntlet: Live Trading Decision

Passing The Gauntlet does NOT automatically enable live trading. Additional requirements:

1. **Capital Allocation Decision** - How much real money?
2. **Risk Limit Review** - Are current limits appropriate?
3. **Monitoring Plan** - Who watches when?
4. **Rollback Plan** - How to return to paper if issues?
5. **Final Human Approval** - Explicit go/no-go decision

---

## Related Documents

- `docs/KILL_SWITCH_POLICY.md` - Emergency procedures
- `docs/PROMOTION_GATE_WORKFLOW.md` - Parameter changes
- `docs/SAFETY_GATES.md` - All safety mechanisms
- `RUNBOOKS/24_7_OPERATIONS.md` - Daily operations
- `RUNBOOKS/INCIDENT_RESPONSE.md` - When things go wrong
- `scripts/live_vs_backtest_reconcile.py` - Comparison tool

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-01-07 | Initial protocol |
