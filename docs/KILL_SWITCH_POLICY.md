# Kill Switch Policy - The Big Red Button

> Last Updated: 2026-01-07
> Status: ACTIVE
> Owner: Human Operator (NEVER automated)

---

## Overview

The kill switch is Kobe's emergency halt mechanism. When activated, ALL trading activity ceases immediately. This document defines the exact conditions under which the kill switch MUST be activated, and the procedure for safe deactivation.

**CRITICAL: The decision to activate the kill switch should NEVER require deep analysis. When in doubt, ACTIVATE IT.**

---

## Kill Switch Location

```
state/KILL_SWITCH
```

When this file exists, all order submissions are blocked.

---

## Mandatory Activation Conditions (MUST Activate)

These conditions require IMMEDIATE kill switch activation. No exceptions.

| ID | Condition | Detection Method | Response Time |
|----|-----------|------------------|---------------|
| **K1** | Portfolio drawdown > 10% in one week | `risk/policy_gate.py` | Immediate |
| **K2** | 3+ consecutive critical component failures | `monitor/health_endpoints.py` | < 5 minutes |
| **K3** | Live P&L diverges from backtest P&L by > 25% | `scripts/live_vs_backtest_reconcile.py` | < 1 hour |
| **K4** | Any SEV-0 incident detected | `RUNBOOKS/INCIDENT_RESPONSE.md` | Immediate |
| **K5** | Primary data provider offline > 30 min with no failover | `guardian/resilience.py` | < 30 minutes |
| **K6** | PAPER_ONLY = False detected unexpectedly | `tools/super_audit_verifier.py` | Immediate |
| **K7** | Order rejected by broker 3+ times consecutively | `execution/broker_alpaca.py` | < 10 minutes |
| **K8** | Position reconciliation mismatch > $1000 | `scripts/reconcile_alpaca.py` | < 30 minutes |
| **K9** | Slippage consistently > 50 BPS for 5+ trades | `execution/tca/transaction_cost_analyzer.py` | < 1 hour |
| **K10** | Any unauthorized or unexpected trade execution | Human observation | Immediate |

---

## Optional Activation Conditions (Use Judgment)

These conditions warrant consideration but do not require automatic activation.

| Condition | Consider Activating If... |
|-----------|---------------------------|
| Unusual market conditions | VIX > 40, flash crash, circuit breakers triggered |
| Major news event not modeled | War, pandemic, major political event |
| System behaving erratically | Unexpected logs, unusual resource usage |
| Multiple minor anomalies | 3+ yellow flags in same day |
| Human uncertainty | "Something feels wrong" |

**When in doubt, activate. Capital preservation > missed trades.**

---

## Activation Procedure

### Method 1: Command Line (Fastest)
```bash
echo '{"reason": "EMERGENCY", "activated_at": "2026-01-07T12:00:00", "activated_by": "human"}' > state/KILL_SWITCH
```

### Method 2: Skill Command
```bash
/kill --reason "Portfolio drawdown exceeded 10%"
```

### Method 3: Python
```python
from core.kill_switch import activate_kill_switch
activate_kill_switch(reason="Critical failure detected")
```

---

## What Happens When Activated

1. **Immediate**: All new order submissions blocked
2. **Within 60 seconds**: Grace period allows exit orders for open positions
3. **After 60 seconds**: ALL orders blocked including exits
4. **Telegram Alert**: Critical alert sent to all configured channels
5. **Audit Log**: Activation recorded in `state/kill_switch_audit.jsonl`

---

## Deactivation Procedure (CRITICAL: Follow Exactly)

**DO NOT simply delete the kill switch file.** Follow this procedure:

### Step 1: Root Cause Analysis
```markdown
Document answers to:
- What triggered the activation?
- What was the root cause?
- What fix was implemented?
- How do we prevent recurrence?
```

### Step 2: Verify Fix
```bash
# Run relevant tests
python -m pytest tests/ -v

# Run super audit
python tools/super_audit_verifier.py
```

### Step 3: Verify All 7 Safety Gates
```bash
python tools/super_audit_verifier.py

# Expected output:
# SAFETY GATES
# [+] PAPER_ONLY                       PASS
# [+] LIVE_TRADING_ENABLED             PASS
# [+] Kill Switch                      PASS (will show ACTIVE here)
# [+] APPROVE_LIVE_ACTION              PASS
# ...
# Passed: 6/7 (Kill switch active is expected)
```

### Step 4: Human Sign-Off
```bash
# Record sign-off in audit log
echo '{"action": "signoff", "operator": "Your Name", "reason": "Root cause addressed", "timestamp": "2026-01-07T14:00:00"}' >> state/kill_switch_audit.jsonl
```

### Step 5: Remove Kill Switch File
```bash
rm state/KILL_SWITCH
```

### Step 6: Verify Deactivation
```bash
python tools/super_audit_verifier.py

# All 7 gates should now pass
```

### Step 7: Observation Period
- Monitor system for **minimum 1 hour** after restart
- Watch for recurrence of original issue
- Be ready to re-activate immediately

---

## Grace Period for Exit Orders

The kill switch includes a 60-second grace period for exit orders:

```python
# core/kill_switch.py
GRACE_PERIOD_SECONDS = 60

def check_kill_switch(allow_exit_orders: bool = False) -> None:
    if is_kill_switch_active():
        if allow_exit_orders:
            activation_time = get_activation_time()
            if (datetime.now() - activation_time).seconds < GRACE_PERIOD_SECONDS:
                return  # Allow exit order
        raise KillSwitchActiveError(...)
```

This allows orderly position closure before full halt.

---

## Audit Trail Requirements

All kill switch events MUST be logged:

```json
// state/kill_switch_audit.jsonl
{"timestamp": "2026-01-07T12:00:00", "action": "activate", "reason": "Portfolio drawdown 12%", "operator": "human"}
{"timestamp": "2026-01-07T14:00:00", "action": "signoff", "operator": "John Doe", "reason": "Reduced position sizes"}
{"timestamp": "2026-01-07T14:05:00", "action": "deactivate", "operator": "John Doe"}
```

---

## Automated vs Manual Activation

| Type | When Used | Authority |
|------|-----------|-----------|
| **Automated** | K1, K2, K5, K7, K8 | System detects and activates |
| **Manual** | K3, K4, K6, K9, K10, Optional conditions | Human decision required |

Even when automated activation occurs, **deactivation is ALWAYS manual**.

---

## Testing the Kill Switch

Regular testing ensures the mechanism works when needed:

### Weekly Test (Paper Trading)
```bash
# Activate
echo '{"reason": "WEEKLY TEST", "activated_at": "...", "activated_by": "test"}' > state/KILL_SWITCH

# Verify orders blocked
python -c "from core.kill_switch import check_kill_switch; check_kill_switch()"
# Should raise KillSwitchActiveError

# Deactivate
rm state/KILL_SWITCH

# Verify orders unblocked
python -c "from core.kill_switch import check_kill_switch; check_kill_switch()"
# Should pass silently
```

---

## Related Documents

- `docs/SAFETY_GATES.md` - All 7 safety gates
- `RUNBOOKS/INCIDENT_RESPONSE.md` - Incident handling
- `core/kill_switch.py` - Implementation
- `tools/super_audit_verifier.py` - Verification script

---

## Policy Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-01-07 | Initial policy document |
