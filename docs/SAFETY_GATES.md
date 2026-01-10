# SAFETY GATES - Complete Documentation

> Last Updated: 2026-01-07
> Status: All 7 Gates Verified
> Verdict: PAPER READY

---

## Overview

The Kobe trading system implements 7 independent safety gates that must ALL pass before any trade execution. These gates form a defense-in-depth strategy ensuring that:

1. No accidental live trading occurs
2. Risk limits are enforced at the broker boundary
3. Human intervention can halt trading instantly
4. Multiple fail-safes prevent any single point of failure

---

## The 7 Safety Gates

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SAFETY GATE HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Gate 1: PAPER_ONLY = True                                          │
│     ↓                                                                │
│  Gate 2: LIVE_TRADING_ENABLED = False                               │
│     ↓                                                                │
│  Gate 3: Kill Switch Mechanism (code exists)                        │
│     ↓                                                                │
│  Gate 4: APPROVE_LIVE_ACTION = False                                │
│     ↓                                                                │
│  Gate 5: Kill Switch File (not present = OK)                        │
│     ↓                                                                │
│  Gate 6: @require_policy_gate decorator                             │
│     ↓                                                                │
│  Gate 7: @require_no_kill_switch decorator                          │
│     ↓                                                                │
│  [TRADE ALLOWED]                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Gate 1: PAPER_ONLY Constant

**File:** `safety/mode.py`
**Line:** 50
**Value:** `PAPER_ONLY: bool = True`

### Purpose
Master switch that determines if the system operates in paper or live mode. When `True`, all trading goes through paper trading APIs.

### Evidence
```python
# safety/mode.py:50
PAPER_ONLY: bool = True  # CRITICAL: Must be True for paper trading
```

### How to Verify
```bash
grep -n "PAPER_ONLY" safety/mode.py
# Should show: PAPER_ONLY: bool = True
```

### What Happens if False
- System connects to LIVE Alpaca endpoint
- Real money at risk
- **REQUIRES EXPLICIT HUMAN CHANGE**

---

## Gate 2: LIVE_TRADING_ENABLED Constant

**File:** `safety/mode.py`
**Line:** 53
**Value:** `LIVE_TRADING_ENABLED: bool = False`

### Purpose
Secondary switch that must be explicitly enabled for live trading. Even if PAPER_ONLY is False, this gate prevents live execution.

### Evidence
```python
# safety/mode.py:53
LIVE_TRADING_ENABLED: bool = False  # CRITICAL: Must be False for paper
```

### How to Verify
```bash
grep -n "LIVE_TRADING_ENABLED" safety/mode.py
# Should show: LIVE_TRADING_ENABLED: bool = False
```

### What Happens if True
- Combined with PAPER_ONLY=False enables live trading
- **REQUIRES EXPLICIT HUMAN CHANGE**

---

## Gate 3: Kill Switch Mechanism

**File:** `core/kill_switch.py`
**Key Functions:** `is_kill_switch_active()`, `check_kill_switch()`

### Purpose
Emergency halt mechanism that can be activated by creating a file. When active, ALL trading operations are blocked.

### Evidence
```python
# core/kill_switch.py
KILL_SWITCH_PATH = Path(os.getenv("KILL_SWITCH_PATH", "state/KILL_SWITCH"))

def is_kill_switch_active() -> bool:
    """Returns True if kill switch file exists."""
    return KILL_SWITCH_PATH.exists()

def check_kill_switch(allow_exit_orders: bool = False) -> None:
    """Raises KillSwitchActiveError if kill switch is active."""
    if is_kill_switch_active():
        raise KillSwitchActiveError(...)
```

### How to Activate
```bash
echo '{"reason": "Emergency halt", "activated_at": "2026-01-07T12:00:00"}' > state/KILL_SWITCH
```

### How to Deactivate
```bash
rm state/KILL_SWITCH
```

### Grace Period (Added 2026-01-07)
The kill switch allows a 60-second grace period for exit orders to close positions gracefully.

---

## Gate 4: APPROVE_LIVE_ACTION Constant

**File:** `research_os/approval_gate.py`
**Line:** 29
**Value:** `APPROVE_LIVE_ACTION = False`

### Purpose
Research OS gate that prevents automated changes to production parameters or live trading without human approval.

### Evidence
```python
# research_os/approval_gate.py:29
APPROVE_LIVE_ACTION = False  # NEVER change programmatically
```

### How to Verify
```bash
grep -n "APPROVE_LIVE_ACTION" research_os/approval_gate.py
# Should show: APPROVE_LIVE_ACTION = False
```

### What Happens if True
- Research OS can promote experimental parameters to production
- **REQUIRES EXPLICIT HUMAN CHANGE**

---

## Gate 5: Kill Switch File Presence

**File:** `state/KILL_SWITCH`
**Expected:** Not present (or present to BLOCK trading)

### Purpose
Physical file that when present, blocks all trading. Easy to create/remove for immediate control.

### How to Verify
```bash
ls -la state/KILL_SWITCH 2>&1
# Should show: No such file or directory (PASS)
# Or show file exists (ACTIVE - trading blocked)
```

### File Format (when active)
```json
{
  "reason": "Why trading is halted",
  "activated_at": "2026-01-07T12:00:00-05:00",
  "activated_by": "human"
}
```

---

## Gate 6: @require_policy_gate Decorator

**File:** `execution/broker_alpaca.py`
**Line:** 47-173

### Purpose
Decorator applied to order execution functions that:
1. Enforces PolicyGate budget limits ($75/order, $1K/day)
2. Checks compliance rules (prohibited list, trading rules)
3. Logs all gate decisions

### Evidence
```python
# execution/broker_alpaca.py:47-58
def require_policy_gate(func):
    """
    Decorator to enforce PolicyGate AND Compliance checks before order placement.

    CRITICAL FIX (2026-01-04): PolicyGate must be enforced at broker boundary,
    not just in higher-level orchestration layers.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # ... PolicyGate checks ...
```

### Applied To
```python
# execution/broker_alpaca.py:768
@require_no_kill_switch
@require_policy_gate
@require_portfolio_gate  # Added 2026-01-07
async def place_ioc_limit(...):
```

### What Gets Blocked
- Orders exceeding $75 notional
- Orders that would exceed $1K daily budget
- Orders for prohibited symbols
- Orders outside trading hours

---

## Gate 7: @require_no_kill_switch Decorator

**File:** `execution/broker_alpaca.py`
**Applied via:** `core/kill_switch.py:require_no_kill_switch`

### Purpose
Decorator that checks kill switch status before ANY order execution. If kill switch is active, the order is rejected immediately.

### Evidence
```python
# core/kill_switch.py
def require_no_kill_switch(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to block execution when kill switch is active."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        check_kill_switch()
        return func(*args, **kwargs)
    return wrapper
```

### Applied To
```python
# execution/broker_alpaca.py:768
@require_no_kill_switch  # Gate 7
@require_policy_gate     # Gate 6
@require_portfolio_gate  # Additional VaR gate
async def place_ioc_limit(...):
```

### What Gets Blocked
- ALL orders when state/KILL_SWITCH exists
- Exit orders after grace period (60s)

---

## Verification Script

**File:** `tools/super_audit_verifier.py`

Run to verify all gates:
```bash
python tools/super_audit_verifier.py
```

### Expected Output
```
============================================================
SUPER AUDIT VERIFICATION REPORT
============================================================
Generated: 2026-01-07T...

SAFETY GATES
----------------------------------------
[+] PAPER_ONLY                       PASS
    Value: True
[+] LIVE_TRADING_ENABLED             PASS
    Value: False
[+] Kill Switch                      PASS
    Value: Implemented
[+] APPROVE_LIVE_ACTION              PASS
    Value: False
[+] Kill Switch File                 PASS
    Value: Not present
[+] @require_policy_gate             PASS
    Value: Applied to execute_signal
[+] @require_no_kill_switch          PASS
    Value: Applied to execute_signal
Passed: 7/7

============================================================
VERDICT: VERIFIED
Pass Rate: 100.0%
============================================================
```

---

## Quick Reference

| Gate | File | Line | Safe Value |
|------|------|------|------------|
| 1 | safety/mode.py | 50 | `PAPER_ONLY = True` |
| 2 | safety/mode.py | 53 | `LIVE_TRADING_ENABLED = False` |
| 3 | core/kill_switch.py | - | Code exists |
| 4 | research_os/approval_gate.py | 29 | `APPROVE_LIVE_ACTION = False` |
| 5 | state/KILL_SWITCH | - | File not present |
| 6 | execution/broker_alpaca.py | 47 | Decorator applied |
| 7 | execution/broker_alpaca.py | - | Decorator applied |

---

## Emergency Procedures

### Immediate Trading Halt
```bash
# Fastest way to stop all trading
echo '{"reason":"EMERGENCY"}' > state/KILL_SWITCH
```

### Resume Trading
```bash
# Only after verifying system is safe
rm state/KILL_SWITCH
python tools/super_audit_verifier.py
```

### Verify Before Market Open
```bash
python scripts/preflight.py --dotenv .env
python tools/super_audit_verifier.py
```

---

## Certification Status

| Criterion | Status |
|-----------|--------|
| All 7 gates present | VERIFIED |
| All gates at safe values | VERIFIED |
| Decorators applied to execution | VERIFIED |
| Kill switch tested | VERIFIED |
| Grace period implemented | VERIFIED |

**VERDICT: PAPER READY**

---

## See Also

- `docs/SCHEDULER_CODEMAP.md` - Scheduler architecture
- `docs/ORCHESTRATION_DAG.md` - Pipeline design
- `tools/super_audit_verifier.py` - Verification script
- `risk/policy_gate.py` - PolicyGate implementation
