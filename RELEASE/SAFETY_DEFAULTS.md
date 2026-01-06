# SAFETY DEFAULTS VERIFICATION

**Generated**: 2026-01-06
**Status**: ALL DEFAULTS FAIL-CLOSED

---

## CRITICAL SAFETY FLAGS

| Flag | Default Value | Fail Mode | Location |
|------|---------------|-----------|----------|
| `PAPER_ONLY` | `True` | CLOSED | config/base.yaml |
| `LIVE_TRADING_ENABLED` | `False` | CLOSED | config/base.yaml |
| `APPROVE_LIVE_ACTION` | `False` | CLOSED | safety/execution_choke.py |
| `APPROVE_LIVE_ACTION_2` | `False` | CLOSED | safety/execution_choke.py |
| `LIVE_ORDER_ACK_TOKEN` | `None` | CLOSED | Environment variable |
| `TRADING_MODE` | `paper` | CLOSED | config/base.yaml |

---

## KILL SWITCH MECHANISM

| Check | File | Behavior |
|-------|------|----------|
| `state/KILL_SWITCH` exists | safety/execution_choke.py | ALL orders blocked |
| File check frequency | Every order | Immediate effect |
| Recovery | Delete file manually | Human action required |

**Kill Switch Test**:
```python
# tests/security/test_runtime_choke_enforcement.py
def test_kill_switch_blocks_paper_orders(self):
    with patch("safety.execution_choke._check_kill_switch", return_value=True):
        result = evaluate_safety_gates(is_paper_order=True)
        assert result.allowed is False  # BLOCKED
```

---

## SEVEN FLAGS FOR LIVE TRADING

To execute a LIVE order, ALL seven flags must pass:

```python
# From safety/execution_choke.py
def evaluate_safety_gates(is_paper_order: bool = False) -> GateResult:
    flags = {
        "kill_switch_inactive": not _check_kill_switch(),
        "paper_only_disabled": not config.get("PAPER_ONLY", True),
        "live_trading_enabled": config.get("LIVE_TRADING_ENABLED", False),
        "trading_mode_live": config.get("TRADING_MODE", "paper") == "live",
        "approve_live_action": os.environ.get("APPROVE_LIVE_ACTION") == "true",
        "approve_live_action_2": os.environ.get("APPROVE_LIVE_ACTION_2") == "true",
        "ack_token_valid": _validate_ack_token(),
    }

    if is_paper_order:
        # Paper only needs kill switch inactive
        return GateResult(allowed=flags["kill_switch_inactive"], ...)

    # Live needs ALL flags
    return GateResult(allowed=all(flags.values()), ...)
```

---

## STARTUP VALIDATION

### Preflight Checks
```bash
python scripts/preflight.py --dotenv ./.env
```

| Check | Validates |
|-------|-----------|
| Environment variables | API keys present |
| Config pin | Parameters frozen |
| Broker probe | Connection valid |
| Kill switch | Not active |
| Data freshness | Recent data available |

### Runner Startup
```python
# scripts/runner.py validates on startup:
# 1. Strategy registry validation
# 2. Kill zone awareness
# 3. Broker connection
# 4. State file integrity
```

---

## POSITION SIZING CAPS

| Cap | Value | Enforced By |
|-----|-------|-------------|
| Risk per trade | 2% of equity | risk/equity_sizer.py |
| Notional per position | 20% of equity | risk/policy_gate.py |
| Daily exposure | 20% | risk/weekly_exposure_gate.py |
| Weekly exposure | 40% | risk/weekly_exposure_gate.py |
| Per-order budget | $75 | risk/policy_gate.py |
| Daily budget | $1,000 | risk/policy_gate.py |

---

## KILL ZONE ENFORCEMENT

| Time (ET) | Zone | Trading Allowed |
|-----------|------|-----------------|
| 9:30-10:00 | Opening Range | NO |
| 10:00-11:30 | London Close | YES |
| 11:30-14:00 | Lunch Chop | NO |
| 14:00-14:30 | Extended Lunch | NO |
| 14:30-15:30 | Power Hour | YES |
| 15:30-16:00 | Close | NO (manage only) |

**Enforced by**: `risk/kill_zone_gate.py`

---

## RESEARCH OS CONSTRAINTS

```python
# CRITICAL - NEVER CHANGED PROGRAMMATICALLY
APPROVE_LIVE_ACTION = False

# Research OS cannot:
# - Auto-merge code
# - Auto-enable live trading
# - Bypass human approval gate
```

---

## VERIFICATION COMMANDS

```bash
# Check all defaults
python -c "
from core.config_loader import load_config
c = load_config()
print(f'PAPER_ONLY: {c.get(\"PAPER_ONLY\", True)}')
print(f'LIVE_TRADING_ENABLED: {c.get(\"LIVE_TRADING_ENABLED\", False)}')
print(f'TRADING_MODE: {c.get(\"TRADING_MODE\", \"paper\")}')
"

# Check environment
echo $env:APPROVE_LIVE_ACTION  # Should be empty or not set
echo $env:APPROVE_LIVE_ACTION_2  # Should be empty or not set

# Check kill switch
Test-Path state/KILL_SWITCH  # Should be False
```

---

## VERDICT

**ALL SAFETY DEFAULTS ARE FAIL-CLOSED** - System blocks by default, requires explicit enablement for live trading.
