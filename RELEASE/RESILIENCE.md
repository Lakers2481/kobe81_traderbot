# RESILIENCE VERIFICATION

**Generated**: 2026-01-06
**Status**: VERIFIED

---

## STATE RECOVERY

### State Files
| File | Purpose | Recovery Behavior |
|------|---------|-------------------|
| `state/positions.json` | Open positions | Reconciled with broker on restart |
| `state/orders.json` | Order history | Idempotency prevents duplicates |
| `state/hash_chain.jsonl` | Audit trail | Append-only, never overwritten |
| `state/watchlist/*.json` | Daily watchlists | Regenerated if missing |
| `state/cognitive/*.json` | Learning state | Persisted, resumable |
| `state/autonomous/*.json` | Brain state | Task queue preserved |

### Restart Behavior
```
┌─────────────────────────────────────────────────────────────────────┐
│                        RUNNER RESTART                               │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load state/positions.json                                       │
│  2. Reconcile with broker (scripts/reconcile_alpaca.py)            │
│  3. Resume task queue from state/autonomous/task_queue.json        │
│  4. Check kill switch (state/KILL_SWITCH)                          │
│  5. Log RESTART event to hash chain                                │
│  6. Continue from last known state                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## IDEMPOTENCY STORE

### Purpose
Prevents duplicate order submissions across restarts.

### Implementation
```python
# oms/idempotency_store.py
class IdempotencyStore:
    def check_and_mark(self, order_key: str) -> bool:
        """Returns True if order already submitted, marks if not."""

    def get_order_key(self, symbol: str, side: str, date: str) -> str:
        """Generate unique key for order."""
```

### Verification Test
```python
# tests/security/test_runtime_choke_enforcement.py
def test_idempotency_prevents_duplicate_orders():
    store = IdempotencyStore()
    key = store.get_order_key("AAPL", "buy", "2026-01-06")
    assert store.check_and_mark(key) is False  # First time
    assert store.check_and_mark(key) is True   # Duplicate blocked
```

---

## RECONCILIATION

### Broker vs Local Sync
```bash
python scripts/reconcile_alpaca.py
```

| Check | Action |
|-------|--------|
| Position in broker, not local | Add to local state |
| Position in local, not broker | Mark as closed |
| Quantity mismatch | Update local to match broker |
| Price mismatch | Update local with broker values |

### Reconciliation Log
```python
# Logged to hash chain
{
    "event": "reconcile",
    "timestamp": "2026-01-06T10:00:00Z",
    "broker_positions": [...],
    "local_positions": [...],
    "discrepancies": [...],
    "actions_taken": [...]
}
```

---

## HASH CHAIN INTEGRITY

### Audit Trail
```python
# core/hash_chain.py
class HashChain:
    def append(self, event: dict) -> str:
        """Append event with hash linking to previous entry."""

    def verify(self) -> bool:
        """Verify chain integrity from genesis to tip."""
```

### Verification
```bash
python scripts/verify_hash_chain.py
```

Expected output:
```
Chain length: 1234 events
Genesis hash: abc123...
Tip hash: xyz789...
Integrity: VALID (all hashes verified)
```

---

## JOURNAL LOGGING

### Structured Events
```python
# core/structured_log.py
def jlog(event: str, **kwargs):
    """Log structured JSON event with timestamp."""
```

### Event Types Logged
| Event | When |
|-------|------|
| `signal_generated` | Scanner produces signal |
| `order_submitted` | Order sent to broker |
| `order_filled` | Order executed |
| `order_rejected` | Order rejected by broker/gate |
| `gate_blocked` | Safety gate blocked order |
| `position_opened` | Position entered |
| `position_closed` | Position exited |
| `reconcile_start` | Reconciliation begins |
| `reconcile_complete` | Reconciliation ends |
| `restart` | System restarted |
| `kill_switch_activated` | Emergency halt |

---

## CRASH RECOVERY TESTS

### Test Scenarios
| Scenario | Expected Behavior | Verified |
|----------|-------------------|----------|
| Kill runner mid-order | Order state persisted, no duplicates on restart | YES |
| Kill during reconcile | Reconcile resumes on restart | YES |
| Power loss | State files on disk, resume from last save | YES |
| Network outage | Orders timeout, retry with idempotency | YES |
| Broker disconnect | Reconnect with exponential backoff | YES |

### Manual Test Procedure
```bash
# 1. Start runner
python scripts/runner.py --mode paper

# 2. Wait for order submission
# 3. Kill process (Ctrl+C or kill)
# 4. Restart runner
python scripts/runner.py --mode paper

# 5. Verify:
#    - No duplicate orders in logs
#    - Positions match broker
#    - RESTART event in hash chain
```

---

## ALERTS ON FAILURE

### Alert Channels
| Condition | Alert |
|-----------|-------|
| Kill switch activated | Telegram + log |
| Order rejected | Telegram + log |
| Reconcile discrepancy | Telegram + log |
| Hash chain corruption | Telegram + log + HALT |
| Broker disconnect > 5min | Telegram + log |

### Implementation
```python
# alerts/professional_alerts.py
def send_alert(channel: str, message: str, severity: str):
    """Send alert to configured channels."""
```

---

## VERDICT

**RESILIENCE VERIFIED** - System can recover from crashes, prevent duplicates, and maintain audit trail integrity.
