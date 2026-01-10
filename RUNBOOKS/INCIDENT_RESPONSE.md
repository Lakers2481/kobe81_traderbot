# INCIDENT RESPONSE RUNBOOK - KOBE TRADING SYSTEM
## Version: 1.0
## Last Updated: 2026-01-06

---

## SEVERITY DEFINITIONS

| Severity | Description | Response Time |
|----------|-------------|---------------|
| SEV-0 | Live loss / Safety bypass / Data corruption | IMMEDIATE |
| SEV-1 | Major performance degradation / Bad fills | < 15 min |
| SEV-2 | Quality / Maintenance issues | < 24 hours |

---

## KILL SWITCH PROCEDURES

### Activate Kill Switch (SEV-0/SEV-1)
```bash
# IMMEDIATE: Stop all trading
touch state/KILL_SWITCH
echo "Reason: $(date) - [DESCRIBE ISSUE]" > state/KILL_SWITCH

# Verify activation
python -c "from core.kill_switch import is_kill_switch_active; print(f'Kill Switch Active: {is_kill_switch_active()}')"
# Expected: Kill Switch Active: True
```

### Deactivate Kill Switch (After Resolution)
```bash
# 1. Verify issue is resolved
# 2. Check no pending orders
python scripts/reconcile_alpaca.py

# 3. Remove kill switch
rm state/KILL_SWITCH

# 4. Verify deactivation
python -c "from core.kill_switch import is_kill_switch_active; print(f'Kill Switch Active: {is_kill_switch_active()}')"
# Expected: Kill Switch Active: False
```

> **IMPORTANT**: For complete kill switch activation/deactivation procedures, mandatory conditions,
> and audit trail requirements, see **`docs/KILL_SWITCH_POLICY.md`**

---

## INCIDENT: BRAIN NOT RUNNING

### Detection
```bash
curl http://localhost:8081/health
# If connection refused or no response -> Brain is down
```

### Response
```bash
# 1. Check for existing process
tasklist | findstr python
# On Linux: ps aux | grep python

# 2. Check logs for crash reason
tail -100 logs/events.jsonl | jq '.event'

# 3. Restart brain
python run_brain.py

# 4. Verify health
curl http://localhost:8081/health
```

---

## INCIDENT: POSITION DISCREPANCY

### Detection
```bash
python scripts/reconcile_alpaca.py
# Output shows mismatched positions
```

### Response
```bash
# 1. IMMEDIATELY activate kill switch
touch state/KILL_SWITCH
echo "Position discrepancy detected" > state/KILL_SWITCH

# 2. Get broker positions
python -c "
from execution.broker_alpaca import get_positions
positions = get_positions()
for p in positions:
    print(f'{p.symbol}: {p.qty} @ {p.current_price}')
"

# 3. Get local state
cat state/positions.json | jq .

# 4. Compare and identify discrepancy
# - Missing from broker: Ghost position locally
# - Missing locally: Orphan position in broker

# 5. Fix using reconcile_and_fix
python -c "
import sys
sys.path.insert(0, 'scripts')
from runner import reconcile_and_fix
from pathlib import Path
result = reconcile_and_fix(Path('.env'), auto_fix=True)
print(result)
"

# 6. Verify fix
python scripts/reconcile_alpaca.py

# 7. Remove kill switch if resolved
rm state/KILL_SWITCH
```

---

## INCIDENT: ORDER REJECTED

### Detection
```bash
# Check recent events
grep "order_rejected\|OrderRejected" logs/events.jsonl | tail -10 | jq .
```

### Response
```bash
# 1. Identify rejection reason
# Common reasons:
# - Insufficient buying power
# - Market closed
# - Position limit exceeded
# - Symbol not tradeable

# 2. Check buying power
python -c "
from execution.broker_alpaca import get_account
acct = get_account()
print(f'Buying Power: {acct.buying_power}')
print(f'Cash: {acct.cash}')
"

# 3. Check market hours
python -c "from market_calendar import is_market_closed; print(f'Market Closed: {is_market_closed()}')"

# 4. Review and retry if appropriate
```

---

## INCIDENT: DATA SOURCE FAILURE

### Detection
```bash
# Polygon API test
python -c "
from data.providers.polygon_eod import fetch_daily_bars_polygon
from pathlib import Path
df = fetch_daily_bars_polygon('AAPL', '2025-01-01', '2025-01-05', Path('data/polygon_cache'))
print(f'Rows: {len(df) if df is not None else 0}')
"
```

### Response
```bash
# 1. Check API key
python -c "import os; print(f'API Key Set: {bool(os.getenv(\"POLYGON_API_KEY\"))}')"

# 2. Check rate limits (Polygon has 5 req/min free tier)
# Wait 1 minute and retry

# 3. Clear cache if corrupted
rm -rf data/polygon_cache/

# 4. Retry data fetch
python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_800.csv --start 2025-01-01 --end 2025-12-31

# 5. If persistent failure, activate kill switch
touch state/KILL_SWITCH
echo "Data source failure - Polygon API" > state/KILL_SWITCH
```

---

## INCIDENT: HIGH DRAWDOWN

### Detection
```bash
# Check P&L
python -c "
from portfolio.state_manager import get_state_manager
sm = get_state_manager()
state = sm.load_state()
print(f'Daily P&L: {state.get(\"daily_pnl\", 0)}')
print(f'Open P&L: {state.get(\"open_pnl\", 0)}')
"
```

### Response
```bash
# 1. If daily loss > 5% of equity, activate kill switch
touch state/KILL_SWITCH
echo "Daily loss limit exceeded" > state/KILL_SWITCH

# 2. Review positions
python scripts/reconcile_alpaca.py

# 3. Consider closing all positions
# (Manual decision - not automated)

# 4. Review after market close
# Analyze what went wrong
# Update risk parameters if needed
```

---

## INCIDENT: LIVE TRADING ATTEMPTED

### SEV-0: CRITICAL

### Detection
Safety gates will block and log the attempt.

### Response
```bash
# 1. IMMEDIATELY investigate
grep "SafetyViolationError\|live_trading_attempted" logs/events.jsonl | tail -20

# 2. Verify all safety gates
python -c "
from safety.mode import PAPER_ONLY, LIVE_TRADING_ENABLED, get_trading_mode
from research_os.approval_gate import APPROVE_LIVE_ACTION
print(f'PAPER_ONLY: {PAPER_ONLY}')
print(f'LIVE_TRADING_ENABLED: {LIVE_TRADING_ENABLED}')
print(f'APPROVE_LIVE_ACTION: {APPROVE_LIVE_ACTION}')
print(get_trading_mode())
"

# 3. If any gate is compromised, activate kill switch
touch state/KILL_SWITCH
echo "SECURITY: Live trading attempted" > state/KILL_SWITCH

# 4. Full code review required before restart
```

---

## INCIDENT: DUPLICATE ORDERS

### Detection
```bash
# Check idempotency store
python -c "
from oms.idempotency_store import get_idempotency_store
store = get_idempotency_store()
print(store.get_recent(10))
"
```

### Response
```bash
# 1. Activate kill switch
touch state/KILL_SWITCH
echo "Duplicate order detected" > state/KILL_SWITCH

# 2. Check broker orders
python -c "
from execution.broker_alpaca import get_orders
orders = get_orders(status='all', limit=20)
for o in orders:
    print(f'{o.id}: {o.symbol} {o.side} {o.qty} @ {o.limit_price} - {o.status}')
"

# 3. Cancel any duplicate orders in broker
# (Manual via Alpaca dashboard if needed)

# 4. Clear idempotency store if corrupted
# rm state/idempotency_store.json

# 5. Restart system
rm state/KILL_SWITCH
python run_brain.py
```

---

## POST-INCIDENT PROCEDURES

### 1. Document Incident
Create file: `logs/incidents/INCIDENT_YYYY-MM-DD_HH-MM.md`

Include:
- Timestamp of detection
- Severity level
- Description of issue
- Actions taken
- Resolution time
- Root cause (if known)
- Prevention measures

### 2. Update Systems
- If new edge case found, update code
- If parameter issue, update config
- If monitoring gap, add alerting

### 3. Review and Learn
- Add to episodic memory
- Update runbooks if needed
- Share learnings with team

---

## EMERGENCY CONTACTS

### Automated Alerts
- Telegram: Configured via `core/alerts.py`
- Health endpoint: `http://localhost:8081/health`

### Manual Escalation
1. Check all logs in `logs/` directory
2. Review `state/` directory for corruption
3. Check Alpaca dashboard for order status
