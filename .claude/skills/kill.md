# /kill

**EMERGENCY KILL SWITCH** - Immediately halt all trading.

## Usage
```
/kill [--reason "message"]
```

## What it does
1. Creates `state/KILL_SWITCH` file
2. Blocks ALL new order submissions
3. Logs reason to audit chain
4. Sends alert (if configured)

**Existing positions are NOT closed** - only new entries blocked.

## Commands
```bash
# Activate kill switch
echo "$(date -Iseconds) Manual kill: ${REASON:-emergency}" > state/KILL_SWITCH
echo "KILL SWITCH ACTIVATED"

# Verify it's active
test -f state/KILL_SWITCH && cat state/KILL_SWITCH

# Log to audit
python -c "
from core.structured_log import log_event
log_event('KILL_SWITCH_ACTIVATED', {'reason': 'manual', 'operator': 'claude'})
"
```

## When to Use
- System behaving unexpectedly
- Market flash crash / extreme volatility
- Data feed issues detected
- Broker connectivity problems
- Any situation requiring immediate halt

## To Resume
Use `/resume` skill after resolving the issue.

## Critical Notes
- Kill switch is **fail-safe** - if file exists, no orders go out
- Works even if main process is frozen
- Survives restarts
- Always document reason for audit trail


