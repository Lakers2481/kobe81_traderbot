# /resume

Deactivate kill switch and resume trading operations.

## Usage
```
/resume [--confirm]
```

## What it does
1. Removes `state/KILL_SWITCH` file
2. Runs preflight checks
3. Logs resumption to audit chain
4. Enables new order submissions

## Pre-Resume Checklist
Before resuming, verify:
- [ ] Root cause of halt identified
- [ ] Issue has been resolved
- [ ] Market conditions are normal
- [ ] Data feeds are healthy
- [ ] Broker connectivity confirmed

## Commands
```bash
# Check current kill switch status
test -f state/KILL_SWITCH && echo "KILL SWITCH ACTIVE: $(cat state/KILL_SWITCH)" || echo "Kill switch: OFF"

# Run preflight before resuming
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# If preflight passes, remove kill switch
rm state/KILL_SWITCH 2>/dev/null && echo "KILL SWITCH REMOVED - Trading resumed"

# Log resumption
python -c "
from core.structured_log import log_event
log_event('KILL_SWITCH_DEACTIVATED', {'reason': 'manual_resume', 'operator': 'claude'})
"
```

## Safety Notes
- Never resume without understanding why halt occurred
- Always run preflight first
- Monitor closely for 15 minutes after resuming
- Consider reduce-only mode initially
