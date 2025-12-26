# /status

Show Kobe's current system health and trading state.

## Usage
```
/status
```

## What it does
Displays:
- Current mode (PAPER/LIVE/HALTED)
- Kill switch state
- Open positions count
- Today's P&L
- Last scan time
- Data freshness
- Broker connection status

## Commands
```bash
# Check kill switch
test -f state/KILL_SWITCH && echo "KILL SWITCH ACTIVE" || echo "Kill switch: OFF"

# Check positions (if broker connected)
python -c "
from execution.broker_alpaca import BrokerAlpaca
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
broker = BrokerAlpaca()
positions = broker.get_positions()
print(f'Open positions: {len(positions)}')
for p in positions:
    print(f'  {p[\"symbol\"]}: {p[\"qty\"]} @ {p[\"avg_entry_price\"]}')
"

# Check last scan
cat state/last_scan.json 2>/dev/null || echo "No scan recorded"

# Health endpoint (if running)
curl -s http://localhost:8000/health 2>/dev/null || echo "Health endpoint not running"
```

## Interpretation
- Kill switch ON = No new orders submitted
- No positions = Flat (all cash)
- Stale scan = Data may need refresh
