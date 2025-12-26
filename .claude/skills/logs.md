# /logs

View Kobe's recent event logs and system activity.

## Usage
```
/logs [--type events|trades|errors] [--tail N] [--follow]
```

## What it does
1. Reads structured JSON logs
2. Filters by event type
3. Formats for readability
4. Can follow in real-time

## Commands
```bash
# Recent events (last 20)
tail -20 logs/events.jsonl | python -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        ts = e.get('timestamp', '')[:19]
        etype = e.get('event_type', 'UNKNOWN')
        msg = e.get('message', e.get('data', ''))
        print(f'{ts} [{etype}] {msg}')
    except: pass
"

# Recent trades only
grep -i 'trade\|fill\|order' logs/events.jsonl | tail -10 | python -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        print(json.dumps(e, indent=2))
    except: pass
"

# Errors only
grep -i 'error\|fail\|exception' logs/events.jsonl | tail -10

# Follow logs in real-time (Ctrl+C to stop)
tail -f logs/events.jsonl | python -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        ts = e.get('timestamp', '')[:19]
        etype = e.get('event_type', 'UNKNOWN')
        symbol = e.get('data', {}).get('symbol', '')
        print(f'{ts} [{etype}] {symbol}')
    except: pass
"

# System health events
grep -E 'HEALTH|PREFLIGHT|HEARTBEAT' logs/events.jsonl | tail -10
```

## Log Locations
- `logs/events.jsonl` - All structured events
- `logs/trades.jsonl` - Trade executions only
- `logs/errors.log` - Error messages
- `state/hash_chain.jsonl` - Audit chain

## Log Retention
- Events: 30 days
- Trades: Indefinite (audit requirement)
- Errors: 7 days
