# /alerts

Manage Kobe's alert configuration and view recent alerts.

## Usage
```
/alerts [--show|--test|--config]
```

## What it does
1. Shows recent alerts triggered
2. Tests alert channels
3. Configures alert thresholds
4. Manages alert destinations

## Commands
```bash
# Show recent alerts
grep -i 'alert\|warn\|error' logs/events.jsonl | tail -20 | python -c "
import sys, json
for line in sys.stdin:
    try:
        e = json.loads(line)
        ts = e.get('timestamp', '')[:19]
        level = e.get('level', 'INFO')
        msg = e.get('message', str(e.get('data', '')))
        print(f'{ts} [{level}] {msg}')
    except: pass
"

# Alert thresholds (view current)
python -c "
thresholds = {
    'daily_loss_usd': 100,       # Alert if daily loss > $100
    'drawdown_pct': 10,          # Alert if drawdown > 10%
    'order_reject_count': 3,     # Alert after 3 rejections
    'data_stale_minutes': 5,     # Alert if data > 5 min old
    'latency_p99_ms': 1000,      # Alert if p99 latency > 1s
    'position_count': 10,        # Alert if > 10 positions
}

print('=== ALERT THRESHOLDS ===')
for key, value in thresholds.items():
    print(f'{key}: {value}')
"

# Test alert (Discord webhook if configured)
python -c "
import os
import json
import urllib.request

webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
if not webhook_url:
    print('DISCORD_WEBHOOK_URL not set')
    exit()

message = {
    'content': 'ðŸ§ª **Kobe Alert Test**\nThis is a test alert from the trading system.',
    'username': 'Kobe Trading Bot'
}

req = urllib.request.Request(
    webhook_url,
    data=json.dumps(message).encode(),
    headers={'Content-Type': 'application/json'}
)

try:
    urllib.request.urlopen(req)
    print('âœ… Test alert sent to Discord')
except Exception as e:
    print(f'âŒ Alert failed: {e}')
"
```

## Alert Types
| Alert | Trigger | Action |
|-------|---------|--------|
| Daily Loss | P&L < -$100 | Notify + consider kill |
| Drawdown | DD > 10% | Notify + REDUCE_ONLY |
| Order Rejects | 3+ in 10 min | Notify + investigate |
| Data Stale | >5 min old | Notify + halt entries |
| System Error | Any exception | Notify immediately |

## Alert Channels
- **Discord**: Webhook for instant notifications
- **Email**: Daily summary (if configured)
- **Log file**: All alerts in logs/alerts.log
- **Health endpoint**: /alerts returns recent

## Configuration
Set in environment or configs/:
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ALERT_EMAIL=your@email.com
ALERT_DAILY_LOSS_THRESHOLD=100
```


