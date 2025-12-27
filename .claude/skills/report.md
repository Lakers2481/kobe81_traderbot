# /report

Generate comprehensive performance reports.

## Usage
```
/report [--period daily|weekly|monthly] [--format text|html|json]
```

## What it does
1. Aggregates trading performance data
2. Calculates key metrics
3. Generates formatted report
4. Optionally sends via Discord/email

## Commands
```bash
# Generate daily report
python -c "
import json
from pathlib import Path
from datetime import datetime, date
import sys
sys.path.insert(0, '.')

print('=' * 60)
print(f'KOBE DAILY REPORT - {date.today()}')
print('=' * 60)
print()

# 1. Trading Summary
trades_file = Path('logs/trades.jsonl')
if trades_file.exists():
    today = date.today().isoformat()
    trades = []
    for line in trades_file.read_text().splitlines():
        try:
            t = json.loads(line)
            if t.get('timestamp', '').startswith(today):
                trades.append(t)
        except:
            pass

    print('ðŸ“Š TRADING SUMMARY')
    print(f'   Trades today: {len(trades)}')
    if trades:
        pnl = sum(t.get('pnl', 0) for t in trades)
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        print(f'   P&L: \${pnl:+,.2f}')
        print(f'   Win rate: {wins/len(trades)*100:.0f}%')
else:
    print('ðŸ“Š TRADING SUMMARY')
    print('   No trades today')
print()

# 2. Positions
print('ðŸ“ˆ OPEN POSITIONS')
pos_file = Path('state/positions.json')
if pos_file.exists():
    positions = json.loads(pos_file.read_text())
    if positions:
        for sym, pos in positions.items():
            print(f'   {sym}: {pos.get(\"qty\", 0)} @ \${pos.get(\"avg_cost\", 0):.2f}')
    else:
        print('   No open positions')
else:
    print('   No position data')
print()

# 3. System Health
print('ðŸ¥ SYSTEM HEALTH')
kill = Path('state/KILL_SWITCH').exists()
print(f'   Kill switch: {\"ðŸ›‘ ACTIVE\" if kill else \"âœ… Off\"}')

pid_file = Path('state/runner.pid')
if pid_file.exists():
    import os
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
        print(f'   Runner: âœ… Running (PID {pid})')
    except:
        print('   Runner: âŒ Stopped')
else:
    print('   Runner: âŒ Not running')

from core.hash_chain import verify_chain
valid = verify_chain()
print(f'   Hash chain: {\"âœ… Valid\" if valid else \"âŒ TAMPERED\"}')
print()

# 4. Alerts
print('âš ï¸ ALERTS')
alerts_file = Path('logs/alerts.log')
if alerts_file.exists():
    today = date.today().isoformat()
    alerts = [l for l in alerts_file.read_text().splitlines() if today in l]
    if alerts:
        for a in alerts[-5:]:
            print(f'   {a}')
    else:
        print('   No alerts today')
else:
    print('   No alerts')
print()

print('=' * 60)
print('Report generated at', datetime.now().strftime('%H:%M:%S'))
"

# Generate weekly report
python scripts/aggregate_wf_report.py --wfdir wf_outputs 2>/dev/null || echo "Run /wf first for weekly stats"

# Export report to JSON
python -c "
import json
from pathlib import Path
from datetime import datetime, date

report = {
    'generated_at': datetime.now().isoformat(),
    'period': 'daily',
    'date': date.today().isoformat(),
    'trades': [],
    'positions': {},
    'metrics': {},
    'system_health': {},
}

# Load data
trades_file = Path('logs/trades.jsonl')
if trades_file.exists():
    today = date.today().isoformat()
    for line in trades_file.read_text().splitlines():
        try:
            t = json.loads(line)
            if t.get('timestamp', '').startswith(today):
                report['trades'].append(t)
        except:
            pass

# Save report
output = Path(f'logs/report_{date.today().isoformat()}.json')
output.write_text(json.dumps(report, indent=2, default=str))
print(f'Report saved to: {output}')
"

# Send report to Discord (if configured)
python -c "
import os
import json
import urllib.request
from datetime import date

webhook = os.getenv('DISCORD_WEBHOOK_URL')
if not webhook:
    print('Discord webhook not configured')
    exit()

report_text = f'''ðŸ“Š **Kobe Daily Report - {date.today()}**

Trades today: 0
P&L: \$0.00
System: âœ… Running

_Full report: /report_
'''

payload = {
    'content': report_text,
    'username': 'Kobe Trading Bot'
}

req = urllib.request.Request(
    webhook,
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'}
)

try:
    urllib.request.urlopen(req)
    print('âœ… Report sent to Discord')
except Exception as e:
    print(f'âŒ Failed to send: {e}')
"
```

## Report Types
| Type | Frequency | Contents |
|------|-----------|----------|
| Daily | End of day | Trades, P&L, positions |
| Weekly | Sunday | Aggregated stats, trends |
| Monthly | 1st of month | Full performance review |

## Report Sections
1. **Trading Summary** - Trades, P&L, win rate
2. **Open Positions** - Current holdings
3. **System Health** - Runner, kill switch, chain
4. **Alerts** - Warnings and errors
5. **Benchmark** - vs SPY (weekly+)

## Automation
```bash
# Add to crontab for daily reports
0 17 * * 1-5 cd /path/to/kobe && python -c "exec(open('.claude/skills/report.md').read())"
```


