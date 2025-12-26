# /metrics

Show Kobe's performance metrics and system stats.

## Usage
```
/metrics [--period week|month|year|all]
```

## What it does
1. Calculates trading performance metrics
2. Shows system health counters
3. Displays Prometheus-format gauges
4. Tracks latency histograms

## Commands
```bash
# Performance metrics
python -c "
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

trades_file = Path('logs/trades.jsonl')
if not trades_file.exists():
    print('No trades recorded yet')
    exit()

trades = []
with open(trades_file) as f:
    for line in f:
        trades.append(json.loads(line))

if not trades:
    print('No trades to analyze')
    exit()

# Calculate metrics
wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
losses = sum(1 for t in trades if t.get('pnl', 0) <= 0)
total_pnl = sum(t.get('pnl', 0) for t in trades)
gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))

win_rate = 100 * wins / len(trades) if trades else 0
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
avg_win = gross_profit / wins if wins > 0 else 0
avg_loss = gross_loss / losses if losses > 0 else 0
expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

print('=== PERFORMANCE METRICS ===')
print(f'Total trades: {len(trades)}')
print(f'Win rate: {win_rate:.1f}%')
print(f'Profit factor: {profit_factor:.2f}')
print(f'Total P&L: \${total_pnl:+,.2f}')
print(f'Avg win: \${avg_win:.2f}')
print(f'Avg loss: \${avg_loss:.2f}')
print(f'Expectancy: \${expectancy:.2f}/trade')
"

# System metrics (if health endpoint running)
curl -s http://localhost:8000/metrics 2>/dev/null || echo "Health endpoint not running"

# Latency stats
python -c "
import json
from pathlib import Path

events_file = Path('logs/events.jsonl')
if not events_file.exists():
    print('No events recorded')
    exit()

latencies = []
with open(events_file) as f:
    for line in f:
        try:
            e = json.loads(line)
            if 'latency_ms' in e.get('data', {}):
                latencies.append(e['data']['latency_ms'])
        except: pass

if latencies:
    latencies.sort()
    p50 = latencies[len(latencies)//2]
    p95 = latencies[int(len(latencies)*0.95)]
    p99 = latencies[int(len(latencies)*0.99)]
    print('=== LATENCY (ms) ===')
    print(f'p50: {p50:.1f}')
    print(f'p95: {p95:.1f}')
    print(f'p99: {p99:.1f}')
"
```

## Key Metrics
| Metric | Target | Description |
|--------|--------|-------------|
| Win Rate | >55% | % profitable trades |
| Profit Factor | >1.5 | Gross profit / loss |
| Sharpe Ratio | >1.0 | Risk-adjusted return |
| Max Drawdown | <15% | Peak to trough |
| Avg Trade | >$0 | Expectancy per trade |
