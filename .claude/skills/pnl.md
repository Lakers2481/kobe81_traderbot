# /pnl

Show Kobe's P&L summary - today, week, month, all-time.

## Usage
```
/pnl [--period today|week|month|all]
```

## What it does
1. Reads trade history from logs
2. Calculates realized + unrealized P&L
3. Shows win rate and profit factor
4. Displays by period

## Commands
```bash
# Today's P&L
python -c "
import json
from datetime import datetime, date
from pathlib import Path

# Read trade log
trades_file = Path('logs/trades.jsonl')
if not trades_file.exists():
    print('No trades recorded yet')
    exit()

today = date.today().isoformat()
today_trades = []
with open(trades_file) as f:
    for line in f:
        trade = json.loads(line)
        if trade.get('exit_time', '').startswith(today):
            today_trades.append(trade)

if not today_trades:
    print(f'No closed trades today ({today})')
else:
    wins = sum(1 for t in today_trades if t.get('pnl', 0) > 0)
    total = len(today_trades)
    total_pnl = sum(t.get('pnl', 0) for t in today_trades)
    gross_profit = sum(t.get('pnl', 0) for t in today_trades if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t.get('pnl', 0) for t in today_trades if t.get('pnl', 0) < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f'=== TODAY ({today}) ===')
    print(f'Trades: {total}')
    print(f'Win Rate: {wins}/{total} ({100*wins/total:.1f}%)')
    print(f'P&L: \${total_pnl:+,.2f}')
    print(f'Profit Factor: {pf:.2f}')
"

# Full P&L summary
python -c "
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

from execution.broker_alpaca import BrokerAlpaca
broker = BrokerAlpaca()
account = broker.get_account()

print('=== ACCOUNT SUMMARY ===')
print(f'Equity: \${float(account[\"equity\"]):,.2f}')
print(f'Cash: \${float(account[\"cash\"]):,.2f}')
print(f'Buying Power: \${float(account[\"buying_power\"]):,.2f}')
print(f'Day P&L: \${float(account.get(\"daily_change\", 0)):+,.2f}')
"
```

## Metrics Explained
- **Win Rate**: % of trades that were profitable
- **Profit Factor**: Gross profit / Gross loss (>1.5 is good)
- **Equity**: Total account value
- **Day P&L**: Change since market open


