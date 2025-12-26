# /positions

Show Kobe's current open positions from Alpaca.

## Usage
```
/positions [--detailed]
```

## What it does
1. Connects to Alpaca broker
2. Fetches all open positions
3. Shows P&L per position
4. Calculates total exposure

## Commands
```bash
# Quick positions summary
python -c "
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

from execution.broker_alpaca import BrokerAlpaca
broker = BrokerAlpaca()
positions = broker.get_positions()

if not positions:
    print('No open positions (flat)')
else:
    total_value = 0
    total_pnl = 0
    print(f'{'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'P&L%':>8}')
    print('-' * 60)
    for p in positions:
        symbol = p['symbol']
        qty = float(p['qty'])
        entry = float(p['avg_entry_price'])
        current = float(p['current_price'])
        pnl = float(p['unrealized_pl'])
        pnl_pct = float(p['unrealized_plpc']) * 100
        value = qty * current
        total_value += value
        total_pnl += pnl
        print(f'{symbol:<8} {qty:>6.0f} {entry:>10.2f} {current:>10.2f} {pnl:>+10.2f} {pnl_pct:>+7.2f}%')
    print('-' * 60)
    print(f'Total Value: \${total_value:,.2f}')
    print(f'Total P&L: \${total_pnl:+,.2f}')
"
```

## Output Columns
- **Symbol**: Stock ticker
- **Qty**: Number of shares
- **Entry**: Average entry price
- **Current**: Current market price
- **P&L**: Unrealized profit/loss ($)
- **P&L%**: Unrealized profit/loss (%)

## Notes
- Uses Alpaca paper/live based on ALPACA_BASE_URL
- Positions are from broker (source of truth)
- For reconciliation, use `/reconcile`
