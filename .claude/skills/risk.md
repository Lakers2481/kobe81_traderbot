# /risk

Check Kobe's risk limits and current exposure.

## Usage
```
/risk [--detailed]
```

## What it does
1. Shows PolicyGate budget limits
2. Calculates current exposure
3. Reports headroom remaining
4. Flags any limit breaches

## Commands
```bash
# Risk summary
python -c "
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

from risk.policy_gate import PolicyGate
from execution.broker_alpaca import BrokerAlpaca

gate = PolicyGate()
broker = BrokerAlpaca()

# Get account info
account = broker.get_account()
equity = float(account['equity'])
positions = broker.get_positions()

# Calculate exposure
total_exposure = sum(abs(float(p['qty']) * float(p['current_price'])) for p in positions)
position_count = len(positions)

# PolicyGate limits
MAX_ORDER = 75.0
MAX_DAILY = 1000.0

print('=== RISK LIMITS ===')
print(f'Max per order: \${MAX_ORDER:.2f}')
print(f'Max daily: \${MAX_DAILY:.2f}')
print()
print('=== CURRENT EXPOSURE ===')
print(f'Account equity: \${equity:,.2f}')
print(f'Total exposure: \${total_exposure:,.2f} ({100*total_exposure/equity:.1f}% of equity)')
print(f'Open positions: {position_count}')
print()

# Check daily usage (from logs)
from datetime import date
import json
from pathlib import Path

today = date.today().isoformat()
daily_spent = 0.0
trades_file = Path('logs/trades.jsonl')
if trades_file.exists():
    with open(trades_file) as f:
        for line in f:
            trade = json.loads(line)
            if trade.get('entry_time', '').startswith(today):
                daily_spent += abs(float(trade.get('entry_value', 0)))

daily_remaining = MAX_DAILY - daily_spent
print(f'Daily budget used: \${daily_spent:,.2f}')
print(f'Daily remaining: \${daily_remaining:,.2f}')
print()

if daily_remaining <= 0:
    print('âš ï¸  DAILY LIMIT REACHED - No new entries allowed')
elif daily_remaining < 150:
    print('âš ï¸  LOW DAILY BUDGET - Approaching limit')
else:
    print('âœ… Risk limits OK')
"
```

## Risk Limits (PolicyGate)
| Limit | Value | Scope |
|-------|-------|-------|
| Max per order | $75 | Per trade |
| Max daily | $1,000 | All trades today |
| Max position | 5% equity | Per symbol |
| Max total | 50% equity | All positions |

## Breach Actions
- Order size > $75: Rejected
- Daily > $1,000: No new entries
- Exposure > limits: REDUCE_ONLY mode


