# /orders

View Kobe's order history and pending orders.

## Usage
```
/orders [--status open|closed|all] [--limit N]
```

## What it does
1. Fetches orders from Alpaca
2. Shows order status, fill price, timestamps
3. Filters by status if specified

## Commands
```bash
# Show open orders
python -c "
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

from execution.broker_alpaca import BrokerAlpaca
broker = BrokerAlpaca()
orders = broker.get_orders(status='open')

if not orders:
    print('No open orders')
else:
    print(f'{'ID':<12} {'Symbol':<8} {'Side':<6} {'Qty':>6} {'Type':<8} {'Status':<10}')
    print('-' * 60)
    for o in orders:
        print(f'{o[\"id\"][:12]:<12} {o[\"symbol\"]:<8} {o[\"side\"]:<6} {o[\"qty\"]:>6} {o[\"type\"]:<8} {o[\"status\"]:<10}')
"

# Show recent filled orders
python -c "
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

from execution.broker_alpaca import BrokerAlpaca
broker = BrokerAlpaca()
orders = broker.get_orders(status='closed', limit=20)

filled = [o for o in orders if o['status'] == 'filled']
if not filled:
    print('No recent filled orders')
else:
    print(f'{'Time':<20} {'Symbol':<8} {'Side':<6} {'Qty':>6} {'Fill Price':>12}')
    print('-' * 60)
    for o in filled[:10]:
        time = o.get('filled_at', '')[:19]
        print(f'{time:<20} {o[\"symbol\"]:<8} {o[\"side\"]:<6} {o[\"qty\"]:>6} {float(o.get(\"filled_avg_price\", 0)):>12.2f}')
"
```

## Order Statuses
- **new**: Order accepted, not yet processed
- **accepted**: Order routed to exchange
- **filled**: Fully executed
- **partially_filled**: Partially executed
- **canceled**: Canceled before fill
- **rejected**: Rejected by broker/exchange

## Notes
- Kobe uses IOC LIMIT orders only
- Orders are logged to hash chain for audit
- Use `/audit` to verify order integrity
