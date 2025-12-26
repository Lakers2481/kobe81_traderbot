# /broker

Check broker connection status and account details.

## Usage
```
/broker [--status|--account|--orders|--positions]
```

## What it does
1. Tests Alpaca API connectivity
2. Shows account balance and buying power
3. Displays open orders
4. Verifies API key validity

## Commands
```bash
# Full broker status
python -c "
import os
import requests
import sys
sys.path.insert(0, '.')
from config.env_loader import load_env
from pathlib import Path

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
key_id = os.getenv('ALPACA_API_KEY_ID', '')
secret = os.getenv('ALPACA_API_SECRET_KEY', '')

if not key_id or not secret:
    print('❌ Alpaca credentials not configured')
    exit(1)

headers = {
    'APCA-API-KEY-ID': key_id,
    'APCA-API-SECRET-KEY': secret,
}

print('=== BROKER STATUS ===')
print(f'Endpoint: {base}')
mode = 'PAPER' if 'paper' in base else 'LIVE'
print(f'Mode: {mode}')
print()

# Account info
try:
    r = requests.get(f'{base}/v2/account', headers=headers, timeout=5)
    if r.status_code == 200:
        acc = r.json()
        print('✅ Connection OK')
        print()
        print('Account:')
        print(f\"  ID: {acc.get('id', 'N/A')}\")
        print(f\"  Status: {acc.get('status', 'N/A')}\")
        print(f\"  Equity: \${float(acc.get('equity', 0)):,.2f}\")
        print(f\"  Cash: \${float(acc.get('cash', 0)):,.2f}\")
        print(f\"  Buying Power: \${float(acc.get('buying_power', 0)):,.2f}\")
        print(f\"  Day Trades: {acc.get('daytrade_count', 0)}\")
        print(f\"  PDT Blocked: {acc.get('pattern_day_trader', False)}\")
    else:
        print(f'❌ Account error: {r.status_code}')
        print(r.text[:200])
except Exception as e:
    print(f'❌ Connection error: {e}')
"

# Check open orders
python -c "
import os
import requests
import sys
sys.path.insert(0, '.')
from config.env_loader import load_env
from pathlib import Path

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
headers = {
    'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID', ''),
    'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY', ''),
}

r = requests.get(f'{base}/v2/orders?status=open', headers=headers, timeout=5)
if r.status_code == 200:
    orders = r.json()
    print(f'=== OPEN ORDERS: {len(orders)} ===')
    for o in orders:
        print(f\"{o['symbol']:<6} {o['side']:<4} {o['qty']} @ {o.get('limit_price', 'MKT')}\")
else:
    print(f'Error: {r.status_code}')
"

# Check positions
python -c "
import os
import requests
import sys
sys.path.insert(0, '.')
from config.env_loader import load_env
from pathlib import Path

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets').rstrip('/')
headers = {
    'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID', ''),
    'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY', ''),
}

r = requests.get(f'{base}/v2/positions', headers=headers, timeout=5)
if r.status_code == 200:
    positions = r.json()
    print(f'=== POSITIONS: {len(positions)} ===')
    if positions:
        print(f'{\"Symbol\":<6} {\"Qty\":>8} {\"Entry\":>10} {\"Current\":>10} {\"P&L\":>12}')
        for p in positions:
            pnl = float(p.get('unrealized_pl', 0))
            print(f\"{p['symbol']:<6} {p['qty']:>8} {float(p['avg_entry_price']):>10.2f} {float(p['current_price']):>10.2f} \${pnl:>+11.2f}\")
else:
    print(f'Error: {r.status_code}')
"
```

## Broker Modes
| Mode | Base URL | Use Case |
|------|----------|----------|
| Paper | paper-api.alpaca.markets | Testing |
| Live | api.alpaca.markets | Production |

## Account Status Meanings
| Status | Description |
|--------|-------------|
| ACTIVE | Trading enabled |
| ACCOUNT_UPDATED | Review needed |
| APPROVAL_PENDING | Not yet approved |
| REJECTED | Application rejected |

## API Rate Limits
- 200 requests/minute for most endpoints
- Burst of 400 allowed
- Monitor via response headers
