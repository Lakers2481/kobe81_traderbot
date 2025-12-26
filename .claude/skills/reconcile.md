# /reconcile

Reconcile Kobe's internal state with Alpaca broker positions.

## Usage
```
/reconcile [--fix]
```

## What it does
1. Fetches positions from Alpaca (source of truth)
2. Compares to internal order state
3. Reports any mismatches
4. Optionally fixes discrepancies

## Commands
```bash
# Run reconciliation check
python scripts/reconcile_alpaca.py

# Detailed reconciliation with fix option
python -c "
import os
from dotenv import load_dotenv
load_dotenv('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

from execution.broker_alpaca import BrokerAlpaca
from oms.order_state import OrderStateManager
import json

broker = BrokerAlpaca()
state = OrderStateManager()

# Get broker positions
broker_positions = {p['symbol']: float(p['qty']) for p in broker.get_positions()}

# Get internal positions
internal_positions = state.get_open_positions()

# Compare
print('=== RECONCILIATION REPORT ===')
print()

all_symbols = set(broker_positions.keys()) | set(internal_positions.keys())
mismatches = []

for symbol in sorted(all_symbols):
    broker_qty = broker_positions.get(symbol, 0)
    internal_qty = internal_positions.get(symbol, 0)

    if broker_qty != internal_qty:
        mismatches.append({
            'symbol': symbol,
            'broker': broker_qty,
            'internal': internal_qty,
            'diff': broker_qty - internal_qty
        })
        print(f'MISMATCH: {symbol}')
        print(f'  Broker: {broker_qty}')
        print(f'  Internal: {internal_qty}')
        print(f'  Diff: {broker_qty - internal_qty:+.0f}')
        print()

if not mismatches:
    print('✅ All positions reconciled - no mismatches')
else:
    print(f'⚠️  {len(mismatches)} mismatches found')
    print('Run with --fix to sync internal state to broker')
"
```

## Mismatch Causes
- Partial fills not recorded
- Manual broker intervention
- System crash during order
- Network timeout after submit

## Resolution
1. Broker is always source of truth
2. `--fix` updates internal state to match
3. All fixes logged to audit chain
4. Review causes to prevent recurrence
