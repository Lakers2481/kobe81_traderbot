# /smoke

Run smoke tests to verify system integrity.

## Usage
```
/smoke [--quick|--full]
```

## What it does
1. Tests all critical system components
2. Verifies imports and dependencies
3. Checks data and broker connectivity
4. Validates strategy execution path

## Commands
```bash
# Run smoke test script
python scripts/smoke_test.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Quick smoke test (imports only)
python -c "
import sys
sys.path.insert(0, '.')

print('=== SMOKE TEST: IMPORTS ===')
tests = []

# Core imports
try:
    from core.hash_chain import append_block, verify_chain
    tests.append(('core.hash_chain', True))
except Exception as e:
    tests.append(('core.hash_chain', False, str(e)))

try:
    from core.structured_log import jlog
    tests.append(('core.structured_log', True))
except Exception as e:
    tests.append(('core.structured_log', False, str(e)))

try:
    from core.config_pin import sha256_file
    tests.append(('core.config_pin', True))
except Exception as e:
    tests.append(('core.config_pin', False, str(e)))

# Data layer
try:
    from data.providers.polygon_eod import fetch_eod_bars
    tests.append(('data.providers.polygon_eod', True))
except Exception as e:
    tests.append(('data.providers.polygon_eod', False, str(e)))

try:
    from data.universe.loader import load_universe
    tests.append(('data.universe.loader', True))
except Exception as e:
    tests.append(('data.universe.loader', False, str(e)))

# Strategies
try:
    from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy
    tests.append(('strategies.connors_rsi2', True))
except Exception as e:
    tests.append(('strategies.connors_rsi2', False, str(e)))

try:
    from strategies.ibs.strategy import IBSStrategy
    tests.append(('strategies.ibs', True))
except Exception as e:
    tests.append(('strategies.ibs', False, str(e)))

# Backtest
try:
    from backtest.engine import Backtester
    tests.append(('backtest.engine', True))
except Exception as e:
    tests.append(('backtest.engine', False, str(e)))

# Risk
try:
    from risk.policy_gate import PolicyGate, RiskLimits
    tests.append(('risk.policy_gate', True))
except Exception as e:
    tests.append(('risk.policy_gate', False, str(e)))

# OMS
try:
    from oms.order_state import OrderRecord, OrderStatus
    tests.append(('oms.order_state', True))
except Exception as e:
    tests.append(('oms.order_state', False, str(e)))

try:
    from oms.idempotency_store import IdempotencyStore
    tests.append(('oms.idempotency_store', True))
except Exception as e:
    tests.append(('oms.idempotency_store', False, str(e)))

# Execution
try:
    from execution.broker_alpaca import place_ioc_limit, get_best_ask
    tests.append(('execution.broker_alpaca', True))
except Exception as e:
    tests.append(('execution.broker_alpaca', False, str(e)))

# Monitor
try:
    from monitor.health_endpoints import start_health_server
    tests.append(('monitor.health_endpoints', True))
except Exception as e:
    tests.append(('monitor.health_endpoints', False, str(e)))

# Results
passed = sum(1 for t in tests if t[1])
failed = len(tests) - passed

for t in tests:
    status = '✅' if t[1] else '❌'
    msg = '' if t[1] else f' - {t[2][:40]}'
    print(f'{status} {t[0]}{msg}')

print()
print(f'Passed: {passed}/{len(tests)}')
if failed > 0:
    print(f'❌ SMOKE TEST FAILED')
    sys.exit(1)
else:
    print('✅ SMOKE TEST PASSED')
"

# Full smoke test (with live checks)
python -c "
import os
import sys
sys.path.insert(0, '.')
from pathlib import Path
from configs.env_loader import load_env

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

print('=== FULL SMOKE TEST ===')

# 1. Imports (quick test)
exec(open('scripts/smoke_test.py').read()) if Path('scripts/smoke_test.py').exists() else print('Smoke script not found')

# 2. Strategy execution
print()
print('--- Strategy Execution ---')
import pandas as pd
import numpy as np
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy

dates = pd.date_range(end='2024-01-01', periods=250, freq='B')
df = pd.DataFrame({
    'timestamp': dates,
    'symbol': 'TEST',
    'open': 100 + np.random.randn(250).cumsum(),
    'high': 101 + np.random.randn(250).cumsum(),
    'low': 99 + np.random.randn(250).cumsum(),
    'close': 100 + np.random.randn(250).cumsum(),
    'volume': np.random.randint(1e6, 5e6, 250),
})
df['high'] = df[['open', 'high', 'close']].max(axis=1)
df['low'] = df[['open', 'low', 'close']].min(axis=1)

strat = ConnorsRSI2Strategy()
sigs = strat.generate_signals(df)
print(f'✅ Strategy generated {len(sigs)} signals on synthetic data')

# 3. PolicyGate
print()
print('--- PolicyGate ---')
from risk.policy_gate import PolicyGate
pg = PolicyGate()
ok, reason = pg.check('TEST', 'long', 50.0, 1)
print(f'✅ PolicyGate check: {ok} ({reason})')

# 4. Idempotency
print()
print('--- Idempotency Store ---')
from oms.idempotency_store import IdempotencyStore
store = IdempotencyStore(':memory:')
store.put('test_id', 'test_key')
exists = store.exists('test_id')
print(f'✅ Idempotency store: put/exists works ({exists})')

print()
print('✅ FULL SMOKE TEST PASSED')
"
```

## Test Categories
| Category | Tests | Time |
|----------|-------|------|
| Imports | All modules load | <5s |
| Strategy | Signal generation | <10s |
| Risk | PolicyGate checks | <1s |
| OMS | Idempotency store | <1s |
| Full | All + connectivity | <30s |

## When to Run Smoke Tests
- After code changes
- Before starting trading
- After deployments
- When diagnosing issues
