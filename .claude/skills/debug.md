# /debug

Toggle debug mode and verbose logging.

## Usage
```
/debug [--on|--off|--status]
```

## What it does
1. Enables/disables verbose logging
2. Shows detailed execution traces
3. Exposes internal state
4. Helps diagnose issues

## Commands
```bash
# Check debug status
python -c "
import os
from pathlib import Path

debug_flag = Path('state/DEBUG_MODE')

print('=== DEBUG STATUS ===')
if debug_flag.exists():
    print('Debug mode: ON')
    print(f'Enabled at: {debug_flag.read_text()}')
else:
    print('Debug mode: OFF')
"

# Enable debug mode
python -c "
from pathlib import Path
from datetime import datetime

debug_flag = Path('state/DEBUG_MODE')
debug_flag.parent.mkdir(parents=True, exist_ok=True)
debug_flag.write_text(datetime.now().isoformat())
print('✅ Debug mode ENABLED')
print('Verbose logging will appear in logs/debug.log')
"

# Disable debug mode
python -c "
from pathlib import Path

debug_flag = Path('state/DEBUG_MODE')
if debug_flag.exists():
    debug_flag.unlink()
    print('✅ Debug mode DISABLED')
else:
    print('Debug mode was already off')
"

# View debug logs
tail -100 logs/debug.log 2>/dev/null || echo "No debug log found"

# Dump internal state (for debugging)
python -c "
import sys
sys.path.insert(0, '.')
import json
from pathlib import Path
from datetime import datetime

print('=== INTERNAL STATE DUMP ===')
print(f'Timestamp: {datetime.now().isoformat()}')
print()

# Runner state
print('--- Runner State ---')
runner_file = Path('state/runner_last.json')
if runner_file.exists():
    print(json.dumps(json.loads(runner_file.read_text()), indent=2))
else:
    print('(none)')
print()

# Policy gate state
print('--- Risk Limits ---')
from risk.policy_gate import PolicyGate, RiskLimits
pg = PolicyGate()
print(f'Max per order: \${pg.limits.max_notional_per_order}')
print(f'Max daily: \${pg.limits.max_daily_notional}')
print(f'Daily used: \${pg._daily_notional}')
print()

# Idempotency stats
print('--- Idempotency Store ---')
idem = Path('state/idempotency.sqlite')
if idem.exists():
    import sqlite3
    con = sqlite3.connect(idem)
    count = con.execute('SELECT COUNT(*) FROM idempotency').fetchone()[0]
    last = con.execute('SELECT created_at FROM idempotency ORDER BY created_at DESC LIMIT 1').fetchone()
    con.close()
    print(f'Total entries: {count}')
    print(f'Last entry: {last[0] if last else \"(none)\"}')
else:
    print('(empty)')
print()

# Hash chain stats
print('--- Hash Chain ---')
chain = Path('state/hash_chain.jsonl')
if chain.exists():
    lines = chain.read_text().splitlines()
    print(f'Total blocks: {len(lines)}')
    if lines:
        last_block = json.loads(lines[-1])
        print(f'Last hash: {last_block.get(\"this_hash\", \"\")[:16]}...')
else:
    print('(empty)')
"

# Test specific component
python -c "
import sys
sys.path.insert(0, '.')

print('=== COMPONENT TEST ===')

# Test strategy signal generation
print('Testing ConnorsRSI2Strategy...')
import pandas as pd
import numpy as np
from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy

dates = pd.date_range(end='2024-01-01', periods=250, freq='B')
df = pd.DataFrame({
    'timestamp': dates,
    'symbol': 'DEBUG',
    'open': 100 + np.random.randn(250).cumsum() * 0.5,
    'high': 101 + np.random.randn(250).cumsum() * 0.5,
    'low': 99 + np.random.randn(250).cumsum() * 0.5,
    'close': 100 + np.random.randn(250).cumsum() * 0.5,
    'volume': np.random.randint(1e6, 5e6, 250),
})
df['high'] = df[['open', 'high', 'close']].max(axis=1) + 0.5
df['low'] = df[['open', 'low', 'close']].min(axis=1) - 0.5

strat = ConnorsRSI2Strategy()
df_with_ind = strat._compute_indicators(df)

print(f'Last RSI2: {df_with_ind[\"rsi2\"].iloc[-1]:.2f}')
print(f'Last SMA200: {df_with_ind[\"sma200\"].iloc[-1]:.2f}')
print(f'Last ATR14: {df_with_ind[\"atr14\"].iloc[-1]:.2f}')

sigs = strat.generate_signals(df)
print(f'Signals generated: {len(sigs)}')

print('✅ Strategy component OK')
"
```

## Debug Outputs
| Location | Contents |
|----------|----------|
| logs/debug.log | Verbose execution trace |
| logs/events.jsonl | Structured events |
| state/DEBUG_MODE | Debug flag file |

## Debug Levels
| Level | Output |
|-------|--------|
| OFF | Normal logging only |
| ON | + Function calls, state changes |
| TRACE | + Every decision, full payloads |

## Common Debug Tasks
| Task | Command |
|------|---------|
| Why no signals? | Check RSI values in state dump |
| Why order rejected? | Check PolicyGate limits |
| Why duplicate? | Check idempotency store |
| Why wrong data? | Validate data cache |
