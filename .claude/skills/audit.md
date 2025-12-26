# /audit

Verify Kobe's hash chain integrity (tamper detection).

## Usage
```
/audit [--full] [--date YYYY-MM-DD]
```

## What it does
1. Reads hash chain from `state/hash_chain.jsonl`
2. Verifies each block's hash
3. Checks chain continuity (prev_hash links)
4. Reports any tampering detected

## Commands
```bash
# Quick audit (last 100 entries)
python scripts/verify_hash_chain.py

# Full chain verification
python -c "
from core.hash_chain import HashChain
import json

chain = HashChain()
result = chain.verify()

if result['valid']:
    print('✅ HASH CHAIN VERIFIED')
    print(f'   Blocks: {result[\"block_count\"]}')
    print(f'   First: {result[\"first_timestamp\"]}')
    print(f'   Last: {result[\"last_timestamp\"]}')
else:
    print('❌ CHAIN INTEGRITY FAILURE')
    print(f'   Error: {result[\"error\"]}')
    print(f'   Block: {result.get(\"failed_block\", \"unknown\")}')
    print()
    print('⚠️  CRITICAL: Possible tampering detected!')
    print('   1. Stop all trading immediately (/kill)')
    print('   2. Investigate the discrepancy')
    print('   3. Restore from backup if needed')
"

# View recent chain entries
tail -5 state/hash_chain.jsonl | python -c "
import sys, json
for line in sys.stdin:
    entry = json.loads(line)
    print(f'{entry[\"timestamp\"]} | {entry[\"event_type\"]} | {entry[\"hash\"][:16]}...')
"
```

## Chain Entry Types
- `ORDER_SUBMITTED`: New order sent to broker
- `ORDER_FILLED`: Order execution confirmed
- `SIGNAL_GENERATED`: Strategy produced signal
- `KILL_SWITCH_*`: Kill switch events
- `CONFIG_CHANGE`: Configuration modified
- `RECONCILIATION`: Position sync events

## On Failure
If chain verification fails:
1. **Activate kill switch immediately**
2. Compare with backup chain
3. Identify tampered entries
4. Investigate root cause
5. Restore from last known good state
