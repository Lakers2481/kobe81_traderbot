# /state

View and manage all state files.

## Usage
```
/state [--show|--clean|--backup]
```

## What it does
1. Shows all state files and their contents
2. Displays state directory structure
3. Validates state integrity
4. Cleans stale state files

## Commands
```bash
# Show all state files
python -c "
from pathlib import Path
from datetime import datetime
import json

state_dir = Path('state')
if not state_dir.exists():
    print('No state directory found')
    exit()

print('=== STATE FILES ===')
for f in sorted(state_dir.iterdir()):
    if f.is_file():
        size = f.stat().st_size
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        print(f'{f.name:<30} {size:>8} bytes  {mtime:%Y-%m-%d %H:%M}')
"

# Show state contents
python -c "
from pathlib import Path
import json

state_dir = Path('state')

# Kill switch
kill = state_dir / 'KILL_SWITCH'
if kill.exists():
    print('ðŸ›‘ KILL_SWITCH: ACTIVE')
    print(f'   Content: {kill.read_text()[:100]}')
else:
    print('âœ… KILL_SWITCH: off')
print()

# Config pin
pin = state_dir / 'config_pin.txt'
if pin.exists():
    print(f'ðŸ“Œ Config pin: {pin.read_text()[:16]}...')
else:
    print('âš ï¸ Config pin: not set')
print()

# Runner state
runner = state_dir / 'runner_last.json'
if runner.exists():
    data = json.loads(runner.read_text())
    print('ðŸƒ Runner state:')
    for k, v in data.items():
        print(f'   {k}: {v}')
else:
    print('ðŸƒ Runner state: no history')
print()

# Idempotency store
idem = state_dir / 'idempotency.sqlite'
if idem.exists():
    import sqlite3
    con = sqlite3.connect(idem)
    count = con.execute('SELECT COUNT(*) FROM idempotency').fetchone()[0]
    con.close()
    print(f'ðŸ”‘ Idempotency store: {count} entries')
else:
    print('ðŸ”‘ Idempotency store: empty')
print()

# Hash chain
chain = state_dir / 'hash_chain.jsonl'
if chain.exists():
    lines = len(chain.read_text().splitlines())
    print(f'â›“ï¸ Hash chain: {lines} blocks')
else:
    print('â›“ï¸ Hash chain: empty')
"

# Validate state integrity
python -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')

print('=== STATE INTEGRITY CHECK ===')

# Hash chain
from core.hash_chain import verify_chain
valid = verify_chain()
if valid:
    print('âœ… Hash chain: valid')
else:
    print('âŒ Hash chain: TAMPERED')

# Idempotency store
from oms.idempotency_store import IdempotencyStore
try:
    store = IdempotencyStore()
    # Just opening it validates the schema
    print('âœ… Idempotency store: valid')
except Exception as e:
    print(f'âŒ Idempotency store: {e}')

# State directory permissions
state_dir = Path('state')
if state_dir.exists():
    try:
        test_file = state_dir / '.write_test'
        test_file.write_text('test')
        test_file.unlink()
        print('âœ… State directory: writable')
    except:
        print('âŒ State directory: not writable')
"

# Clean stale state (CAREFUL!)
python -c "
from pathlib import Path
from datetime import datetime, timedelta

state_dir = Path('state')
now = datetime.now()
stale_days = 30

print('=== STALE STATE FILES ===')
print(f'(Older than {stale_days} days)')
print()

for f in state_dir.iterdir():
    if f.is_file():
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        age = (now - mtime).days
        if age > stale_days:
            print(f'{f.name}: {age} days old')
            # Uncomment to delete:
            # f.unlink()
            # print(f'  DELETED')

print()
print('To delete, uncomment the unlink() line in the script')
"
```

## State Files
| File | Purpose | Critical |
|------|---------|----------|
| KILL_SWITCH | Emergency halt flag | Yes |
| config_pin.txt | Config signature | Yes |
| hash_chain.jsonl | Audit trail | Yes |
| idempotency.sqlite | Duplicate prevention | Yes |
| runner_last.json | Scheduler state | No |
| runner.pid | Process ID | No |

## State Directory Structure
```
state/
â”œâ”€â”€ KILL_SWITCH         # Created by /kill, removed by /resume
â”œâ”€â”€ config_pin.txt      # SHA256 of settings.json
â”œâ”€â”€ hash_chain.jsonl    # Append-only audit log
â”œâ”€â”€ idempotency.sqlite  # SQLite database
â”œâ”€â”€ runner_last.json    # Last run timestamps
â””â”€â”€ runner.pid          # Running process ID
```

## Backup State
State files should be backed up:
- Before deployments
- After each trading day
- Before manual interventions


