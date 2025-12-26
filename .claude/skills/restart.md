# /restart

Restart the Kobe trading system cleanly.

## Usage
```
/restart [--mode paper|live] [--force]
```

## What it does
1. Gracefully stops the runner
2. Waits for clean shutdown
3. Runs preflight checks
4. Starts runner fresh

## Commands
```bash
# Full restart sequence
echo "=== KOBE RESTART ==="

# 1. Stop current runner
echo "Step 1: Stopping runner..."
python -c "
import os
import signal
import time
from pathlib import Path

pid_file = Path('state/runner.pid')
if pid_file.exists():
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f'  Sent SIGTERM to PID {pid}')
        time.sleep(3)
        try:
            os.kill(pid, 0)
            print('  Still running, sending SIGKILL...')
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        pid_file.unlink()
        print('  ✅ Runner stopped')
    except ProcessLookupError:
        print('  Runner not running')
        pid_file.unlink()
else:
    print('  No runner was running')
"

# 2. Quick preflight
echo "Step 2: Quick preflight..."
python -c "
import os
import sys
sys.path.insert(0, '.')
from pathlib import Path
from configs.env_loader import load_env

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

# Check essentials
checks = []
checks.append(('Env vars', bool(os.getenv('POLYGON_API_KEY') and os.getenv('ALPACA_API_KEY_ID'))))
checks.append(('Kill switch', not Path('state/KILL_SWITCH').exists()))

from core.hash_chain import verify_chain
checks.append(('Hash chain', verify_chain()))

all_ok = all(c[1] for c in checks)
for name, ok in checks:
    print(f'  {\"✅\" if ok else \"❌\"} {name}')

if not all_ok:
    print('  ❌ Preflight failed')
    sys.exit(1)
print('  ✅ Preflight passed')
"

# 3. Start fresh
echo "Step 3: Starting runner..."
nohup python scripts/runner.py \
    --mode paper \
    --universe data/universe/optionable_liquid_final.csv \
    --cap 50 \
    --scan-times 09:35,10:30,15:55 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env \
    > logs/runner.log 2>&1 &

echo $! > state/runner.pid
echo "  ✅ Runner started (PID $(cat state/runner.pid))"

# 4. Verify
sleep 2
echo "Step 4: Verifying..."
python -c "
from pathlib import Path
import psutil
pid = int(Path('state/runner.pid').read_text())
try:
    p = psutil.Process(pid)
    print(f'  ✅ Runner verified (PID {pid})')
except:
    print('  ❌ Runner failed to start - check logs/runner.log')
"

echo "=== RESTART COMPLETE ==="

# Quick restart (one-liner)
python -c "
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Stop
pid_file = Path('state/runner.pid')
if pid_file.exists():
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except:
        pass
    pid_file.unlink()

# Start
cmd = [
    sys.executable, 'scripts/runner.py',
    '--mode', 'paper',
    '--universe', 'data/universe/optionable_liquid_final.csv',
    '--cap', '50',
    '--scan-times', '09:35,10:30,15:55',
    '--dotenv', 'C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'
]
proc = subprocess.Popen(cmd, stdout=open('logs/runner.log', 'w'), stderr=subprocess.STDOUT)
Path('state/runner.pid').write_text(str(proc.pid))
print(f'✅ Restarted (PID {proc.pid})')
"
```

## When to Restart
| Situation | Restart? |
|-----------|----------|
| Config change | Yes |
| Code update | Yes |
| Memory leak | Yes |
| Data refresh | No (handled internally) |
| Strategy change | Yes + re-validate |

## Restart Preserves
- Hash chain history
- Idempotency store
- Trade logs
- Position state (at broker)

## Restart Resets
- In-memory caches
- Daily budget tracking
- Runner schedule state

## Force Restart
If normal restart fails:
```bash
# Kill all Python processes (CAREFUL!)
pkill -9 -f runner.py

# Clean state
rm -f state/runner.pid

# Start fresh
/start
```
