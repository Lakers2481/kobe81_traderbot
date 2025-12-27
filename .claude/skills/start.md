# /start

Start the Kobe trading system.

## Usage
```
/start [--mode paper|live] [--once] [--cap N]
```

## What it does
1. Runs preflight checks
2. Validates environment
3. Starts the 24/7 runner
4. Confirms system is trading

## Commands
```bash
# Full start sequence (recommended)
echo "=== KOBE STARTUP SEQUENCE ==="

# 1. Run preflight
echo "Step 1: Preflight checks..."
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
if [ $? -ne 0 ]; then
    echo "âŒ Preflight failed. Aborting."
    exit 1
fi

# 2. Check kill switch
echo "Step 2: Kill switch check..."
if [ -f state/KILL_SWITCH ]; then
    echo "âŒ Kill switch is ACTIVE. Remove with /resume"
    exit 1
fi

# 3. Smoke test
echo "Step 3: Smoke test..."
python -c "
import sys
sys.path.insert(0, '.')
from strategies._rsi2.strategy import RSI2Strategy
from risk.policy_gate import PolicyGate
print('âœ… Core imports OK')
"

# 4. Start runner
echo "Step 4: Starting runner..."
nohup python scripts/runner.py \
    --mode paper \
    --universe data/universe/optionable_liquid_900.csv \
    --cap 50 \
    --scan-times 09:35,10:30,15:55 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env \
    > logs/runner.log 2>&1 &

echo $! > state/runner.pid
echo "âœ… Runner started (PID $(cat state/runner.pid))"

# 5. Verify running
sleep 2
python -c "
from pathlib import Path
import psutil
pid = int(Path('state/runner.pid').read_text())
try:
    p = psutil.Process(pid)
    print(f'âœ… Verified: Runner is running (PID {pid})')
except:
    print('âŒ Runner failed to start')
"

echo "=== STARTUP COMPLETE ==="

# Quick start (one-liner)
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --once --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Start with minimal validation
python -c "
import subprocess
import sys
from pathlib import Path

# Quick checks
if Path('state/KILL_SWITCH').exists():
    print('âŒ Kill switch active')
    sys.exit(1)

# Start
cmd = [
    sys.executable, 'scripts/runner.py',
    '--mode', 'paper',
    '--universe', 'data/universe/optionable_liquid_900.csv',
    '--cap', '50',
    '--scan-times', '09:35,10:30,15:55',
    '--dotenv', 'C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'
]
print('Starting Kobe...')
subprocess.Popen(cmd, stdout=open('logs/runner.log', 'w'), stderr=subprocess.STDOUT)
print('âœ… Kobe started')
"
```

## Startup Checklist
```
[x] Preflight checks pass
[x] Kill switch is OFF
[x] Broker connection OK
[x] Data is fresh
[x] Config is pinned
[x] Hash chain is valid
```

## Start Modes
| Mode | Command | Description |
|------|---------|-------------|
| Paper (default) | `--mode paper` | Simulated trading |
| Live | `--mode live` | Real money |
| Once | `--once` | Single run, then exit |

## After Starting
1. Check `/status` for system health
2. Monitor `/logs` for activity
3. Verify `/positions` as trades occur
4. Keep `/kill` ready for emergencies

## Troubleshooting
| Issue | Solution |
|-------|----------|
| Won't start | Check `/preflight` |
| Exits immediately | Check `logs/runner.log` |
| No trades | Verify `/signals` generation |
| Connection errors | Check `/broker` |


