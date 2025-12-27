# /runner

Control the 24/7 trading runner (scheduler).

## Usage
```
/runner [--start|--stop|--status|--once]
```

## What it does
1. Starts/stops the 24/7 trading scheduler
2. Manages scan schedules (9:35, 10:30, 15:55)
3. Tracks execution state
4. Shows runner status and logs

## Commands
```bash
# Start runner (paper mode)
python scripts/runner.py \
    --mode paper \
    --universe data/universe/optionable_liquid_900.csv \
    --cap 50 \
    --scan-times 09:35,10:30,15:55 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Start runner in background
nohup python scripts/runner.py \
    --mode paper \
    --universe data/universe/optionable_liquid_900.csv \
    --cap 50 \
    --scan-times 09:35,10:30,15:55 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env \
    > logs/runner.log 2>&1 &

echo $! > state/runner.pid
echo "Runner started with PID $(cat state/runner.pid)"

# Run once and exit (for testing)
python scripts/runner.py \
    --mode paper \
    --universe data/universe/optionable_liquid_900.csv \
    --cap 50 \
    --once \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Check runner status
python -c "
from pathlib import Path
import json
import psutil

print('=== RUNNER STATUS ===')

# Check PID file
pid_file = Path('state/runner.pid')
if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    try:
        proc = psutil.Process(pid)
        print(f'âœ… Runner running (PID {pid})')
        print(f'   CPU: {proc.cpu_percent()}%')
        print(f'   Memory: {proc.memory_info().rss / 1024 / 1024:.1f} MB')
        print(f'   Started: {proc.create_time()}')
    except psutil.NoSuchProcess:
        print(f'âŒ Runner not running (stale PID {pid})')
else:
    print('âŒ No runner PID file found')

# Check last run state
state_file = Path('state/runner_last.json')
if state_file.exists():
    state = json.loads(state_file.read_text())
    print()
    print('Last runs:')
    for key, date in state.items():
        print(f'  {key}: {date}')
"

# Stop runner gracefully
python -c "
from pathlib import Path
import signal
import os

pid_file = Path('state/runner.pid')
if pid_file.exists():
    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f'Sent SIGTERM to runner (PID {pid})')
        pid_file.unlink()
    except ProcessLookupError:
        print('Runner not running')
        pid_file.unlink()
else:
    print('No runner PID file found')
"

# View runner logs
tail -100 logs/runner.log 2>/dev/null || echo "No runner log found"
```

## Scan Schedule
| Time (ET) | Purpose |
|-----------|---------|
| 09:35 | Post-open scan (avoid first 5 min volatility) |
| 10:30 | Mid-morning scan |
| 15:55 | End-of-day scan (before close) |

## Runner States
| State | Description |
|-------|-------------|
| Running | Actively monitoring and executing |
| Sleeping | Between scan times |
| Stopped | Not running |
| Killed | Kill switch activated |

## Parameters
| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | paper | paper or live |
| `--universe` | required | Universe CSV file |
| `--cap` | 50 | Max symbols to scan |
| `--scan-times` | 09:35,10:30,15:55 | Comma-separated times |
| `--lookback-days` | 540 | Days of data to fetch |
| `--once` | false | Run once and exit |


