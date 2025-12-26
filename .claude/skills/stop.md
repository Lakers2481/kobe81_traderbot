# /stop

Gracefully stop the Kobe trading system.

## Usage
```
/stop [--force] [--wait]
```

## What it does
1. Signals runner to stop accepting new trades
2. Waits for pending operations to complete
3. Saves state
4. Shuts down cleanly

## Commands
```bash
# Graceful stop
python -c "
import os
import signal
from pathlib import Path

pid_file = Path('state/runner.pid')

if not pid_file.exists():
    print('No runner PID file found - runner may not be running')
    exit(0)

pid = int(pid_file.read_text().strip())
print(f'Stopping runner (PID {pid})...')

try:
    # Send SIGTERM for graceful shutdown
    os.kill(pid, signal.SIGTERM)
    print('SIGTERM sent')

    # Wait a moment
    import time
    time.sleep(2)

    # Check if still running
    try:
        os.kill(pid, 0)  # Signal 0 checks existence
        print('Runner still running, waiting...')
        time.sleep(5)
        os.kill(pid, 0)
        print('⚠️ Runner not responding. Use /stop --force')
    except ProcessLookupError:
        print('✅ Runner stopped successfully')
        pid_file.unlink()

except ProcessLookupError:
    print('Runner was not running (stale PID)')
    pid_file.unlink()
except Exception as e:
    print(f'Error: {e}')
"

# Force stop (if graceful fails)
python -c "
import os
import signal
from pathlib import Path

pid_file = Path('state/runner.pid')

if not pid_file.exists():
    print('No runner PID file')
    exit(0)

pid = int(pid_file.read_text().strip())
print(f'Force stopping runner (PID {pid})...')

try:
    os.kill(pid, signal.SIGKILL)  # Force kill
    print('SIGKILL sent')
    pid_file.unlink()
    print('✅ Runner force stopped')
except ProcessLookupError:
    print('Runner was not running')
    pid_file.unlink()
except Exception as e:
    print(f'Error: {e}')
"

# Stop and verify
echo "=== STOP SEQUENCE ==="

# 1. Check current status
echo "Current status:"
python -c "
from pathlib import Path
import psutil

pid_file = Path('state/runner.pid')
if pid_file.exists():
    pid = int(pid_file.read_text())
    try:
        p = psutil.Process(pid)
        print(f'  Runner: RUNNING (PID {pid})')
    except:
        print(f'  Runner: STOPPED (stale PID {pid})')
else:
    print('  Runner: STOPPED (no PID file)')
"

# 2. Stop
echo "Sending stop signal..."
python -c "
import os
import signal
from pathlib import Path

pid_file = Path('state/runner.pid')
if pid_file.exists():
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, signal.SIGTERM)
    except:
        pass
"

# 3. Wait and verify
sleep 3
echo "Verifying..."
python -c "
from pathlib import Path

pid_file = Path('state/runner.pid')
if pid_file.exists():
    import os
    pid = int(pid_file.read_text())
    try:
        os.kill(pid, 0)
        print('⚠️ Runner still running')
    except:
        print('✅ Runner stopped')
        pid_file.unlink()
else:
    print('✅ Runner stopped')
"
```

## Stop vs Kill
| Command | Action | When to Use |
|---------|--------|-------------|
| `/stop` | Graceful shutdown | Normal stop |
| `/stop --force` | SIGKILL | Unresponsive runner |
| `/kill` | Emergency halt | Mid-trade emergency |

## Graceful Shutdown Behavior
1. Stop accepting new signals
2. Wait for pending orders to fill/timeout
3. Log final state
4. Save runner_last.json
5. Remove PID file
6. Exit cleanly

## After Stopping
- Positions remain open at broker
- Use `/positions` to check
- Use `/reconcile` to verify state
- Runner state preserved in `state/`

## Restart After Stop
```bash
/start --mode paper
```
