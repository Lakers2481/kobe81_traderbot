# /health

Control and monitor the health check server.

## Usage
```
/health [--start|--stop|--check|--port PORT]
```

## What it does
1. Starts health check HTTP server
2. Provides /readiness and /liveness endpoints
3. Enables external monitoring integration
4. Shows current health status

## Commands
```bash
# Start health server (background)
python scripts/start_health.py --port 8000 &
echo "Health server started on port 8000"

# Check if health server is running
python -c "
import requests
import sys

port = 8000
try:
    r = requests.get(f'http://localhost:{port}/liveness', timeout=2)
    if r.status_code == 200:
        print(f'âœ… Health server running on port {port}')
        print(f'   Liveness: {r.json()}')

    r = requests.get(f'http://localhost:{port}/readiness', timeout=2)
    if r.status_code == 200:
        print(f'   Readiness: {r.json()}')
except requests.exceptions.ConnectionError:
    print(f'âŒ Health server not running on port {port}')
except Exception as e:
    print(f'âŒ Error: {e}')
"

# Manual health check (without server)
python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path

print('=== HEALTH CHECK ===')

# Check kill switch
kill_switch = Path('state/KILL_SWITCH')
if kill_switch.exists():
    print('ðŸ›‘ KILL SWITCH ACTIVE')
else:
    print('âœ… Kill switch: OFF')

# Check state files
state_dir = Path('state')
if state_dir.exists():
    files = list(state_dir.glob('*'))
    print(f'âœ… State directory: {len(files)} files')
else:
    print('âš ï¸ State directory missing')

# Check logs
logs_dir = Path('logs')
if logs_dir.exists():
    events = logs_dir / 'events.jsonl'
    if events.exists():
        lines = len(events.read_text().splitlines())
        print(f'âœ… Event log: {lines} entries')
    else:
        print('âš ï¸ No event log')
else:
    print('âš ï¸ Logs directory missing')

# Check data
data_dir = Path('data/cache')
if data_dir.exists():
    cached = len(list(data_dir.glob('*.csv')))
    print(f'âœ… Data cache: {cached} symbols')
else:
    print('âš ï¸ Data cache missing')
"

# Full system health
python -c "
import os
import requests
import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.env_loader import load_env

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

print('=== FULL SYSTEM HEALTH ===')
checks = []

# 1. Env vars
required = ['POLYGON_API_KEY', 'ALPACA_API_KEY_ID', 'ALPACA_API_SECRET_KEY']
missing = [k for k in required if not os.getenv(k)]
checks.append(('Env vars', len(missing) == 0, f'{len(missing)} missing'))

# 2. Kill switch
kill = Path('state/KILL_SWITCH').exists()
checks.append(('Kill switch', not kill, 'ACTIVE' if kill else 'off'))

# 3. Alpaca connection
try:
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    r = requests.get(f'{base}/v2/account', headers={
        'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY_ID', ''),
        'APCA-API-SECRET-KEY': os.getenv('ALPACA_API_SECRET_KEY', ''),
    }, timeout=5)
    checks.append(('Alpaca', r.status_code == 200, f'HTTP {r.status_code}'))
except Exception as e:
    checks.append(('Alpaca', False, str(e)[:30]))

# 4. Hash chain
from core.hash_chain import verify_chain
valid = verify_chain()
checks.append(('Hash chain', valid, 'valid' if valid else 'TAMPERED'))

# Print results
for name, ok, detail in checks:
    status = 'âœ…' if ok else 'âŒ'
    print(f'{status} {name}: {detail}')
"
```

## Endpoints
| Endpoint | Response | Use |
|----------|----------|-----|
| `/liveness` | `{"alive": true}` | Is process running? |
| `/readiness` | `{"ready": true}` | Can accept traffic? |

## Integration
```yaml
# Kubernetes probe example
livenessProbe:
  httpGet:
    path: /liveness
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /readiness
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Monitoring Integration
- **Prometheus**: Scrape /metrics endpoint
- **Uptime Kuma**: Ping /liveness every 60s
- **AWS ALB**: Use /readiness for target health


