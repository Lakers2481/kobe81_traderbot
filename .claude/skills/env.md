# /env

Manage environment variables and configuration.

## Usage
```
/env [--show|--check|--template]
```

## What it does
1. Shows loaded environment variables (masked)
2. Validates required variables are set
3. Generates .env template
4. Checks env file locations

## Commands
```bash
# Show current environment
python -c "
import os
import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.env_loader import load_env

env_path = Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')
loaded = load_env(env_path)

print('=== ENVIRONMENT ===')
print(f'Loaded from: {env_path}')
print(f'Variables loaded: {len(loaded)}')
print()

# Show key variables (masked)
key_vars = [
    'POLYGON_API_KEY',
    'ALPACA_API_KEY_ID',
    'ALPACA_API_SECRET_KEY',
    'ALPACA_BASE_URL',
    'DISCORD_WEBHOOK_URL',
]

for var in key_vars:
    val = os.getenv(var, '')
    if val:
        if 'KEY' in var or 'SECRET' in var or 'WEBHOOK' in var:
            masked = val[:4] + '*' * min(len(val) - 8, 20) + val[-4:] if len(val) > 8 else '***'
            print(f'{var}: {masked}')
        else:
            print(f'{var}: {val}')
    else:
        print(f'{var}: (not set)')
"

# Check required variables
python -c "
import os
import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.env_loader import load_env

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

print('=== REQUIRED VARIABLES CHECK ===')

required = [
    ('POLYGON_API_KEY', 'Market data API'),
    ('ALPACA_API_KEY_ID', 'Broker authentication'),
    ('ALPACA_API_SECRET_KEY', 'Broker authentication'),
]

optional = [
    ('ALPACA_BASE_URL', 'Broker endpoint (default: paper)'),
    ('DISCORD_WEBHOOK_URL', 'Alert notifications'),
    ('ALERT_EMAIL', 'Email notifications'),
]

all_ok = True
for var, desc in required:
    val = os.getenv(var, '')
    if val:
        print(f'âœ… {var}')
    else:
        print(f'âŒ {var} - REQUIRED ({desc})')
        all_ok = False

print()
print('Optional:')
for var, desc in optional:
    val = os.getenv(var, '')
    if val:
        print(f'âœ… {var}')
    else:
        print(f'âšª {var} - optional ({desc})')

print()
if all_ok:
    print('âœ… All required variables set')
else:
    print('âŒ Missing required variables')
"

# Generate .env template
python -c "
template = '''# Kobe Trading Bot Environment Configuration
# Copy this file to .env and fill in your values

# === REQUIRED ===

# Polygon.io API Key (for market data)
# Get from: https://polygon.io/dashboard/api-keys
POLYGON_API_KEY=your_polygon_key_here

# Alpaca API Credentials (for trading)
# Get from: https://app.alpaca.markets/paper/dashboard/overview
ALPACA_API_KEY_ID=your_alpaca_key_id
ALPACA_API_SECRET_KEY=your_alpaca_secret_key

# Broker endpoint (paper or live)
# Paper: https://paper-api.alpaca.markets
# Live:  https://api.alpaca.markets
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# === OPTIONAL ===

# Discord webhook for alerts
# DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Email for daily reports
# ALERT_EMAIL=you@example.com

# Alert thresholds
# ALERT_DAILY_LOSS_THRESHOLD=100
# ALERT_DRAWDOWN_THRESHOLD=10
'''
print(template)
"

# Check env file exists and is readable
python -c "
from pathlib import Path

env_path = Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env')

print('=== ENV FILE STATUS ===')
if env_path.exists():
    print(f'âœ… File exists: {env_path}')

    # Check it's not empty
    content = env_path.read_text()
    lines = [l for l in content.splitlines() if l.strip() and not l.startswith('#')]
    print(f'   Non-comment lines: {len(lines)}')

    # Check permissions (basic)
    try:
        with open(env_path, 'r') as f:
            pass
        print('   Readable: Yes')
    except:
        print('   Readable: No')
else:
    print(f'âŒ File not found: {env_path}')
    print('   Run: /env --template > .env')
"
```

## Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| POLYGON_API_KEY | Yes | Market data API key |
| ALPACA_API_KEY_ID | Yes | Broker key ID |
| ALPACA_API_SECRET_KEY | Yes | Broker secret |
| ALPACA_BASE_URL | No | paper-api or api |
| DISCORD_WEBHOOK_URL | No | Alert webhook |
| ALERT_EMAIL | No | Report email |

## .env File Location
```
C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

## Security Notes
- Never commit .env to git
- Add .env to .gitignore
- Use separate keys for paper vs live
- Rotate keys periodically


