# /secrets

Manage and validate API keys and secrets.

## Usage
```
/secrets [--validate|--rotate|--show-masked]
```

## What it does
1. Validates API key formats
2. Tests API connectivity
3. Shows masked key values
4. Guides key rotation process

## Commands
```bash
# Validate all secrets
python -c "
import os
import sys
sys.path.insert(0, '.')
from pathlib import Path
from configs.env_loader import load_env

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

print('=== SECRETS VALIDATION ===')

# Polygon API Key
polygon_key = os.getenv('POLYGON_API_KEY', '')
if polygon_key:
    masked = polygon_key[:4] + '*' * (len(polygon_key) - 8) + polygon_key[-4:]
    print(f'POLYGON_API_KEY: {masked}')
    # Test connectivity
    import requests
    r = requests.get(f'https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={polygon_key}', timeout=5)
    if r.status_code == 200:
        print('  ✅ Valid and working')
    elif r.status_code == 401:
        print('  ❌ Invalid key')
    else:
        print(f'  ⚠️ HTTP {r.status_code}')
else:
    print('POLYGON_API_KEY: ❌ NOT SET')

print()

# Alpaca API Keys
alpaca_id = os.getenv('ALPACA_API_KEY_ID', '')
alpaca_secret = os.getenv('ALPACA_API_SECRET_KEY', '')

if alpaca_id:
    masked = alpaca_id[:4] + '*' * (len(alpaca_id) - 8) + alpaca_id[-4:] if len(alpaca_id) > 8 else '*' * len(alpaca_id)
    print(f'ALPACA_API_KEY_ID: {masked}')
else:
    print('ALPACA_API_KEY_ID: ❌ NOT SET')

if alpaca_secret:
    masked = alpaca_secret[:4] + '*' * (len(alpaca_secret) - 8) + alpaca_secret[-4:] if len(alpaca_secret) > 8 else '*' * len(alpaca_secret)
    print(f'ALPACA_API_SECRET_KEY: {masked}')
else:
    print('ALPACA_API_SECRET_KEY: ❌ NOT SET')

# Test Alpaca
if alpaca_id and alpaca_secret:
    base = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    import requests
    r = requests.get(f'{base}/v2/account', headers={
        'APCA-API-KEY-ID': alpaca_id,
        'APCA-API-SECRET-KEY': alpaca_secret,
    }, timeout=5)
    if r.status_code == 200:
        print('  ✅ Valid and working')
    elif r.status_code == 401 or r.status_code == 403:
        print('  ❌ Invalid credentials')
    else:
        print(f'  ⚠️ HTTP {r.status_code}')

print()

# Broker URL
base_url = os.getenv('ALPACA_BASE_URL', '')
if base_url:
    mode = 'LIVE' if 'api.alpaca.markets' in base_url and 'paper' not in base_url else 'PAPER'
    print(f'ALPACA_BASE_URL: {base_url}')
    print(f'  Mode: {mode}')
else:
    print('ALPACA_BASE_URL: Using default (paper)')
"

# Show env file location
python -c "
from pathlib import Path

env_locations = [
    Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'),
    Path('.env'),
    Path('../.env'),
]

print('=== ENV FILE LOCATIONS ===')
for loc in env_locations:
    if loc.exists():
        print(f'✅ {loc} (exists)')
    else:
        print(f'❌ {loc} (not found)')
"

# Key rotation guide
echo "
=== KEY ROTATION GUIDE ===

1. POLYGON API KEY:
   - Go to: https://polygon.io/dashboard/api-keys
   - Generate new key
   - Update .env file
   - Test with: /secrets --validate

2. ALPACA API KEYS:
   - Go to: https://app.alpaca.markets/paper/dashboard/overview
   - Generate new API keys
   - Update .env file
   - Test with: /broker

3. AFTER ROTATION:
   - Run /preflight
   - Run /smoke
   - Verify /broker shows correct account
"
```

## Security Best Practices
| Practice | Why |
|----------|-----|
| Never commit .env | Keys in git history |
| Use paper keys for dev | Protect real money |
| Rotate regularly | Limit exposure |
| Monitor usage | Detect unauthorized use |

## Required Secrets
| Key | Purpose | Where to Get |
|-----|---------|--------------|
| POLYGON_API_KEY | Market data | polygon.io |
| ALPACA_API_KEY_ID | Broker auth | alpaca.markets |
| ALPACA_API_SECRET_KEY | Broker auth | alpaca.markets |
| ALPACA_BASE_URL | Broker endpoint | Set manually |

## Key Formats
- **Polygon**: ~32 character alphanumeric
- **Alpaca Key ID**: ~20 character alphanumeric
- **Alpaca Secret**: ~40 character alphanumeric
