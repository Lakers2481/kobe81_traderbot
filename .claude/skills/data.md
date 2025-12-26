# /data

Check data fetch status, cache health, and validate data integrity.

## Usage
```
/data [--validate|--cache|--refresh SYMBOL]
```

## What it does
1. Shows data cache status (size, age, completeness)
2. Validates data integrity (no gaps, correct OHLCV)
3. Checks Polygon API connectivity
4. Optionally refreshes specific symbols

## Commands
```bash
# Check cache status
python -c "
from pathlib import Path
import os
from datetime import datetime

cache_dir = Path('data/cache')
if not cache_dir.exists():
    print('No cache directory found')
    exit()

files = list(cache_dir.glob('*.csv'))
total_size = sum(f.stat().st_size for f in files)
oldest = min((f.stat().st_mtime for f in files), default=0)
newest = max((f.stat().st_mtime for f in files), default=0)

print('=== DATA CACHE STATUS ===')
print(f'Cache location: {cache_dir}')
print(f'Symbols cached: {len(files)}')
print(f'Total size: {total_size / 1024 / 1024:.1f} MB')
if oldest:
    print(f'Oldest file: {datetime.fromtimestamp(oldest):%Y-%m-%d %H:%M}')
    print(f'Newest file: {datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}')
"

# Validate data integrity
python -c "
import pandas as pd
from pathlib import Path
import sys

cache_dir = Path('data/cache')
errors = []
checked = 0

for f in list(cache_dir.glob('*.csv'))[:50]:  # Sample check
    try:
        df = pd.read_csv(f)
        checked += 1
        # Check required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f'{f.stem}: missing {missing}')
            continue
        # Check OHLC validity
        invalid = ((df['high'] < df['low']) | (df['open'] <= 0) | (df['close'] <= 0)).sum()
        if invalid > 0:
            errors.append(f'{f.stem}: {invalid} invalid bars')
    except Exception as e:
        errors.append(f'{f.stem}: {e}')

print('=== DATA INTEGRITY CHECK ===')
print(f'Checked: {checked} files')
print(f'Errors: {len(errors)}')
for e in errors[:10]:
    print(f'  - {e}')
if len(errors) > 10:
    print(f'  ... and {len(errors) - 10} more')
"

# Check Polygon API
python -c "
import os
import requests
from configs.env_loader import load_env
from pathlib import Path

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))
key = os.getenv('POLYGON_API_KEY', '')
if not key:
    print('POLYGON_API_KEY not set')
    exit(1)

url = f'https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={key}'
try:
    r = requests.get(url, timeout=5)
    print(f'Polygon API: {r.status_code}')
    if r.status_code == 200:
        print('✅ Polygon connection OK')
    else:
        print(f'❌ Polygon error: {r.text[:100]}')
except Exception as e:
    print(f'❌ Polygon error: {e}')
"
```

## Data Quality Checks
| Check | Pass Criteria |
|-------|---------------|
| Completeness | All 950 symbols have data |
| Freshness | Last bar within 24h (trading days) |
| OHLCV Valid | high >= low, prices > 0 |
| No Gaps | Max 5 consecutive missing days |
| Volume | Volume > 0 for all bars |

## Cache Structure
```
data/cache/
├── AAPL.csv      # Symbol-level CSV files
├── MSFT.csv
├── ...
└── metadata.json # Cache metadata
```
