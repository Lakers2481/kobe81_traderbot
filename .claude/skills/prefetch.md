# /prefetch

Prefetch EOD bars for the universe to speed up backtesting and scanning.

## Usage
```
/prefetch [--universe PATH] [--start DATE] [--end DATE] [--force]
```

## What it does
1. Downloads EOD bars from Polygon for all universe symbols
2. Caches data locally for fast access
3. Shows progress and handles rate limits
4. Validates downloaded data

## Commands
```bash
# Prefetch full universe (default: last 10 years)
python scripts/prefetch_polygon_universe.py \
    --universe data/universe/optionable_liquid_900.csv \
    --start 2015-01-01 \
    --end $(date +%Y-%m-%d) \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Quick prefetch (last 2 years only)
python scripts/prefetch_polygon_universe.py \
    --universe data/universe/optionable_liquid_900.csv \
    --start $(date -d '-2 years' +%Y-%m-%d) \
    --end $(date +%Y-%m-%d) \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Check prefetch status
python -c "
from pathlib import Path
import pandas as pd

universe = Path('data/universe/optionable_liquid_900.csv')
cache = Path('data/cache')

if universe.exists():
    symbols = pd.read_csv(universe)['symbol'].tolist()
    cached = [f.stem for f in cache.glob('*.csv')]
    missing = [s for s in symbols if s not in cached]

    print('=== PREFETCH STATUS ===')
    print(f'Universe: {len(symbols)} symbols')
    print(f'Cached: {len(cached)} symbols')
    print(f'Missing: {len(missing)} symbols')
    if missing[:10]:
        print(f'First missing: {missing[:10]}')
else:
    print('Universe file not found')
"
```

## Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--universe` | optionable_liquid_900.csv | Universe CSV file |
| `--start` | 2015-01-01 | Start date for data |
| `--end` | today | End date for data |
| `--concurrency` | 3 | Parallel API requests |

## Rate Limits
- Polygon free tier: 5 requests/minute
- Polygon paid tier: 100+ requests/minute
- Script auto-throttles based on tier

## Expected Duration
| Universe Size | Time (free tier) | Time (paid tier) |
|---------------|------------------|------------------|
| 100 symbols | ~20 min | ~2 min |
| 500 symbols | ~100 min | ~10 min |
| 900 symbols | ~190 min | ~20 min |


