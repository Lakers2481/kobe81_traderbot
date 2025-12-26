# /universe

Manage Kobe's tradable stock universe.

## Usage
```
/universe [--show|--build|--validate|--refresh]
```

## What it does
1. Shows current universe composition
2. Builds new universe from filters
3. Validates data availability
4. Refreshes with latest market data

## Commands
```bash
# Show current universe stats
python -c "
import pandas as pd
from pathlib import Path

universe_file = Path('data/universe/optionable_liquid_final.csv')
if not universe_file.exists():
    print('Universe file not found!')
    exit(1)

df = pd.read_csv(universe_file)
print(f'=== UNIVERSE: {universe_file.name} ===')
print(f'Total symbols: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()
print('Sample symbols:')
print(df.head(10).to_string(index=False))
"

# Build new universe (950 stocks)
python scripts/build_universe_polygon.py \
    --candidates data/universe/optionable_liquid_candidates.csv \
    --start 2015-01-01 \
    --end 2024-12-31 \
    --min-years 10 \
    --cap 950 \
    --concurrency 3

# Validate universe has data
python -c "
import pandas as pd
from pathlib import Path

universe = pd.read_csv('data/universe/optionable_liquid_final.csv')
data_dir = Path('data/polygon/daily')

missing = []
for symbol in universe['symbol']:
    if not (data_dir / f'{symbol}.csv').exists():
        missing.append(symbol)

if missing:
    print(f'Missing data for {len(missing)} symbols:')
    for s in missing[:20]:
        print(f'  {s}')
    if len(missing) > 20:
        print(f'  ... and {len(missing)-20} more')
else:
    print('✅ All symbols have data')
"

# Refresh universe data (prefetch)
python scripts/prefetch_polygon_universe.py \
    --universe data/universe/optionable_liquid_final.csv \
    --start 2015-01-01 \
    --end 2024-12-31
```

## Universe Filters
- Min market cap: $1B
- Min avg daily volume: 500K
- Price range: $5 - $1000
- Options available: Required
- History: 10+ years

## Notes
- Universe is rebuilt nightly (if scheduled)
- Changes logged to audit chain
- Always validate after rebuild
