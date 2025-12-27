# /watchlist

Manage custom watchlists beyond the main universe.

## Usage
```
/watchlist [list|add|remove|scan]
```

## What it does
1. Create and manage custom symbol lists
2. Add/remove symbols from watchlists
3. Scan watchlist for signals
4. Track specific stocks of interest

## Commands
```bash
# List all watchlists
ls data/watchlists/

# View watchlist contents
cat data/watchlists/focus.csv

# Add symbol to watchlist
echo "TSLA" >> data/watchlists/focus.csv

# Remove symbol (create new without it)
grep -v "TSLA" data/watchlists/focus.csv > temp && mv temp data/watchlists/focus.csv

# Scan custom watchlist
python scripts/run_paper_trade.py --universe data/watchlists/focus.csv --cap 20 --scan-only

# Create new watchlist
echo "symbol" > data/watchlists/earnings.csv
echo "AAPL" >> data/watchlists/earnings.csv
echo "MSFT" >> data/watchlists/earnings.csv
```

## Watchlist Types
| Name | Purpose |
|------|---------|
| focus.csv | High-conviction ideas |
| earnings.csv | Upcoming earnings plays |
| sector.csv | Sector-specific tracking |
| avoid.csv | Symbols to exclude |

## Integration
- Watchlists stored in `data/watchlists/`
- Can be used with any script via `--universe`
- Avoid list checked by PolicyGate


