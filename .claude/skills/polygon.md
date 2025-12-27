# /polygon

Validate Polygon data source and API health.

## Usage
```
/polygon [status|test|coverage|refresh]
```

## What it does
1. Check Polygon API connectivity
2. Validate universe data coverage
3. Verify data freshness
4. Test API rate limits

## Commands
```bash
# Check Polygon API status
python scripts/polygon_health.py --status --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Test API with sample request
python scripts/polygon_health.py --test

# Verify 900 universe coverage
python scripts/polygon_health.py --coverage --universe data/universe/optionable_liquid_900.csv

# Check data freshness
python scripts/polygon_health.py --freshness --cache data/cache

# Full validation
python scripts/polygon_health.py --full
```

## Output Example
```
POLYGON DATA SOURCE VALIDATION

API STATUS:
  [PASS] API Key valid
  [PASS] Endpoint reachable (latency: 45ms)
  [PASS] Rate limit OK (4/5 per second)
  [PASS] Subscription: Stocks Starter

UNIVERSE COVERAGE (900 symbols):
  [PASS] 900/900 symbols have data
  [PASS] All symbols have 10+ years history
  [PASS] No gaps > 5 days detected
  [WARN] 3 symbols have stale data (>7 days)
    - SYMBOL1: last update 2024-12-18
    - SYMBOL2: last update 2024-12-17
    - SYMBOL3: last update 2024-12-16

DATA FRESHNESS:
  Cache location: data/cache/
  Total cached files: 900
  Oldest cache: 2024-12-20
  Newest cache: 2024-12-25
  [PASS] 95% of cache < 3 days old

RECOMMENDATIONS:
  - Run /prefetch to update stale symbols
  - Cache is healthy for backtesting
```

## Validation Checks
| Check | Criteria | Action if Fail |
|-------|----------|----------------|
| API Key | Valid response | Fix .env |
| Connectivity | < 500ms latency | Check network |
| Coverage | 100% of universe | Remove missing |
| Freshness | < 3 days old | Run /prefetch |
| Gaps | < 5 consecutive days | Flag symbol |

## Integration
- Run in /preflight
- Daily automated check
- Alert on API issues
- Block trading if coverage < 95%


