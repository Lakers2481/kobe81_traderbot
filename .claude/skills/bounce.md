# /bounce - Bounce Analysis System

## Purpose
Analyze bounce profiles for stocks with consecutive down-day streaks. Uses 10-year historical data to calculate recovery rates, BounceScores, and apply quality gates.

## Usage

### Check specific ticker
```bash
# Check PLTR's bounce profile at current streak
python -c "from bounce import quick_bounce_check; import json; print(json.dumps(quick_bounce_check('PLTR'), indent=2))"

# Check specific streak level
python -c "from bounce import quick_bounce_check; import json; print(json.dumps(quick_bounce_check('PLTR', streak=5), indent=2))"
```

### Run full bounce watchlist scan
```bash
python tools/today_bounce_watchlist.py --prefer 5 --fallback 10 --min_streak 3 --top 10
```

### Generate/update bounce database
```bash
# Full 10-year build (takes ~15 min)
python tools/build_bounce_db.py --years 10 --window 7 --max_streak 7

# 5-year (derived from 10Y, ~2 min)
python tools/build_bounce_db.py --years 5

# Generate all 900 ticker profiles
python tools/build_bounce_profiles_all.py --years 10
```

### Check single ticker profile
```bash
python tools/bounce_profile.py --ticker PLTR --years 10 --window 7
```

## Key Concepts

### Down Day & Streaks
- **Down Day**: `Close[t] < Close[t-1]`
- **Streak N**: N consecutive down days ending at day t
- A 7-day streak generates 7 events (one per level 1-7)

### Recovery Metrics (7-day window)
- **Recovery Rate**: % of events where Close returns to event_close within 7 days
- **Avg Days to Recover**: Average number of days to recover (when recovered)
- **Max Drawdown**: Worst intraday low during 7-day window

### BounceScore (0-100)
```
Recovery Rate   * 40  (0-40 pts)
Speed to Recover * 20  (0-20 pts, faster = better)
Avg Return      * 20  (0-20 pts)
Sample Size     * 10  (0-10 pts)
Pain Tolerance  * 10  (0-10 pts, less drawdown = better)
```

### Bounce Gates (FIRM)
- `events >= 20` (sufficient sample)
- `recovery_rate >= 75%` (reliable bounce)
- `avg_days <= 3.2` (fast recovery)

## Output Files

| File | Description |
|------|-------------|
| `reports/bounce/week_down_then_bounce_events_10y.parquet` | All 1M+ events |
| `reports/bounce/week_down_then_bounce_per_stock_10y.csv` | Per-ticker summaries |
| `reports/bounce/profiles_10y/*.md` | 900 individual profiles |
| `reports/bounce/today_bounce_watchlist_*.csv` | Daily watchlist |

## Python API

```python
from bounce import quick_bounce_check, run_bounce_analysis

# Quick lookup
result = quick_bounce_check('PLTR', streak=5)
print(f"Score: {result['bounce_score']}, Gate: {result['gate_passed']}")

# Full analysis
analysis = run_bounce_analysis(min_streak=3, top_n=5)
for item in analysis.get('watchlist', []):
    print(f"{item['ticker']}: {item['bounce_score']}")
```

## Integration with Scanner

The bounce module automatically integrates with `DualStrategyScanner` via:
- `bounce.strategy_integration.integrate_with_scanner()`
- Applies bounce gates as signal filter
- Adjusts position sizing based on BounceScore

## Database Stats (Current Build)

- **10Y**: 1,026,056 events from 898 tickers
- **5Y**: 543,920 events
- **Profiles**: 901 files per window
- **Validation**: Lookahead bias checks PASSED
