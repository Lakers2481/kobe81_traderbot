# /drawdown

Drawdown analysis and recovery statistics.

## Usage
```
/drawdown [--period DAYS] [--detail]
```

## What it does
1. Calculate current and historical drawdowns
2. Show recovery times
3. Analyze drawdown causes
4. Compare to benchmarks

## Commands
```bash
# Current drawdown status
python scripts/analyze_drawdown.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Historical drawdowns
python scripts/analyze_drawdown.py --period 365 --detail

# Drawdown attribution
python scripts/analyze_drawdown.py --attribution

# Compare to SPY
python scripts/analyze_drawdown.py --benchmark SPY
```

## Output Example
```
DRAWDOWN ANALYSIS (2024-12-25)

CURRENT STATUS:
  Peak Equity: $108,500 (2024-12-15)
  Current Equity: $105,200
  Current Drawdown: -3.04%
  Days in Drawdown: 10

HISTORICAL DRAWDOWNS:
| Rank | Drawdown | Start | Bottom | Recovery | Duration |
|------|----------|-------|--------|----------|----------|
| 1 | -8.5% | Mar 1 | Mar 15 | Apr 10 | 40 days |
| 2 | -6.2% | Jul 20 | Aug 5 | Aug 25 | 36 days |
| 3 | -4.8% | Oct 10 | Oct 18 | Nov 1 | 22 days |
| 4 | -3.0% | Dec 15 | ongoing | - | 10+ days |

STATISTICS:
  Max Drawdown: -8.5%
  Avg Drawdown: -4.2%
  Avg Recovery: 28 days
  Longest Recovery: 40 days
  Ulcer Index: 3.2

VS BENCHMARK (SPY):
  Kobe Max DD: -8.5%
  SPY Max DD: -12.3%
  Kobe recovers 30% faster

ATTRIBUTION (Current DD):
  NVDA: -1.5% contribution
  AAPL: -0.8% contribution
  MSFT: -0.5% contribution
  Other: -0.3% contribution
```

## Thresholds
| Drawdown | Action |
|----------|--------|
| < 5% | Normal - continue trading |
| 5-10% | Caution - reduce size 50% |
| 10-15% | Warning - minimal new entries |
| > 15% | Halt - trigger kill switch |

## Recovery Tracking
- Days to recover logged
- Win rate during recovery
- Behavioral analysis (overtrading after DD?)
