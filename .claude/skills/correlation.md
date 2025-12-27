# /correlation

Position correlation matrix and concentration analysis.

## Usage
```
/correlation [--period DAYS]
```

## What it does
1. Calculate pairwise correlation of positions
2. Show sector clustering
3. Warn on high correlation risk
4. Suggest diversification

## Commands
```bash
# Current position correlations
python scripts/analyze_correlation.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Historical correlation (60 days)
python scripts/analyze_correlation.py --period 60

# Include universe (not just positions)
python scripts/analyze_correlation.py --universe data/universe/optionable_liquid_900.csv --top 50

# Export matrix
python scripts/analyze_correlation.py --output correlation_matrix.csv
```

## Output Example
```
POSITION CORRELATION MATRIX

        AAPL   MSFT   NVDA   GOOGL  AMZN
AAPL    1.00   0.82   0.75   0.68   0.71
MSFT    0.82   1.00   0.78   0.72   0.69
NVDA    0.75   0.78   1.00   0.65   0.62
GOOGL   0.68   0.72   0.65   1.00   0.74
AMZN    0.71   0.69   0.62   0.74   1.00

RISK METRICS:
  Average Pairwise Correlation: 0.72
  Effective Positions: 2.3 (of 5)
  Diversification Ratio: 0.46

WARNINGS:
  [!] AAPL-MSFT correlation 0.82 > 0.70 threshold
  [!] Portfolio highly concentrated in Tech (85%)
  [!] Effective positions low - add uncorrelated names

SUGGESTIONS:
  - Add Healthcare (XLV correlation: 0.35)
  - Add Utilities (XLU correlation: 0.28)
  - Add Financials (XLF correlation: 0.52)
```

## Correlation Thresholds
| Level | Range | Action |
|-------|-------|--------|
| Low | < 0.40 | Good diversification |
| Medium | 0.40-0.70 | Acceptable |
| High | > 0.70 | Warning - reduce |
| Very High | > 0.85 | Block new entries |

## Integration
- PolicyGate checks avg correlation
- Alert on correlation spike
- Weekly correlation report


