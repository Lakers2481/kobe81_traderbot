# /exposure

Analyze portfolio exposure by sector, market cap, and factor.

## Usage
```
/exposure [--detail]
```

## What it does
1. Calculate sector allocation
2. Show market cap distribution
3. Analyze factor exposures (beta, momentum)
4. Warn on concentration risk

## Commands
```bash
# Current exposure analysis
python scripts/analyze_exposure.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Detailed breakdown
python scripts/analyze_exposure.py --detail --output exposure_report.html

# Check universe exposure (before trading)
python scripts/analyze_exposure.py --universe data/universe/optionable_liquid_900.csv
```

## Output Metrics
### Sector Allocation
| Sector | Target | Limit |
|--------|--------|-------|
| Technology | <30% | 40% |
| Healthcare | <20% | 30% |
| Financials | <20% | 30% |
| Any single | - | 40% |

### Market Cap
| Size | Definition |
|------|------------|
| Large | >$10B |
| Mid | $2B-$10B |
| Small | <$2B |

### Risk Metrics
- **Portfolio Beta**: Target 0.8-1.2
- **Concentration**: Top 5 positions < 50%
- **Correlation**: Avg pairwise < 0.6

## Alerts
- Sector > 40%: WARNING
- Single stock > 10%: WARNING
- Beta > 1.5: WARNING
- Correlation > 0.8: WARNING

## Integration
- Checked by PolicyGate before new entries
- Daily exposure report in logs
- Telegram alert on limit breach


