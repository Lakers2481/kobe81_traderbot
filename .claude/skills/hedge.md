# /hedge

Suggest protective puts for open positions.

## Usage
```
/hedge [--symbol SYMBOL] [--all]
```

## What it does
1. Analyze open equity positions
2. Calculate position risk (delta dollars)
3. Suggest protective put strikes/expiries
4. Estimate hedge cost vs protection

## Commands
```bash
# Suggest hedges for all positions
python scripts/suggest_hedge.py --all --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Hedge specific symbol
python scripts/suggest_hedge.py --symbol AAPL

# Target protection level
python scripts/suggest_hedge.py --symbol AAPL --protect 10  # 10% downside protection

# Show cost analysis
python scripts/suggest_hedge.py --all --cost-analysis
```

## Hedge Suggestions
| Protection | Strike | Expiry | Cost | Break-even |
|------------|--------|--------|------|------------|
| 5% | ATM-5% | 30 DTE | 1.2% | -6.2% |
| 10% | ATM-10% | 30 DTE | 0.5% | -10.5% |
| 15% | ATM-15% | 45 DTE | 0.3% | -15.3% |

## Criteria
- Liquid options (OI > 100, spread < 10%)
- 30-45 DTE preferred
- OTM puts for cost efficiency
- Portfolio-level delta hedge option

## Output
```
Position: AAPL 100 shares @ $175 ($17,500)
Risk: $1,750 (10% drawdown)

Suggested Hedge:
  PUT AAPL 170 Feb21 @ $2.50
  Cost: $250 (1.4% of position)
  Max Loss: $750 (4.3%)
  Break-even: $167.50
```

## Notes
- Hedging has cost - use selectively
- Consider portfolio-level hedges (SPY puts)
- Check IV before buying protection
