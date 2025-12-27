# /earnings

Earnings calendar and earnings-aware trading.

## Usage
```
/earnings [--days N] [--symbol SYMBOL]
```

## What it does
1. Show upcoming earnings for universe
2. Flag positions with imminent earnings
3. Suggest avoid/target based on strategy
4. Historical earnings move analysis

## Commands
```bash
# Show earnings next 7 days
python scripts/earnings_calendar.py --days 7 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Check specific symbol
python scripts/earnings_calendar.py --symbol AAPL

# Flag universe for earnings
python scripts/earnings_calendar.py --universe data/universe/optionable_liquid_900.csv --days 5

# Show historical earnings moves
python scripts/earnings_calendar.py --symbol AAPL --history
```

## Output
```
Earnings Next 7 Days (from universe):
| Symbol | Date | Time | Est EPS | Avg Move |
|--------|------|------|---------|----------|
| AAPL | Jan 28 | AMC | $2.10 | +/-4.2% |
| MSFT | Jan 29 | AMC | $2.95 | +/-3.8% |
| NVDA | Feb 21 | AMC | $5.50 | +/-8.1% |

POSITIONS AT RISK:
  AAPL - earnings in 3 days - CONSIDER EXITING
```

## Strategy Integration
- **Default**: Avoid entries 3 days before earnings
- **Configurable**: `--earnings-buffer 5` for 5-day buffer
- **Override**: `--allow-earnings` to ignore

## Earnings Behavior
| Setting | Action |
|---------|--------|
| Conservative | Exit all before earnings |
| Moderate | Exit if unrealized P&L > 5% |
| Aggressive | Hold through earnings |

## Data Source
- Polygon.io reference data
- Earnings dates updated daily


