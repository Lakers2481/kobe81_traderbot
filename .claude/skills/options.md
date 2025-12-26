# /options

Options chain lookup with IV and Greeks.

## Usage
```
/options [SYMBOL] [--expiry DATE]
```

## What it does
1. Fetch options chain from Polygon
2. Display calls/puts with strikes
3. Show implied volatility (IV)
4. Calculate Greeks (delta, gamma, theta, vega)

## Commands
```bash
# Get options chain for symbol
python scripts/options_chain.py --symbol AAPL --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Filter by expiry
python scripts/options_chain.py --symbol AAPL --expiry 2025-01-17

# Show ATM options only
python scripts/options_chain.py --symbol AAPL --atm-only

# Get IV rank/percentile
python scripts/options_chain.py --symbol AAPL --iv-stats
```

## Output
| Field | Description |
|-------|-------------|
| Strike | Option strike price |
| Expiry | Expiration date |
| Type | CALL or PUT |
| Bid/Ask | Current bid/ask |
| IV | Implied volatility |
| Delta | Price sensitivity |
| Gamma | Delta sensitivity |
| Theta | Time decay |
| Vega | IV sensitivity |
| OI | Open interest |
| Volume | Daily volume |

## Use Cases
- Check IV before earnings
- Find liquid strikes for hedging
- Analyze put/call skew
- Screen for high IV percentile

## Data Source
- Polygon.io Options API
- Requires options-enabled API key
