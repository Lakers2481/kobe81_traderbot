# /learn

Show what Kobe learned from recent trades.

## Usage
```
/learn [--period DAYS] [--strategy NAME]
```

## What it does
1. Analyze recent trade outcomes
2. Identify patterns in wins/losses
3. Surface behavioral insights
4. Track strategy adaptation signals

## Commands
```bash
# Learn from last 30 days
python scripts/learn_trades.py --period 30 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Learn by strategy
python scripts/learn_trades.py --strategy donchian --period 60

# Compare strategies
python scripts/learn_trades.py --compare --period 90

# Export learnings
python scripts/learn_trades.py --period 30 --output learnings.json
```

## Learning Categories

### Entry Timing
```
LEARNED: donchian entries at RSI < 5 outperform RSI 5-10
  Win Rate: 68% vs 52%
  Avg Return: +2.1% vs +0.8%
  Sample: 45 trades
  Confidence: HIGH
```

### Exit Patterns
```
LEARNED: Time stops (5 bar) capture more profit than ATR stops
  Time stop avg: +1.8%
  ATR stop avg: -0.5%
  Recommendation: Consider wider ATR multiplier
```

### Market Conditions
```
LEARNED: Strategy underperforms when VIX > 25
  Normal VIX win rate: 58%
  High VIX win rate: 41%
  Suggestion: Reduce position size in high VIX
```

### Symbol Patterns
```
LEARNED: Tech sector outperforms in current regime
  Tech win rate: 62%
  Other sectors: 51%
  Top performers: NVDA, AAPL, MSFT
```

## Actionable Insights
| Insight | Action | Impact |
|---------|--------|--------|
| Deep oversold better | Tighten RSI threshold to 8 | +5% win rate |
| Friday exits weak | Avoid Friday entries | -2% loss rate |
| Large caps better | Weight toward mega caps | +3% return |

## Notes
- Learnings are observational, not auto-applied
- Requires sufficient sample size (20+ trades)
- Review with /journal for context


