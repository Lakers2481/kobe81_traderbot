# /suggest

AI suggests next actions based on current system state.

## Usage
```
/suggest [--context AREA]
```

## What it does
1. Analyze current system state
2. Review recent performance
3. Check pending tasks/issues
4. Suggest prioritized actions

## Commands
```bash
# General suggestions
python scripts/ai_suggest.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Focus on specific area
python scripts/ai_suggest.py --context trading
python scripts/ai_suggest.py --context risk
python scripts/ai_suggest.py --context maintenance

# Detailed reasoning
python scripts/ai_suggest.py --verbose
```

## Suggestion Categories
| Context | Analyzes |
|---------|----------|
| trading | Signals, positions, market conditions |
| risk | Exposure, drawdown, limits |
| maintenance | Data freshness, system health |
| performance | Win rate, PF trends |

## Example Output
```
KOBE SUGGESTIONS (2024-12-25 09:30)

PRIORITY 1 - IMMEDIATE:
  [!] Run /preflight - not run today
  [!] 2 positions approaching stop loss

PRIORITY 2 - TODAY:
  [ ] Review 3 new signals from morning scan
  [ ] Earnings in 2 days for AAPL - consider exit
  [ ] Data cache 3 days old - run /prefetch

PRIORITY 3 - THIS WEEK:
  [ ] Win rate dropped 5% - review /metrics
  [ ] Run /simulate before scaling up
  [ ] Weekly /journal entry pending

MARKET CONTEXT:
  VIX: 18.5 (normal)
  SPY trend: Bullish (above 20/50 SMA)
  Regime: Low volatility uptrend
```

## Integration
- Runs automatically at session start
- Checks state/, logs/, positions
- Uses structured log patterns
- Respects kill switch state


