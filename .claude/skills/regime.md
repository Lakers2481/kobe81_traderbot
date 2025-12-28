# /regime

Market regime detection (bull/bear/chop).

## Usage
```
/regime [--detail]
```

## What it does
1. Analyze SPY/QQQ trend and volatility
2. Classify current market regime
3. Show regime history
4. Suggest strategy adjustments

## Commands
```bash
# Current regime
python scripts/detect_regime.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Detailed analysis
python scripts/detect_regime.py --detail

# Regime history
python scripts/detect_regime.py --history 90

# Check regime for specific date
python scripts/detect_regime.py --date 2024-12-01
```

## Regime Classifications
| Regime | Criteria | Strategy Adjustment |
|--------|----------|---------------------|
| BULL_LOW_VOL | SPY > SMA50, VIX < 15 | Full size, aggressive |
| BULL_HIGH_VOL | SPY > SMA50, VIX > 20 | Reduce size 50% |
| BEAR_LOW_VOL | SPY < SMA50, VIX < 20 | Long-only cautious |
| BEAR_HIGH_VOL | SPY < SMA50, VIX > 25 | Minimal/cash |
| CHOP | ADX < 20, range-bound | Mean reversion favored |
| TREND | ADX > 30 | Momentum favored |

## Output Example
```
MARKET REGIME ANALYSIS (2024-12-25)

Current Regime: BULL_LOW_VOL
Confidence: 85%

INDICATORS:
  SPY: $595.20 (+12% YTD)
  SPY vs SMA50: +3.2% (bullish)
  SPY vs SMA200: +8.5% (bullish)
  VIX: 14.2 (low volatility)
  ADX(14): 28 (trending)

REGIME HISTORY (90 days):
  BULL_LOW_VOL: 65 days (72%)
  BULL_HIGH_VOL: 15 days (17%)
  CHOP: 10 days (11%)

STRATEGY IMPLICATIONS:
  IBS_RSI/ICT: FAVORABLE (mean reversion works in low vol)
  ICT: FAVORABLE (range-bound intraday)
  Position Size: 100% (normal)

WATCH FOR:
  - VIX spike above 20 â†’ reduce exposure
  - SPY break below SMA50 â†’ defensive mode
```

## Integration
- Checked by PolicyGate before entries
- Logged in daily structured logs
- Telegram alert on regime change


