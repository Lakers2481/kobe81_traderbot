# /explain

Explain why a signal was generated or rejected.

## Usage
```
/explain [--signal-id ID] [--symbol SYMBOL] [--date DATE]
```

## What it does
1. Trace signal generation logic
2. Show indicator values at signal time
3. Explain entry/exit criteria met
4. Show why other symbols were rejected

## Commands
```bash
# Explain specific signal
python scripts/explain_signal.py --signal-id SIG_20241225_AAPL_ABC123

# Explain latest signal for symbol
python scripts/explain_signal.py --symbol AAPL --latest

# Explain why no signal today
python scripts/explain_signal.py --symbol MSFT --date 2024-12-25 --why-not

# Explain all signals from scan
python scripts/explain_signal.py --scan-date 2024-12-25
```

## Output Example
```
SIGNAL EXPLANATION: SIG_20241225_AAPL_ABC123
Strategy:  Donchian/ICT
Symbol: AAPL
Date: 2024-12-25

ENTRY CRITERIA:
  [PASS] RSI(2) = 8.5 <= 10 threshold
  [PASS] Close $175.20 > SMA(200) $168.45
  [PASS] Price $175.20 >= $5.00 min
  [PASS] ATR(14) = $3.20 (valid for stop calc)

CALCULATED VALUES:
  Entry Price: $175.20 (close)
  Stop Loss: $168.80 (entry - 2*ATR)
  Position Size: 4 shares ($75 budget)

CONFIDENCE: HIGH
  - Strong trend (close 4% above SMA200)
  - Deep oversold (donchian in bottom 10%)
  - Adequate liquidity (ADV $50M+)
```

## Why-Not Analysis
```
WHY NO SIGNAL: MSFT on 2024-12-25
  [FAIL] RSI(2) = 45.2 > 10 threshold
  Close $425.00 > SMA(200) $380.00 [PASS]

  Reason: RSI not oversold enough
```

## Use Cases
- Debug unexpected signals
- Understand rejection reasons
- Validate strategy logic
- Learning and journaling


