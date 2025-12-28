# /debugger

Advanced code debugging and error diagnosis.

## Usage
```
/debugger [--trace] [--error ERROR_ID] [--symbol SYMBOL]
```

## What it does
1. Trace execution flow
2. Diagnose errors from logs
3. Debug signal generation
4. Profile performance bottlenecks
5. Inspect variable states
6. Reproduce issues

## Commands
```bash
# Debug last error
python scripts/debugger.py --last-error --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Debug specific error
python scripts/debugger.py --error ERR_20241225_ABC123

# Trace signal for symbol
python scripts/debugger.py --trace-signal --symbol AAPL --date 2024-12-25

# Profile performance
python scripts/debugger.py --profile --script scripts/run_paper_trade.py

# Interactive debug mode
python scripts/debugger.py --interactive
```

## Debug Modes

### 1. ERROR DIAGNOSIS
```bash
# Analyze last error
python scripts/debugger.py --last-error
```
Output:
```
ERROR DIAGNOSIS
===============
Error ID: ERR_20241225_143022_XYZ
Time: 2024-12-25 14:30:22 UTC
Type: KeyError
Message: 'close' not in dataframe

STACK TRACE:
  File strategies/_rsi2/strategy.py, line 86
    close = float(row['close'])
  File backtest/engine.py, line 52
    signals = self.get_signals(merged)

ROOT CAUSE:
  DataFrame missing 'close' column after merge

SUGGESTED FIX:
  Check data fetch returned valid OHLCV
  Add: assert 'close' in df.columns

SIMILAR ERRORS (last 7 days): 2
```

### 2. SIGNAL TRACING
```bash
python scripts/debugger.py --trace-signal --symbol AAPL
```
Output:
```
SIGNAL TRACE: AAPL @ 2024-12-25
===============================
Step 1: Data Fetch
  Bars fetched: 252
  Date range: 2024-01-02 to 2024-12-25
  [OK] Data complete

Step 2: Indicator Calculation
  RSI(2): 8.45
  SMA(200): 168.30
  ATR(14): 3.25
  [OK] All indicators computed

Step 3: Signal Logic
  Close (175.20) > SMA200 (168.30)? YES
  ibs_rsi (8.45) <= 10? YES
  Price >= $5? YES
  [SIGNAL GENERATED]

Step 4: Entry Calculation
  Entry: $175.20
  Stop: $168.70 (entry - 2*ATR)
  Size: 4 shares ($75 budget)

Step 5: Execution
  Order submitted: ORD_20241225_AAPL
  Fill: $175.35 (slippage: 0.09%)
  [COMPLETE]
```

### 3. PERFORMANCE PROFILING
```bash
python scripts/debugger.py --profile
```
Output:
```
PERFORMANCE PROFILE
===================
Total time: 45.2s

TOP 10 SLOW FUNCTIONS:
  1. fetch_daily_bars_polygon: 28.5s (63%)
  2. compute_indicators: 8.2s (18%)
  3. generate_signals: 4.1s (9%)
  4. _simulate_symbol: 2.8s (6%)
  5. to_csv: 1.2s (3%)

BOTTLENECK: API calls
SUGGESTION: Increase cache usage, batch requests
```

### 4. VARIABLE INSPECTION
```bash
python scripts/debugger.py --inspect --breakpoint strategy.py:86
```
Output:
```
BREAKPOINT: strategy.py:86
--------------------------
Local variables:
  sym = 'AAPL'
  row = Series(timestamp=2024-12-25, close=175.20, ...)
  ibs_rsi = 8.45
  sma200 = 168.30
  atrv = 3.25

DataFrame shape: (252, 12)
Memory usage: 24.5 KB
```

### 5. REPRODUCE ISSUE
```bash
python scripts/debugger.py --reproduce --error ERR_20241225_ABC123
```
Output:
```
REPRODUCING ERROR...
====================
1. Loading state from error timestamp
2. Replaying data fetch
3. Running signal generation
4. ERROR REPRODUCED at line 86

Debug session ready. Type 'help' for commands.
(debug) >
```

## Debug Commands (Interactive)
| Command | Action |
|---------|--------|
| `step` | Step to next line |
| `next` | Step over function |
| `continue` | Run to next breakpoint |
| `print VAR` | Print variable value |
| `where` | Show current location |
| `up/down` | Move in stack frame |
| `quit` | Exit debugger |

## Common Issues & Fixes

### "No signals generated"
```bash
python scripts/debugger.py --trace-signal --symbol AAPL --verbose
# Check: indicator values, thresholds, data completeness
```

### "API timeout"
```bash
python scripts/debugger.py --diagnose-api
# Check: rate limits, network, API status
```

### "Incorrect P&L"
```bash
python scripts/debugger.py --trace-pnl --trade-id TRD_123
# Check: entry/exit prices, fees, position size
```

## Integration
- Auto-captures errors to logs/errors.jsonl
- Telegram alert with error ID
- Links to /logs for context
- Creates reproducible test cases


