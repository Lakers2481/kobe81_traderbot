# /replay

Replay historical signals to analyze past performance.

## Usage
```
/replay [--date DATE] [--symbol SYMBOL] [--strategy NAME]
```

## What it does
1. Loads historical data for specified date
2. Regenerates signals as they would have been
3. Shows what trades would have occurred
4. Compares to actual trades (if any)

## Commands
```bash
# Replay signals for a specific date
python -c "
import sys
sys.path.insert(0, '.')
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_eod_bars
from strategies._rsi2.strategy import RSI2Strategy
from data.universe.loader import load_universe

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

# Target date (change as needed)
target_date = '2024-12-20'
print(f'=== SIGNAL REPLAY: {target_date} ===')
print()

# Load universe
universe = load_universe('data/universe/optionable_liquid_900.csv', cap=50)

# Fetch data up to and including target date
end_date = datetime.strptime(target_date, '%Y-%m-%d')
start_date = end_date - timedelta(days=300)

signals = []
strategy = RSI2Strategy()

print(f'Scanning {len(universe)} symbols...')
for sym in universe:
    try:
        df = fetch_eod_bars(sym, start_date.strftime('%Y-%m-%d'), target_date)
        if df is not None and len(df) > 200:
            # Filter to only data up to target date
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'].dt.strftime('%Y-%m-%d') <= target_date]

            if len(df) > 200:
                sigs = strategy.generate_signals(df)
                if not sigs.empty:
                    signals.extend(sigs.to_dict('records'))
    except Exception as e:
        pass

print()
if signals:
    print(f'Signals that would have been generated on {target_date}:')
    print()
    for sig in signals:
        print(f\"  {sig['symbol']:<6} {sig['side']:<5} @ \${sig['entry_price']:.2f}\")
        print(f\"         Stop: \${sig['stop_loss']:.2f}\")
        print(f\"         {sig['reason']}\")
        print()
else:
    print('No signals would have been generated')
"

# Replay specific symbol
python -c "
import sys
sys.path.insert(0, '.')
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from config.env_loader import load_env
from data.providers.polygon_eod import fetch_eod_bars
from strategies._rsi2.strategy import RSI2Strategy

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

symbol = 'AAPL'  # Change as needed
target_date = '2024-12-20'

print(f'=== {symbol} REPLAY: {target_date} ===')

end_date = datetime.strptime(target_date, '%Y-%m-%d')
start_date = end_date - timedelta(days=300)

df = fetch_eod_bars(symbol, start_date.strftime('%Y-%m-%d'), target_date)
if df is None or len(df) < 200:
    print('Insufficient data')
    exit()

strategy = RSI2Strategy()
df_ind = strategy._compute_indicators(df)

# Get the target date row
df_ind['timestamp'] = pd.to_datetime(df_ind['timestamp'])
target_row = df_ind[df_ind['timestamp'].dt.strftime('%Y-%m-%d') == target_date]

if target_row.empty:
    print(f'No data for {target_date}')
    exit()

row = target_row.iloc[-1]
print()
print('Indicator values (shifted, no lookahead):')
print(f\"  Close: \${row['close']:.2f}\")
print(f\"  RSI(2): {row['rsi2_sig']:.2f}\")
print(f\"  SMA(200): {row['sma200_sig']:.2f}\")
print(f\"  ATR(14): {row['atr14_sig']:.2f}\")
print()

# Check conditions
close = row['close']
rsi = row['rsi2_sig']
sma = row['sma200_sig']

if close > sma and rsi <= 10:
    print('âœ… LONG signal conditions MET')
elif close < sma and rsi >= 90:
    print('âœ… SHORT signal conditions MET')
else:
    print('âŒ No signal conditions met')
    print(f'   RSI {rsi:.1f} not <= 10 for long or >= 90 for short')
    if close <= sma:
        print(f'   Close {close:.2f} not above SMA {sma:.2f} for long')
"

# Compare replay to actual trades
python -c "
import json
from pathlib import Path
from datetime import date

target = '2024-12-20'  # Change as needed

print(f'=== ACTUAL vs REPLAY: {target} ===')
print()

# Load actual trades
trades_file = Path('logs/trades.jsonl')
actual_trades = []
if trades_file.exists():
    for line in trades_file.read_text().splitlines():
        try:
            t = json.loads(line)
            if t.get('timestamp', '').startswith(target):
                actual_trades.append(t)
        except:
            pass

print('Actual trades:')
if actual_trades:
    for t in actual_trades:
        print(f\"  {t.get('symbol')} {t.get('side')} {t.get('qty')} @ \${t.get('price', 0):.2f}\")
else:
    print('  (none)')
print()
print('Replay trades: Run full replay above to compare')
"
```

## Replay Use Cases
| Use Case | Purpose |
|----------|---------|
| Post-mortem | Why didn't we trade X? |
| Validation | Would strategy have worked? |
| Debug | Indicator calculation check |
| Audit | Verify actual vs expected |

## Replay Limitations
- Uses current strategy parameters
- Data quality may differ from live
- Execution assumptions are theoretical
- Doesn't account for fills/slippage

## Replay Output
```
=== SIGNAL REPLAY: 2024-12-20 ===

Signals that would have been generated:

  AAPL   long  @ $198.50
         Stop: $195.20
         donchian=8.5<= 10 & above SMA200

  MSFT   long  @ $425.30
         Stop: $420.10
         donchian=6.2<= 10 & above SMA200
```


