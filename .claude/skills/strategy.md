# /strategy

View and compare strategy configurations and parameters.

## Usage
```
/strategy [--list|--show NAME|--compare]
```

## What it does
1. Lists available strategies
2. Shows strategy parameters and logic
3. Compares strategy performance metrics
4. Displays entry/exit rules

## Commands
```bash
# List available strategies
python -c "
from pathlib import Path
import importlib.util

strategies_dir = Path('strategies')
print('=== AVAILABLE STRATEGIES ===')
for d in strategies_dir.iterdir():
    if d.is_dir() and (d / 'strategy.py').exists():
        print(f'  - {d.name}')
"

# Show  IBS_RSI/ICT parameters
python -c "
from strategies._rsi2.strategy import RSI2Params

params = RSI2Params()
print('===  IBS_RSI/ICT STRATEGY ===')
print()
print('Entry Rules:')
print('  LONG:  RSI(2) <= 10 AND Close > SMA(200)')
print('  SHORT: RSI(2) >= 90 AND Close < SMA(200)')
print()
print('Exit Rules:')
print('  - Stop Loss: ATR(14) x 2.0')
print('  - Time Stop: 5 bars')
print()
print('Parameters:')
for field in params.__dataclass_fields__:
    print(f'  {field}: {getattr(params, field)}')
"

# Show ICT parameters
python -c "
print('=== ICT STRATEGY ===')
print()
print('Entry Rules:')
print('  LONG:  ICT < 0.2 AND Close > SMA(200)')
print('  SHORT: ICT > 0.8 AND Close < SMA(200)')
print()
print('Exit Rules:')
print('  - Stop Loss: ATR(14) x 2.0')
print('  - Time Stop: 5 bars')
print()
print('Where ICT = (Close - Low) / (High - Low)')
"

# Compare strategy performance (from WF results)
python -c "
from pathlib import Path
import json

wf_dir = Path('wf_outputs')
if not wf_dir.exists():
    print('No walk-forward results yet')
    exit()

print('=== STRATEGY COMPARISON ===')
print(f'{\"Strategy\":<20} {\"Win Rate\":>10} {\"PF\":>8} {\"Sharpe\":>8} {\"MaxDD\":>8}')
print('-' * 60)

for strat_dir in wf_dir.iterdir():
    if not strat_dir.is_dir():
        continue
    summary_file = strat_dir / 'summary.json'
    if not summary_file.exists():
        # Try to find in split subdirs
        for split in strat_dir.glob('split_*/summary.json'):
            summary_file = split
            break
    if summary_file.exists():
        try:
            data = json.loads(summary_file.read_text())
            wr = data.get('win_rate', 0) * 100
            pf = data.get('profit_factor', 0)
            sh = data.get('sharpe', 0)
            dd = data.get('max_drawdown', 0) * 100
            print(f'{strat_dir.name:<20} {wr:>9.1f}% {pf:>8.2f} {sh:>8.2f} {dd:>7.1f}%')
        except:
            pass
"
```

## Strategy Matrix
| Strategy | Entry | Exit | Best For |
|----------|-------|------|----------|
|  IBS_RSI/ICT | RSI(2)â‰¤10 | ATR stop / 5 bars | Mean reversion |
| ICT | ICT<0.2 | ATR stop / 5 bars | Intraday range |

## Modifying Parameters
Parameters are in `strategies/<name>/strategy.py`. Changes require:
1. Update the `*Params` dataclass
2. Re-run walk-forward validation
3. Update config pin hash


