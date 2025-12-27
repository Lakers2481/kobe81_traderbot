# /backtest

Run a simple backtest (not walk-forward) for quick validation.

## Usage
```
/backtest [--strategy NAME] [--symbols TICKERS] [--start DATE] [--end DATE]
```

## What it does
1. Runs a single-pass backtest on specified data
2. Generates trade list and equity curve
3. Calculates key performance metrics
4. Outputs results to backtest/ directory

## Commands
```bash
# Quick backtest on sample symbols
python scripts/run_backtest_polygon.py \
    --symbols AAPL,MSFT,GOOGL,AMZN,META \
    --start 2022-01-01 \
    --end 2024-12-31 \
    --strategy _rsi2 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Full universe backtest
python scripts/run_backtest_polygon.py \
    --universe data/universe/optionable_liquid_900.csv \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --cap 100 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# View latest backtest results
python -c "
from pathlib import Path
import json

# Find most recent backtest output
bt_dirs = sorted(Path('backtest').glob('bt_*'), key=lambda p: p.stat().st_mtime, reverse=True)
if not bt_dirs:
    print('No backtest results found')
    exit()

latest = bt_dirs[0]
summary_file = latest / 'summary.json'

print(f'=== BACKTEST RESULTS: {latest.name} ===')
if summary_file.exists():
    data = json.loads(summary_file.read_text())
    print(f\"Trades: {data.get('trades', 0)}\")
    print(f\"Win Rate: {data.get('win_rate', 0)*100:.1f}%\")
    print(f\"Profit Factor: {data.get('profit_factor', 0):.2f}\")
    print(f\"Sharpe Ratio: {data.get('sharpe', 0):.2f}\")
    print(f\"Max Drawdown: {data.get('max_drawdown', 0)*100:.1f}%\")
    print(f\"Final Equity: \${data.get('final_equity', 0):,.2f}\")

# Show last 10 trades
trades_file = latest / 'trade_list.csv'
if trades_file.exists():
    import pandas as pd
    trades = pd.read_csv(trades_file)
    print()
    print('Last 10 trades:')
    print(trades.tail(10).to_string(index=False))
"
```

## Backtest vs Walk-Forward
| Aspect | Backtest | Walk-Forward |
|--------|----------|--------------|
| Speed | Fast (minutes) | Slow (hours) |
| Overfitting | High risk | Low risk |
| Validation | Single period | Rolling OOS |
| Use case | Quick check | Production validation |

## Output Files
```
backtest/bt_YYYYMMDD_HHMMSS/
â”œâ”€â”€ summary.json      # Performance metrics
â”œâ”€â”€ trade_list.csv    # All trades
â”œâ”€â”€ equity_curve.csv  # Daily equity
â””â”€â”€ config.json       # Parameters used
```

## Key Metrics Explained
| Metric | Target | Meaning |
|--------|--------|---------|
| Win Rate | >55% | % profitable trades |
| Profit Factor | >1.5 | Gross win / gross loss |
| Sharpe | >1.0 | Risk-adjusted return |
| Max Drawdown | <15% | Worst peak-to-trough |


