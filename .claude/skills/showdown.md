# /showdown

Run strategy showdown - compare multiple strategies head-to-head.

## Usage
```
/showdown [--strategies LIST] [--universe PATH] [--start DATE] [--end DATE]
```

## What it does
1. Runs all strategies on identical data
2. Compares performance metrics side-by-side
3. Ranks strategies by key criteria
4. Generates comparison report

## Commands
```bash
# Run strategy showdown
python scripts/run_showdown_polygon.py \
    --universe data/universe/optionable_liquid_900.csv \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --cap 100 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# View showdown results
python -c "
import pandas as pd
from pathlib import Path

compare_file = Path('wf_outputs/wf_summary_compare.csv')
if not compare_file.exists():
    print('No comparison file found. Run /showdown first.')
    exit()

df = pd.read_csv(compare_file)
print('=== STRATEGY SHOWDOWN ===')
print()

# Sort by Sharpe ratio
df_sorted = df.sort_values('sharpe', ascending=False)
print(df_sorted.to_string(index=False))

print()
print('=== WINNER ===')
winner = df_sorted.iloc[0]
print(f\"Strategy: {winner['strategy']}\")
print(f\"Sharpe: {winner['sharpe']:.2f}\")
print(f\"Win Rate: {winner['win_rate']*100:.1f}%\")
"

# Quick comparison (cached results)
python -c "
from pathlib import Path
import json

print('=== STRATEGY COMPARISON (Cached) ===')
print()
print(f'{\"Strategy\":<15} {\"Trades\":>8} {\"Win%\":>8} {\"PF\":>8} {\"Sharpe\":>8} {\"MaxDD\":>8}')
print('-' * 60)

for strat in ['_rsi2', 'ICT']:
    wf_dir = Path(f'wf_outputs/{strat}')
    if not wf_dir.exists():
        continue

    # Aggregate across splits
    trades = 0
    wins = 0
    total_pnl = 0

    for split_dir in wf_dir.glob('split_*'):
        summary = split_dir / 'summary.json'
        if summary.exists():
            data = json.loads(summary.read_text())
            n = data.get('trades', 0)
            trades += n
            wins += int(n * data.get('win_rate', 0))

    if trades > 0:
        wr = wins / trades * 100
        # Get overall metrics from latest split
        latest = sorted(wf_dir.glob('split_*/summary.json'))[-1]
        data = json.loads(latest.read_text())
        pf = data.get('profit_factor', 0)
        sh = data.get('sharpe', 0)
        dd = data.get('max_drawdown', 0) * 100
        print(f'{strat:<15} {trades:>8} {wr:>7.1f}% {pf:>8.2f} {sh:>8.2f} {dd:>7.1f}%')
"
```

## Comparison Criteria
| Criterion | Weight | Description |
|-----------|--------|-------------|
| Sharpe Ratio | 30% | Risk-adjusted returns |
| Win Rate | 20% | Consistency |
| Profit Factor | 20% | Win/loss ratio |
| Max Drawdown | 15% | Risk tolerance |
| Trade Count | 15% | Statistical significance |

## Strategy Selection Guidelines
- **Conservative**: Highest Sharpe, lowest drawdown
- **Aggressive**: Highest absolute returns
- **Balanced**: Best combination of metrics
- **Production**: Must pass walk-forward validation

## Output
```
wf_outputs/
â”œâ”€â”€ wf_summary_compare.csv    # Side-by-side comparison
â”œâ”€â”€ _rsi2/
â”‚   â””â”€â”€ split_*/
â””â”€â”€ ICT/
    â””â”€â”€ split_*/
```


