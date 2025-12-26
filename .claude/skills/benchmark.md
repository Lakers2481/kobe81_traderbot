# /benchmark

Compare Kobe's performance against SPY benchmark.

## Usage
```
/benchmark [--period week|month|year|all]
```

## What it does
1. Calculates Kobe's returns over period
2. Fetches SPY returns for same period
3. Computes alpha, beta, correlation
4. Shows risk-adjusted comparison

## Commands
```bash
# Compare vs SPY
python -c "
import sys
sys.path.insert(0, '.')
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

from config.env_loader import load_env
load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

print('=== BENCHMARK COMPARISON ===')

# Get Kobe's equity curve
equity_file = Path('backtest/equity_curve.csv')
if not equity_file.exists():
    # Try WF outputs
    for f in Path('wf_outputs').glob('*/split_*/equity_curve.csv'):
        equity_file = f
        break

if not equity_file.exists():
    print('No equity curve found. Run /backtest first.')
    exit()

kobe_eq = pd.read_csv(equity_file, index_col=0, parse_dates=True)
kobe_returns = kobe_eq['equity'].pct_change().dropna()

start_date = kobe_returns.index.min().strftime('%Y-%m-%d')
end_date = kobe_returns.index.max().strftime('%Y-%m-%d')

print(f'Period: {start_date} to {end_date}')
print()

# Fetch SPY data
key = os.getenv('POLYGON_API_KEY', '')
if key:
    url = f'https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{start_date}/{end_date}?apiKey={key}'
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        data = r.json()
        spy_bars = pd.DataFrame(data.get('results', []))
        if not spy_bars.empty:
            spy_bars['date'] = pd.to_datetime(spy_bars['t'], unit='ms')
            spy_bars.set_index('date', inplace=True)
            spy_returns = spy_bars['c'].pct_change().dropna()

            # Align dates
            common = kobe_returns.index.intersection(spy_returns.index)
            k_ret = kobe_returns.loc[common]
            s_ret = spy_returns.loc[common]

            # Calculate metrics
            kobe_total = (1 + k_ret).prod() - 1
            spy_total = (1 + s_ret).prod() - 1

            kobe_sharpe = k_ret.mean() / k_ret.std() * np.sqrt(252) if k_ret.std() > 0 else 0
            spy_sharpe = s_ret.mean() / s_ret.std() * np.sqrt(252) if s_ret.std() > 0 else 0

            correlation = k_ret.corr(s_ret)
            beta = k_ret.cov(s_ret) / s_ret.var() if s_ret.var() > 0 else 0
            alpha = (k_ret.mean() - beta * s_ret.mean()) * 252  # Annualized

            print(f'{\"\":<15} {\"Kobe\":>12} {\"SPY\":>12}')
            print('-' * 40)
            print(f'{\"Total Return\":<15} {kobe_total*100:>11.1f}% {spy_total*100:>11.1f}%')
            print(f'{\"Sharpe Ratio\":<15} {kobe_sharpe:>12.2f} {spy_sharpe:>12.2f}')
            print(f'{\"Volatility\":<15} {k_ret.std()*np.sqrt(252)*100:>11.1f}% {s_ret.std()*np.sqrt(252)*100:>11.1f}%')
            print()
            print(f'Alpha (annual): {alpha*100:.2f}%')
            print(f'Beta: {beta:.2f}')
            print(f'Correlation: {correlation:.2f}')
            print()
            if kobe_total > spy_total:
                print(f'🏆 Kobe beats SPY by {(kobe_total - spy_total)*100:.1f}%')
            else:
                print(f'📉 SPY beats Kobe by {(spy_total - kobe_total)*100:.1f}%')
else:
    print('POLYGON_API_KEY not set - cannot fetch SPY data')
"

# Quick benchmark summary
python -c "
import json
from pathlib import Path

# Find latest backtest summary
summary_file = None
for f in sorted(Path('.').glob('**/summary.json'), key=lambda x: x.stat().st_mtime, reverse=True):
    if 'wf_outputs' in str(f) or 'backtest' in str(f):
        summary_file = f
        break

if summary_file:
    data = json.loads(summary_file.read_text())
    print('=== KOBE PERFORMANCE ===')
    print(f\"Win Rate: {data.get('win_rate', 0)*100:.1f}%\")
    print(f\"Profit Factor: {data.get('profit_factor', 0):.2f}\")
    print(f\"Sharpe Ratio: {data.get('sharpe', 0):.2f}\")
    print(f\"Max Drawdown: {data.get('max_drawdown', 0)*100:.1f}%\")
    print()
    print('(Run full /benchmark for SPY comparison)')
else:
    print('No backtest results found')
"
```

## Key Metrics
| Metric | Description | Target |
|--------|-------------|--------|
| Alpha | Excess return over benchmark | > 0% |
| Beta | Market sensitivity | 0.5 - 1.0 |
| Sharpe | Risk-adjusted return | > 1.0 |
| Correlation | Return correlation with SPY | < 0.7 ideal |

## Interpretation
| Result | Meaning |
|--------|---------|
| Positive Alpha | Beating the market |
| Beta < 1 | Less volatile than market |
| Low Correlation | Diversification benefit |
| Higher Sharpe | Better risk-adjusted |

## Benchmark Comparison Table
```
Strategy Performance vs SPY (example)
====================================
                    Kobe         SPY
Total Return       +18.5%      +12.3%
Sharpe Ratio        1.45        0.89
Max Drawdown       -8.2%      -13.1%

Alpha: +6.2% annually
Beta: 0.45
🏆 Kobe outperforms with lower risk
```
