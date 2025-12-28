# /simulate

Run Monte Carlo simulation for forward testing validation.

## Usage
```
/simulate [--iterations N] [--confidence LEVEL]
```

## What it does
1. Bootstrap resamples historical trade results
2. Runs N iterations with randomized sequences
3. Calculates probability distributions
4. Generates confidence intervals for key metrics

## Commands
```bash
# Run Monte Carlo simulation (1000 iterations)
python scripts/monte_carlo.py --trades wf_outputs/ibs_rsi/trade_list.csv --iterations 1000 --confidence 0.95

# Quick simulation (100 iterations)
python scripts/monte_carlo.py --trades wf_outputs/ibs_rsi/trade_list.csv --iterations 100

# Compare strategies
python scripts/monte_carlo.py --trades wf_outputs/and/trade_list.csv --iterations 1000 --output simulate_outputs/
```

## Output Metrics
| Metric | Description |
|--------|-------------|
| Expected Return | Mean return across simulations |
| Sharpe Range | 5th-95th percentile Sharpe ratio |
| Max Drawdown | Worst-case drawdown distribution |
| Probability of Ruin | P(drawdown > 25%) |
| Win Rate CI | Confidence interval on win rate |
| Profit Factor CI | Confidence interval on PF |

## Interpretation
- **Probability of Ruin < 5%**: Safe to proceed
- **Sharpe 5th percentile > 0.5**: Robust strategy
- **Max DD 95th percentile < 20%**: Acceptable risk

## When to Run
1. After walk-forward backtest passes
2. Before transitioning paper -> live
3. After any strategy parameter change
4. Monthly validation check


