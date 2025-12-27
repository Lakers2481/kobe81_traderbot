# /optimize

Run parameter optimization for strategy tuning.

## Usage
```
/optimize [--strategy NAME] [--metric TARGET]
```

## What it does
1. Grid search over parameter ranges
2. Walk-forward optimization (no lookahead)
3. Find optimal parameters per metric
4. Generate optimization report

## Commands
```bash
# Optimize Donchian/ICT parameters
python scripts/optimize_params.py --strategy donchian --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --metric sharpe

# Optimize ICT parameters
python scripts/optimize_params.py --strategy ICT --metric profit_factor

# Quick optimization (smaller grid)
python scripts/optimize_params.py --strategy donchian --quick --metric win_rate
```

## Parameter Ranges
### Donchian/ICT
| Parameter | Default | Range |
|-----------|---------|-------|
| rsi_period | 2 | 2-5 |
| entry_threshold | 10 | 5-15 |
| exit_threshold | 70 | 60-80 |
| sma_period | 200 | 100-200 |

### ICT
| Parameter | Default | Range |
|-----------|---------|-------|
| ibs_long_max | 0.2 | 0.1-0.3 |
| ibs_short_min | 0.8 | 0.7-0.9 |
| sma_period | 200 | 100-200 |

## Target Metrics
- `sharpe` - Sharpe ratio (recommended)
- `profit_factor` - Gross profit / gross loss
- `win_rate` - Percentage of winning trades
- `calmar` - Return / max drawdown

## Output
- `optimize_outputs/grid_results.csv` - All combinations
- `optimize_outputs/best_params.json` - Optimal parameters
- `optimize_outputs/heatmap.png` - Parameter sensitivity

## Warnings
- Always use walk-forward (OOS) for final validation
- Beware overfitting to historical data
- Prefer robust parameters over optimal ones


