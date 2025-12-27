# /wf

Run Kobe's walk-forward backtest with HTML report.

## Usage
```
/wf [--start DATE] [--end DATE] [--train DAYS] [--test DAYS]
```

## What it does
1. Splits data into train/test windows
2. Optimizes parameters on train period
3. Tests on out-of-sample period
4. Rolls forward and repeats
5. Generates HTML comparison report

## Commands
```bash
# Standard 10-year walk-forward
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63

# Generate HTML report
python scripts/aggregate_wf_report.py --wfdir wf_outputs

# Quick backtest (5 years)
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2020-01-01 --end 2024-12-31 --train-days 252 --test-days 63
```

## Output
- `wf_outputs/wf_summary_compare.csv` - Strategy comparison
- `wf_outputs/<strategy>/split_NN/` - Per-split results
  - `trade_list.csv` - All trades
  - `equity_curve.csv` - Daily equity
  - `summary.json` - Performance metrics

## Key Metrics
- Win Rate: Target 55-60%
- Profit Factor: Target > 1.5
- Max Drawdown: Target < 15%
- Sharpe Ratio: Target > 1.0

## Notes
- Uses Polygon EOD data (cached locally)
- No lookahead: signals at close(t), fills at open(t+1)
- Includes slippage (10 bps) and commissions


