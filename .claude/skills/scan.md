# /scan

Run Kobe's daily universe scanner to find trading signals.

## Usage
```
/scan [--universe PATH] [--cap N]
```

## What it does
1. Loads the 900-stock universe
2. Fetches latest EOD data from Polygon
3. Runs  IBS_RSI/ICT + ICT strategies
4. Outputs signals meeting entry criteria
5. Respects PolicyGate budgets ($75/order, $1k/day)

## Commands
```bash
# Standard scan (900 stocks)
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-only

# Quick scan (top 100 liquid)
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 100 --scan-only

# With custom universe
python scripts/run_paper_trade.py --universe data/universe/custom.csv --cap 50 --scan-only
```

## Output
Signals CSV with columns:
- timestamp, symbol, side, entry_price
- stop_loss, take_profit, reason
- strategy_name, confidence

## Notes
- Scan runs at close(t), orders execute at open(t+1)
- No lookahead: all indicators shifted 1 bar
- Signals are advisory until PolicyGate approves


