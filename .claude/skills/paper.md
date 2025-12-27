# /paper

Start Kobe's paper trading session with micro budget.

## Usage
```
/paper [--cap N] [--scan-times TIMES]
```

## What it does
1. Runs preflight checks first
2. Starts 24/7 runner in PAPER mode
3. Scans at configured times (default: 09:35, 10:30, 15:55 ET)
4. Submits paper orders via Alpaca
5. Logs all decisions to hash chain

## Commands
```bash
# Standard paper trading (50 stock cap)
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,10:30,15:55

# Micro budget paper trading
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50

# With custom scan times
python scripts/runner.py --mode paper --universe data/universe/optionable_liquid_900.csv --cap 50 --scan-times 09:35,12:00,15:55
```

## Budget Limits (PolicyGate)
- Max per order: $75
- Max daily: $1,000
- Canary mode: Auto-demote on KPI breach

## Kill Switch
To halt all trading immediately:
```bash
touch state/KILL_SWITCH
```

To resume:
```bash
rm state/KILL_SWITCH
```

## Monitoring
- Watch logs: `tail -f logs/events.jsonl`
- Check positions: `/status`
- Verify chain: `python scripts/verify_hash_chain.py`


