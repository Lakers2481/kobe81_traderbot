# /preflight

Run Kobe's 10 preflight checks before any trading session.

## Usage
```
/preflight [--env PATH]
```

## What it does
Runs all 10 critical checks:
1. Broker connectivity (Alpaca reachable)
2. Market calendar (is today a trading day?)
3. Data freshness (EOD data current)
4. Order dry-run simulation
5. Position reconciliation
6. Instance lock (no duplicates)
7. Config signature (detect unauthorized changes)
8. Clock skew (< 250ms)
9. Hash chain integrity (tamper detection)
10. Secrets validation (API keys present)

## Command
```bash
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

## Expected Output
- All PASS: System ready for trading
- Any FAIL: System enters SAFE_MODE (no new entries)

## On Failure
If any check fails:
1. Review the specific failure message
2. Fix the underlying issue
3. Re-run preflight before trading


