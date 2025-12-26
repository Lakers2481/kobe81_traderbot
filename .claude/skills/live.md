# /live

Start live trading mode (REAL MONEY - USE WITH CAUTION).

## Usage
```
/live [--confirm] [--cap N] [--universe PATH]
```

## What it does
1. Runs preflight checks
2. Verifies live broker connection
3. Starts live trading with real money
4. Enforces strict risk limits

## ⚠️ WARNINGS
- **REAL MONEY**: Live mode uses your actual brokerage account
- **VERIFY FIRST**: Always run /preflight and /paper first
- **KILL SWITCH READY**: Know how to use /kill immediately
- **START SMALL**: Use --cap 5-10 initially

## Pre-Live Checklist
```
[ ] Paper trading successful for 2+ weeks
[ ] Walk-forward validation passed
[ ] Preflight checks all green
[ ] Kill switch tested
[ ] Risk limits verified
[ ] Broker account funded
[ ] Emergency contact ready
```

## Commands
```bash
# Full preflight before live
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Verify you're ready for live
python -c "
import os
import sys
sys.path.insert(0, '.')
from pathlib import Path
from config.env_loader import load_env

load_env(Path('C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env'))

print('=== LIVE TRADING READINESS ===')
checks = []

# 1. Check broker URL
base = os.getenv('ALPACA_BASE_URL', '')
is_live = 'api.alpaca.markets' in base and 'paper' not in base
checks.append(('Live broker URL', is_live, base[:40]))

# 2. Kill switch ready
kill = Path('state/KILL_SWITCH')
checks.append(('Kill switch OFF', not kill.exists(), 'Active' if kill.exists() else 'Ready'))

# 3. Hash chain valid
from core.hash_chain import verify_chain
valid = verify_chain()
checks.append(('Hash chain valid', valid, 'Valid' if valid else 'TAMPERED'))

# 4. Config pinned
pin = Path('state/config_pin.txt').exists()
checks.append(('Config pinned', pin, 'Pinned' if pin else 'Not pinned'))

# Print
all_pass = True
for name, ok, detail in checks:
    status = '✅' if ok else '❌'
    print(f'{status} {name}: {detail}')
    if not ok:
        all_pass = False

print()
if all_pass:
    print('🟢 READY FOR LIVE TRADING')
else:
    print('🔴 NOT READY - Fix issues above')
"

# Start live trading (MICRO BUDGET)
python scripts/run_live_trade_micro.py \
    --universe data/universe/optionable_liquid_final.csv \
    --cap 10 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Start live runner (24/7)
python scripts/runner.py \
    --mode live \
    --universe data/universe/optionable_liquid_final.csv \
    --cap 10 \
    --scan-times 09:35,10:30,15:55 \
    --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env
```

## Risk Limits (Enforced)
| Limit | Value | Enforced By |
|-------|-------|-------------|
| Per-order notional | $75 | PolicyGate |
| Daily notional | $1,000 | PolicyGate |
| Order type | IOC LIMIT only | broker_alpaca.py |
| Min price | $3 | PolicyGate |
| Max price | $1,000 | PolicyGate |
| Shorts | Disabled | PolicyGate |

## Emergency Actions
1. **Immediate halt**: `/kill`
2. **Check positions**: `/positions`
3. **Verify broker**: `/broker`
4. **Check logs**: `/logs --errors`

## Live vs Paper
| Aspect | Paper | Live |
|--------|-------|------|
| Money | Simulated | Real |
| Broker URL | paper-api.* | api.* |
| Risk | None | Your capital |
| Recommendation | Test first | After validation |
