# /version

Show Kobe system version and update history.

## Usage
```
/version [--changelog] [--check-updates]
```

## What it does
1. Display current Kobe version
2. Show last update timestamp
3. List recent changes
4. Check for available updates

## Commands
```bash
# Show current version
python scripts/version.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Show changelog
python scripts/version.py --changelog

# Check for updates
python scripts/version.py --check-updates
```

## Output
```
KOBE TRADING SYSTEM
===================
Version: 1.2.0
Build: 2024-12-25T14:30:00Z
Python: 3.11.5

COMPONENTS:
  Core: 1.2.0
  Strategies: 1.1.0
  Backtest: 1.2.0
  Execution: 1.0.3

LAST UPDATE: 2024-12-20
  - Added IBS strategy
  - Fixed ATR calculation
  - Improved caching

DEPENDENCIES:
  pandas: 2.0.3
  alpaca-py: 0.10.0
  polygon-api-client: 1.12.0
```

## Version File
Located at `VERSION.json`:
```json
{
  "version": "1.2.0",
  "build_date": "2024-12-25T14:30:00Z",
  "components": {
    "core": "1.2.0",
    "strategies": "1.1.0"
  },
  "changelog": [
    {"version": "1.2.0", "date": "2024-12-20", "changes": ["Added IBS"]}
  ]
}
```

## Integration
- Displayed in /status header
- Logged at startup
- Included in error reports
