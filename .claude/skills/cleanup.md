# /cleanup

Purge old logs, cache, and temporary files.

## Usage
```
/cleanup [--dry-run] [--force] [--area AREA]
```

## What it does
1. Remove old log files
2. Clear stale cache
3. Delete temporary files
4. Free up disk space

## Commands
```bash
# Preview what would be deleted (safe)
python scripts/cleanup.py --dry-run --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Clean all areas
python scripts/cleanup.py --all

# Clean specific area
python scripts/cleanup.py --area logs
python scripts/cleanup.py --area cache
python scripts/cleanup.py --area temp

# Force clean (no confirmation)
python scripts/cleanup.py --force
```

## Cleanup Targets

### 1. LOGS
| Target | Retention | Action |
|--------|-----------|--------|
| events.jsonl | 30 days | Archive then delete |
| error logs | 90 days | Archive then delete |
| debug logs | 7 days | Delete |

### 2. CACHE
| Target | Retention | Action |
|--------|-----------|--------|
| EOD bars | 30 days unused | Delete |
| Indicator cache | 7 days | Delete |
| API responses | 24 hours | Delete |

### 3. TEMP FILES
| Target | Action |
|--------|--------|
| *.tmp | Delete |
| *.pyc | Delete |
| __pycache__ | Clear |
| .pytest_cache | Clear |

### 4. BACKTEST OUTPUTS
| Target | Retention | Action |
|--------|-----------|--------|
| Old WF runs | Keep last 5 | Archive rest |
| Trade lists | 90 days | Archive |
| Equity curves | 90 days | Archive |

## Output
```
CLEANUP REPORT
==============
Mode: DRY RUN (no files deleted)

LOGS:
  [DEL] logs/events_2024-10-*.jsonl (12 files, 45MB)
  [DEL] logs/debug_*.log (28 files, 120MB)
  [KEEP] logs/events.jsonl (current)

CACHE:
  [DEL] cache/eod/AAPL_2023*.csv (stale, 2MB)
  [DEL] cache/indicators/*.pkl (expired, 15MB)
  [KEEP] cache/eod/*_2024*.csv (recent)

TEMP:
  [DEL] **/__pycache__ (34 dirs, 8MB)
  [DEL] .pytest_cache (1MB)

SUMMARY:
  Files to delete: 156
  Space to free: 191 MB

Run with --force to execute cleanup.
```

## Safety
- Never deletes state files
- Never deletes hash chain
- Archives before delete (logs/archive/)
- Requires --force for actual deletion

## Integration
- Weekly cron job recommended
- Alert if disk < 5GB
- Log cleanup actions to audit
