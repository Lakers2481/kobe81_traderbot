# /snapshot

Create full state snapshot for recovery.

## Usage
```
/snapshot [create|restore|list] [--name NAME]
```

## What it does
1. Capture complete system state
2. Store for disaster recovery
3. Restore to previous state
4. Compare snapshots

## Commands
```bash
# Create snapshot
python scripts/snapshot.py create --name "pre_live_v1" --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Create with auto-name (timestamp)
python scripts/snapshot.py create

# List all snapshots
python scripts/snapshot.py list

# Restore from snapshot
python scripts/snapshot.py restore --name "pre_live_v1"

# Compare two snapshots
python scripts/snapshot.py diff --from "snap_001" --to "snap_002"

# Delete old snapshot
python scripts/snapshot.py delete --name "old_snapshot"
```

## Snapshot Contents

### 1. STATE FILES
| File | Included |
|------|----------|
| positions.json | âœ… |
| orders.json | âœ… |
| pnl_history.json | âœ… |
| idempotency_store.json | âœ… |
| hash_chain.jsonl | âœ… |
| KILL_SWITCH | âœ… (if exists) |

### 2. CONFIGURATION
| File | Included |
|------|----------|
| config.json | âœ… |
| strategy params | âœ… |
| risk limits | âœ… |
| .env (masked) | âœ… |

### 3. METADATA
| Info | Included |
|------|----------|
| Timestamp | âœ… |
| Version | âœ… |
| Git commit | âœ… |
| File hashes | âœ… |

## Output
```
SNAPSHOT CREATED
================
Name: pre_live_v1
Time: 2024-12-25T14:30:00Z
Size: 2.4 MB

CONTENTS:
  state/positions.json (SHA256: abc123...)
  state/orders.json (SHA256: def456...)
  state/hash_chain.jsonl (SHA256: ghi789...)
  config/config.json (SHA256: jkl012...)

Location: snapshots/pre_live_v1.zip

RESTORE COMMAND:
  python scripts/snapshot.py restore --name pre_live_v1
```

## List Output
```
SNAPSHOTS
=========
| Name          | Date       | Size  | Version |
|---------------|------------|-------|---------|
| pre_live_v1   | 2024-12-25 | 2.4MB | 1.2.0   |
| daily_backup  | 2024-12-24 | 2.3MB | 1.2.0   |
| before_update | 2024-12-20 | 2.1MB | 1.1.0   |

Total: 3 snapshots, 6.8 MB
```

## Restore Process
```
RESTORE FROM SNAPSHOT
=====================
Source: pre_live_v1 (2024-12-25)

[!] WARNING: This will overwrite current state!

Current positions: 3
Snapshot positions: 5

Proceed? [y/N]: y

RESTORING...
  [OK] state/positions.json
  [OK] state/orders.json
  [OK] state/hash_chain.jsonl
  [OK] config/config.json

RESTORE COMPLETE
Run /preflight to verify system state.
```

## Best Practices
- Snapshot before going live
- Snapshot before config changes
- Snapshot before updates
- Keep at least 5 recent snapshots
- Store offsite copy weekly

## Integration
- Auto-snapshot in /deploy
- Auto-snapshot before /live
- Telegram alert on restore
- Logged to audit chain


