# /backup

Backup Kobe's state, logs, and configuration.

## Usage
```
/backup [--full|--state|--logs] [--dest PATH]
```

## What it does
1. Archives state files (hash chain, positions)
2. Copies trade logs
3. Saves configuration
4. Creates timestamped backup

## Commands
```bash
# Quick state backup
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy critical files
cp -r state/ "$BACKUP_DIR/state/"
cp -r logs/ "$BACKUP_DIR/logs/"
cp -r configs/ "$BACKUP_DIR/configs/"
cp data/universe/*.csv "$BACKUP_DIR/" 2>/dev/null

echo "Backup created: $BACKUP_DIR"
ls -la "$BACKUP_DIR"

# Full backup with compression
python -c "
import shutil
from datetime import datetime
from pathlib import Path

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_name = f'kobe_backup_{timestamp}'
backup_dir = Path('backups')
backup_dir.mkdir(exist_ok=True)

# Directories to backup
dirs_to_backup = ['state', 'logs', 'configs', 'data/universe']
files_to_backup = ['CLAUDE.md', 'requirements.txt']

# Create archive
archive_path = backup_dir / backup_name
shutil.make_archive(
    str(archive_path),
    'zip',
    '.',
    '.'  # Would need to filter in production
)

print(f'âœ… Backup created: {archive_path}.zip')
"

# List recent backups
ls -lht backups/ | head -10
```

## Backup Contents
| Directory | Contents |
|-----------|----------|
| state/ | Hash chain, kill switch, positions |
| logs/ | Events, trades, errors |
| configs/ | Strategy parameters, gates |
| data/universe/ | Stock universe CSVs |

## Backup Schedule
- **Automatic**: Daily at 18:00 ET (after market close)
- **Before deploy**: Always backup before updates
- **Before live**: Full backup before going live

## Restore
```bash
# Restore from backup
unzip backups/kobe_backup_20251225.zip -d restore_temp/
# Review files, then copy back
```

## Retention
- Daily backups: 7 days
- Weekly backups: 4 weeks
- Monthly backups: 12 months


