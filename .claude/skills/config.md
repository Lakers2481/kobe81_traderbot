# /config

View and manage configuration with signature tracking.

## Usage
```
/config [--show|--pin|--diff|--validate]
```

## What it does
1. Shows current configuration
2. Displays config pin (SHA256 hash)
3. Compares config to last known pin
4. Validates configuration integrity

## Commands
```bash
# Show config pin
python scripts/show_config_pin.py

# View current configuration
python -c "
import json
from pathlib import Path

config_file = Path('configs/settings.json')
if not config_file.exists():
    print('No settings.json found')
    exit()

config = json.loads(config_file.read_text())
print('=== CONFIGURATION ===')
print(json.dumps(config, indent=2))
"

# Check config signature
python -c "
import sys
sys.path.insert(0, '.')
from core.config_pin import sha256_file
from pathlib import Path

config_file = 'configs/settings.json'
pin_file = Path('state/config_pin.txt')

current_hash = sha256_file(config_file)
print(f'Current config hash: {current_hash[:16]}...')

if pin_file.exists():
    saved_hash = pin_file.read_text().strip()
    print(f'Saved pin:          {saved_hash[:16]}...')
    if current_hash == saved_hash:
        print('✅ Config unchanged')
    else:
        print('⚠️ CONFIG HAS CHANGED since last pin!')
else:
    print('No saved pin found')
"

# Show all config files
python -c "
from pathlib import Path
from core.config_pin import sha256_file

print('=== CONFIG FILES ===')
configs_dir = Path('configs')
for f in configs_dir.glob('*.json'):
    try:
        h = sha256_file(str(f))
        print(f'{f.name:<25} {h[:12]}...')
    except:
        print(f'{f.name:<25} ERROR')

for f in configs_dir.glob('*.py'):
    if '__pycache__' not in str(f):
        print(f'{f.name:<25} (Python)')
"

# Pin current config (save hash)
python -c "
import sys
sys.path.insert(0, '.')
from core.config_pin import sha256_file
from pathlib import Path

config_file = 'configs/settings.json'
pin_file = Path('state/config_pin.txt')
pin_file.parent.mkdir(parents=True, exist_ok=True)

current_hash = sha256_file(config_file)
pin_file.write_text(current_hash)
print(f'Config pinned: {current_hash[:16]}...')
print('New config signature saved to state/config_pin.txt')
"
```

## Configuration Files
| File | Purpose |
|------|---------|
| configs/settings.json | Main strategy settings |
| configs/env_loader.py | Environment loading |
| state/config_pin.txt | Saved config signature |

## Config Pin Workflow
```
1. Develop & test config changes
2. Run /config --validate
3. Run /wf to validate performance
4. Pin config with /config --pin
5. Deploy to production
```

## Why Pin Configs?
- **Audit trail**: Know exactly what config was running
- **Drift detection**: Alert if config changes unexpectedly
- **Rollback**: Restore to known-good configuration
- **Compliance**: Document configuration history
