# /deploy

Safely deploy updates to Kobe with rollback capability.

## Usage
```
/deploy [--dry-run] [--skip-backup]
```

## What it does
1. Creates backup of current state
2. Runs preflight checks
3. Applies code/config updates
4. Validates deployment
5. Enables rollback if needed

## Pre-Deploy Checklist
- [ ] All tests passing locally
- [ ] Changes reviewed
- [ ] Kill switch ready
- [ ] Backup created
- [ ] Off-hours deployment preferred

## Commands
```bash
# 1. Create backup first
BACKUP_DIR="backups/pre_deploy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r state/ logs/ configs/ "$BACKUP_DIR/"
echo "Backup: $BACKUP_DIR"

# 2. Activate kill switch during deploy
echo "$(date -Iseconds) Deployment in progress" > state/KILL_SWITCH

# 3. Pull latest code (if using git)
git stash
git pull origin main
git stash pop

# 4. Update dependencies
pip install -r requirements.txt --quiet

# 5. Run validation
python -m pytest tests/ -q --tb=line

# 6. Run preflight
python scripts/preflight.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# 7. If all checks pass, remove kill switch
rm state/KILL_SWITCH
echo "✅ Deployment complete"

# 8. Monitor for 15 minutes
echo "Monitoring... Check /logs and /status"
```

## Rollback
If deployment fails:
```bash
# 1. Ensure kill switch is active
touch state/KILL_SWITCH

# 2. Restore from backup
BACKUP_DIR="backups/pre_deploy_YYYYMMDD_HHMMSS"
cp -r "$BACKUP_DIR/state/"* state/
cp -r "$BACKUP_DIR/configs/"* configs/

# 3. Validate
python scripts/preflight.py

# 4. Resume if OK
rm state/KILL_SWITCH
```

## Deployment Windows
- **Preferred**: Weekends or after 4 PM ET
- **Avoid**: Market hours (9:30 AM - 4 PM ET)
- **Never**: During high volatility events

## Post-Deploy Monitoring
- Watch `/logs` for errors
- Check `/status` every 5 min
- Verify `/positions` unchanged
- Monitor `/metrics` for anomalies
