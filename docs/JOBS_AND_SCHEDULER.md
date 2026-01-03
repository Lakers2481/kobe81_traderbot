# JOBS_AND_SCHEDULER.md - Scheduled Jobs and Automation

> **Last Updated:** 2026-01-03
> **Primary Scheduler:** `scripts/scheduler_kobe.py`
> **Runner:** `scripts/runner.py`

---

## Daily Schedule (Eastern Time)

| Time (ET) | Job Tag | Script | Purpose |
|-----------|---------|--------|---------|
| 05:30 | `DB_BACKUP` | `backup_state.py` | Database backup |
| 06:00 | `DATA_UPDATE` | `prefetch_polygon_universe.py` | Update EOD data |
| 08:00 | `PRE_GAME` | `premarket_validator.py` | Validate gaps/news |
| 08:15 | `PREGAME_BLUEPRINT` | `generate_pregame_blueprint.py` | Generate trade thesis |
| 09:45 | `FIRST_SCAN` | `scan.py` | Pre-open signal scan |
| 10:00-16:00 | `POSITION_MANAGER` | `run_paper_trade.py` | Position management (every 15 min) |
| 15:30 | `EOD_REPORT` | `pnl.py` | End-of-day P&L |
| 15:45 | `OVERNIGHT_WATCHLIST` | `overnight_watchlist.py` | Build next day watchlist |

---

## Scheduler Scripts

### Master Scheduler

```bash
python scripts/scheduler_kobe.py
```

**File:** `scripts/scheduler_kobe.py`

Runs all scheduled jobs. Uses APScheduler for cron-style scheduling.

### 24/7 Runner

```bash
python scripts/runner.py --mode paper --scan-times 09:35,10:30,15:55
```

**File:** `scripts/runner.py`

Features:
- Single-instance enforcement (PID file)
- Heartbeat monitoring
- Kill switch integration
- State persistence

| Argument | Description |
|----------|-------------|
| `--mode` | `paper` or `live` |
| `--scan-times` | Comma-separated scan times (HH:MM) |
| `--universe` | Path to universe CSV |
| `--cap` | Number of stocks to scan |
| `--approve-live` | Required flag for live mode |

### Job Runner (Windows Task Scheduler)

```bash
python scripts/run_job.py --tag DB_BACKUP --dotenv ./.env
```

**File:** `scripts/run_job.py`

For use with Windows Task Scheduler or cron.

---

## Job Definitions

### DB_BACKUP (05:30 AM)

```bash
python scripts/backup_state.py
```

Backs up:
- `state/*.json`
- `state/*.sqlite`
- `logs/*.jsonl`

Output: `backups/backup_YYYYMMDD_HHMMSS.tar.gz`

---

### DATA_UPDATE (06:00 AM)

```bash
python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_900.csv --start 2015-01-01 --end 2024-12-31
```

Updates EOD cache for all 900 stocks.

---

### PRE_GAME (08:00 AM)

```bash
python scripts/premarket_validator.py --gap-threshold 0.03
```

Validates overnight watchlist:
- Checks for gaps > 3%
- Flags news risk
- Outputs `state/watchlist/today_validated.json`

---

### PREGAME_BLUEPRINT (08:15 AM)

```bash
python scripts/generate_pregame_blueprint.py --cap 900 --top 5 --execute 2
```

Generates comprehensive trade thesis:
- Historical patterns
- Expected move
- Support/resistance
- AI confidence
- Bull/bear cases

Output:
- `reports/pregame_YYYYMMDD.json`
- `reports/pregame_YYYYMMDD.md`

---

### FIRST_SCAN (09:45 AM)

```bash
python scripts/scan.py --cap 900 --deterministic --top3
```

Pre-open signal scan. Signals won't execute until 10:00 AM (kill zone block).

---

### POSITION_MANAGER (10:00 AM - 16:00 PM, every 15 min)

```bash
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv --cap 50 --watchlist-only
```

Manages positions:
- Entry execution at 10:00 AM
- Stop loss monitoring
- Exit execution
- Position updates

---

### EOD_REPORT (15:30 PM)

```bash
python scripts/pnl.py
```

Generates end-of-day P&L summary.

---

### OVERNIGHT_WATCHLIST (15:45 PM)

```bash
python scripts/overnight_watchlist.py --cap 900 --prefetch
```

Builds watchlist for next trading day:
- Top 5 candidates
- Trade of the Day (TOTD)
- Output: `state/watchlist/next_day.json`

---

## Windows Task Scheduler Setup

### Create Scheduled Task

```powershell
# Example: PRE_GAME at 08:00 AM weekdays
schtasks /create /tn "Kobe_PRE_GAME" /tr "python C:\path\to\scripts\run_job.py --tag PRE_GAME --dotenv C:\path\to\.env" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:00
```

### Required Tasks

| Task Name | Time | Days |
|-----------|------|------|
| Kobe_DB_BACKUP | 05:30 | Mon-Fri |
| Kobe_DATA_UPDATE | 06:00 | Mon-Fri |
| Kobe_PRE_GAME | 08:00 | Mon-Fri |
| Kobe_PREGAME_BLUEPRINT | 08:15 | Mon-Fri |
| Kobe_FIRST_SCAN | 09:45 | Mon-Fri |
| Kobe_RUNNER | 10:00 | Mon-Fri |
| Kobe_OVERNIGHT | 15:45 | Mon-Fri |

---

## Monitoring

### Health Check

```bash
curl http://localhost:5000/health
```

### Heartbeat

```bash
cat state/autonomous/heartbeat.json
```

### View Scheduler Status

```bash
python scripts/status.py
```

---

## Kill Switch Integration

All scheduled jobs respect the kill switch:

```python
from core.kill_switch import is_kill_switch_active

if is_kill_switch_active():
    print("Kill switch active, skipping job")
    sys.exit(0)
```

---

## Related Documentation

- [ENTRYPOINTS.md](ENTRYPOINTS.md) - All runnable scripts
- [PROFESSIONAL_EXECUTION_FLOW.md](PROFESSIONAL_EXECUTION_FLOW.md) - Trading flow
- [ROBOT_MANUAL.md](ROBOT_MANUAL.md) - Complete guide
