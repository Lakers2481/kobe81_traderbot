# SCHEDULER CODEMAP

> Last Updated: 2026-01-07
> Status: Production (462 tasks)

---

## Overview

The Kobe trading system runs a custom 24/7 scheduler that orchestrates 462 tasks across trading, research, maintenance, and learning activities. This document maps the complete scheduler architecture.

---

## Scheduler Entrypoints

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/run_autonomous.py` | CLI entrypoint for 24/7 brain | 1-194 |
| `autonomous/scheduler_full.py` | MASTER_SCHEDULE definition (462 tasks) | 1-871 |
| `autonomous/scheduler.py` | Runtime scheduler with priority queue | 1-864 |
| `autonomous/brain.py` | Master orchestrator + discovery | 1-500+ |
| `autonomous/master_brain_full.py` | Full orchestration with 462 tasks | 1-500+ |
| `autonomous/handlers.py` | Task execution handlers | 1-500+ |
| `autonomous/awareness.py` | Market context & time awareness | - |

---

## Scheduling Framework

**Type:** Custom scheduler (NOT APScheduler, NOT cron)

**Key Features:**
- Time-based: Hardcoded HH:MM task times
- Context-aware: Tasks filtered by market phase, work mode, weekday/weekend
- Priority-based: Tasks sorted 1-10 (1=highest)
- State-persistent: Task state saved to JSON after each execution
- Sequential execution: One task at a time, 60-second cycles
- Timeout protection: 30s default, 300s max (added 2026-01-07)
- Retry logic: Exponential backoff (60s, 120s, 240s... max 1hr)

---

## Key Classes

### autonomous/scheduler.py

| Class | Purpose |
|-------|---------|
| `TaskPriority` | Enum: CRITICAL(1), HIGH(2), NORMAL(3), LOW(4), BACKGROUND(5) |
| `TaskStatus` | Enum: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED |
| `TaskCategory` | Enum: TRADING, RESEARCH, MAINTENANCE, LEARNING, MONITORING |
| `Task` | Dataclass with name, handler, params, priority, status, etc. |
| `TaskQueue` | Priority queue with context-aware filtering |
| `AutonomousScheduler` | Main scheduler with execute_task(), run_one_cycle() |

### autonomous/awareness.py

| Class | Purpose |
|-------|---------|
| `MarketPhase` | Enum: PRE_MARKET, MARKET_OPEN, MARKET_CLOSE, AFTER_HOURS, WEEKEND |
| `WorkMode` | Enum: ACTIVE_TRADING, MONITORING, RESEARCH, MAINTENANCE, SLEEP |
| `MarketContext` | Dataclass with phase, mode, is_trading_day, etc. |
| `ContextBuilder` | Builds MarketContext from current time |

---

## Timezone Handling

| Component | Timezone | Implementation | Status |
|-----------|----------|----------------|--------|
| Core Scheduler | ET | `zoneinfo.ZoneInfo("America/New_York")` | OK |
| Kill Zone Gate | ET | `zoneinfo.ZoneInfo("America/New_York")` | FIXED 2026-01-07 |
| Equities Calendar | ET | `zoneinfo` | OK |
| Crypto Clock | UTC | `zoneinfo` | OK |

---

## State Files

```
state/autonomous/
├── scheduler/
│   └── scheduler_state.json  # Task execution state
├── task_queue.json           # Priority queue state
├── brain_state.json          # Brain metadata
├── heartbeat.json            # Alive signal (updated every cycle)
├── discoveries.json          # Important findings
└── discoveries.log           # Append-only log
```

### heartbeat.json Format
```json
{
  "timestamp": "2026-01-07T12:00:00-05:00",
  "alive": true,
  "cycles": 1234,
  "last_task": "scan_signals",
  "mode": "ACTIVE_TRADING"
}
```

---

## Task Categories by Time

### Pre-Market (6:00-9:30 ET)
- Data refresh
- Universe validation
- Watchlist preparation
- Gap analysis

### Market Open (9:30-10:00 ET)
- Opening range observation (NO TRADES)
- Price monitoring
- Gap validation

### Primary Trading (10:00-11:30 ET)
- Signal scanning
- Trade execution
- Position monitoring

### Lunch (11:30-14:00 ET)
- Research experiments
- Backtesting
- NO new entries

### Power Hour (14:30-15:30 ET)
- Secondary signal scan
- Position management
- Exit evaluation

### Market Close (15:30-16:00 ET)
- Position review
- P&L calculation
- Overnight watchlist build

### After Hours (16:00-20:00 ET)
- Trade analysis
- Learning updates
- Daily reflection

### Overnight (20:00-6:00 ET)
- ML model retraining
- Walk-forward analysis
- Data validation

### Weekend
- Extended research
- Full backtesting
- System maintenance

---

## Reliability Features (Added 2026-01-07)

### Task Timeout
```python
DEFAULT_TASK_TIMEOUT = 30   # seconds
MAX_TASK_TIMEOUT = 300      # 5 minutes max

# Uses ThreadPoolExecutor for cross-platform timeout
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(handler, **task.params)
    result = future.result(timeout=task_timeout)
```

### Retry Logic
```python
max_retries = 3
backoff = 60 * (2 ** retry_count)  # 60s, 120s, 240s
backoff = min(backoff, 3600)       # Cap at 1 hour
```

### Failure Alerts
```python
def _send_failure_alert(self, task: Task) -> None:
    """Send Telegram alert for permanently failed task."""
    from notifications.telegram_alerts import send_critical_alert
    send_critical_alert(f"SCHEDULER FAILURE: {task.name} failed {task.retry_count}x")
```

---

## Important Constants

| Constant | Location | Value | Purpose |
|----------|----------|-------|---------|
| `DEFAULT_TASK_TIMEOUT` | scheduler.py:27 | 30s | Default task timeout |
| `MAX_TASK_TIMEOUT` | scheduler.py:28 | 300s | Maximum task timeout |
| `CYCLE_INTERVAL` | brain.py | 60s | Seconds between cycles |
| `HEARTBEAT_INTERVAL` | brain.py | 60s | Heartbeat update frequency |

---

## Handler Registration

Handlers are registered in `autonomous/handlers.py` and `autonomous/master_brain_full.py`:

```python
self._handlers = {
    "scan_signals": self._handler_scan_signals,
    "check_positions": self._handler_check_positions,
    "run_backtest": self._handler_run_backtest,
    "retrain_models": self._handler_retrain_models,
    # ... 462 total handlers
}
```

---

## Entry Point: run_autonomous.py

```bash
# Start 24/7 brain
python scripts/run_autonomous.py

# Show status
python scripts/run_autonomous.py --status

# Show awareness
python scripts/run_autonomous.py --awareness

# Single cycle (testing)
python scripts/run_autonomous.py --once
```

---

## Monitoring

### Health Check Endpoint
- URL: `http://localhost:8081/health`
- Returns: JSON with heartbeat, task queue, last errors

### Heartbeat File
- Path: `state/autonomous/heartbeat.json`
- Update frequency: Every 60s
- Alert threshold: 5 minutes without update

---

## Known Limitations

1. **Sequential Execution**: Tasks run one at a time (no parallelism)
2. **No DAG Dependencies**: Tasks are independent (no ordering)
3. **Hard-coded Holidays**: Manual 2024-2026 list in calendar module
4. **No Backpressure**: Unlimited pending tasks allowed

---

## See Also

- `docs/ORCHESTRATION_DAG.md` - Proposed DAG-based orchestration
- `docs/SAFETY_GATES.md` - All 7 safety gates documented
- `docs/ARCHITECTURE.md` - System architecture overview
