# 24/7 OPERATIONS GUIDE - KOBE TRADING SYSTEM
## Version: 1.0
## Last Updated: 2026-01-06

---

## OVERVIEW

The Kobe trading system operates 24/7 through the Master Brain, which:
- Schedules 462 tasks across weekdays, Saturdays, and Sundays
- Adapts behavior based on market phase and time
- Self-monitors and self-heals
- Learns from every trade

---

## STARTING THE SYSTEM

### Full 24/7 Operation
```bash
# Start the master brain (full visibility)
python run_brain.py

# Or run in background
python run_brain.py > logs/brain_output.log 2>&1 &
```

### View Today's Schedule
```bash
python run_brain.py --schedule
```

### Single Cycle (Testing)
```bash
python run_brain.py --once
```

---

## TASK BREAKDOWN BY TIME

### Weekdays (235 Tasks)

| Time (ET) | Phase | Tasks |
|-----------|-------|-------|
| 04:00-07:00 | Pre-market Early | Research, backtests, experiments |
| 07:00-09:30 | Pre-market Active | Watchlist prep, gap check, preflight |
| 09:30-10:00 | Market Opening | Observe only, NO trades |
| 10:00-11:30 | Market Morning | Scan, trade from watchlist |
| 11:30-14:00 | Market Lunch | Research mode, no new trades |
| 14:00-15:30 | Market Afternoon | Power hour trading |
| 15:30-16:00 | Market Close | Position management |
| 16:00-20:00 | After Hours | Trade analysis, reflection |
| 20:00-04:00 | Night | Walk-forward, model retraining |

### Saturdays (162 Tasks)

| Activity | Description |
|----------|-------------|
| Weekly Review | Analyze week's performance |
| Deep Backtests | Extended walk-forward tests |
| Model Retraining | Retrain ML models with new data |
| Research | Test new hypotheses |
| Cleanup | Prune logs, optimize storage |

### Sundays (65 Tasks)

| Activity | Description |
|----------|-------------|
| Week Preparation | Build watchlist for Monday |
| System Maintenance | Health checks, updates |
| Learning Review | Consolidate lessons |
| Research Summary | Document findings |

---

## SCHEDULED TASK CATEGORIES

### Trading Tasks
- Signal scanning (market hours only)
- Position monitoring
- Exit management
- Reconciliation

### Research Tasks
- Random parameter experiments
- Strategy discovery
- Pattern analysis
- Feature importance

### Learning Tasks
- Trade reflection
- Lesson extraction
- Memory consolidation
- Model updates

### Maintenance Tasks
- Data quality checks
- Log rotation
- State cleanup
- Health monitoring

---

## MONITORING

### Health Endpoint
```bash
# Check system health
curl http://localhost:8081/health
```

### Heartbeat
```bash
# Check last heartbeat
cat state/heartbeat.json
```

### Metrics
```bash
# Get current metrics
curl http://localhost:8081/metrics | jq .
```

### Prometheus Metrics
```bash
# Prometheus format
curl http://localhost:8081/metrics/prometheus
```

---

## WORK MODES

The brain automatically adjusts behavior based on:

### 1. Market Phase
- **Pre-market**: Research focus
- **Market Open**: Observation only
- **Trading Hours**: Active trading
- **After Hours**: Learning focus
- **Night**: Optimization focus

### 2. Day Type
- **Weekday**: Full trading + research
- **Saturday**: Deep research + review
- **Sunday**: Preparation + maintenance

### 3. Special Events
- **FOMC Days**: Reduced activity
- **OpEx Days**: Special handling
- **Holidays**: Research only

---

## AUTOMATIC BEHAVIORS

### Self-Monitoring
- Heartbeat every 30 seconds
- Health endpoint always available
- Automatic restart on crash (if configured)

### Self-Learning
- Every trade creates an episode
- Reflection after each trade
- Daily summary generation
- Weekly pattern extraction

### Self-Healing
- Reconciliation fixes discrepancies
- Idempotency prevents duplicates
- Catch-up for missed exits
- State recovery on restart

---

## COMMON OPERATIONS

### Pause Trading (Keep Brain Running)
```bash
touch state/KILL_SWITCH
echo "Manual pause for maintenance" > state/KILL_SWITCH
```

### Resume Trading
```bash
rm state/KILL_SWITCH
```

### Force Immediate Task
```bash
# Trigger a scan now
python scripts/scan.py --cap 900 --deterministic --top5

# Trigger reconciliation now
python scripts/reconcile_alpaca.py
```

### Check Scheduler Status
```bash
python -c "
from autonomous.scheduler_full import get_scheduler, get_next_tasks
scheduler = get_scheduler()
next_tasks = get_next_tasks(count=5)
for t in next_tasks:
    print(f'{t.scheduled_time}: {t.name}')
"
```

---

## LOG FILES

| File | Purpose | Rotation |
|------|---------|----------|
| `logs/events.jsonl` | All events | Daily |
| `logs/brain_output.log` | Brain stdout | Weekly |
| `logs/errors.log` | Errors only | Daily |
| `logs/trades/` | Trade records | Never |

### Viewing Logs
```bash
# Recent events
tail -50 logs/events.jsonl | jq .

# Errors only
grep -i error logs/events.jsonl | jq .

# Specific event type
grep "signal_generated" logs/events.jsonl | jq .
```

---

## STATE FILES

| File | Purpose | Backup |
|------|---------|--------|
| `state/positions.json` | Open positions | Every trade |
| `state/runner_last.json` | Last run state | Every run |
| `state/idempotency_store.json` | Order dedup | Daily |
| `state/cognitive/` | Learning data | Weekly |

---

## RESTART PROCEDURES

### Clean Restart
```bash
# 1. Stop brain (Ctrl+C or kill process)
# 2. Verify clean state
python scripts/reconcile_alpaca.py
# 3. Restart
python run_brain.py
```

### Recovery Restart
```bash
# 1. Check for kill switch
ls state/KILL_SWITCH

# 2. Check positions
python scripts/reconcile_alpaca.py

# 3. Run catch-up for missed exits
python -c "
from scripts.exit_manager import catch_up_missed_exits
catch_up_missed_exits()
"

# 4. Restart
python run_brain.py
```

---

## ALERTS CONFIGURATION

### Telegram Alerts
Configured in `.env`:
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Alert Types
- Trade executed
- Position closed
- Error occurred
- Kill switch activated
- High drawdown detected

---

## PERFORMANCE OPTIMIZATION

### Reduce Load
- Increase scan interval during research
- Reduce universe cap during testing
- Disable non-essential tasks

### Increase Throughput
- Pre-fetch data before market open
- Cache frequently accessed data
- Use deterministic mode for consistency

---

## TROUBLESHOOTING

### Brain Not Starting
```bash
# Check for lock file
ls state/kobe_runner.lock

# Remove stale lock
rm state/kobe_runner.lock

# Retry
python run_brain.py
```

### Tasks Not Running
```bash
# Check scheduler
python run_brain.py --schedule

# Check if kill switch active
ls state/KILL_SWITCH
```

### Memory Issues
```bash
# Check Python memory
python -c "import tracemalloc; tracemalloc.start(); ..."

# Reduce episode cache
# Edit cognitive/episodic_memory.py max_episodes
```

---

## GOVERNANCE & POLICY DOCUMENTS

The following policy documents govern system operations and human oversight:

| Document | Purpose | Location |
|----------|---------|----------|
| **Kill Switch Policy** | Mandatory halt conditions, activation/deactivation | `docs/KILL_SWITCH_POLICY.md` |
| **Promotion Gate Workflow** | Human approval for parameter changes | `docs/PROMOTION_GATE_WORKFLOW.md` |
| **Forward Test Protocol** | The Gauntlet - 1-3 month validation | `docs/FORWARD_TEST_PROTOCOL.md` |
| **Safety Gates** | All 7 safety mechanisms | `docs/SAFETY_GATES.md` |

### Research OS Approval Process

The bot can propose parameter changes, but **humans MUST approve**:

```bash
# View pending approvals
python scripts/research_os_cli.py approvals --pending

# Approve a proposal
python scripts/research_os_cli.py approve --id <ID> --approver "Name"

# Reject a proposal
python scripts/research_os_cli.py reject --id <ID> --reason "Why"
```

> **CRITICAL**: `APPROVE_LIVE_ACTION = False` must NEVER be changed programmatically.
> See `docs/PROMOTION_GATE_WORKFLOW.md` for complete workflow.

### Live vs Backtest Monitoring

During The Gauntlet forward testing period:

```bash
# Run weekly reconciliation
python scripts/live_vs_backtest_reconcile.py --live-start 2026-01-07 --save
```

See `docs/FORWARD_TEST_PROTOCOL.md` for complete Gauntlet procedures.
