# KOBE TRADING SYSTEM - RUNBOOK

> Operational guide for running, monitoring, and troubleshooting the Kobe autonomous trading system.

---

## QUICK START (5 minutes)

### 1. Preflight Check
```bash
python scripts/preflight.py --dotenv ./.env
```
Expected: All 10 checks PASS

### 2. Check Status
```bash
python -m autonomous.run --status
```
Expected: Brain status, research status, awareness context

### 3. Run Demo
```bash
python -m autonomous.run --demo
```
Expected: Full system demonstration in 5 minutes

---

## COMMANDS REFERENCE

### Autonomous Brain Commands

| Command | Description |
|---------|-------------|
| `python -m autonomous.run --start` | Start 24/7 brain |
| `python -m autonomous.run --stop` | Graceful shutdown |
| `python -m autonomous.run --status` | Current status |
| `python -m autonomous.run --demo` | 5-minute demo |
| `python -m autonomous.run --weekend` | Weekend deep research |
| `python -m autonomous.run --awareness` | Market awareness |
| `python -m autonomous.run --research` | Research status |
| `python -m autonomous.run --health` | Health check (exit code) |
| `python -m autonomous.run --tour` | System tour |

### Trading Commands

| Command | Description |
|---------|-------------|
| `python scripts/scan.py --cap 900 --deterministic --top3` | Daily scan |
| `python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv` | Paper trade |
| `python scripts/overnight_watchlist.py` | Build next-day watchlist |
| `python scripts/premarket_validator.py` | Validate watchlist (8 AM) |

### Backtesting Commands

| Command | Description |
|---------|-------------|
| `python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_900.csv --cap 150` | Quick backtest |
| `python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv` | Walk-forward |
| `python scripts/aggregate_wf_report.py --wfdir wf_outputs` | WF report |

### Data Commands

| Command | Description |
|---------|-------------|
| `python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_900.csv` | Prefetch data |
| `python scripts/verify_hash_chain.py` | Verify audit chain |
| `python scripts/reconcile_alpaca.py` | Reconcile positions |

---

## DAILY OPERATIONS

### Pre-Market (7:00-9:30 AM ET)

1. **Check overnight report:**
   ```bash
   cat reports/weekend_morning_*.md | tail -100
   ```

2. **Validate watchlist:**
   ```bash
   python scripts/premarket_validator.py
   ```

3. **Check system health:**
   ```bash
   python -m autonomous.run --health
   ```

### Market Hours (9:30 AM - 4:00 PM ET)

1. **Monitor brain:**
   ```bash
   python -m autonomous.run --status
   ```

2. **View positions:**
   ```bash
   python scripts/show_positions.py
   ```

3. **Check logs:**
   ```bash
   tail -f logs/events.jsonl
   ```

### Post-Market (4:00 PM - 6:00 PM ET)

1. **Run daily reflection:**
   ```bash
   python scripts/daily_reflection.py
   ```

2. **Build tomorrow's watchlist:**
   ```bash
   python scripts/overnight_watchlist.py
   ```

3. **Check P&L:**
   ```bash
   cat reports/daily_*.md | grep "P&L"
   ```

---

## WEEKEND OPERATIONS

### Saturday Morning

1. **Check weekend report:**
   ```bash
   cat reports/weekend_morning_*.md
   ```

2. **Start deep research:**
   ```bash
   python -m autonomous.run --weekend
   ```

3. **Monitor experiments:**
   ```bash
   python -m autonomous.run --research
   ```

### Sunday Evening

1. **Review discoveries:**
   ```bash
   cat state/discovery/ideas.jsonl | python -m json.tool
   ```

2. **Prepare Monday plan:**
   ```bash
   python scripts/prepare_monday.py
   ```

3. **Verify data freshness:**
   ```bash
   python pipelines/data_audit_pipeline.py
   ```

---

## TROUBLESHOOTING

### Brain Not Responding

**Symptoms:** Status shows no heartbeat or stale timestamp

**Solution:**
```bash
# Check for running process
tasklist | findstr python

# Check PID file
cat state/autonomous/kobe.pid

# Force stop if needed
python -m autonomous.run --stop

# Restart
python -m autonomous.run --start
```

### Kill Switch Active

**Symptoms:** All trading blocked, "Kill switch active" in logs

**Solution:**
```bash
# Check why kill switch was activated
cat logs/events.jsonl | findstr KILL

# Remove kill switch (after investigation!)
del state\KILL_SWITCH

# Restart brain
python -m autonomous.run --start
```

### Data Quality Issues

**Symptoms:** Backtest failures, missing symbols

**Solution:**
```bash
# Run data audit
python pipelines/data_audit_pipeline.py

# Refresh specific symbol
python scripts/prefetch_polygon_universe.py --symbols AAPL,MSFT

# Check cache integrity
python scripts/verify_cache.py
```

### Broker Disconnection

**Symptoms:** Orders not submitting, connection errors

**Solution:**
```bash
# Check broker status
python scripts/check_broker.py

# Reconcile positions
python scripts/reconcile_alpaca.py

# If needed, restart with fresh connection
python -m autonomous.run --stop
python -m autonomous.run --start
```

---

## MONITORING

### Log Files

| File | Contents |
|------|----------|
| `logs/events.jsonl` | Structured event log |
| `logs/daily_picks.csv` | Daily trading picks |
| `logs/signals.jsonl` | All generated signals |
| `state/autonomous/heartbeat.json` | Brain heartbeat |
| `state/autonomous/brain_state.json` | Brain state |

### Health Checks

```bash
# Full health check
python -m autonomous.run --health

# Quick status
python -m autonomous.run --status

# Detailed awareness
python -m autonomous.run --awareness
```

### Metrics to Monitor

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| Heartbeat age | < 5 min | > 10 min |
| Cycle count | Increasing | Stalled |
| Error rate | < 1% | > 5% |
| Win rate | 55-65% | < 50% |
| Drawdown | < 10% | > 15% |

---

## EMERGENCY PROCEDURES

### Emergency Stop

```bash
# Create kill switch immediately
echo "EMERGENCY" > state/KILL_SWITCH

# Verify trading stopped
python -m autonomous.run --status
```

### Position Closeout

```bash
# View all positions
python scripts/show_positions.py

# Close all positions (manual)
python scripts/close_all_positions.py --confirm
```

### Data Rollback

```bash
# Find latest good snapshot
ls data/snapshots/

# Restore from snapshot
python scripts/restore_snapshot.py --date 20260101
```

---

## MAINTENANCE

### Weekly Tasks

- [ ] Review weekend report
- [ ] Check experiment results
- [ ] Verify data quality
- [ ] Run full backtest
- [ ] Update documentation if needed

### Monthly Tasks

- [ ] Archive old logs (> 30 days)
- [ ] Verify all 942 tests pass
- [ ] Review and update universe
- [ ] Check API rate limits
- [ ] Update frozen parameters if needed

### Quarterly Tasks

- [ ] Full walk-forward validation
- [ ] Review all risk limits
- [ ] Update ML models
- [ ] Performance review
- [ ] Security audit

---

## CONFIGURATION

### Key Config Files

| File | Purpose |
|------|---------|
| `config/base.yaml` | Main configuration (696 lines) |
| `config/autonomous.yaml` | Autonomous brain config |
| `config/frozen_strategy_params_v2.2.json` | Locked strategy parameters |
| `.env` | API keys (never commit!) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POLYGON_API_KEY` | Yes | Polygon.io API key |
| `ALPACA_API_KEY_ID` | Yes | Alpaca API key |
| `ALPACA_API_SECRET_KEY` | Yes | Alpaca secret |
| `ALPACA_BASE_URL` | Yes | Paper or live URL |
| `TELEGRAM_BOT_TOKEN` | No | Telegram alerts |
| `TELEGRAM_CHAT_ID` | No | Telegram chat ID |

---

## SUPPORT

### Getting Help

1. Check this runbook
2. Review `docs/` directory
3. Search `logs/events.jsonl`
4. Check `CLAUDE.md` for guidance

### Key Documents

- `CLAUDE.md` - AI assistant guidance
- `docs/STATUS.md` - Single source of truth
- `docs/ARCHITECTURE.md` - System architecture
- `docs/INTERVIEW_QA.md` - Q&A reference

---

*Kobe Trading System Runbook v1.0*
