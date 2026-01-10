# MORNING CHECKLIST - KOBE TRADING SYSTEM
## For: 2026-01-07 (Tuesday)
## Timezone: America/New_York (ET)

---

## PRE-MARKET CHECKLIST (07:00 - 09:00 ET)

### 1. System Health Check (07:00 ET)
```bash
# Check if brain is running
curl http://localhost:8081/health

# Expected response:
# {"status": "healthy", "ready": true, "alive": true}
```
- [ ] Health endpoint returns healthy
- [ ] No KILL_SWITCH file exists

### 2. Check Kill Switch Status (07:05 ET)
```bash
# Should NOT exist
ls state/KILL_SWITCH

# If exists, investigate before removing:
cat state/KILL_SWITCH
```
- [ ] Kill switch is NOT active

### 3. Verify Paper Mode (07:10 ET)
```bash
python -c "from safety.mode import get_trading_mode; print(get_trading_mode())"
```
- [ ] Mode is "paper"
- [ ] PAPER_ONLY is True
- [ ] live_allowed is False

### 4. Run Preflight Check (07:15 ET)
```bash
python scripts/preflight.py --dotenv ./.env
```
- [ ] All 10 checks pass
- [ ] Polygon API key valid
- [ ] Alpaca connection OK

### 5. Validate Watchlist (08:00 ET)
```bash
python scripts/premarket_validator.py
```
- [ ] Overnight watchlist loaded
- [ ] Gap check completed
- [ ] Validated watchlist saved to state/watchlist/today_validated.json

### 6. Check Open Positions (08:15 ET)
```bash
python scripts/reconcile_alpaca.py
```
- [ ] Position count matches
- [ ] No orphan positions
- [ ] No missing local positions

---

## MARKET OPEN (09:30 ET)

### 7. Opening Range Observation (09:30 - 10:00 ET)
**DO NOT TRADE DURING THIS WINDOW**
```bash
python scripts/opening_range_observer.py
```
- [ ] Observe volatility
- [ ] Note gaps and opening moves
- [ ] NO TRADES until 10:00 ET

---

## PRIMARY TRADING WINDOW (10:00 - 11:30 ET)

### 8. First Scan (10:00 ET)
```bash
python scripts/scan.py --cap 900 --deterministic --top5 --markov --markov-prefilter 100
```
- [ ] Scan completes without errors
- [ ] daily_top5.csv generated
- [ ] tradeable.csv generated (top 2)

### 9. Execute Paper Trades (10:05 ET)
```bash
python scripts/run_paper_trade.py --watchlist-only --max-trades 2
```
- [ ] Trades from watchlist only
- [ ] Max 2 trades placed
- [ ] Position sizes correct (2% risk, 20% notional cap)

### 10. Monitor Positions (10:30 ET)
```bash
python scripts/reconcile_alpaca.py
```
- [ ] Positions match
- [ ] Stop losses in place
- [ ] P&L tracking

---

## LUNCH (11:30 - 14:00 ET)
**REDUCED ACTIVITY - NO NEW TRADES**
- [ ] Monitor existing positions only
- [ ] No new scans

---

## POWER HOUR (14:30 - 15:30 ET)

### 11. Secondary Scan (14:30 ET)
```bash
python scripts/scan.py --cap 900 --deterministic --top5
```
- [ ] Power hour scan if daily limit not reached
- [ ] Higher quality bar (Score >= 70)

---

## MARKET CLOSE (15:30 - 16:00 ET)

### 12. Position Management (15:30 ET)
- [ ] Review all open positions
- [ ] No new entries
- [ ] Prepare for overnight

### 13. End of Day Reconciliation (16:05 ET)
```bash
python scripts/reconcile_alpaca.py
```
- [ ] Final position check
- [ ] P&L recorded
- [ ] No discrepancies

### 14. Generate Tomorrow's Watchlist (16:30 ET)
```bash
python scripts/overnight_watchlist.py
```
- [ ] Next day watchlist generated
- [ ] Top 5 candidates saved

### 15. Daily Reflection (17:00 ET)
- [ ] Review today's trades
- [ ] Log lessons learned
- [ ] Check brain learning status

---

## CRITICAL REMINDERS

### DO NOT:
- Trade during 09:30-10:00 ET (opening range)
- Trade during 11:30-14:00 ET (lunch chop)
- Override kill switch without investigation
- Place more than 2 trades per day from watchlist
- Place live trades (PAPER ONLY)

### ALWAYS:
- Check health endpoint before trading
- Verify paper mode is active
- Run preflight before market open
- Reconcile positions after trades
- Log all decisions

---

## EMERGENCY CONTACTS

### Kill Switch Activation
```bash
touch state/KILL_SWITCH
echo "Manual activation - investigating issue" > state/KILL_SWITCH
```

### Kill Switch Deactivation
```bash
# Only after investigation
rm state/KILL_SWITCH
```

### View Logs
```bash
# Recent events
tail -100 logs/events.jsonl | jq .

# Errors only
grep -i error logs/events.jsonl | tail -20 | jq .
```
