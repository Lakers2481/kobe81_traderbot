# CRITICAL FIX LOG - January 2, 2026

> **SEVERITY:** HIGH
> **INCIDENT:** Position sizing error caused 79% capital exposure
> **STATUS:** DOCUMENTED AND FIXED + PROFESSIONAL ALLOCATION SYSTEM IMPLEMENTED

---

## PROFESSIONAL PORTFOLIO ALLOCATION SYSTEM (LIVE)

New modules implemented: `risk/weekly_exposure_gate.py`, `risk/dynamic_position_sizer.py`

| Constraint | Value | Type | Rationale |
|------------|-------|------|-----------|
| **Max per position** | **10%** | HARD | Diversification |
| **Max daily exposure** | **20%** | HARD | 2 positions max per day |
| **Max weekly exposure** | **40%** | HARD | Capital preservation |
| **Max positions/day** | **2** | HARD | Quality over quantity |
| **Max positions/week** | **10** | SOFT | Warning at 8, soft cap at 10 |
| **Watchlist size** | **Top 5** | - | More optionality |
| **Execute** | **Best 2** | - | Only A+ signals (score >= 70) |

### Dynamic Sizing (When Multiple A+ Signals Appear)

| Signals | Allocation Each | Total Daily |
|---------|-----------------|-------------|
| 1 A+ | 10% | 10% |
| 2 A+ | 10% each | 20% |
| 3 A+ | 6.67% each | 20% |
| 4+ A+ | Split 20% evenly (floor 5%) | 20% |

### Budget Recovery

- Budget freed when positions close (hit target/stop)
- Available at **next scan** (10:30 or 15:55 ET), not immediately
- Unused daily budget carries forward within week
- Week resets Friday after close **ONLY if all positions closed**

### State File

`state/weekly_budget.json` tracks all allocations and is persistent across restarts.

---

## ORIGINAL INCIDENT (For Reference)

| Rule | Value | Rationale |
|------|-------|-----------|
| **Max per position** | **10%** of account | Diversification |
| **Max daily exposure** | **20%** of account | 2 positions max per day |
| **Min cash reserve** | **80%** available | Opportunity + safety |

---

## INCIDENT SUMMARY

On January 2, 2026, manual orders were placed that bypassed the automated position sizing system, resulting in:

| Issue | What Happened | Corrected To |
|-------|---------------|--------------|
| PLTR | 291 shares ($49,694 = 47%) | 62 shares ($10.5k = 10%) |
| TSLA | 76 shares ($33,682 = 32%) | 24 shares ($10.5k = 10%) |
| **Total Exposure** | **$83,376 (79%)** | **$21,200 (20%)** |

---

## ROOT CAUSE ANALYSIS

### What Went Wrong

1. **Manual position sizing** used only the 2% risk formula:
   ```
   Shares = (Account × 2%) ÷ (Entry - Stop)
   ```

2. **Forgot the notional cap** (now 10% per position, 20% daily):
   ```
   Max Notional = Account × 10%
   Max Shares = Max Notional ÷ Entry Price
   ```

3. **Orders placed outside the automated system** which has both caps built in.

### Why This Matters (Quant Interview Level)

| Metric | Risk-Only Sizing | Proper Dual-Cap Sizing |
|--------|------------------|------------------------|
| Capital at risk | 2% per trade ✓ | 2% per trade ✓ |
| Capital deployed | UNLIMITED ❌ | Max 10% per position ✓ |
| Total exposure | Could be 100%+ ❌ | Capped at 20%/day ✓ |
| Liquidity | Could be 0% ❌ | Always 80%+ available ✓ |
| Opportunity cost | High (can't take new trades) | Low (always have capital) |

---

## THE CORRECT POSITION SIZING FORMULA

```python
def calculate_position_size(entry_price, stop_loss, account_equity):
    """
    Professional position sizing with DUAL CAPS.

    Cap 1: Risk-based (2% of account at risk)
    Cap 2: Notional-based (10% of account deployed per position)

    ALWAYS take the LESSER of both calculations.
    """
    # CAP 1: Risk-based sizing
    risk_per_share = abs(entry_price - stop_loss)
    risk_dollars = account_equity * 0.02  # 2% risk
    shares_by_risk = int(risk_dollars / risk_per_share)

    # CAP 2: Notional-based sizing (CRITICAL - DON'T SKIP!)
    max_notional = account_equity * 0.10  # 10% max per position
    shares_by_notional = int(max_notional / entry_price)

    # FINAL: Take the LESSER of both
    final_shares = min(shares_by_risk, shares_by_notional)

    return final_shares

# Example that shows why BOTH caps matter:
# Account: $105,000
# Entry: $170, Stop: $163 (tight stop = $7 risk per share)
#
# Risk-only: $2,100 / $7 = 300 shares = $51,000 notional (49%!) ❌
# With cap:  min(300, 62) = 62 shares = $10,540 notional (10%) ✓
```

---

## FIXES APPLIED

### 1. Bracket Orders (Automatic Stops)

**File:** `scripts/run_paper_trade.py`

Changed from simple IOC limit orders to bracket orders:

```python
# OLD (no automatic stops):
rec = place_ioc_limit(decision)

# NEW (automatic stop + target):
bracket_result = place_bracket_order(
    symbol=sym,
    side='buy',
    qty=max_qty,
    limit_price=limit_px,
    stop_loss=float(stop_loss),
    take_profit=take_profit_px,  # 2R target
    time_in_force='gtc',
)
```

### 2. Notional Cap Updated to 10%

**File:** `scripts/run_paper_trade.py` (line 240)

Changed from 20% to 10% per position:
```python
pos_size = calculate_position_size(
    entry_price=limit_px,
    stop_loss=float(stop_loss),
    risk_pct=risk_limits.risk_per_trade_pct,  # 2% from config
    cognitive_multiplier=size_multiplier,
    max_notional_pct=0.10,  # Max 10% of account per position (20% daily total)
)
```

### 3. Policy Gate Limits Updated (Conservative)

**File:** `config/base.yaml`

```yaml
real:
  max_notional_per_order: 11000   # 10% of $105k account
  max_daily_notional: 22000       # 20% total daily exposure
  max_positions: 2                # Max 2 positions per day
  risk_per_trade_pct: 0.02
  max_notional_pct: 0.10          # 10% per position (CRITICAL!)
  max_daily_exposure_pct: 0.20    # 20% total daily (CRITICAL!)
```

---

## RULES TO NEVER REPEAT THIS

### NEVER DO:
1. ❌ Place manual orders outside `run_paper_trade.py`
2. ❌ Calculate position size without BOTH caps
3. ❌ Assume "2% risk" means position size is safe
4. ❌ Deploy >20% of account in total positions per day

### ALWAYS DO:
1. ✅ Use `calculate_position_size()` from `risk/equity_sizer.py`
2. ✅ Verify: `notional < account × 10%` per position
3. ✅ Verify: `total_exposure < account × 20%` per day
4. ✅ Use bracket orders (automatic stops)
5. ✅ Let the automated runner handle order placement
6. ✅ Keep 80% cash available at all times

---

## VERIFICATION COMMANDS

### Check Position Sizing Logic
```bash
python -c "
from risk.equity_sizer import calculate_position_size
size = calculate_position_size(
    entry_price=170.0,
    stop_loss=163.0,
    risk_pct=0.02,
    max_notional_pct=0.10  # 10% per position
)
print(f'Shares: {size.shares}')
print(f'Notional: \${size.notional:,.0f}')
print(f'Risk: \${size.risk_dollars:,.0f}')
print(f'Capped: {size.capped} ({size.cap_reason})')
"
```

### Check Current Positions
```bash
python scripts/reconcile_alpaca.py
```

### Run End-to-End Test
```bash
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_900.csv \
    --start 2025-12-15 --end 2026-01-02 --cap 50 --dotenv .env --cognitive
```

---

## QUANT INTERVIEW TALKING POINTS

If asked about position sizing in an interview:

1. **"We use dual-cap position sizing"** - risk-based AND notional-based
2. **"Risk cap ensures 2% max loss per trade"** - standard Kelly-inspired sizing
3. **"Notional cap ensures 10% max per position"** - prevents concentration
4. **"Total daily exposure capped at 20%"** - always have 80% liquidity
5. **"Max 2 positions per day"** - focus on quality over quantity
6. **"Bracket orders for automatic risk management"** - stops attached at entry

This incident demonstrates:
- Understanding of position sizing pitfalls
- Ability to identify and fix root causes
- Professional documentation standards
- Defense-in-depth risk management
- Willingness to reduce risk when identified

---

## APPENDIX: Files Modified

| File | Change | Status |
|------|--------|--------|
| `scripts/run_paper_trade.py` | Added bracket orders, import | ✅ Complete |
| `config/base.yaml` | Increased notional limits | ✅ Complete |
| `docs/STATUS.md` | Added to NEVER DO table | ✅ Complete |
| `docs/CRITICAL_FIX_20260102.md` | This document | ✅ Complete |
| `CLAUDE.md` | Add position sizing warning | ⏳ Pending |

---

## SESSION 2 FIXES (Jan 2, 2026 - Afternoon)

> **FOCUS:** 24/7 automation, position sync, and scheduler verification

### Issues Found and Fixed

| Issue | Root Cause | Fix Applied |
|-------|------------|-------------|
| Halftime report showing 0 positions | Looking at `state/positions.json` instead of `state/reconcile/positions.json` | Updated `cognitive/game_briefings.py` line 964 |
| Position prices as strings not floats | Broker returns all values as strings | Added type conversion in position parsing |
| Console emoji encoding error | Windows cp1252 can't encode emojis | Replaced emojis with ASCII `[+]` and `[-]` |
| Missing `get_account_info` function | Not implemented in broker module | Added to `execution/broker_alpaca.py` |
| Health check missing dotenv | Environment variables not loaded | Added `load_dotenv()` to health check script |
| Weekly gate missing convenience keys | `get_status()` nested structure | Added top-level `current_exposure_pct` and `open_symbols` |

### Files Modified (Session 2)

| File | Change |
|------|--------|
| `execution/broker_alpaca.py` | Added `get_account_info()` function |
| `cognitive/game_briefings.py` | Fixed position file path, added type conversions |
| `scripts/generate_briefing.py` | Replaced emojis with ASCII |
| `scripts/health_check_full.py` | NEW - Full system health check |
| `risk/weekly_exposure_gate.py` | Added convenience keys to `get_status()` |

### Verification Commands

```bash
# Full system health check
python scripts/health_check_full.py

# Generate halftime report (now shows positions)
python scripts/generate_briefing.py --phase halftime

# Quick integration test
python -c "
from risk.weekly_exposure_gate import WeeklyExposureGate
gate = WeeklyExposureGate()
print(gate.get_status())
"
```

### System Status After Fixes

| Component | Status |
|-----------|--------|
| Scheduler (120 tasks) | ✅ Running (PID 105216) |
| Broker Connection | ✅ Connected ($104,653 equity) |
| Weekly Exposure Gate | ✅ 20.2% exposure, 2 open positions |
| Position Manager | ✅ 3 positions tracked |
| Briefing Engine | ✅ Generating with positions |
| Health Check | ✅ All systems healthy |

### 24/7 Schedule Verified

- **120 total tasks** defined in `scheduler_kobe.py`
- **49 tasks remaining** for today (Jan 2, 2026)
- **21 Windows Tasks** registered as backup
- Next tasks: 13:30 - DIVERGENCE_7, HOLIDAY_RISK_CALIBRATE, INTRADAY_SCAN_1330, POSITION_MANAGER_13
