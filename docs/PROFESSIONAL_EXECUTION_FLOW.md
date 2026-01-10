# Professional Execution Flow

> **Version:** 1.0
> **Date:** January 2, 2026
> **Status:** IMPLEMENTING

## Overview

This document defines the professional-grade execution flow for the Kobe Trading System. It covers every edge case and mirrors how institutional traders and quants approach daily execution.

---

## The Problem We're Solving

| Issue | Old Behavior | New Behavior |
|-------|--------------|--------------|
| Morning picks ignored | Scanner took random stocks | Only trade from validated watchlist |
| Trading at open chaos | Could trade at 9:30 | Block trades until 10:00 |
| Stale overnight analysis | Morning picks not re-validated | Fresh premarket validation |
| No fallback plan | If watchlist fails, nothing | Fallback scan with higher bar |
| Wasted compute | Scan 800 stocks multiple times | Overnight Top 5 + targeted scans |

---

## Daily Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PREVIOUS DAY (3:30 PM)                           │
├─────────────────────────────────────────────────────────────────────┤
│  OVERNIGHT_WATCHLIST_BUILD                                          │
│  ├── Scan 800 stocks for NEXT DAY setups                           │
│  ├── Generate Top 5 watchlist + TOTD                               │
│  ├── Save to state/watchlist/next_day.json                         │
│  └── Prefetch data for Top 5 only (saves API calls)                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREMARKET (6:00-8:00 AM)                         │
├─────────────────────────────────────────────────────────────────────┤
│  DATA_UPDATE (6:00 AM)                                              │
│  ├── Refresh EOD data for Top 5 watchlist                          │
│  └── Check for overnight gaps, news, corporate actions             │
│                                                                     │
│  PREMARKET_VALIDATOR (8:00 AM)                                      │
│  ├── Load overnight watchlist                                       │
│  ├── Check each stock:                                              │
│  │   ├── Gap > 3%? → Flag as "GAP_INVALIDATED"                     │
│  │   ├── News event? → Flag as "NEWS_RISK"                         │
│  │   ├── Setup still valid? → Keep on watchlist                    │
│  │   └── Setup improved? → Upgrade priority                        │
│  ├── Generate VALIDATED watchlist (may be 3-5 stocks)              │
│  └── Save to state/watchlist/today_validated.json                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    OPENING RANGE (9:30-10:00 AM)                    │
├─────────────────────────────────────────────────────────────────────┤
│  OPENING_RANGE_OBSERVER (9:30, 9:45 AM)                             │
│  ├── ⛔ NO TRADES ALLOWED - OBSERVE ONLY                           │
│  ├── Log opening prices for watchlist stocks                       │
│  ├── Track: Which are showing strength/weakness?                   │
│  ├── Track: Any breaking out of opening range?                     │
│  └── Save observations to state/watchlist/opening_range.json       │
│                                                                     │
│  WHY NO TRADES?                                                     │
│  ├── First 15-30 min = "Amateur Hour"                              │
│  ├── Algos fighting, gaps filling, fake moves                      │
│  ├── Professionals wait for dust to settle                         │
│  └── Real direction emerges after 10:00 AM                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 FIRST EXECUTION WINDOW (10:00-10:30 AM)             │
├─────────────────────────────────────────────────────────────────────┤
│  PRIMARY_SCAN (10:00 AM)                                            │
│  ├── Scan ONLY validated watchlist stocks (3-5 stocks)             │
│  ├── Check: Did pattern DEVELOP since premarket?                   │
│  ├── Check: Is entry price still valid?                            │
│  ├── Check: Is risk/reward still acceptable (>1.5:1)?              │
│  ├── Rank by: Opening range strength + signal quality              │
│  └── EXECUTE: Up to 2 trades from watchlist                        │
│                                                                     │
│  QUALITY REQUIREMENTS (Watchlist Stocks):                           │
│  ├── Quality Score >= 65                                            │
│  ├── Confidence >= 0.60                                             │
│  ├── R:R >= 1.5:1                                                   │
│  └── Volume > 50% of 20-day average                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │ Watchlist triggered?  │
                    └───────────┬───────────┘
                          │           │
                         YES          NO
                          │           │
                          ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FALLBACK SCAN (10:30 AM)                         │
├─────────────────────────────────────────────────────────────────────┤
│  FALLBACK_SCAN (10:30 AM) - Only if watchlist didn't trigger       │
│  ├── Quick scan of 800 stocks                                       │
│  ├── Look for setups that EMERGED from the open                    │
│  ├── HIGHER QUALITY BAR required:                                  │
│  │   ├── Quality Score >= 75 (vs 65 for watchlist)                 │
│  │   ├── Confidence >= 0.70 (vs 0.60 for watchlist)                │
│  │   ├── R:R >= 2:1 (vs 1.5:1 for watchlist)                       │
│  │   └── Must show strength in opening range                       │
│  └── EXECUTE: Max 1 trade from fallback                            │
│                                                                     │
│  WHY HIGHER BAR?                                                    │
│  ├── Wasn't on our radar overnight = less conviction               │
│  ├── Need extra confirmation from market action                    │
│  └── Protects against random noise trades                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MIDDAY (11:00 AM - 2:00 PM)                      │
├─────────────────────────────────────────────────────────────────────┤
│  LUNCH CHOP ZONE - REDUCED ACTIVITY                                 │
│  ├── Position management only (stops, exits)                       │
│  ├── NO new entries 11:30 AM - 2:00 PM                             │
│  ├── Low volume = fake moves, choppy action                        │
│  └── Professionals go to lunch, algos dominate                     │
│                                                                     │
│  HALF_TIME (12:00 PM)                                               │
│  ├── Review morning trades                                          │
│  ├── Adjust stops if needed                                         │
│  └── Prepare for afternoon session                                 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 POWER HOUR (2:30-3:30 PM)                           │
├─────────────────────────────────────────────────────────────────────┤
│  AFTERNOON_SCAN (2:30 PM)                                           │
│  ├── Second execution window opens                                  │
│  ├── Institutions positioning for close                            │
│  ├── Re-scan watchlist + emerging setups                           │
│  ├── Same quality requirements as morning                          │
│  └── EXECUTE: Up to 1 trade (if daily limit not hit)               │
│                                                                     │
│  SWING_SCANNER (3:30 PM)                                            │
│  ├── Build NEXT DAY watchlist                                       │
│  ├── Scan 800 stocks for overnight setups                          │
│  └── This becomes tomorrow's starting point                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MARKET CLOSE (3:55-4:00 PM)                      │
├─────────────────────────────────────────────────────────────────────┤
│  POSITION_CLOSE_CHECK (3:55 PM)                                     │
│  ├── Any positions to close today?                                  │
│  ├── Check time stops (7-bar rule)                                 │
│  ├── Check if holding overnight is appropriate                     │
│  └── Final position management                                      │
│                                                                     │
│  POST_GAME (4:00 PM)                                                │
│  ├── Generate end-of-day report                                     │
│  ├── Log lessons learned                                            │
│  └── Confirm next day watchlist ready                              │
└─────────────────────────────────────────────────────────────────────┘

---

## Kill Zones (When to Trade)

| Zone | Time (ET) | Action | Reason |
|------|-----------|--------|--------|
| Pre-Open | 9:00-9:30 | OBSERVE | Final prep, no trading |
| Opening Range | 9:30-10:00 | **NO TRADES** | Amateur hour, let dust settle |
| London Close | 10:00-11:00 | **PRIMARY WINDOW** | Best setups develop here |
| Lunch Chop | 11:30-2:00 | **AVOID NEW ENTRIES** | Low volume, fake moves |
| Power Hour | 2:30-3:30 | **SECONDARY WINDOW** | Institutional positioning |
| Close | 3:30-4:00 | MANAGE ONLY | No new entries, manage existing |

---

## Quality Gates by Source

| Source | Quality Score | Confidence | R:R | Max Trades |
|--------|---------------|------------|-----|------------|
| **Watchlist (TOTD)** | >= 60 | >= 0.55 | >= 1.5:1 | Priority |
| **Watchlist (Top 5)** | >= 65 | >= 0.60 | >= 1.5:1 | Up to 2 |
| **Fallback (900 scan)** | >= 75 | >= 0.70 | >= 2.0:1 | Max 1 |
| **Power Hour** | >= 70 | >= 0.65 | >= 1.5:1 | Max 1 |

---

## State Files

| File | Purpose | When Updated |
|------|---------|--------------|
| `state/watchlist/next_day.json` | Tomorrow's Top 5 | 3:30 PM swing scan |
| `state/watchlist/today_validated.json` | Validated watchlist | 8:00 AM premarket |
| `state/watchlist/opening_range.json` | Opening observations | 9:30, 9:45 AM |
| `state/watchlist/execution_log.json` | What we traded and why | After each trade |

---

## Edge Cases Covered

| Scenario | How We Handle It |
|----------|------------------|
| All 5 watchlist stocks gap > 3% | Fallback scan at 10:30, higher quality bar |
| TOTD gaps up 5% | Remove from watchlist, note as "GAP_INVALIDATED" |
| News hits premarket | Flag as "NEWS_RISK", may remove or downgrade |
| Watchlist stock triggers at 9:35 | BLOCK - must wait until 10:00 |
| No signals all day | That's fine - capital preservation |
| 3 watchlist stocks trigger at once | Take best 2 only (daily limit) |
| Power hour setup after 2 daily trades | Skip - respect daily limit |
| Market crashes at open | Regime detection blocks all entries |

---

## Implementation Checklist

- [ ] Create `scripts/overnight_watchlist.py` - 3:30 PM next-day scan
- [ ] Create `scripts/premarket_validator.py` - 8:00 AM validation
- [ ] Create `scripts/opening_range_observer.py` - 9:30-10:00 logging
- [ ] Modify `scripts/scan.py` - Add `--watchlist-only` and `--fallback` modes
- [ ] Modify `scripts/run_paper_trade.py` - Block trades before 10:00 AM
- [ ] Create `risk/kill_zone_gate.py` - Time-based trade blocking
- [ ] Update `scheduler_kobe.py` - New task flow
- [ ] Create state directory `state/watchlist/`

---

## Success Metrics

After implementing this flow, we should see:

| Metric | Before | Target |
|--------|--------|--------|
| Trades from watchlist | ~20% | > 80% |
| Trades in kill zones | Random | > 90% in valid zones |
| Opening range trades | ~30% | 0% |
| Win rate | ~60% | > 65% (better entries) |
| Average R:R | 1.3:1 | > 1.5:1 |

---

## Quant Interview Talking Points

1. **"We use a two-phase watchlist system"** - Overnight generation + morning validation
2. **"We respect market microstructure"** - No trades in first 30 minutes
3. **"We have tiered quality requirements"** - Watchlist vs fallback bars
4. **"We cover every edge case"** - Gaps, news, no signals, etc.
5. **"Capital preservation is the default"** - No trades is a valid outcome
