# START_HERE.md - Documentation Reading Order

> **Last Updated:** 2026-01-03
> **Purpose:** Onboarding guide for new developers and AI assistants

---

## Welcome to Kobe Trading Robot

This document provides the recommended reading order to understand the system quickly.

---

## Quick Start (5 Minutes)

### 1. Read CLAUDE.md First
```
CLAUDE.md (root)
```
- Project overview
- Critical rules (position sizing, kill zones, strategy usage)
- Common commands
- Skill reference

### 2. Check System Status
```
docs/STATUS.md
```
- Single Source of Truth (SSOT)
- Active strategies and parameters
- Performance metrics

### 3. Verify Readiness
```
docs/READINESS.md
```
- Backtest/Paper/Live status
- Critical safeguards
- Verification commands

---

## Understanding the System (30 Minutes)

### Reading Order

| Order | File | Time | Purpose |
|-------|------|------|---------|
| 1 | `CLAUDE.md` | 5 min | Rules and commands |
| 2 | `docs/STATUS.md` | 3 min | Current state |
| 3 | `docs/READINESS.md` | 3 min | Production readiness |
| 4 | `docs/ARCHITECTURE.md` | 10 min | Pipeline wiring |
| 5 | `docs/REPO_MAP.md` | 5 min | Directory structure |
| 6 | `docs/ENTRYPOINTS.md` | 5 min | All runnable scripts |

---

## Deep Dive (1 Hour)

### Full Reading Order

```
1. CLAUDE.md                    # Rules + commands
2. docs/STATUS.md               # SSOT
3. docs/READINESS.md            # Production status
4. docs/ARCHITECTURE.md         # Pipeline wiring
5. docs/REPO_MAP.md             # Directory tree
6. docs/ENTRYPOINTS.md          # All scripts
7. docs/ROBOT_MANUAL.md         # End-to-end guide
8. docs/KNOWN_GAPS.md           # Missing components
9. docs/RISK_REGISTER.md        # Risk assessment
10. docs/PROFESSIONAL_EXECUTION_FLOW.md  # Trading flow
```

---

## Document Categories

### Core Documents (Always Read)
| File | Purpose |
|------|---------|
| `CLAUDE.md` | Mandatory rules for AI/developers |
| `docs/STATUS.md` | Single Source of Truth |
| `docs/READINESS.md` | Production readiness matrix |

### Architecture Documents (Understanding System)
| File | Purpose |
|------|---------|
| `docs/ARCHITECTURE.md` | Pipeline wiring with visual diagrams |
| `docs/REPO_MAP.md` | Full directory structure |
| `docs/ENTRYPOINTS.md` | All 180+ runnable scripts |

### Operational Documents (Daily Use)
| File | Purpose |
|------|---------|
| `docs/PROFESSIONAL_EXECUTION_FLOW.md` | How professionals trade |
| `docs/JOBS_AND_SCHEDULER.md` | Scheduled jobs |
| `docs/CRITICAL_FIX_20260102.md` | Position sizing incident |

### Risk Documents (Safety)
| File | Purpose |
|------|---------|
| `docs/KNOWN_GAPS.md` | Missing components and mitigation |
| `docs/RISK_REGISTER.md` | Risk assessment |

### Living Documents (Ongoing Updates)
| File | Purpose |
|------|---------|
| `docs/WORKLOG.md` | Index of work notes |
| `docs/CHANGELOG.md` | Version history |
| `docs/CONTRIBUTING.md` | Documentation rules |

---

## Key Concepts

### 1. Single Scanner Rule
```
ONLY use: python scripts/scan.py --cap 900 --deterministic --top3
```
- One scanner: `scan.py`
- Uses `DualStrategyScanner` (IBS+RSI + Turtle Soup)
- Never use standalone strategies

### 2. Kill Zones
```
9:30-10:00 AM  → BLOCKED (amateur hour)
10:00-11:30    → PRIMARY WINDOW (best setups)
11:30-14:30    → BLOCKED (lunch chop)
14:30-15:30    → SECONDARY WINDOW (power hour)
```

### 3. Position Sizing
```
final_shares = min(shares_by_risk, shares_by_notional)
- 2% risk cap
- 20% notional cap
NEVER bypass via manual orders
```

### 4. Kill Switch
```bash
# Emergency halt
python scripts/kill.py --reason "Emergency"

# Resume
python scripts/resume.py --confirm
```

---

## Verification Commands

```bash
# Pre-trading check
python scripts/preflight.py --dotenv ./.env

# System status
python scripts/status.py

# Verify strategy
python scripts/backtest_dual_strategy.py --start 2023-01-01 --end 2024-12-31 --cap 150
# Expected: ~64% WR, ~1.60 PF
```

---

## Common Tasks

### Run Daily Scan
```bash
python scripts/scan.py --cap 900 --deterministic --top3
```

### Start Paper Trading
```bash
python scripts/run_paper_trade.py --universe data/universe/optionable_liquid_800.csv --cap 50
```

### View Positions
```bash
python scripts/positions.py
```

### Check P&L
```bash
python scripts/pnl.py
```

---

## Getting Help

### Skills (70 Available)
```
/status     System health
/positions  Current holdings
/pnl        Profit & loss
/kill       Emergency stop
/preflight  Pre-trade checks
```

See `CLAUDE.md` for full skill reference.

### Documentation
```
docs/           # Primary reference
.claude/skills/ # Skill definitions
```

---

## What NOT to Do

| DON'T | WHY | DO INSTEAD |
|-------|-----|------------|
| Use standalone strategies | Wrong class = 13% WR loss | Use `DualStrategyScanner` |
| Place manual orders | Bypasses risk gates | Use `run_paper_trade.py` |
| Trade 9:30-10:00 | Amateur hour | Wait for 10:00 AM |
| Skip preflight | Miss critical issues | Always run preflight |
| Ignore kill switch | Continues trading | Respect emergency halt |

---

## Related Documentation

- [CLAUDE.md](../CLAUDE.md) - Mandatory rules
- [STATUS.md](STATUS.md) - Single Source of Truth
- [ARCHITECTURE.md](ARCHITECTURE.md) - System wiring
- [ROBOT_MANUAL.md](ROBOT_MANUAL.md) - Complete guide
