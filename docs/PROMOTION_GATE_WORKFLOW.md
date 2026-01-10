# Promotion Gate Workflow - Human Approval Process

> Last Updated: 2026-01-07
> Status: ACTIVE
> Critical Constraint: APPROVE_LIVE_ACTION = False (NEVER change programmatically)

---

## Overview

Kobe can discover, research, and propose strategy improvements autonomously. However, **NO changes to production parameters can occur without explicit human approval**. This document defines the exact workflow for reviewing and approving bot-proposed changes.

---

## Core Safety Constraint

```python
# research_os/approval_gate.py:29
APPROVE_LIVE_ACTION = False  # NEVER change programmatically
```

This flag MUST remain False at all times except during explicit human-supervised promotion.

---

## The Four-Step Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PROMOTION GATE WORKFLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1: BOT PROPOSES                                                    │
│    └── Bot runs A/B experiment                                          │
│    └── Validates statistical significance (p < 0.05)                    │
│    └── Creates proposal in state/research_os/proposals/                 │
│                                                                          │
│  STEP 2: HUMAN REVIEWS                                                   │
│    └── python scripts/research_os_cli.py approvals --pending            │
│    └── Reviews evidence, sample size, improvement %                     │
│    └── Checks for overfitting signs                                     │
│                                                                          │
│  STEP 3: HUMAN DECIDES                                                   │
│    └── APPROVE: research_os_cli.py approve --id <ID> --approver "Name"  │
│    └── REJECT: research_os_cli.py reject --id <ID> --reason "Why"       │
│                                                                          │
│  STEP 4: MANUAL IMPLEMENTATION                                           │
│    └── Human manually sets APPROVE_LIVE_ACTION = True                   │
│    └── Runs promotion script                                            │
│    └── Immediately sets APPROVE_LIVE_ACTION = False                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Bot Proposes Change

The bot autonomously discovers potential improvements through:
- A/B testing (`research/experiment_analyzer.py`)
- Random parameter experiments (`autonomous/research.py`)
- CuriosityEngine discoveries (`cognitive/curiosity_engine.py`)

### Proposal Requirements

For a proposal to be created, the bot MUST have:

| Requirement | Threshold | Verification |
|-------------|-----------|--------------|
| Sample Size | >= 20 trades | `experiment_analyzer.py` |
| Statistical Significance | p < 0.05 | Two-sample t-test |
| Improvement | >= 5% better than control | P&L comparison |
| Win Rate Delta | >= 3% improvement | Win rate comparison |
| No Degradation | No metric worse by > 2% | All metrics checked |

### Proposal Storage

```
state/research_os/
├── proposals/
│   └── proposal_20260107_param_atr_mult.json
├── pending_approvals.json
└── approval_audit.jsonl
```

### Proposal Format

```json
{
  "proposal_id": "P-20260107-001",
  "created_at": "2026-01-07T14:00:00",
  "parameter_name": "atr_multiplier",
  "current_value": 2.0,
  "proposed_value": 2.5,
  "evidence": {
    "control_trades": 45,
    "experiment_trades": 47,
    "control_win_rate": 0.62,
    "experiment_win_rate": 0.68,
    "control_avg_pnl": 125.50,
    "experiment_avg_pnl": 158.75,
    "improvement_pct": 26.5,
    "p_value": 0.023
  },
  "status": "pending",
  "risk_assessment": "LOW"
}
```

---

## Step 2: Human Reviews

### View Pending Approvals

```bash
python scripts/research_os_cli.py approvals --pending
```

Example output:
```
================================================================================
PENDING APPROVALS
================================================================================

[1] Proposal: P-20260107-001
    Parameter: atr_multiplier
    Change: 2.0 -> 2.5
    Improvement: +26.5% P&L
    Win Rate: 62% -> 68% (+6%)
    Samples: 45 control, 47 experiment
    p-value: 0.023
    Risk: LOW
    Created: 2026-01-07 14:00:00

[2] Proposal: P-20260105-003
    Parameter: ibs_threshold
    Change: 0.08 -> 0.10
    Improvement: +8.2% P&L
    Win Rate: 60% -> 61% (+1%)
    Samples: 52 control, 48 experiment
    p-value: 0.041
    Risk: MEDIUM
    Created: 2026-01-05 10:30:00

================================================================================
```

### Review Checklist

Before approving, verify:

- [ ] **Sample Size Adequate**: At least 20 trades in each group
- [ ] **Statistical Significance**: p-value < 0.05
- [ ] **Meaningful Improvement**: >= 5% improvement in primary metric
- [ ] **No Overfitting Signs**:
  - Improvement not suspiciously large (> 50%)
  - Similar performance across different market conditions
  - Walk-forward validation passed
- [ ] **Risk Assessment**: Understand what could go wrong
- [ ] **Backtest Validation**: Independent backtest confirms improvement

### Deep Dive Commands

```bash
# View full proposal details
python scripts/research_os_cli.py proposal --id P-20260107-001

# View experiment history
python scripts/research_os_cli.py experiments --parameter atr_multiplier

# Run independent backtest validation
python scripts/backtest_dual_strategy.py --param atr_multiplier=2.5 --validate
```

---

## Step 3: Human Decision

### To Approve

```bash
python scripts/research_os_cli.py approve \
    --id P-20260107-001 \
    --approver "John Doe" \
    --notes "Validated with independent backtest. Improvement consistent across regimes."
```

### To Reject

```bash
python scripts/research_os_cli.py reject \
    --id P-20260107-001 \
    --reason "Sample size too small for this parameter. Need 50+ trades minimum."
```

### Decision is Logged

```json
// state/research_os/approval_audit.jsonl
{
  "timestamp": "2026-01-07T15:30:00",
  "proposal_id": "P-20260107-001",
  "action": "approved",
  "approver": "John Doe",
  "notes": "Validated with independent backtest. Improvement consistent across regimes."
}
```

---

## Step 4: Manual Implementation

Even after approval, the change does NOT automatically apply. Implementation requires:

### 4.1 Set Approval Flag (Temporarily)

```python
# Manually edit research_os/approval_gate.py
APPROVE_LIVE_ACTION = True  # TEMPORARY - revert immediately after
```

### 4.2 Run Promotion Script

```bash
python scripts/research_os_cli.py implement --id P-20260107-001
```

### 4.3 Immediately Revert Flag

```python
# Manually edit research_os/approval_gate.py
APPROVE_LIVE_ACTION = False  # CRITICAL: Always revert
```

### 4.4 Verify Implementation

```bash
# Check frozen params updated
cat config/frozen_strategy_params_v2.2.json | grep atr_multiplier

# Run validation
python scripts/backtest_dual_strategy.py --validate
```

---

## Risk Assessment Levels

| Level | Criteria | Additional Review |
|-------|----------|-------------------|
| **LOW** | Single parameter, well-understood, small change | Standard review |
| **MEDIUM** | Multiple parameters, moderate change magnitude | Run additional backtest |
| **HIGH** | Core logic change, large magnitude, limited data | Walk-forward + peer review |
| **CRITICAL** | Risk management parameter, execution logic | CTO approval required |

---

## Rejection Reasons (Common)

| Reason | Action |
|--------|--------|
| "Sample size too small" | Continue experiment for more trades |
| "Overfitting suspected" | Run walk-forward validation |
| "Inconsistent across regimes" | Segment analysis by market condition |
| "Risk too high for benefit" | Reduce parameter change magnitude |
| "Contradicts existing knowledge" | Document reasoning, may still reject |

---

## Audit Trail

All decisions are permanently logged:

```bash
# View approval history
python scripts/research_os_cli.py history --days 30

# Export audit log
cat state/research_os/approval_audit.jsonl
```

---

## Emergency Override (NEVER Use in Normal Operations)

In extreme circumstances only:

```bash
# This bypasses the approval gate - USE ONLY IN EMERGENCIES
python scripts/research_os_cli.py emergency-override \
    --id <proposal_id> \
    --reason "Critical production issue" \
    --override-code <code from CTO>
```

This requires a one-time code from system administrator and is logged with elevated audit priority.

---

## Related Documents

- `docs/KILL_SWITCH_POLICY.md` - Emergency halt procedures
- `docs/FORWARD_TEST_PROTOCOL.md` - Live testing requirements
- `research_os/approval_gate.py` - Implementation
- `scripts/research_os_cli.py` - CLI interface
- `research/experiment_analyzer.py` - A/B testing engine

---

## Policy Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-01-07 | Initial workflow document |
