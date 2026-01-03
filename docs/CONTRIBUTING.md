# CONTRIBUTING.md - Documentation and Code Guidelines

> **Last Updated:** 2026-01-03
> **Purpose:** Rules for maintaining documentation and code quality

---

## Documentation Requirements

### Every Change Must Update Docs

| Change Type | Required Updates |
|-------------|------------------|
| New feature | CHANGELOG.md, relevant docs |
| Bug fix | CHANGELOG.md, worklog entry if significant |
| Strategy change | STATUS.md, CLAUDE.md, worklog entry |
| Risk gate change | RISK_REGISTER.md, worklog entry |
| New script | ENTRYPOINTS.md |
| New module | REPO_MAP.md, ARCHITECTURE.md |

---

## File Naming Conventions

### Documentation

| Pattern | Example | Use For |
|---------|---------|---------|
| `UPPERCASE.md` | `STATUS.md` | Primary reference docs |
| `worklog/YYYY-MM-DD__desc.md` | `2026-01-03__fix.md` | Work logs |
| `COMPONENT_NAME.md` | `DATA_PIPELINE.md` | Component docs |

### Code

| Pattern | Example | Use For |
|---------|---------|---------|
| `snake_case.py` | `policy_gate.py` | Python modules |
| `test_*.py` | `test_policy_gate.py` | Test files |
| `run_*.py` | `run_paper_trade.py` | Executable scripts |

---

## Documentation Standards

### Every Claim Needs Evidence

```markdown
GOOD:
- PolicyGate enforces $75/order cap (see risk/policy_gate.py:45)

BAD:
- The system has position limits
```

### Use Tables for Structured Data

```markdown
GOOD:
| Metric | Value |
|--------|-------|
| Win Rate | 64% |

BAD:
The win rate is 64%.
```

### Include Code Examples

```markdown
GOOD:
```python
from strategies.registry import get_production_scanner
scanner = get_production_scanner()
```

BAD:
Use the production scanner from the registry.
```

---

## Code Standards

### No Lookahead

```python
# GOOD: Uses shifted values
signal = df['close'].shift(1) < df['sma200'].shift(1)

# BAD: Uses current bar values
signal = df['close'] < df['sma200']
```

### Use Type Hints

```python
# GOOD
def calculate_size(equity: float, risk_pct: float) -> int:
    return int(equity * risk_pct)

# BAD
def calculate_size(equity, risk_pct):
    return int(equity * risk_pct)
```

### Document Side Effects

```python
def save_positions(positions: list) -> None:
    """
    Save positions to state file.

    Side Effects:
        - Writes to state/positions.json
        - Updates hash chain
    """
```

---

## Worklog Requirements

### When to Create

- Major feature additions
- Bug fixes affecting trading
- Strategy parameter changes
- Risk gate modifications
- Production incidents

### Required Sections

1. Summary (1-2 sentences)
2. Files Changed (table)
3. Commands Run (code block)
4. Tests Verified (checklist)
5. Next Steps (checklist)

See `docs/worklog/TEMPLATE.md` for full template.

---

## Changelog Requirements

### Entry Format

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New feature description

### Changed
- Modified behavior description

### Fixed
- Bug fix description
```

### Version Numbering

| Type | When to Bump | Example |
|------|--------------|---------|
| MAJOR | Breaking changes | 1.0.0 → 2.0.0 |
| MINOR | New features | 1.0.0 → 1.1.0 |
| PATCH | Bug fixes | 1.0.0 → 1.0.1 |

---

## PR Checklist

Before submitting changes:

- [ ] Updated relevant documentation
- [ ] Added changelog entry
- [ ] Created worklog entry (if significant)
- [ ] All tests pass
- [ ] No lookahead bias introduced
- [ ] Risk gates not bypassed
- [ ] Code follows type hints

---

## What NOT to Do

| DON'T | WHY |
|-------|-----|
| Skip documentation | Future developers/AI will be lost |
| Use standalone strategies | Wrong class = 13% WR loss |
| Bypass risk gates | Exposes to excessive loss |
| Commit without changelog | Changes not tracked |
| Guess file locations | Always cite with file:line |

---

## Documentation Categories

### Primary (Always Updated)

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Rules for AI/developers |
| `docs/STATUS.md` | Single Source of Truth |
| `docs/CHANGELOG.md` | Version history |

### Reference (Update on Changes)

| File | Update When |
|------|-------------|
| `docs/ARCHITECTURE.md` | Pipeline changes |
| `docs/ENTRYPOINTS.md` | New scripts |
| `docs/REPO_MAP.md` | New modules |
| `docs/READINESS.md` | Status changes |

### Risk (Update Carefully)

| File | Update When |
|------|-------------|
| `docs/KNOWN_GAPS.md` | Gaps found/resolved |
| `docs/RISK_REGISTER.md` | Risks identified/mitigated |

---

## Related Documentation

- [CHANGELOG.md](CHANGELOG.md) - Version history
- [WORKLOG.md](WORKLOG.md) - Work log index
- [worklog/TEMPLATE.md](worklog/TEMPLATE.md) - Work log template
