# /quality

Run comprehensive quality checks on code, data, and system.

## Usage
```
/quality [--area AREA] [--strict]
```

## What it does
1. Code quality (linting, style, complexity)
2. Data quality (completeness, accuracy)
3. Test coverage
4. Documentation coverage
5. Performance benchmarks

## Commands
```bash
# Full quality check
python scripts/quality_check.py --full --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Code quality only
python scripts/quality_check.py --area code

# Data quality only
python scripts/quality_check.py --area data

# Strict mode (fail on warnings)
python scripts/quality_check.py --strict
```

## Quality Areas

### 1. CODE QUALITY
| Check | Standard |
|-------|----------|
| Linting | flake8, no errors |
| Type hints | All functions typed |
| Complexity | Cyclomatic < 10 |
| Duplication | No copy-paste blocks |
| Dead code | No unused imports/vars |
| Docstrings | All public functions |

### 2. DATA QUALITY
| Check | Standard |
|-------|----------|
| Completeness | No missing required fields |
| Accuracy | Values in expected ranges |
| Consistency | No contradictions |
| Timeliness | Data < 3 days old |
| Uniqueness | No duplicates |
| Validity | Passes schema validation |

### 3. TEST QUALITY
| Metric | Target |
|--------|--------|
| Coverage | > 80% |
| Pass rate | 100% |
| Flaky tests | 0 |
| Test speed | < 60s total |

### 4. SYSTEM QUALITY
| Check | Standard |
|-------|----------|
| Memory usage | < 2GB |
| CPU usage | < 50% avg |
| Disk space | > 10GB free |
| Response time | < 500ms |
| Error rate | < 0.1% |

## Output
```
QUALITY REPORT
==============
Overall Score: 87/100 (B+)

CODE QUALITY: 90/100
  [PASS] Linting: 0 errors
  [PASS] Type hints: 95% coverage
  [WARN] Complexity: 2 functions > 10
  [PASS] No dead code

DATA QUALITY: 85/100
  [PASS] Completeness: 100%
  [WARN] 5 symbols missing recent data
  [PASS] No duplicates
  [PASS] Schema valid

TEST QUALITY: 82/100
  [PASS] Coverage: 84%
  [WARN] 2 slow tests (>5s each)
  [PASS] All tests passing

SYSTEM QUALITY: 92/100
  [PASS] Memory: 1.2GB
  [PASS] Disk: 45GB free
  [PASS] Response: 120ms avg
```

## Quality Gates
| Gate | Threshold | Action |
|------|-----------|--------|
| A+ | 95-100 | Deploy to live |
| A | 90-94 | Deploy to paper |
| B | 80-89 | Review before deploy |
| C | 70-79 | Must fix issues |
| F | <70 | Block all deploys |

## Integration
- Runs in CI/CD pipeline
- Blocks /deploy if score < 80
- Weekly quality trend report
- Telegram alert on score drop
