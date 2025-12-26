# /test

Run unit tests and integration tests.

## Usage
```
/test [--unit] [--integration] [--coverage] [--module MODULE]
```

## What it does
1. Run unit test suite
2. Run integration tests
3. Generate coverage report
4. Test specific modules

## Commands
```bash
# Run all tests
python -m pytest tests/ --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Unit tests only (fast)
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html

# Test specific module
python -m pytest tests/test_strategy.py -v

# Test specific function
python -m pytest tests/test_strategy.py::test_rsi2_signal -v

# Fail fast (stop on first failure)
python -m pytest tests/ -x
```

## Test Categories

### Unit Tests
| Module | Tests | Purpose |
|--------|-------|---------|
| strategies | 25 | Signal generation logic |
| indicators | 18 | RSI, SMA, ATR calculations |
| risk | 12 | PolicyGate limits |
| backtest | 15 | Engine simulation |

### Integration Tests
| Test | Purpose |
|------|---------|
| test_polygon_fetch | API connectivity |
| test_alpaca_auth | Broker auth |
| test_full_pipeline | End-to-end signal flow |
| test_wf_run | Walk-forward execution |

## Output
```
TEST RESULTS
============
tests/unit/test_indicators.py ............ [100%]
tests/unit/test_strategy.py .............. [100%]
tests/unit/test_risk.py ........ [100%]
tests/integration/test_pipeline.py .... [100%]

========================= RESULTS =========================
Passed: 78
Failed: 0
Skipped: 2
Duration: 12.4s

COVERAGE: 84%
  strategies/: 92%
  backtest/: 88%
  risk/: 85%
  execution/: 72%
```

## Critical Tests
These must pass before /live:
- `test_no_lookahead` - Verify indicator shift
- `test_stop_loss_calc` - ATR stop accuracy
- `test_policy_gate` - Budget enforcement
- `test_idempotency` - No duplicate orders

## Integration
- Runs in /validate
- Blocks /deploy if failures
- Coverage report in reports/coverage/
- CI/CD integration ready
