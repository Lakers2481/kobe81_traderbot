# /validate

Run Kobe's full validation suite - tests, type checks, and code quality.

## Usage
```
/validate [--quick]
```

## What it does
1. Runs pytest with coverage
2. Checks for import errors
3. Validates strategy interface compliance
4. Verifies no lookahead bias in indicators

## Commands
```bash
# Full validation
python -m pytest tests/ -v --tb=short

# Quick smoke test
python -m pytest tests/ -q --tb=line

# Check imports work
python -c "from strategies.connors_rsi2.strategy import ConnorsRSI2Strategy; print('OK')"
python -c "from execution.broker_alpaca import BrokerAlpaca; print('OK')"
```

## Expected Output
- All tests PASS
- No import errors
- Strategy generates valid signals

## On Failure
1. Check test output for specific failures
2. Fix code issues
3. Re-run validation
