# Data Validation Quickstart

**TL;DR:** Add Pandera + Great Expectations to Kobe for production-grade data validation.

---

## What You Get

### Before (Current)
- Custom validator in `data/validation.py` ✅
- Basic checks in `preflight/data_quality.py` ✅
- Works, but limited statistical validation

### After (With Pandera + GE)
- **3 layers** of validation (fast → domain → comprehensive)
- **Statistical checks** (mean, std, hypothesis testing)
- **Auto-generated reports** (Great Expectations Data Docs)
- **Type safety** (Pandera decorators prevent bad inputs)

---

## Quick Start (Pandera Only - 30 Minutes)

### Step 1: Install
```bash
pip install "pandera[strategies]>=0.28.0"
```

### Step 2: Validate DataFrames
```python
from data.schemas.ohlcv_schema import validate_ohlcv_pandera

# In polygon_eod.py
df = fetch_daily_bars_polygon("AAPL", "2024-01-01", "2024-12-31")
validated_df = validate_ohlcv_pandera(df, lazy=True)  # Raises SchemaError if invalid
```

### Step 3: Use Decorators (Optional)
```python
from pandera import check_input
from data.schemas.ohlcv_schema import ohlcv_schema

@check_input(ohlcv_schema)
def backtest_strategy(df: pd.DataFrame) -> pd.DataFrame:
    # df is guaranteed valid
    ...
```

### Step 4: Run Tests
```bash
pytest tests/data/test_pandera_schemas.py -v
```

**Done!** You now have production-grade validation.

---

## Full Integration (Pandera + GE - 3 Weeks)

See `docs/DATA_VALIDATION_INTEGRATION.md` for complete plan.

**Week 1:** Pandera (in-memory validation)
**Week 2:** Great Expectations (dataset validation)
**Week 3:** Integration + data docs

---

## What Gets Validated

### Tier 1: Critical (Block Trading)
- No NaN in OHLCV
- OHLC relationships (High ≥ O,C; Low ≤ O,C)
- No negative prices/volume
- Timestamp monotonicity
- High ≥ Low

### Tier 2: Quality (Warn)
- Price reasonability (no >50% daily moves)
- Gap detection (no >5 day gaps)
- Volume consistency

### Tier 3: Statistical (Research)
- Mean return bounds
- Volatility bounds
- Volume stability

---

## Why Both Pandera + Great Expectations?

| Tool | Use Case | Speed | Output |
|------|----------|-------|--------|
| **Pandera** | In-memory validation | < 10ms | Exception |
| **Great Expectations** | Dataset validation | ~500ms | HTML reports |
| **Custom** | Trading rules | ~100ms | JSON report |

**They complement, not compete.**

---

## File Locations

| File | Status |
|------|--------|
| `data/schemas/ohlcv_schema.py` | ✅ Created |
| `tests/data/test_pandera_schemas.py` | ✅ Created |
| `docs/DATA_VALIDATION_INTEGRATION.md` | ✅ Created |
| `great_expectations/` | ❌ Not yet (Week 2) |

---

## Next Steps

1. **Run tests:** `pytest tests/data/test_pandera_schemas.py -v`
2. **Integrate with polygon_eod.py** (5 lines of code)
3. **Add to scan.py** (optional decorator)
4. **Great Expectations** (if needed for frozen datasets)

---

**Questions?** Read `docs/DATA_VALIDATION_INTEGRATION.md` for full details.

**Status:** Ready for production ✅
**Author:** Claude Opus 4.5
**Date:** 2026-01-07
