# Kobe81 Traderbot - Progress Status

**Last Updated:** 2025-12-26 18:00 UTC
**Project:** C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

---

## CURRENT STATUS: Full WF Backtest (MR strict + Donchian 2.5R) Running

Task started: 2025-12-26 (local)
Command:
`python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 60 --cap 900 --outdir wf_outputs_full --cache data/cache --fallback-free --topn-on --regime-on --rsi2-long-max 5 --ibs-long-max 0.15 --crsi-long-max 7 --time-stop-bars 3 --topn-k 5 --min-price 10 --donchian-on --donchian-lookback 55 --donchian-stop-mult 2.0 --donchian-time-stop 20 --donchian-r-mult 2.5 --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env`

Progress: initializing (this run includes RSI2/IBS/CRSI + Donchian)
Estimated completion: 30–90+ minutes
Outputs: `wf_outputs_full/`

Notes:
- MR aims for high WR (strict selection); Donchian targets 2.5R winners.
- After completion, run: `python scripts/aggregate_wf_report.py --wfdir wf_outputs_full` to generate HTML.

**Task ID:** be8b951
**Command:** `python scripts/run_wf_polygon.py --topn-on --universe ... --cap 900 --start 2015-01-01 --end 2024-12-31`
**Progress:** ~Split 3 of 25 (RSI2 strategy)
**Estimated Completion:** 30-60+ minutes remaining

The backtest runs 4 strategies (RSI2, IBS, AND, TOPN) × 25 splits = 100 total runs.
Warnings about "possibly delisted" stocks are expected for IPOs after 2015 (PLTR, COIN, RIVN, etc.).

### Baseline Results (Already Complete)
| Strategy | Win Rate | Profit Factor | Sharpe |
|----------|----------|---------------|--------|
| RSI2     | 40.6%    | 0.93          | -0.12  |
| IBS      | 39.8%    | 0.89          | -0.18  |
| AND      | 40.3%    | 0.91          | -0.15  |

**When TOPN completes:**
1. Results will appear in `wf_outputs/wf_summary_compare.csv` with TOPN row
2. TOPN per-split metrics in `wf_outputs/topn/wf_splits.csv`
3. Generate report: `python scripts/aggregate_wf_report.py --wfdir wf_outputs`

---

## Completed Tasks (This Session)

### 1. Kobe 1% Push - Core Features (COMPLETE)
All config-gated, defaults OFF:

- **Commission/Fees Model** - `backtest/engine.py`
  - Tracks gross_pnl, net_pnl, total_fees
  - Config: `backtest.commissions.enabled: false`

- **LULD/Volatility Clamp** - `execution/broker_alpaca.py`
  - `_apply_clamp()` function with fixed % or ATR-based clamping
  - Config: `execution.clamp.enabled: false`

- **Rate Limiter** - `core/rate_limiter.py`
  - Token bucket (120/min) with exponential backoff
  - Config: `execution.rate_limiter.enabled: false`

- **Earnings Proximity Filter** - `core/earnings_filter.py`
  - Config: `filters.earnings.enabled: false`

- **Metrics Endpoint** - `monitor/health_endpoints.py`
  - GET /metrics returns JSON with KPIs
  - Config: `health.metrics.enabled: true`

- **Regime Filter** - `core/regime_filter.py`
  - SPY SMA(200) trend gate + realized volatility gate
  - Config: `regime_filter.enabled: false`

- **Volatility-Targeted Sizing** - `backtest/engine.py`
  - qty = (risk_pct * equity) / (entry - stop)
  - Config: `sizing.enabled: false`

### 2. Composite Scoring + Top-N (COMPLETE)
Cross-sectional ranking per day:

- **Config** - `config/base.yaml`
  ```yaml
  selection:
    enabled: false
    top_n: 10
    score_weights:
      rsi2: 0.6
      ibs: 0.4
    include_and_guard: true
    min_price: 5.0
  ```

- **run_wf_polygon.py** - Updated with:
  - `--regime-on` and `--topn-on` flags
  - `apply_regime_filter()` function
  - `apply_topn_crosssectional()` with daily ranking
  - TOPN row in wf_summary_compare.csv
  - Separate wf_outputs/topn folder

- **run_showdown_polygon.py** - Updated with:
  - Same regime/topn integration
  - TOPN row in showdown_summary.csv
  - showdown_outputs/topn folder

- **aggregate_wf_report.py** - Updated with:
  - TOPN per-split metrics section
  - Net metrics explanation note

### 3. Robustness Tools (COMPLETE)
- **scripts/optimize.py** - Parameter grid search
- **scripts/monte_carlo.py** - Trade reordering + block bootstrap

### 4. Crypto Backtesting (COMPLETE)
- **data/providers/polygon_crypto.py**
- **data/universe/crypto_top3.csv, crypto_top10.csv**
- **scripts/run_wf_crypto.py, run_showdown_crypto.py**

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `config/base.yaml` | Added selection.include_and_guard, selection.min_price |
| `scripts/run_wf_polygon.py` | Full regime filter + top-N cross-sectional ranking |
| `scripts/run_showdown_polygon.py` | Full regime filter + top-N integration |
| `scripts/aggregate_wf_report.py` | TOPN section + net metrics note |
| `README.md` | Updated with all new features |
| `docs/COMPLETE_ROBOT_ARCHITECTURE.md` | Updated with all new features |

---

## Validation Commands

```bash
# Baseline WF (defaults OFF - unchanged behavior)
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --train-days 252 --test-days 63 --cap 900 --outdir wf_outputs --cache data/cache

# Enable selection in config/base.yaml, then run:
# selection.enabled: true
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 900 --outdir wf_outputs --cache data/cache

# Or force TOPN on via flag:
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 900 --outdir wf_outputs --cache data/cache --topn-on

# Showdown with TOPN:
python scripts/run_showdown_polygon.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 900 --outdir showdown_outputs --cache data/cache --topn-on

# Robustness tools:
python scripts/optimize.py --universe data/universe/optionable_liquid_final.csv --start 2015-01-01 --end 2024-12-31 --cap 100 --outdir optimize_outputs
python scripts/monte_carlo.py --trades wf_outputs/rsi2/split_00/trade_list.csv --iterations 1000 --outdir monte_carlo_outputs
```

---

## Next Steps (If Continuing)

1. Run validation commands to verify all features work correctly
2. Test with `selection.enabled: true` in config
3. Generate side-by-side comparison (AND vs TOPN)
4. Review robustness tool outputs

---

## Session Notes

- All features are config-gated with defaults OFF
- No breaking changes to existing equities pipeline
- Cross-sectional ranking uses shifted indicators (no lookahead)
- TOPN appears as additional row in summary CSVs when enabled
