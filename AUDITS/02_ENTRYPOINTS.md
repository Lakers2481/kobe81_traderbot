# PHASE 2: ENTRYPOINT DISCOVERY

**Generated:** 2026-01-05 20:25 ET
**Auditor:** Claude SUPER AUDIT
**Status:** COMPLETE

---

## EXECUTIVE SUMMARY

**TOTAL RUNNABLE ENTRYPOINTS: 290**

Every file with `if __name__ == '__main__'` block has been catalogued.

---

## ENTRYPOINTS BY DIRECTORY

| Directory | Count | Purpose |
|-----------|-------|---------|
| scripts/ | 176 | Main operational scripts |
| tests/ | 28 | Runnable tests |
| autonomous/ | 13 | Brain & scheduler |
| data/ | 12 | Data providers |
| risk/ | 11 | Risk management |
| tools/ | 8 | Utility tools |
| guardian/ | 7 | Guardian patterns |
| analytics/ | 5 | Performance analytics |
| execution/ | 4 | Broker interfaces |
| portfolio/ | 4 | Portfolio management |
| cognitive/ | 3 | Cognitive brain |
| web/ | 3 | Web interfaces |
| agents/ | 2 | Agent framework |
| monitor/ | 2 | Monitoring |
| testing/ | 2 | Additional tests |
| alerts/ | 1 | Alert system |
| dashboard/ | 1 | Dashboard |
| ml_features/ | 1 | Feature pipeline |
| observability/ | 1 | Observability |
| pipelines/ | 1 | Quant pipeline |
| preflight/ | 1 | Preflight checks |
| reports/ | 1 | Report generation |
| strategies/ | 1 | Strategy execution |

---

## CRITICAL ENTRYPOINTS (Trading Path)

### Daily Operations
| Script | Purpose | Safety |
|--------|---------|--------|
| `scripts/scan.py` | Daily stock scanner | Quality gate |
| `scripts/run_paper_trade.py` | Paper trading | PAPER mode |
| `scripts/run_live_trade_micro.py` | Live trading | 7 safety gates |
| `scripts/runner.py` | 24/7 scheduler | Mode enforcement |
| `scripts/overnight_watchlist.py` | Build Top 5 | No orders |
| `scripts/premarket_validator.py` | Validate gaps | No orders |
| `scripts/opening_range_observer.py` | Observe open | No orders |

### Autonomous Brain
| Script | Purpose | Safety |
|--------|---------|--------|
| `scripts/run_autonomous.py` | Start brain | Paper only |
| `autonomous/run.py` | Run brain | Paper only |
| `autonomous/master_brain_full.py` | Full brain | Paper only |

### Backtesting
| Script | Purpose | Safety |
|--------|---------|--------|
| `scripts/backtest_dual_strategy.py` | Strategy test | No orders |
| `scripts/run_wf_polygon.py` | Walk-forward | No orders |
| `scripts/aggregate_wf_report.py` | WF report | No orders |

### Risk & Compliance
| Script | Purpose | Safety |
|--------|---------|--------|
| `scripts/preflight.py` | Preflight checks | No orders |
| `scripts/reconcile_alpaca.py` | Broker reconcile | No orders |
| `scripts/kill.py` | Emergency halt | Creates kill switch |
| `scripts/resume.py` | Resume trading | Removes kill switch |

### Data Management
| Script | Purpose | Safety |
|--------|---------|--------|
| `scripts/prefetch_polygon_universe.py` | Prefetch data | No orders |
| `scripts/build_universe_polygon.py` | Build universe | No orders |
| `scripts/validate_universe_coverage.py` | Validate data | No orders |

---

## MANIFEST FILE

Full list: `AUDITS/02_ENTRYPOINTS_MANIFEST.json`

Contains 290 entrypoints with:
- File path
- Parent directory
- Has argparse flag

---

## NEXT: PHASE 3 - COMPONENT DISCOVERY

**Signature:** SUPER_AUDIT_PHASE2_2026-01-05_COMPLETE
