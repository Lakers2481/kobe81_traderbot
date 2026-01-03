# KNOWN_GAPS.md - Missing Components and Known Issues

> **Last Updated:** 2026-01-03
> **Audit Status:** Complete (READ-ONLY)
> **Critical Blockers:** 0

---

## Summary

| Severity | Count | Status |
|----------|-------|--------|
| CRITICAL | 0 | All resolved |
| HIGH | 2 | Documented, mitigated |
| MEDIUM | 5 | Acceptable for micro-cap |
| LOW | 3 | Enhancement opportunities |

---

## Critical Findings

### 1. PortfolioStateManager

| Attribute | Value |
|-----------|-------|
| **Status** | NOT FOUND |
| **Severity** | MEDIUM |
| **Expected Location** | `portfolio/state_manager.py` |
| **Search Result** | No class named `PortfolioStateManager` exists |
| **Impact** | No central state orchestration |
| **Current Solution** | File-based JSON state in `state/` directory |
| **Recommendation** | Document as "Design Decision" - JSON works for micro-cap |

**Evidence:**
```bash
grep -r "PortfolioStateManager" --include="*.py" .
# Result: 0 matches
```

**Mitigation:**
- `state/positions.json` - Current positions
- `state/order_state.json` - Order records
- `oms/idempotency_store.py` - SQLite duplicate prevention
- Single-instance enforcement via file locking in `runner.py`

---

### 2. EnhancedConfidenceScorer

| Attribute | Value |
|-----------|-------|
| **Status** | NOT FOUND (but ML IS wired) |
| **Severity** | LOW |
| **Expected Location** | `ml_meta/enhanced_scorer.py` |
| **Search Result** | No class named `EnhancedConfidenceScorer` exists |
| **Impact** | None - ML confidence IS wired |
| **Current Solution** | `ml_meta/model.py` provides equivalent functionality |
| **Recommendation** | Remove from readiness blockers |

**Evidence:**
```python
# ml_meta/model.py
class MLConfidenceModel:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Returns probability scores per strategy."""

# scripts/scan.py:1116-1162
ml_conf = ml_model.predict_proba(features)
final_score = blend_scores(signal_score, ml_conf, sentiment)
```

**ML Confidence IS Wired:**
- `ml_meta/model.py` - GradientBoost per strategy
- `scripts/scan.py:1116` - Blends ML + sentiment
- Models in `models/` directory (when trained)

---

## HIGH Severity Gaps

### 3. No Central State Manager

| Attribute | Value |
|-----------|-------|
| **Severity** | HIGH |
| **Impact** | Potential race conditions under concurrent writes |
| **Current Solution** | File-based JSON with single-instance enforcement |
| **Risk** | Low for micro-cap (single instance) |

**Mitigation:**
- `runner.py` enforces single-instance via PID file
- `oms/idempotency_store.py` uses SQLite (ACID compliant)
- Kill switch file mechanism is atomic

---

### 4. ML Models Not Trained

| Attribute | Value |
|-----------|-------|
| **Severity** | HIGH |
| **Impact** | ML confidence scores unavailable until training |
| **Current Solution** | Quality gate threshold lowered (55 vs 70) |
| **Files Affected** | `scripts/scan.py`, `ml_meta/model.py` |

**Mitigation:**
- System operates without ML models (signal score + sentiment only)
- Training scripts exist: `scripts/train_lstm_confidence.py`, `scripts/train_ensemble.py`
- Quality gate threshold configurable in `config/base.yaml`

---

## MEDIUM Severity Gaps

### 5. No Real-Time Position Sync

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Impact** | Position drift between local state and broker |
| **Mitigation** | `scripts/reconcile_alpaca.py` for manual sync |

---

### 6. Limited Error Recovery

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Impact** | Manual intervention required on failures |
| **Mitigation** | Kill switch + Telegram alerts enable fast response |

---

### 7. No Automated Failover

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Impact** | System down if primary process crashes |
| **Mitigation** | Windows Task Scheduler can restart; health endpoints enable monitoring |

---

### 8. Cognitive Brain Not in Hot Path

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Impact** | AI decision support not automated |
| **Mitigation** | By design - Claude is advisory-only |

---

### 9. Options Backtesting Synthetic Only

| Attribute | Value |
|-----------|-------|
| **Severity** | MEDIUM |
| **Impact** | Real options behavior may differ |
| **Mitigation** | Black-Scholes pricing conservative; for hedging analysis only |

---

## LOW Severity Gaps

### 10. No Web Dashboard Authentication

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Impact** | Dashboard accessible without login |
| **Mitigation** | Run on localhost only; not exposed to internet |

---

### 11. Limited Historical Pattern Coverage

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Impact** | Some patterns have small sample sizes |
| **Mitigation** | Bounce analysis database built with 1M+ events (10Y + 5Y) |

---

### 12. No Multi-Broker Support

| Attribute | Value |
|-----------|-------|
| **Severity** | LOW |
| **Impact** | Tied to Alpaca only |
| **Mitigation** | Broker interface abstracted in `execution/broker_base.py` |

---

## Components NOT Required

These are NOT gaps - they were never intended to exist:

| Component | Reason Not Needed |
|-----------|-------------------|
| PortfolioStateManager | File-based JSON sufficient for micro-cap |
| EnhancedConfidenceScorer | ml_meta/model.py provides equivalent |
| Real-time streaming | EOD data sufficient for swing trading |
| Options market data | Synthetic pricing works for hedging analysis |
| Multi-account support | Single account by design |

---

## Gap Resolution Timeline

| Gap | Priority | Target | Status |
|-----|----------|--------|--------|
| Train ML models | HIGH | Next | Pending (scripts ready) |
| Position reconciliation | MEDIUM | Weekly | Automated via scheduler |
| State manager | LOW | Future | Not blocking |
| Dashboard auth | LOW | Never | Localhost only |

---

## Verification Commands

```bash
# Verify no critical gaps blocking trading
python scripts/preflight.py --dotenv ./.env

# Check ML model status
python -c "from ml_meta.model import load_model; m = load_model(); print('ML Model:', 'Loaded' if m else 'Not trained')"

# Verify state file integrity
python scripts/verify_hash_chain.py

# Check broker connectivity
python -c "from execution.broker_alpaca import AlpacaBroker; b = AlpacaBroker(); print(b.get_account())"
```

---

## Related Documentation

- [READINESS.md](READINESS.md) - Production readiness matrix
- [RISK_REGISTER.md](RISK_REGISTER.md) - Risk assessment
- [ARCHITECTURE.md](ARCHITECTURE.md) - Component wiring proof
