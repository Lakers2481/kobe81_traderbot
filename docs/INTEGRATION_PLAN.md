# KOBE 5-Phase Integration Plan

> Generated: 2026-01-05
> Purpose: Close gaps identified in blueprint audit
> Current Score: 87.5% -> Target: 95%+

---

## PHASE 1: CRITICAL SAFETY (Same Day)

**Goal:** Add critical safety features that prevent catastrophic losses

### 1.1 Drawdown Auto-Halt
- **File:** `risk/drawdown_halt.py` (new)
- **Integration:** Wire into `autonomous/master_brain_full.py`
- **Trigger:** If drawdown > 15%, create KILL_SWITCH
- **Test:** `tests/risk/test_drawdown_halt.py`

```python
# risk/drawdown_halt.py
class DrawdownHalt:
    MAX_DD = 0.15  # 15% max drawdown

    def __init__(self, initial_equity: float):
        self.peak = initial_equity
        self.current = initial_equity

    def update(self, equity: float) -> bool:
        self.current = equity
        self.peak = max(self.peak, equity)
        dd = (self.peak - self.current) / self.peak
        if dd > self.MAX_DD:
            self._trigger_halt(dd)
            return False  # HALT
        return True  # OK

    def _trigger_halt(self, dd: float):
        Path("state/KILL_SWITCH").write_text(f"DRAWDOWN_HALT: {dd:.2%}")
        log.critical(f"DRAWDOWN HALT TRIGGERED: {dd:.2%}")
```

### 1.2 Portfolio Heat Tracking
- **File:** `risk/portfolio_heat.py` (new)
- **Integration:** Check before every new position in PolicyGate
- **Max Heat:** 10% of portfolio at risk at any time
- **Test:** `tests/risk/test_portfolio_heat.py`

```python
# risk/portfolio_heat.py
class PortfolioHeat:
    MAX_HEAT = 0.10  # 10% max portfolio risk

    def calculate_heat(self, positions: List[dict], portfolio_value: float) -> float:
        total_risk = sum(
            p['shares'] * abs(p['entry'] - p['stop_loss'])
            for p in positions
        )
        return total_risk / portfolio_value if portfolio_value > 0 else 0

    def can_open(self, new_risk: float, positions: List[dict], pv: float) -> bool:
        current_heat = self.calculate_heat(positions, pv)
        return (current_heat + new_risk / pv) <= self.MAX_HEAT
```

### 1.3 Wire Into PolicyGate
- **File:** `risk/policy_gate.py` (modify)
- **Add:** Check portfolio heat before allowing trade

**Deliverables:**
- [ ] `risk/drawdown_halt.py`
- [ ] `risk/portfolio_heat.py`
- [ ] Modified `risk/policy_gate.py`
- [ ] Tests passing

---

## PHASE 2: DATA ACCURACY (1-2 Days)

**Goal:** Ensure backtests use properly adjusted historical data

### 2.1 Corporate Actions Module
- **File:** `data/adjustments/corporate_actions.py` (new)
- **Source:** Polygon Splits + Dividends APIs
- **Cache:** Store adjustments in `data/cache/adjustments/`

```python
# data/adjustments/corporate_actions.py
class CorporateActions:
    def __init__(self, polygon_client):
        self.client = polygon_client

    def get_split_adjusted(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        splits = self.client.get_splits(symbol)
        for split in splits:
            mask = df.index < split['execution_date']
            df.loc[mask, ['open', 'high', 'low', 'close']] /= split['split_ratio']
            df.loc[mask, 'volume'] *= split['split_ratio']
        return df

    def get_dividend_adjusted(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        divs = self.client.get_dividends(symbol)
        for div in divs:
            if div['ex_dividend_date'] in df.index:
                close = df.loc[div['ex_dividend_date'], 'close']
                factor = 1 - (div['cash_amount'] / close)
                mask = df.index < div['ex_dividend_date']
                df.loc[mask, ['open', 'high', 'low', 'close']] *= factor
        return df
```

### 2.2 Integration with PolygonEOD
- **File:** `data/providers/polygon_eod.py` (modify)
- **Add:** Option to apply adjustments on fetch

### 2.3 Re-Freeze Data Lake
- **Script:** `scripts/refreeze_adjusted.py`
- **Purpose:** Create new frozen datasets with adjustments

**Deliverables:**
- [ ] `data/adjustments/corporate_actions.py`
- [ ] Modified `data/providers/polygon_eod.py`
- [ ] `scripts/refreeze_adjusted.py`
- [ ] Tests passing

---

## PHASE 3: EXECUTION ENHANCEMENT (1 Day)

**Goal:** Improve order execution capabilities

### 3.1 Bracket Orders
- **File:** `execution/broker_alpaca.py` (modify)
- **Add:** `place_bracket_order()` method

```python
def place_bracket_order(
    self,
    symbol: str,
    qty: int,
    side: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float
) -> Order:
    return self.api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='limit',
        limit_price=entry_price,
        time_in_force='day',
        order_class='bracket',
        stop_loss={'stop_price': stop_loss},
        take_profit={'limit_price': take_profit}
    )
```

### 3.2 Slippage Tracker
- **File:** `execution/slippage_tracker.py` (new)
- **Purpose:** Track expected vs actual fill prices

```python
# execution/slippage_tracker.py
class SlippageTracker:
    def __init__(self):
        self.records = []

    def record(self, order_id: str, expected: float, actual: float, shares: int):
        slippage_pct = (actual - expected) / expected
        self.records.append({
            'order_id': order_id,
            'expected': expected,
            'actual': actual,
            'slippage_pct': slippage_pct,
            'slippage_dollars': (actual - expected) * shares
        })

    def summary(self) -> dict:
        if not self.records:
            return {'avg_slippage': 0, 'total_slippage': 0}
        return {
            'avg_slippage': np.mean([r['slippage_pct'] for r in self.records]),
            'total_slippage': sum(r['slippage_dollars'] for r in self.records])
        }
```

**Deliverables:**
- [ ] Bracket orders in `broker_alpaca.py`
- [ ] `execution/slippage_tracker.py`
- [ ] Tests passing

---

## PHASE 4: ML MATURITY (1-2 Days)

**Goal:** Add proper ML model tracking and versioning

### 4.1 MLflow Integration
- **Requirement:** `pip install mlflow`
- **File:** `ml_advanced/mlflow_registry.py` (new)

```python
# ml_advanced/mlflow_registry.py
import mlflow
from pathlib import Path

MLFLOW_URI = "file:///state/mlflow"

class ModelRegistry:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_URI)

    def log_experiment(self, name: str, params: dict, metrics: dict, model):
        mlflow.set_experiment(name)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if hasattr(model, 'save'):
                model.save(Path("temp_model"))
                mlflow.log_artifact("temp_model")
            return mlflow.active_run().info.run_id

    def load_model(self, name: str, version: str = "latest"):
        return mlflow.pyfunc.load_model(f"models:/{name}/{version}")

    def promote_to_production(self, name: str, run_id: str):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=run_id,
            stage="Production"
        )
```

### 4.2 Wire Into Existing ML
- **Modify:** `ml_advanced/lstm_confidence/model.py`
- **Modify:** `ml_advanced/ensemble/ensemble_predictor.py`
- **Add:** MLflow logging to training loops

**Deliverables:**
- [ ] `ml_advanced/mlflow_registry.py`
- [ ] MLflow integration in LSTM
- [ ] MLflow integration in Ensemble
- [ ] `state/mlflow/` directory

---

## PHASE 5: POLISH & EXTRAS (Ongoing)

**Goal:** Nice-to-have features for completeness

### 5.1 Email Alerts
- **File:** `notifications/email.py` (new)
- **Config:** Add SMTP settings to `.env`

### 5.2 Strategy Hot-Reload
- **File:** `strategies/hot_reload.py` (new)
- **Purpose:** Reload strategy without restart

### 5.3 Integration Tests
- **File:** `tests/integration/` (new directory)
- **Add:** End-to-end tests for full trading flow

**Deliverables:**
- [ ] `notifications/email.py`
- [ ] `strategies/hot_reload.py`
- [ ] Integration test suite
- [ ] Updated documentation

---

## INTEGRATION CHECKLIST

### Phase 1 (Safety)
- [ ] Drawdown halt implemented
- [ ] Portfolio heat tracking implemented
- [ ] PolicyGate updated
- [ ] Unit tests passing
- [ ] Brain wired up

### Phase 2 (Data)
- [ ] Corporate actions module
- [ ] Polygon splits API integrated
- [ ] Polygon dividends API integrated
- [ ] Provider updated
- [ ] Data lake re-frozen

### Phase 3 (Execution)
- [ ] Bracket orders working
- [ ] Slippage tracker active
- [ ] Paper trade test passed
- [ ] Documentation updated

### Phase 4 (ML)
- [ ] MLflow installed
- [ ] Registry module created
- [ ] LSTM integrated
- [ ] Ensemble integrated
- [ ] Experiment logged

### Phase 5 (Polish)
- [ ] Email alerts working
- [ ] Hot-reload tested
- [ ] Integration tests passing
- [ ] All docs updated

---

## SUCCESS METRICS

| Phase | Score Before | Score After | Delta |
|-------|-------------|-------------|-------|
| Phase 1 | 87.5% | 89% | +1.5% |
| Phase 2 | 89% | 91% | +2% |
| Phase 3 | 91% | 93% | +2% |
| Phase 4 | 93% | 95% | +2% |
| Phase 5 | 95% | 97%+ | +2% |

**Final Target: 97%+ Blueprint Coverage**

---

## VERIFICATION COMMAND

After each phase, run:
```bash
python tools/verify_robot.py --phase N
```

This verifies all components for that phase are working.

---

*Generated by Claude Code - 2026-01-05*
