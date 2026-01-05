# KOBE Gap Report with External Resources

> Generated: 2026-01-05
> Overall Score: 87.5% (105/120 components)
> Critical Gaps: 4 | Nice-to-Have: 6

---

## CRITICAL GAPS (Production Impact)

### Gap 1: Corporate Actions (Dividend/Split Adjustment)

**Status:** MISSING
**Impact:** Historical backtests may be inaccurate on stocks with splits/dividends
**Priority:** HIGH

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| pysystemtrade adjustments | `pysystemtrade/sysdata/fx/spotfx_from_csv.py` | Pattern for adjustment factors |
| Polygon Splits API | `https://polygon.io/docs/stocks/get_v3_reference_splits` | Get split data |
| Polygon Dividends API | `https://polygon.io/docs/stocks/get_v3_reference_dividends` | Get dividend data |
| Yahoo Finance adj close | Built into yfinance `Adj Close` column | Pre-adjusted prices |

**Implementation Approach:**
```python
# data/providers/adjustment.py
def adjust_for_splits(df, symbol):
    splits = polygon.get_splits(symbol)
    for split_date, ratio in splits:
        mask = df.index < split_date
        df.loc[mask, ['open', 'high', 'low', 'close']] /= ratio
    return df

def adjust_for_dividends(df, symbol):
    divs = polygon.get_dividends(symbol)
    for ex_date, amount in divs:
        mask = df.index < ex_date
        factor = 1 - (amount / df.loc[ex_date, 'close'])
        df.loc[mask, ['open', 'high', 'low', 'close']] *= factor
    return df
```

**Estimated Effort:** Medium (1-2 days)

---

### Gap 2: Drawdown Auto-Halt

**Status:** MISSING
**Impact:** No automatic protection against catastrophic drawdowns
**Priority:** HIGH

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| pysystemtrade capital control | `pysystemtrade/sysproduction/capital.py` | Daily capital checks |
| Carver's blog on drawdowns | `https://qoppac.blogspot.com/2016/08/maximum-drawdown.html` | Theory |

**Implementation Approach:**
```python
# risk/drawdown_halt.py
class DrawdownHalt:
    def __init__(self, max_dd_pct=0.15):  # 15% max drawdown
        self.max_dd_pct = max_dd_pct
        self.peak_equity = None

    def check(self, current_equity):
        if self.peak_equity is None:
            self.peak_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)

        dd = (self.peak_equity - current_equity) / self.peak_equity
        if dd > self.max_dd_pct:
            self.trigger_halt(dd)
            return False
        return True

    def trigger_halt(self, dd):
        Path("state/KILL_SWITCH").touch()
        log.critical(f"DRAWDOWN HALT: {dd:.1%} exceeds {self.max_dd_pct:.1%}")
```

**Estimated Effort:** Low (4 hours)

---

### Gap 3: Portfolio Heat Tracking

**Status:** MISSING
**Impact:** Cannot track total portfolio risk exposure in real-time
**Priority:** MEDIUM-HIGH

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| pysystemtrade position tracking | `pysystemtrade/sysproduction/optimal_positions.py` | Position heat |
| Carver's position sizing | `https://qoppac.blogspot.com/2017/04/` | Theory |

**Implementation Approach:**
```python
# risk/portfolio_heat.py
class PortfolioHeat:
    def __init__(self, max_heat_pct=0.10):  # 10% max portfolio risk
        self.max_heat_pct = max_heat_pct

    def calculate_heat(self, positions: List[Position]) -> float:
        total_risk = sum(
            pos.shares * (pos.entry - pos.stop_loss)
            for pos in positions
        )
        return total_risk / self.portfolio_value

    def can_add_position(self, new_risk: float) -> bool:
        current_heat = self.calculate_heat(self.positions)
        return (current_heat + new_risk) <= self.max_heat_pct
```

**Estimated Effort:** Low (4 hours)

---

### Gap 4: Model Registry (MLflow)

**Status:** MISSING
**Impact:** No centralized tracking of ML model versions, experiments, artifacts
**Priority:** MEDIUM

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| MLflow | `https://mlflow.org/docs/latest/index.html` | Model registry |
| MLflow Tracking | `https://mlflow.org/docs/latest/tracking.html` | Experiment tracking |
| Weights & Biases | `https://wandb.ai/` | Alternative to MLflow |

**Implementation Approach:**
```python
# ml_advanced/registry.py
import mlflow

mlflow.set_tracking_uri("file:///state/mlflow")

def log_model(model, metrics, params, name):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name)

def load_production_model(name):
    return mlflow.sklearn.load_model(f"models:/{name}/Production")
```

**Estimated Effort:** Medium (1 day)

---

## NICE-TO-HAVE GAPS

### Gap 5: Strategy Hot-Reload

**Status:** MISSING
**Impact:** Must restart brain to load strategy changes
**Priority:** LOW

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| freqtrade strategy loading | `freqtrade/strategy/strategy_loading.py` | Dynamic import |
| importlib.reload | Python stdlib | Hot reload modules |

**Implementation Approach:**
```python
import importlib
def reload_strategy(strategy_module):
    importlib.reload(strategy_module)
    return strategy_module.Strategy()
```

**Estimated Effort:** Low (2 hours)

---

### Gap 6: Multi-Broker Support

**Status:** PARTIAL (Alpaca only)
**Impact:** Cannot use Interactive Brokers, TD Ameritrade, etc.
**Priority:** LOW (Alpaca sufficient for current needs)

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| ib_insync | `https://github.com/erdewit/ib_insync` | Interactive Brokers |
| alpaca-py | Already using | Alpaca |
| ccxt | `https://github.com/ccxt/ccxt` | Crypto exchanges |

**Implementation Approach:**
```python
# execution/broker_factory.py
def get_broker(name: str) -> BaseBroker:
    if name == "alpaca":
        return AlpacaBroker()
    elif name == "ibkr":
        return IBKRBroker()  # New implementation
    raise ValueError(f"Unknown broker: {name}")
```

**Estimated Effort:** High (1 week per broker)

---

### Gap 7: Bracket Orders

**Status:** PARTIAL (IOC LIMIT only)
**Impact:** Cannot place stop-loss and take-profit atomically
**Priority:** LOW

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| Alpaca Bracket | `https://docs.alpaca.markets/docs/bracket-orders` | OCO orders |

**Implementation Approach:**
```python
# execution/broker_alpaca.py
def place_bracket_order(symbol, qty, side, entry, stop_loss, take_profit):
    return self.api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type='limit',
        limit_price=entry,
        order_class='bracket',
        stop_loss={'stop_price': stop_loss},
        take_profit={'limit_price': take_profit}
    )
```

**Estimated Effort:** Low (4 hours)

---

### Gap 8: Email Alerts

**Status:** MISSING (Telegram only)
**Impact:** No email notifications for critical events
**Priority:** LOW

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| smtplib | Python stdlib | SMTP email |
| SendGrid | `https://sendgrid.com/` | Email API |

**Implementation Approach:**
```python
# notifications/email.py
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body, to_email):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SMTP_FROM
    msg['To'] = to_email
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
```

**Estimated Effort:** Low (2 hours)

---

### Gap 9: Live A/B Testing

**Status:** MISSING
**Impact:** Cannot test strategy variants in production
**Priority:** LOW

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| pysystemtrade instrument weights | Forecast diversification | A/B concept |

**Implementation Approach:**
```python
# strategies/ab_router.py
import random

class ABRouter:
    def __init__(self, strategy_a, strategy_b, split=0.5):
        self.strategy_a = strategy_a
        self.strategy_b = strategy_b
        self.split = split
        self.assignments = {}

    def route_signal(self, symbol):
        if symbol not in self.assignments:
            self.assignments[symbol] = 'A' if random.random() < self.split else 'B'
        return self.strategy_a if self.assignments[symbol] == 'A' else self.strategy_b
```

**Estimated Effort:** Medium (1 day)

---

### Gap 10: Real Options Data

**Status:** MISSING (Synthetic only)
**Impact:** Cannot backtest with real options prices
**Priority:** LOW (synthetic sufficient for learning)

**External Resources:**
| Resource | URL | Notes |
|----------|-----|-------|
| Polygon Options | `https://polygon.io/docs/options` | Real options data (paid) |
| CBOE | `https://www.cboe.com/` | Options data |
| ThetaData | `https://thetadata.net/` | Historical options |

**Estimated Effort:** High (2 weeks + data cost)

---

## PRIORITY MATRIX

```
IMPACT
  ^
  |   [1] Corporate Actions    [2] Drawdown Halt
  |   [3] Portfolio Heat       [4] Model Registry
HIGH|-------------------------------------------
  |   [7] Bracket Orders       [5] Hot-Reload
  |   [8] Email Alerts
MED |-------------------------------------------
  |   [10] Real Options        [6] Multi-Broker
  |                            [9] A/B Testing
LOW |-------------------------------------------
     LOW        MEDIUM          HIGH      EFFORT -->
```

---

## RECOMMENDED PRIORITY ORDER

1. **Drawdown Auto-Halt** - 4 hours, critical safety
2. **Portfolio Heat Tracking** - 4 hours, risk visibility
3. **Corporate Actions** - 1-2 days, backtest accuracy
4. **Bracket Orders** - 4 hours, execution improvement
5. **Model Registry (MLflow)** - 1 day, ML maturity
6. **Email Alerts** - 2 hours, notification diversity
7. **Strategy Hot-Reload** - 2 hours, development velocity
8. **Live A/B Testing** - 1 day, experimentation
9. **Multi-Broker** - 1 week+, optional expansion
10. **Real Options Data** - 2 weeks+, future enhancement

---

*Generated by Claude Code - 2026-01-05*
