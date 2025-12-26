# /quant

High-level quantitative analysis and research dashboard.

## Usage
```
/quant [analysis|research|stats|alpha]
```

## What it does
1. Portfolio-level quant metrics
2. Alpha/beta decomposition
3. Risk-adjusted returns
4. Factor exposures
5. Statistical significance tests
6. Research pipeline status

## Commands
```bash
# Full quant dashboard
python scripts/quant_dashboard.py --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Specific analysis
python scripts/quant_dashboard.py --analysis performance
python scripts/quant_dashboard.py --analysis risk
python scripts/quant_dashboard.py --analysis alpha

# Export research report
python scripts/quant_dashboard.py --export quant_report.html
```

## Quant Metrics

### Performance Metrics
| Metric | Formula | Target |
|--------|---------|--------|
| CAGR | Compound Annual Growth Rate | > 15% |
| Sharpe | (Return - Rf) / StdDev | > 1.0 |
| Sortino | (Return - Rf) / Downside Dev | > 1.5 |
| Calmar | CAGR / Max Drawdown | > 0.5 |
| Information Ratio | Alpha / Tracking Error | > 0.5 |

### Risk Metrics
| Metric | Description | Limit |
|--------|-------------|-------|
| VaR (95%) | Value at Risk | < 3% daily |
| CVaR (95%) | Conditional VaR | < 5% daily |
| Max Drawdown | Peak to trough | < 20% |
| Beta | Market sensitivity | 0.5 - 1.2 |
| Volatility | Annualized std dev | < 20% |

### Alpha Analysis
| Component | Source |
|-----------|--------|
| Total Return | Strategy performance |
| Beta Return | Market exposure contribution |
| Alpha | Excess return over beta |
| Factor Alpha | From size, value, momentum |
| Pure Alpha | Unexplained excess return |

## Output
```
QUANT DASHBOARD - KOBE
======================
Period: 2024-01-01 to 2024-12-25

PERFORMANCE
  CAGR: 18.5%
  Sharpe: 1.32
  Sortino: 1.85
  Calmar: 0.92

RISK
  Volatility: 14.2%
  Max Drawdown: -8.5%
  VaR (95%): 1.8%
  Beta: 0.75

ALPHA DECOMPOSITION
  Total Return: +18.5%
  Market (Beta): +11.2%
  ─────────────────────
  Alpha: +7.3% ***

  t-stat: 2.45 (significant at 95%)

FACTOR EXPOSURES
  Market: 0.75 (long)
  Size: -0.12 (slight large-cap tilt)
  Value: 0.08 (neutral)
  Momentum: 0.35 (momentum tilt)

STRATEGY BREAKDOWN
  RSI-2: +5.2% contribution
  IBS: +2.1% contribution
  Correlation: 0.45
```

## Statistical Tests
| Test | Purpose | Significance |
|------|---------|--------------|
| t-test | Alpha significance | p < 0.05 |
| Durbin-Watson | Autocorrelation | 1.5 < DW < 2.5 |
| Jarque-Bera | Return normality | p > 0.05 |
| ADF | Stationarity | p < 0.05 |

## Research Pipeline
```
[ ] Hypothesis: RSI < 5 better than RSI < 10
    Status: Backtest complete, p=0.03

[ ] Hypothesis: Add momentum filter
    Status: In progress, 60% complete

[x] Hypothesis: IBS + RSI combination
    Status: Validated, deployed
```

## Alerts
- Alpha decay detected (3-month rolling < 0)
- Sharpe dropped below 1.0
- Beta drift outside 0.5-1.2
- Strategy correlation increased > 0.7

## Integration
- Updates daily after market close
- Weekly email report
- Feeds into /regime for adaptation
- Logs to logs/quant_metrics.jsonl
