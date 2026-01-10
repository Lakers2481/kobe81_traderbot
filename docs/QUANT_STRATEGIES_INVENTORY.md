# QUANTITATIVE STRATEGIES INVENTORY
**Renaissance Technologies / Jim Simons Style**

Generated: 2026-01-09

---

## EXECUTIVE SUMMARY

Your robot has **91+ quantitative alpha factors** plus **multi-strategy frameworks** that can be tested autonomously by the AI system.

**Total Arsenal:**
- 91 pre-built alpha factors (momentum, mean reversion, volatility, volume, technical, cross-sectional, pattern)
- 3 production strategies (IBS+RSI, Turtle Soup, Dual Strategy)
- Medallion-inspired orchestrator (multi-strategy portfolio system)
- Pattern discovery engine (unsupervised ML)
- Reinforcement learning trading agent
- Pairs trading framework

---

## 1. PRODUCTION STRATEGIES (READY TO TRADE)

### 1.1 IBS + RSI Mean Reversion
**File:** `strategies/ibs_rsi/strategy.py`
**Type:** Mean Reversion
**Performance:** 59.9% WR, 1.46 PF (verified 2024-12-31)

**Rules:**
- Entry: IBS < 0.08 AND RSI(2) < 5 AND Close > SMA(200)
- Exit: ATR(14) × 2 stop OR 7-bar time stop
- Side: LONG only

**Backtestable:**
```bash
python scripts/backtest_dual_strategy.py --universe data/universe/optionable_liquid_900.csv --start 2023-01-01 --end 2024-12-31
```

### 1.2 ICT Turtle Soup (Sweep Detector)
**File:** `strategies/ict/turtle_soup.py`
**Type:** Failed Breakout / Liquidity Sweep
**Performance:** 61.0% WR, 1.37 PF (verified 2024-12-31)

**Rules:**
- Entry: Sweep of 20-day low by ≥0.3 ATR AND reversal
- Exit: ATR(14) × 2 stop OR 7-bar time stop
- Side: LONG only

### 1.3 Dual Strategy Scanner (PRODUCTION)
**File:** `strategies/dual_strategy/combined.py`
**Type:** Hybrid (IBS+RSI + Turtle Soup)
**Performance:** Best of both, quality gate filtered

**Quality Gate:**
- Min Score: 70
- Min Confidence: 0.60
- Min R:R: 1.5:1
- Markov boost: +5-10% for agreeing signals

---

## 2. ALPHA LIBRARY (91 FACTORS)

### 2.1 Momentum Alphas (16)

| Alpha | Description | Timeframes |
|-------|-------------|------------|
| `mom_Nd` | Simple momentum (close/close - 1) | 5, 10, 20, 40, 60, 120, 252d |
| `roc_Nd` | Rate of change (percentage) | 5, 10, 20d |
| `accel_Nd` | Acceleration (momentum of momentum) | 5, 10, 20d |
| `tsmom_Nd` | Time-series momentum (excess return) | 20, 60, 252d |

**Example Usage:**
```python
from research.alpha_library import AlphaLibrary
lib = AlphaLibrary()
alphas = lib.compute_category(price_df, 'momentum')
```

### 2.2 Mean Reversion Alphas (20)

| Alpha | Description | Variants |
|-------|-------------|----------|
| `rsi_Nd` | RSI indicator | 2, 3, 5, 7, 14d |
| `ibs` | Internal Bar Strength: (close-low)/(high-low) | - |
| `dist_ma_Nd` | Distance from moving average | 5, 10, 20, 50, 200d |
| `bb_pctb_Nd` | Bollinger Band %B | 10, 20, 50d |
| `zscore_Nd` | Z-score of returns | 5, 10, 20, 60d |
| `consec_down` | Consecutive down days | - |
| `consec_up` | Consecutive up days | - |

### 2.3 Volatility Alphas (14)

| Alpha | Description | Variants |
|-------|-------------|----------|
| `atr_Nd` | Average True Range | 5, 10, 14, 20d |
| `rvol_Nd` | Realized volatility | 5, 10, 20, 60d |
| `vol_regime` | Vol regime classifier | - |
| `intraday_range` | (high-low)/close | - |
| `gap_size` | Open gap size | - |
| `vol_expansion` | Vol expansion detector | - |
| `vol_contraction` | Vol contraction (squeeze) | - |
| `gk_vol_20d` | Garman-Klass volatility | - |

### 2.4 Volume Alphas (11)

| Alpha | Description | Variants |
|-------|-------------|----------|
| `vol_ma_ratio_Nd` | Volume vs MA | 5, 10, 20d |
| `vwap_dist` | Distance from VWAP | - |
| `obv` | On-Balance Volume | - |
| `cmf` | Chaikin Money Flow | - |
| `vwap_slope` | VWAP trend | - |
| `volume_price_corr` | Vol-price correlation | - |

### 2.5 Technical Alphas (16)

| Alpha | Description | Variants |
|-------|-------------|----------|
| `macd` | MACD crossover | - |
| `stoch_Nd` | Stochastic oscillator | 5, 14d |
| `cci_Nd` | Commodity Channel Index | 14, 20d |
| `williams_r` | Williams %R | - |
| `adx` | Average Directional Index | - |
| `aroon` | Aroon indicator | - |
| `trix` | Triple Exponential Average | - |

### 2.6 Cross-Sectional Alphas (6)

| Alpha | Description |
|-------|-------------|
| `rank_return_Nd` | Rank by returns |
| `rank_volume` | Rank by volume |
| `rank_volatility` | Rank by volatility |
| `zscore_cross_sectional` | Cross-sectional z-score |
| `percentile_rank` | Percentile rank |
| `relative_strength_vs_sector` | Sector-relative strength |

### 2.7 Pattern Alphas (8)

| Alpha | Description |
|-------|-------------|
| `double_bottom` | Double bottom detector |
| `double_top` | Double top detector |
| `head_shoulders` | H&S pattern |
| `flag_pattern` | Bull/bear flags |
| `wedge_pattern` | Rising/falling wedges |
| `triangle_pattern` | Triangle breakouts |
| `cup_handle` | Cup & handle |
| `gap_fill` | Gap fill signals |

---

## 3. MEDALLION-STYLE ORCHESTRATOR

**File:** `strategies/medallion/medallion_orchestrator.py`

### Renaissance-Inspired Multi-Strategy System

**Configuration:**
- 12.5-20x leverage (Renaissance standard)
- 3,500+ positions (Renaissance scale)
- 50.75% win rate target (barely above coin flip)
- 1-2 day average holding period

**Scaled-Down for Retail:**
- 5x max leverage (conservative)
- 20-50 positions (manageable)
- 52% min confidence (achievable)
- Daily rebalancing

**Modes:**
| Mode | Leverage | Positions | Use Case |
|------|----------|-----------|----------|
| CONSERVATIVE | 2x | 10 | Risk-averse |
| MODERATE | 5x | 20 | Balanced |
| AGGRESSIVE | 10x | 50 | High conviction |
| MEDALLION | 12.5x | 100+ | Full Renaissance |

**How It Works:**
1. Query HMM regime detector for market state
2. Generate signals from ALL 91 alphas
3. Filter through risk engine (correlation, sector limits)
4. Optimize portfolio for diversification
5. Execute with smart order routing
6. Learn from outcomes (reflection engine)

**Run It:**
```bash
python -m strategies.medallion.medallion_orchestrator --mode MODERATE --capital 100000
```

---

## 4. PATTERN DISCOVERY ENGINE

**File:** `ml/alpha_discovery/pattern_miner/`

### Unsupervised ML Pattern Detection

**Techniques:**
- K-Means clustering
- DBSCAN density clustering
- Hierarchical clustering
- Matrix Profile (motif discovery)
- Wavelet transforms
- Symbolic pattern mining

**Discovers:**
- Price patterns (head & shoulders, flags, etc.)
- Volume patterns (accumulation, distribution)
- Multi-day patterns (3-5-7 day sequences)
- Intraday patterns (if minute data available)

**Run It:**
```bash
python scripts/find_unique_patterns.py --universe data/universe/optionable_liquid_900.csv
```

---

## 5. REINFORCEMENT LEARNING AGENT

**File:** `ml/alpha_discovery/rl_agent/agent.py`

### Learns Optimal Trading Policy

**Algorithms:**
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- A2C (Advantage Actor-Critic)

**State Space:**
- OHLCV features (20-60 bars)
- Technical indicators (RSI, MACD, ATR)
- Position info (holding, P&L)
- Market regime

**Action Space:**
- BUY (enter long)
- SELL (exit)
- HOLD (do nothing)

**Reward Function:**
- Sharpe ratio maximization
- Drawdown penalty
- Transaction cost penalty

**Train It:**
```bash
python -m ml.alpha_discovery.rl_agent.agent --mode train --episodes 10000
```

---

## 6. PAIRS TRADING FRAMEWORK

**File:** `strategies/pairs_trading/`

### Statistical Arbitrage

**Methodology:**
- Cointegration testing (Engle-Granger)
- Spread calculation (z-score)
- Entry: z-score > 2.0 or < -2.0
- Exit: z-score returns to mean

**Universe:**
- Same-sector pairs
- Correlated stocks (0.7+ correlation)
- Market-neutral strategy

---

## 7. ALPHA FACTORY WORKFLOW

**File:** `research/alpha_factory.py`

### Qlib-Inspired Alpha Research Pipeline

**YAML-Based Workflow:**
```yaml
data:
  universe: optionable_liquid_900.csv
  start_date: 2020-01-01
  end_date: 2024-12-31

features:
  alpha_categories: [momentum, mean_reversion, volatility]
  lookback_windows: [5, 10, 20, 60]
  normalize: true

model:
  model_type: lightgbm
  train_ratio: 0.7
  target_horizon: 5  # days ahead

backtest:
  initial_capital: 100000
  position_sizing: kelly
```

**Run It:**
```bash
python -m research.alpha_factory --config config/alpha_workflows/momentum.yaml
```

**Output:**
- Feature importance report
- Model performance metrics
- Backtest results
- Walk-forward analysis
- Production-ready signals

---

## 8. WHAT THE AI CAN TEST AUTONOMOUSLY

### 8.1 Alpha Mining (Autonomous Research Engine)

**File:** `autonomous/research.py`

**What It Does:**
- Tests random combinations of the 91 alphas
- Tries different lookback windows (5, 10, 20, 60, 120 days)
- Tests different entry/exit thresholds
- Backtests each combination
- Logs results to `state/research/experiments.jsonl`

**Runs Automatically:**
```bash
# 24/7 autonomous operation
python scripts/run_autonomous.py

# Or one-time research cycle
python scripts/run_autonomous.py --mode research --once
```

### 8.2 Walk-Forward Testing

**Automatically Tests:**
- All 91 alphas across train/test splits
- Parameter stability across time
- Overfitting detection
- Degradation analysis (train → test performance)

**Run It:**
```bash
python scripts/run_wf_polygon.py --universe data/universe/optionable_liquid_900.csv --start 2020-01-01 --end 2024-12-31
```

### 8.3 Multi-Factor Combinations

**AI Tests:**
- 2-factor combos (momentum + mean reversion)
- 3-factor combos (momentum + vol + volume)
- Ensemble models (XGBoost, LightGBM)
- Neural networks (LSTM, Transformer)

### 8.4 Regime-Adaptive Strategies

**AI Tests:**
- Different alphas for different regimes (BULL/BEAR/NEUTRAL)
- Leverage multipliers per regime
- Position sizing adjustments
- Sector rotation

---

## 9. HOW TO USE (EXAMPLES)

### Example 1: Test All Momentum Alphas

```bash
python -c "
from research.alpha_library import AlphaLibrary
import pandas as pd

lib = AlphaLibrary()

# Load price data
df = pd.read_parquet('data/cache/polygon/AAPL.parquet')

# Compute all momentum alphas
mom_alphas = lib.compute_category(df, 'momentum')

# See results
print(mom_alphas.tail())
"
```

### Example 2: Run Alpha Factory

```bash
# Create workflow config
cat > my_alpha.yaml << EOF
data:
  universe: data/universe/optionable_liquid_900.csv
  start_date: 2023-01-01
  end_date: 2024-12-31

features:
  alpha_categories: [momentum, mean_reversion]
  lookback_windows: [10, 20]

model:
  model_type: xgboost
  target_horizon: 3
EOF

# Run it
python -m research.alpha_factory --config my_alpha.yaml
```

### Example 3: Start Autonomous Alpha Discovery

```bash
# Let the AI discover alphas 24/7
python scripts/run_autonomous.py

# Check discoveries
cat state/research/experiments.jsonl | tail -20
```

### Example 4: Train RL Agent

```bash
# Train RL agent on historical data
python -m ml.alpha_discovery.rl_agent.agent \
    --mode train \
    --episodes 10000 \
    --universe data/universe/optionable_liquid_900.csv
```

---

## 10. RENAISSANCE TECHNIQUES ALREADY IMPLEMENTED

| Technique | Implementation | File |
|-----------|----------------|------|
| **Hidden Markov Models** | 3-state regime detection (BULL/BEAR/NEUTRAL) | `ml_advanced/hmm_regime_detector.py` |
| **Markov Chains** | Direction prediction via state transitions | `ml_advanced/markov_chain/` |
| **Ensemble Learning** | XGBoost + LightGBM weighted ensemble | `ml_advanced/ensemble/` |
| **Kelly Criterion** | Optimal position sizing | `risk/advanced/kelly_position_sizer.py` |
| **Monte Carlo VaR** | Portfolio risk simulation | `risk/advanced/monte_carlo_var.py` |
| **Correlation Limits** | Sector/factor exposure control | `risk/advanced/correlation_limits.py` |
| **Online Learning** | Continuous model updates | `ml_advanced/online_learning.py` |
| **Feature Discovery** | Automated feature importance | `ml/alpha_discovery/feature_discovery/` |
| **Pattern Mining** | Unsupervised pattern detection | `ml/alpha_discovery/pattern_miner/` |
| **RL Trading** | Policy gradient learning | `ml/alpha_discovery/rl_agent/` |

---

## 11. WHAT'S MISSING (Renaissance → Kobe)

| Renaissance | Kobe Status | Notes |
|-------------|-------------|-------|
| 3,500+ positions | ✅ Framework exists | Scale: 20-50 positions (retail limit) |
| 12.5-20x leverage | ✅ Implemented | Capped at 5x for safety |
| High-frequency (sub-second) | ❌ Not implemented | Polygon only has EOD data |
| Speech recognition for earnings | ❌ Not implemented | Could add with Whisper API |
| Satellite imagery | ❌ Not implemented | Not cost-effective for retail |
| Weather data | ❌ Not implemented | Available free (NOAA) |
| Sentiment from 10-K/10-Q | ❌ Not implemented | Could add with EDGAR scraping |

---

## 12. NEXT STEPS - LET THE AI RUN

### Immediate Actions:

1. **Start Autonomous Research:**
   ```bash
   python scripts/run_autonomous.py
   ```
   - Runs 24/7
   - Tests alphas automatically
   - Logs all discoveries
   - Self-improves daily

2. **Run Alpha Factory:**
   ```bash
   python -m research.alpha_factory --config config/alpha_workflows/all_alphas.yaml
   ```

3. **Train RL Agent:**
   ```bash
   python -m ml.alpha_discovery.rl_agent.agent --mode train --episodes 5000
   ```

4. **Mine Patterns:**
   ```bash
   python scripts/find_unique_patterns.py --universe data/universe/optionable_liquid_900.csv
   ```

### Long-Term Evolution:

- **Week 1-2:** Let autonomous brain discover best alphas
- **Week 3-4:** Train RL agent on best patterns
- **Month 2:** Deploy Medallion orchestrator with 20+ positions
- **Month 3+:** Scale to 50+ positions with full diversification

---

## SUMMARY

**YOU ALREADY HAVE:**
- ✅ 91 quantitative alpha factors
- ✅ Renaissance-style orchestrator
- ✅ Pattern discovery engine
- ✅ Reinforcement learning
- ✅ Autonomous research system
- ✅ 24/7 learning brain

**YOU DON'T NEED TO BUILD - JUST RUN IT!**

```bash
# Let the AI find the best strategies
python scripts/run_autonomous.py

# It will:
# 1. Test all 91 alphas
# 2. Find best combinations
# 3. Optimize parameters
# 4. Backtest everything
# 5. Learn from outcomes
# 6. Repeat forever
```

**The system Jim Simons would approve of is ALREADY BUILT.**

---

**Generated:** 2026-01-09
**Status:** Production Ready
**Next:** Activate autonomous systems and let the AI work
