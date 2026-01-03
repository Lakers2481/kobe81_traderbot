# TRADE ANALYSIS STANDARD v1.0

> **THIS IS THE STANDARD FOR ALL TRADE ANALYSIS IN THE KOBE TRADING SYSTEM**
>
> Every trade in the Top 2 MUST have a comprehensive report following this format.
> This is non-negotiable. If the data isn't available, the trade doesn't happen.

---

## QUICK REFERENCE: Auto-Pass Criteria

A signal **automatically passes** the quality gate if:
- **20+ historical samples** of the pattern (statistically significant at p < 0.05)
- **90%+ historical win rate**
- **5+ consecutive days** in the streak

Example: PLTR with 23 samples and 100% win rate on 5+ down days = **AUTO-PASS**

---

## PATTERN GRADING SYSTEM

| Grade | Criteria | Auto-Pass | Weight Boost |
|-------|----------|-----------|--------------|
| A+ | 20+ samples, 90%+ WR | YES | Historical 40% (from 20%) |
| A | 15+ samples, 85%+ WR | No | Historical 25% |
| B | 10+ samples, 75%+ WR | No | Historical 20% |
| C | 5+ samples, 65%+ WR | No | Historical 20% |
| D | < 5 samples or < 65% WR | No | Historical 15% |

**A+ Pattern Philosophy:**
- Data shows bounce typically occurs on day 6-7 of a streak
- Enter on Day 5 to ANTICIPATE the bounce (not chase it)
- 20+ samples provides statistical significance (p < 0.05)

---

## ENTRY TIMING SYSTEM (BACKTEST-DRIVEN)

**Philosophy:** Don't use hardcoded values. Let each stock's ACTUAL BACKTEST DATA determine optimal entry.

**How Optimal Entry Day is Calculated:**
1. Analyze all historical streaks for the specific stock
2. Calculate win rate, expected return, and sample size at each streak length
3. Score each streak length: `win_rate * 100 + expected_return * 500 + sample_bonus - late_penalty`
4. Select streak length with highest score as optimal entry day

**Example: PLTR Backtest Results:**
| Streak Length | Win Rate | Samples | Avg Return | Score |
|---------------|----------|---------|------------|-------|
| 3 days | 75% | 45 | +1.8% | 94.1 |
| 4 days | 82% | 31 | +2.4% | 99.2 |
| **5 days** | **100%** | **23** | **+3.8%** | **128.7** ← OPTIMAL |
| 6 days | 100% | 8 | +4.2% | 118.4 |
| 7 days | 100% | 3 | +5.1% | 105.6 |

**Recommendation Logic (based on backtest):**
| Current vs Optimal | Recommendation | Reason |
|--------------------|----------------|--------|
| Current < Optimal - 2 | TOO_EARLY | Pattern not mature |
| Current = Optimal - 1 | ALMOST_READY | One more day |
| Current = Optimal | **ENTER_NOW** | Backtest-proven sweet spot |
| Current > Optimal | ENTER_NOW (late) | Still valid, may miss some upside |

**Entry Timing Output:**
```
ENTRY TIMING RECOMMENDATION (BACKTESTED)
Symbol: PLTR
Current Streak: 5 days down
Optimal Entry Day: 5 (calculated from backtest)
Recommendation: ENTER_NOW
Win Rate at Day 5: 100% (23 samples)
Avg Return at Day 5: +3.8%
Justification: Day 5 = OPTIMAL ENTRY (backtested). This is the sweet spot for PLTR.
```

**CRITICAL:** Optimal entry day varies by stock. TSLA might be day 4, NVDA might be day 6. Always trust the backtest data.

---

## TOP 5 STUDY → TOP 2 EXECUTE

**Process:**
1. Generate comprehensive reports for ALL Top 5 candidates
2. User studies all 5 reports before market open
3. Select Top 2 for execution based on complete analysis
4. Top 3-5 remain as backup if Top 2 invalidate premarket

**Report Scope by Rank:**
| Rank | Full Report | Execution Priority |
|------|-------------|-------------------|
| #1 (TOTD) | ✅ YES | PRIMARY |
| #2 | ✅ YES | PRIMARY |
| #3 | ✅ YES | BACKUP |
| #4 | ✅ YES | BACKUP |
| #5 | ✅ YES | BACKUP |

---

## MANDATORY ANALYSIS COMPONENTS (ALL Top 5 Trades)

### 1. EXECUTIVE SUMMARY

```
Symbol: PLTR
Date: 2026-01-02
Pattern: 5 consecutive down days
Historical Win Rate: 100% (29/29 samples)
Expected Day 1 Bounce: +3.8%
Confidence: HIGH
Recommendation: LONG
```

### 2. PRICE ACTION & PATTERN ANALYSIS

**Current State:**
| Metric | Value | Source |
|--------|-------|--------|
| Current Price | $171.11 | Polygon EOD |
| Week Open | $195.01 | Polygon |
| From Week Open | -14.0% | Calculated |
| RSI(2) | 0.0 | pandas-ta |
| IBS | 0.033 | (close-low)/(high-low) |

**Consecutive Day Pattern:**
| Metric | Value |
|--------|-------|
| Current Streak | 5 days down |
| Historical Samples | 29 |
| Historical Win Rate | 100% |
| Avg Day 1 Bounce | +3.8% |
| Max Day 1 Bounce | +15.2% |
| Min Day 1 Bounce | +0.1% |
| Avg Hold Time | 2 days |
| Avg Total Bounce | +6.9% |

**Historical Instances Table:**
| # | End Date | Streak | Day 1 | Days | Total | Drop |
|---|----------|--------|-------|------|-------|------|
| 1 | 2021-02-18 | 6 | +15.2% | 1 | +15.2% | -34% |
| 2 | 2021-02-26 | 5 | +3.8% | 1 | +3.8% | -18% |
| ... | ... | ... | ... | ... | ... | ... |
| 29 | 2025-11-21 | 5 | +4.8% | 4 | +8.6% | -11% |

### 3. EXPECTED MOVE ANALYSIS

**Weekly Expected Move (from realized volatility):**
| Metric | Value |
|--------|-------|
| 20-day Realized Vol | 44.6% annualized |
| Weekly EM | +/- 6.28% |
| Upper Bound | $178.40 |
| Lower Bound | $157.32 |
| Move from Week Open | -14.0% (2.2x EM) |
| Remaining Room UP | +23.5% |
| Remaining Room DOWN | 0% (exhausted) |

**Interpretation:** Price has moved 2.2x the expected weekly move to the downside.
Statistically, this is a massive overshoot that typically reverts.

### 4. SUPPORT & RESISTANCE LEVELS

| Level | Price | Type | Strength | Distance | Justification |
|-------|-------|------|----------|----------|---------------|
| Stop Zone | $159.50 | Support | - | -6.8% | Below psychological $160 |
| Pivot Low | $163.81 | Support | 2 | -4.3% | Touched 2x |
| Psychological | $165.00 | Support | 1 | -3.6% | Round number |
| Current | $171.11 | - | - | 0% | Current price |
| Psychological | $175.00 | Resistance | 1 | +2.3% | Round number |
| Pivot High | $188.94 | Resistance | 5 | +10.4% | Touched 5x |

### 5. NEWS & CATALYSTS (REQUIRED)

**Recent Headlines (Last 7 Days):**
| Date | Headline | Sentiment | Source |
|------|----------|-----------|--------|
| 2026-01-02 | "PLTR reports new $500M government contract" | Positive | Reuters |
| 2025-12-30 | "Tech sector faces selling pressure" | Neutral | Bloomberg |
| 2025-12-28 | "Year-end tax loss selling impacts momentum names" | Negative | CNBC |

**Aggregated Sentiment:**
- Compound: 0.15 (slightly positive)
- Positive: 0.35
- Negative: 0.20
- Neutral: 0.45

### 6. POLITICAL & INSTITUTIONAL ACTIVITY (4 REQUIRED DATA SOURCES)

#### 6A. CONGRESSIONAL TRADES (Quiver Quant API)

**Data Source:** `altdata/congressional_trades.py` → Quiver Quant API
**API Key Required:** `QUIVER_API_KEY` in `.env`

**Recent Congressional Trades (Last 90 Days):**
| Date | Official | Party | House | Transaction | Amount | Reported |
|------|----------|-------|-------|-------------|--------|----------|
| 2025-12-15 | Rep. Nancy Pelosi | D | House | BUY | $250K-$500K | 2025-12-20 |
| 2025-12-10 | Sen. Tommy Tuberville | R | Senate | SALE | $100K-$250K | 2025-12-15 |

**Congressional Summary:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Net Buy/Sell (90d) | +$350K | NET BUYING |
| Total Transactions | 5 | Active interest |
| Notable Officials | Pelosi, Tuberville | High-profile |
| Reporting Lag | 5-45 days | Consider delay |

#### 6B. INSIDER ACTIVITY (SEC EDGAR Form 4)

**Data Source:** `altdata/insider_activity.py` → SEC EDGAR (public, no API key)

**Recent Insider Trades (Last 30 Days):**
| Date | Insider | Title | Transaction | Shares | Price | Value |
|------|---------|-------|-------------|--------|-------|-------|
| 2025-12-10 | Alex Karp | CEO | SELL | 100,000 | $175.00 | $17.5M |
| 2025-12-08 | Stephen Cohen | CFO | BUY | 5,000 | $172.50 | $862K |

**Insider Summary:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Net Shares (30d) | -95,000 | NET SELLING |
| Net Value (30d) | -$16.6M | Significant outflow |
| Officer Trades | 3 | Executive activity |
| Director Trades | 1 | Board activity |
| Cluster Buys | NO | No coordinated buying |

**Insider Interpretation Key:**
- **Cluster Buys (3+ insiders buying in 7 days):** Very bullish signal
- **CEO/CFO Selling:** Often routine, check if planned (10b5-1)
- **Director Buying:** Often more meaningful than officer selling

#### 6C. OPTIONS FLOW (Polygon API)

**Data Source:** `altdata/options_flow.py` → Polygon Options API
**API Key Required:** `POLYGON_API_KEY` in `.env`

**Unusual Options Activity (Last 7 Days):**
| Date | Type | Strike | Expiry | Volume | OI | Vol/OI | Premium | Direction |
|------|------|--------|--------|--------|-----|--------|---------|-----------|
| 2026-01-02 | CALL | $180 | 2026-01-17 | 15,000 | 3,000 | 5.0x | $2.1M | BULLISH |
| 2025-12-30 | PUT | $160 | 2026-01-10 | 8,000 | 2,500 | 3.2x | $800K | BEARISH |

**Options Flow Summary:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Put/Call Ratio | 0.65 | BULLISH (< 0.70) |
| IV Percentile | 72% | Elevated volatility |
| Unusual Calls | 3 | Bullish bets detected |
| Unusual Puts | 1 | Less bearish activity |
| Smart Money Flow | +$1.3M net | NET BULLISH |

**Unusual Activity Criteria:**
- Volume > 2x Open Interest
- Minimum 1,000 contracts
- Minimum $50,000 premium

**Put/Call Ratio Interpretation:**
| PCR Range | Interpretation |
|-----------|----------------|
| < 0.70 | BULLISH (more calls) |
| 0.70 - 1.15 | NEUTRAL |
| > 1.15 | BEARISH (more puts) |

#### 6D. INSTITUTIONAL HOLDINGS (13F Filings)

**Notable Recent Changes (Last Quarter):**
| Institution | Change | Shares | % Portfolio |
|-------------|--------|--------|-------------|
| ARK Invest | +500,000 | 12.5M | 8.2% |
| BlackRock | -200,000 | 45.2M | 0.3% |
| Vanguard | +150,000 | 38.1M | 0.2% |

**Institutional Summary:**
- Net Change: +450,000 shares
- Top Holder Sentiment: ACCUMULATING

### 7. SECTOR & MARKET CONTEXT

**Sector Relative Strength:**
| Metric | Value |
|--------|-------|
| Sector ETF | XLK |
| Symbol Return (20d) | -4.7% |
| Sector Return (20d) | -0.1% |
| Relative Strength | -4.6% (underperforming) |
| Beta vs Sector | 3.48 |

**Market Regime:**
| Metric | Value |
|--------|-------|
| SPY vs SMA(50) | Above (Bullish) |
| VIX Level | 18.5 (Normal) |
| Regime | BULLISH |

### 8. VOLUME ANALYSIS

| Metric | Value |
|--------|-------|
| Avg Volume (20d) | 36.3M |
| Avg Volume (50d) | 47.8M |
| Relative Volume | 1.67x |
| Volume Trend | Decreasing |
| Buying Pressure | 54% |

### 9. ENTRY / STOP / TARGET JUSTIFICATION

**Entry:**
- **Price:** $171.11
- **Why:** 5 consecutive down days with 100% historical reversal rate (29 samples)
- **Confirmation:** RSI(2) = 0.0, IBS = 0.033 (extreme oversold)

**Stop Loss:**
- **Price:** $159.50
- **Why:** Below psychological $160 AND major pivot low at $163.81
- **Risk:** $11.60 per share (6.8%)

**Target 1 (Day 1):**
- **Price:** $177.60
- **Why:** Average historical Day 1 bounce (+3.8%)
- **R:R:** 0.56:1 (partial take profit)

**Target 2 (Total Bounce):**
- **Price:** $183.00
- **Why:** Average historical total bounce (+6.9%)
- **R:R:** 1.03:1 (full exit)

### 10. RISK:REWARD ANALYSIS

| Scenario | Entry | Target | Stop | R:R |
|----------|-------|--------|------|-----|
| Conservative (Day 1) | $171.11 | $177.60 | $159.50 | 0.56:1 |
| Base (2-day avg) | $171.11 | $183.00 | $159.50 | 1.03:1 |
| Aggressive (max hist) | $171.11 | $197.20 | $159.50 | 2.25:1 |

**Minimum Required R:R:** 2.25:1 (per system rules)
**This Trade:** Achievable if held 2 days and hits total bounce target

### 11. BULL CASE

1. 100% historical win rate on this exact pattern (29/29 samples)
2. Extreme oversold readings (RSI2=0, IBS=0.03)
3. Already down 14% from week open (2.2x expected move)
4. Still above 200-day SMA (long-term uptrend intact)
5. Government contract pipeline remains strong

### 12. BEAR CASE

1. Beta of 3.48 means amplified downside in market selloff
2. Underperforming sector by 4.6%
3. High valuation (may have further to fall)
4. Year-end tax loss selling may continue into January
5. Pattern could fail for the first time

### 13. WHAT COULD GO WRONG

1. **Macro Event:** Fed surprise, geopolitical crisis
2. **Company-Specific:** Negative earnings pre-announcement, contract cancellation
3. **Gap Down:** Opening -5% would immediately hit stop
4. **Pattern Failure:** First failure in 29 occurrences (2.6% base rate)
5. **Sector Rotation:** Continued tech selloff

### 14. AI CONFIDENCE BREAKDOWN

**Standard Weights:**
| Factor | Score | Weight | Contribution |
|--------|-------|--------|--------------|
| Historical Pattern | 100% | 20% | 20.0 |
| Technical Oversold | 95% | 15% | 14.3 |
| Expected Move | 90% | 10% | 9.0 |
| Sector Strength | 40% | 10% | 4.0 |
| Sentiment (News) | 60% | 10% | 6.0 |
| Volume Profile | 70% | 5% | 3.5 |
| Regime Alignment | 80% | 5% | 4.0 |
| Congressional Trades | 75% | 10% | 7.5 |
| Insider Activity | 50% | 10% | 5.0 |
| Options Flow | 85% | 5% | 4.3 |
| **TOTAL** | | 100% | **77.6** |

**A+ Pattern Weights (Historical boosted to 40%):**
| Factor | Score | Weight | Contribution |
|--------|-------|--------|--------------|
| Historical Pattern | 100% | **40%** | **40.0** |
| Technical Oversold | 95% | 10% | 9.5 |
| Expected Move | 90% | 5% | 4.5 |
| Sector Strength | 40% | 5% | 2.0 |
| Sentiment (News) | 60% | 10% | 6.0 |
| Volume Profile | 70% | 5% | 3.5 |
| Regime Alignment | 80% | 5% | 4.0 |
| Congressional Trades | 75% | 7% | 5.3 |
| Insider Activity | 50% | 8% | 4.0 |
| Options Flow | 85% | 5% | 4.3 |
| **TOTAL** | | 100% | **83.1** |

**Final Confidence:** 83.1% (HIGH) - A+ Pattern Boost Applied

### 15. POSITION SIZING

```python
Account: $50,000
Max Risk Per Trade: 2% = $1,000
Entry: $171.11
Stop: $159.50
Risk Per Share: $11.61
Max Shares: 86 shares
Position Size: $14,716 (29.4% of account)

# Check notional cap (20%)
Max Notional: $10,000
Adjusted Shares: 58 shares
Final Position: $9,924 (19.8% of account)
```

---

## DATA SOURCES (ALL REQUIRED)

| Data Type | Source | Module | API Key | Update Frequency |
|-----------|--------|--------|---------|------------------|
| Price Data | Polygon.io | `data/providers/polygon_eod.py` | `POLYGON_API_KEY` | EOD |
| Historical Patterns | Calculated | `analysis/historical_patterns.py` | None | Real-time |
| News Headlines | Polygon/Finnhub | `altdata/news_processor.py` | `POLYGON_API_KEY` | Intraday |
| Sentiment | VADER | `altdata/news_processor.py` | None | Real-time |
| Congressional Trades | Quiver Quant | `altdata/congressional_trades.py` | `QUIVER_API_KEY` | Daily |
| Insider Activity | SEC EDGAR | `altdata/insider_activity.py` | None (public) | Daily |
| Options Flow | Polygon Options | `altdata/options_flow.py` | `POLYGON_API_KEY` | EOD |
| Expected Move | Realized Vol | `analysis/options_expected_move.py` | None | EOD |

**Required Environment Variables:**
```bash
POLYGON_API_KEY=your_polygon_key
QUIVER_API_KEY=your_quiver_key  # Optional but recommended
```

---

## WHAT MAKES A TRADE EXECUTABLE

A trade is EXECUTABLE if:

**Path 1: A+ Pattern AUTO-PASS**
- **20+ historical samples** of the exact pattern
- **90%+ historical win rate**
- **Entry timing is ENTER_NOW** (Day 5+ of streak)
- Result: **Quality Gate = AUTO-PASS, Grade = A+**

**Path 2: Standard Quality Gate**
- **Quality Score >= 70:** Passes standard quality gate
- **Confidence >= 0.65:** AI confidence threshold
- **R:R >= 1.5:1:** Minimum risk-reward ratio

**PLUS all of (both paths):**
- [ ] News sentiment is not extremely negative (compound < -0.5)
- [ ] No earnings within 5 days
- [ ] Liquidity adequate ($5M+ ADV)
- [ ] Within kill zone (10:00-11:30 or 14:30-15:30 ET)
- [ ] Daily exposure limit not exceeded (20%)
- [ ] Weekly exposure limit not exceeded (40%)
- [ ] No conflicting alt-data signals (all 4 sources checked)

---

## REPORT GENERATION REQUIREMENTS

**For ALL Top 5 Trades, EVERY section above is MANDATORY.**

This ensures:
- User can study all 5 candidates with complete information
- Backup trades (3-5) are ready if Top 2 invalidate premarket
- No information asymmetry between execution and backup candidates

**If any data source fails:**
1. Log the failure with error details
2. Note "DATA UNAVAILABLE" in the section
3. Lower confidence score by appropriate amount:
   - Price data unavailable: **ABORT** (critical)
   - Historical patterns unavailable: -20% confidence
   - News/sentiment unavailable: -10% confidence
   - Congressional trades unavailable: -5% confidence
   - Insider activity unavailable: -5% confidence
   - Options flow unavailable: -5% confidence
4. Consider if trade is still executable (Path 2 only - A+ patterns proceed)

**This is how professionals trade. No shortcuts. Full evidence or no trade.**

---

## COMMAND REFERENCE

```bash
# Generate Pre-Game Blueprint (ALL Top 5 with full reports)
python scripts/generate_pregame_blueprint.py --cap 900 --top 5 --execute 2

# Output files:
# - reports/pregame_YYYYMMDD.json  (machine-readable)
# - reports/pregame_YYYYMMDD.md    (human-readable)
```

---

*Last Updated: 2026-01-02*
*Version: 2.0*
*Changes: Added A+ grading, entry timing, 4 alt-data sources, Top 5 full reports*
