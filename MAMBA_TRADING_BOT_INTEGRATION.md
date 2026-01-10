# 🐍 MAMBA AI V3 + KOBE TRADING BOT Integration Guide

## YES - Mamba AI Can Audit Your Trading Bot!

### What It Will Find

| Issue Type | What V3 Detects | How It Helps |
|------------|-----------------|--------------|
| **Lookahead Bias** | Uses future data in signals | Searches for `.shift()` missing, `col_sig` issues |
| **Bugs** | Syntax errors, logic errors | Full Python AST analysis |
| **Missing Components** | Incomplete implementations | Finds TODO comments, empty functions |
| **Security Issues** | Hardcoded API keys | Scans for credentials in code |
| **Data Issues** | Stale data, missing files | Checks file timestamps, validates formats |
| **Performance** | Slow code, inefficient loops | Identifies bottlenecks |
| **Risk Issues** | No stop loss, over-leverage | Analyzes risk management code |
| **Testing Gaps** | Missing tests, low coverage | Finds untested functions |

---

## 🔥 Real Examples with Your Trading Bot

### Example 1: Full Audit

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

ai-scan-issues
```

**What Happens:**
```
╔══════════════════════════════════════════════════════════════════╗
║            🔍 CODEBASE ISSUE DETECTION 🔍                        ║
╚══════════════════════════════════════════════════════════════════╝

📂 Scanning 250 files...

📊 ISSUE SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔴 Critical: 1    # Hardcoded API key found
🟠 High:     3    # Syntax errors in 3 files
🟡 Medium:   12   # TODO comments, missing tests
🟢 Low:      8    # Style issues

DETAILED ISSUES:

[Critical] SecurityRisk
  File: C:\...\config.py
  Issue: Hardcoded API key detected: ALPACA_API_KEY = "PKA..."
  Fix: Move to environment variables: $env:ALPACA_API_KEY

[High] SyntaxError
  File: C:\...\strategies\dual_strategy.py
  Issue: Missing parenthesis on line 245
  Fix: Review and fix syntax errors

[Medium] UnfinishedWork
  File: C:\...\risk\advanced\var.py
  Issue: TODO: Implement stress testing
  Fix: Complete or remove TODO items
```

### Example 2: Find Lookahead Bias

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

ai-autonomous "scan all strategy files for lookahead bias - check if signals use future data"
```

**What V3 Does:**
```
🔄 Iteration 1: Search for all strategy files
   Found: dual_strategy.py, ibs_rsi.py, turtle_soup.py

🔄 Iteration 2: Analyze dual_strategy.py
   Reading code...
   Checking for .shift(1) on indicators
   Finding: col_sig = col  # MISSING .shift()! ⚠️

🔄 Iteration 3: Search for .shift() usage patterns
   Searching GitHub for proper shift patterns
   Found: Best practice is col_sig = col.shift(1)

🔄 Iteration 4: Check other files
   ibs_rsi.py: ✅ Uses .shift(1) correctly
   turtle_soup.py: ⚠️  Missing shift on 'sweep_strength'

✅ COMPLETE: Found 2 lookahead bias issues
   1. dual_strategy.py line 156: col_sig = col (should be .shift(1))
   2. turtle_soup.py line 89: sweep_sig = sweep (should be .shift(1))

RECOMMENDATION: Add .shift(1) to prevent using future data in backtests
```

### Example 3: Verify Data Quality

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

ai-autonomous "check if all data files are fresh and valid"
```

**What V3 Does:**
```
🔄 Iteration 1: Find all data files
   Found: data/prices/*.parquet, data/universe/*.csv

🔄 Iteration 2: Check data/prices/AAPL.parquet
   Last modified: 2025-12-28 (5 days old) ⚠️
   Expected: Daily updates

🔄 Iteration 3: Check for missing stocks
   Expected: 800 stocks
   Found: 897 stocks
   Missing: TSLA, NVDA, PLTR ⚠️

🔄 Iteration 4: Validate data format
   Checking OHLCV columns...
   ✅ All columns present
   ⚠️  3 files have gaps (missing trading days)

✅ COMPLETE: Data quality issues found
   1. Data is 5 days stale (expected daily updates)
   2. Missing 3 stocks: TSLA, NVDA, PLTR
   3. 3 files have date gaps (AAPL, MSFT, GOOGL)

RECOMMENDATION: Run data/updater.py to refresh
```

### Example 4: Find All Missing Tests

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

ai-autonomous "find all functions that don't have tests and create a list"
```

**What V3 Does:**
```
🔄 Iteration 1: Find all test files
   Found: tests/*.py (42 test files)

🔄 Iteration 2: Find all source functions
   Analyzing strategies/*.py
   Found: 127 functions

🔄 Iteration 3: Match functions to tests
   Checking test coverage...

   Tested: 89 functions (70% coverage)
   Untested: 38 functions (30% missing)

🔄 Iteration 4: Generate missing test list
   Creating report...

✅ COMPLETE: Coverage report generated

UNTESTED FUNCTIONS:
1. strategies/dual_strategy.py::calculate_ibs() - NO TEST
2. risk/advanced/var.py::monte_carlo_sim() - NO TEST
3. execution/broker_alpaca.py::retry_order() - NO TEST
[... 35 more ...]

RECOMMENDATION: Prioritize testing high-risk functions (execution, risk)
```

### Example 5: Validate Risk Management

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

ai-autonomous "audit all risk management code - verify stops, position sizing, and exposure limits are enforced"
```

**What V3 Does:**
```
🔄 Iteration 1: Find risk management files
   Found: risk/policy_gate.py, risk/equity_sizer.py

🔄 Iteration 2: Analyze policy_gate.py
   Reading code...
   Checking for:
   ✅ 2% equity risk cap (PolicyGate.MAX_RISK_PCT = 0.02)
   ✅ 20% notional cap (PolicyGate.MAX_NOTIONAL_PCT = 0.20)
   ✅ Order count limits (PolicyGate.MAX_ORDERS_PER_DAY = 5)

🔄 Iteration 3: Check if gates are enforced
   Searching for PolicyGate.check() calls...
   ✅ Found in run_paper_trade.py line 245
   ✅ Found in run_live_trade_micro.py line 189
   ❌ NOT found in runner.py ⚠️  RISK!

🔄 Iteration 4: Check stop loss implementation
   Analyzing backtest/engine.py...
   ✅ ATR-based stops (2x ATR)
   ✅ 7-bar time stop for IBS/RSI
   ⚠️  No emergency kill switch check

✅ COMPLETE: Risk audit complete

FINDINGS:
✅ Good: 2% risk cap enforced in paper/live trading
✅ Good: Position sizing limits enforced
✅ Good: Stop losses implemented (ATR + time-based)

⚠️  ISSUES:
1. runner.py doesn't call PolicyGate.check() before placing orders
2. No KILL_SWITCH check in runner.py main loop
3. Weekly exposure gate (40% cap) bypassed in 1 location

🔴 CRITICAL: runner.py can place orders without risk checks!

RECOMMENDATION: Add PolicyGate.check() to runner.py before order placement
```

---

## 📋 Specific Questions You Can Ask

### About Bias & Data Quality

```powershell
ai "explain how the strategy avoids lookahead bias"
ai-autonomous "find any place where future data might leak into signals"
ai "are there any cases where we're using bar close prices before the bar closes?"
ai-autonomous "validate that all indicators use .shift(1)"
```

### About Bugs & Errors

```powershell
ai-scan-issues  # Full audit
ai "what could cause this error: KeyError: 'close_price'"
ai-autonomous "find all syntax errors and fix them"
ai "are there any infinite loops in the code?"
```

### About Missing Components

```powershell
ai "what features are marked TODO or FIXME?"
ai-autonomous "find all incomplete implementations"
ai "is the kill switch properly implemented everywhere?"
ai "what's missing from the risk management system?"
```

### About Strategy Logic

```powershell
ai "explain the dual strategy scanner logic"
ai "how does the Markov chain integration work?"
ai "are there any edge cases where the strategy could fail?"
ai "what happens if the API connection drops during trading?"
```

### About Performance

```powershell
ai-autonomous "find performance bottlenecks in the backtest engine"
ai "which functions are slowest?"
ai "can we optimize the data loading?"
ai "are there any O(n²) algorithms we should fix?"
```

### About Testing

```powershell
ai "what percentage of code is covered by tests?"
ai-autonomous "find all untested functions and prioritize by risk"
ai "are there tests for the live trading logic?"
ai "what edge cases are not covered by tests?"
```

---

## 🎯 Recommended Daily Workflow

### Morning Routine (Before Trading)

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

# 1. Quick health check
ai-scan-issues | Select-String -Pattern "Critical"

# 2. Verify data freshness
ai-autonomous "check if data is up to date for today"

# 3. Validate recent changes
ai "review changes made yesterday - any new risks?"
```

### Weekly Audit (Sunday)

```powershell
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

# 1. Full codebase scan
ai-scan-issues > weekly_audit_$(Get-Date -Format 'yyyyMMdd').txt

# 2. Test coverage check
ai-autonomous "generate test coverage report and identify gaps"

# 3. Performance review
ai "analyze last week's trade outcomes - any patterns in failures?"
```

### After Adding New Feature

```powershell
# 1. Scan new code for issues
ai-scan-issues

# 2. Check for lookahead bias
ai-autonomous "audit new strategy code for lookahead bias"

# 3. Verify tests exist
ai "does the new feature have tests?"

# 4. Review code quality
ai-review path/to/new_feature.py
```

---

## 🔥 Power Combos with Your Trading Bot

### Combo 1: Pre-Deployment Checklist

```powershell
# Full pre-deployment audit
ai-autonomous "run complete deployment checklist:
1. Find any syntax errors
2. Check for lookahead bias
3. Verify risk gates are enforced
4. Validate data is fresh
5. Confirm tests pass
6. Check for hardcoded credentials
7. Verify kill switch works
Create detailed report"
```

### Combo 2: Debug Failed Trades

```powershell
# After a bad trade
ai "analyze why trade on TSLA 2025-12-28 failed - check signal, execution, risk"

# Deep dive
ai-autonomous "trace the signal generation for TSLA on 2025-12-28 through the entire pipeline"
```

### Combo 3: Strategy Improvement

```powershell
# Research better patterns
ai-research "best practices for mean reversion strategies 2025"

# Find optimization opportunities
ai-autonomous "analyze backtest results and suggest parameter improvements"
```

---

## ⚠️ Important Limitations

### What V3 WON'T Do (But Will Help You Do)

| Task | V3 | Your Responsibility |
|------|-----|---------------------|
| Find lookahead bias | ✅ Finds missing .shift() | ❌ Won't understand ALL subtle biases |
| Fix bugs | ✅ Fixes syntax errors | ⚠️  **Review logic fixes manually** |
| Validate strategy | ✅ Checks code structure | ❌ Can't verify trading edge |
| Test fixes | ✅ Runs syntax checks | ❌ Won't run full backtest suite |
| Deploy changes | ❌ Needs your approval | ✅ Always review before live trading |

### Critical Rule

**NEVER deploy V3's changes to live trading without:**
1. ✅ Manual code review
2. ✅ Running full test suite (`pytest`)
3. ✅ Backtest verification
4. ✅ Paper trading test

---

## 💡 Example Session

```powershell
# Start of day
PS> cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

PS> ai "quick health check - any critical issues?"

🐍 Mamba AI thinking...

💡 Mamba AI:
I've scanned your trading bot. Here's the status:

✅ GOOD:
- All 942 tests passing
- Data updated yesterday (acceptable)
- No syntax errors
- Risk gates functional

⚠️  WARNINGS:
- 3 TODO comments in new code
- Data is 1 day old (consider refreshing for today)
- Test coverage at 94% (down from 96% last week)

🔴 CRITICAL:
- None

RECOMMENDATION: Refresh data with `python scripts/prefetch_polygon_universe.py` before trading.

# Refresh data
PS> python scripts/prefetch_polygon_universe.py --universe data/universe/optionable_liquid_800.csv --start 2025-12-20 --end 2026-01-08

# Verify fix
PS> ai "is the data fresh now?"

🐍 Mamba AI thinking...

💡 Mamba AI:
✅ Data is now fresh!
- Latest data: 2026-01-08 (today)
- All 800 stocks updated
- No gaps detected

You're good to trade!
```

---

## 🚀 Get Started Right Now

```powershell
# 1. Restart PowerShell (loads V3 automatically)
reload

# 2. Navigate to your trading bot
cd C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

# 3. Run first audit
ai-scan-issues

# 4. Ask anything
ai "is my bot ready for live trading?"
```

---

**This is YOUR AI assistant for YOUR trading bot. It knows your codebase and can find issues you might miss.**

**Use it every day. It's like having a second pair of eyes that never gets tired.**

🐍🏀💛

---

## Quick Reference Card

```
═══════════════════════════════════════════════════════════════
  MAMBA AI V3 - TRADING BOT QUICK REFERENCE
═══════════════════════════════════════════════════════════════

📍 LOCATION: C:\Users\Owner\OneDrive\Desktop\kobe81_traderbot

🔍 DAILY HEALTH CHECK:
   ai-scan-issues

🐛 FIND LOOKAHEAD BIAS:
   ai-autonomous "find any lookahead bias in strategies"

🔒 SECURITY AUDIT:
   ai-autonomous "find hardcoded credentials"

📊 DATA VALIDATION:
   ai "is data fresh and valid?"

🧪 TEST COVERAGE:
   ai-autonomous "find untested functions"

⚡ PERFORMANCE:
   ai-autonomous "find performance bottlenecks"

🛡️ RISK AUDIT:
   ai-autonomous "verify all risk gates are enforced"

❓ ASK ANYTHING:
   ai "your question about the trading bot"

═══════════════════════════════════════════════════════════════
```

Save this file: `MAMBA_TRADING_BOT_INTEGRATION.md`
