# /integrity-check

Comprehensive data integrity validation - detect lookahead, bias, bugs, fake data, manipulation.

## Usage
```
/integrity-check [--full] [--fix]
```

## What it does
Runs 25+ checks to prevent catastrophic trading errors from bad data.

## Commands
```bash
# Full integrity check
python scripts/integrity_check.py --full --dotenv C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env

# Quick check (critical only)
python scripts/integrity_check.py --quick

# Check specific area
python scripts/integrity_check.py --area lookahead
python scripts/integrity_check.py --area data-quality
python scripts/integrity_check.py --area signals
```

## CHECK CATEGORIES

### 1. LOOKAHEAD BIAS DETECTION
| Check | Catches |
|-------|---------|
| Indicator shift | All `_sig` columns shifted by 1 bar |
| Future data leak | Signals using data after signal time |
| Fill price | Fills use next-bar open, not signal close |
| Train/test bleed | WF test data never in training |

### 2. DATA QUALITY
| Check | Catches |
|-------|---------|
| Missing bars | Gaps > 5 trading days |
| Zero/negative prices | Invalid OHLCV |
| OHLC consistency | High >= Low, O/C within range |
| Volume anomalies | Zero or >10x average |
| Duplicate timestamps | Same date twice |
| Future timestamps | Data after today |

### 3. SIGNAL VALIDATION
| Check | Catches |
|-------|---------|
| Impossible prices | Entry outside daily H-L |
| Stop sanity | Stop within ATR*3 |
| Phantom signals | Symbols not in universe |
| Duplicate signals | Same symbol/side/time |

### 4. BACKTEST SANITY
| Check | Catches |
|-------|---------|
| Unrealistic returns | >500% annual (bug) |
| Win rate bounds | >95% = likely lookahead |
| Zero trades | Signal logic broken |
| P&L math | Verify calculations |

### 5. SOURCE AUTHENTICITY
| Check | Catches |
|-------|---------|
| Polygon verification | Data from API, not fabricated |
| Cache tampering | Compare cache vs fresh fetch |
| Symbol validation | All symbols exist |

### 6. CODE INTEGRITY
| Check | Catches |
|-------|---------|
| Strategy hash | Unauthorized code changes |
| Config pin | Config modifications |
| Indicator formulas | Match canonical definitions |

## OUTPUT
```
INTEGRITY CHECK REPORT
======================
Total: 25 | Pass: 23 | Warn: 1 | Fail: 1

[FAIL] ATR14 not shifted - LOOKAHEAD BIAS
       Fix: df['atr14_sig'] = df['atr14'].shift(1)

[WARN] 3 symbols cache >7 days old

RECOMMENDATION: DO NOT TRADE until resolved
```

## ZERO TOLERANCE
Trading BLOCKED if any:
- Lookahead in indicators
- Future data in signals
- Fabricated data
- Unauthorized code changes
- >200% annual backtest returns

## Integration
- Runs in /preflight
- Blocks trading on FAIL
- Telegram alert on issues
