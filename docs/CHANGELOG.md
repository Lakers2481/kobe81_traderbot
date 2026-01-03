# CHANGELOG.md - Version History

All notable changes to the Kobe Trading Robot are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

### Added
- Comprehensive documentation system (15+ files)
- Living documentation (WORKLOG, CONTRIBUTING)

### Changed
- None

### Fixed
- None

---

## [2.3.0] - 2026-01-03

### Added
- Full repository documentation audit
- `docs/REPO_MAP.md` - Complete directory tree
- `docs/ENTRYPOINTS.md` - All 180+ runnable scripts
- `docs/ARCHITECTURE.md` - Pipeline wiring proof
- `docs/READINESS.md` - Production readiness matrix
- `docs/KNOWN_GAPS.md` - Missing components tracker
- `docs/RISK_REGISTER.md` - Risk assessment (17 risks)
- `docs/START_HERE.md` - Onboarding guide
- `docs/ROBOT_MANUAL.md` - Complete system guide
- `docs/WORKLOG.md` - Work log index
- `docs/CONTRIBUTING.md` - Documentation rules

### Changed
- Updated CLAUDE.md with mandatory reading sections

### Fixed
- Documentation gaps identified and documented

---

## [2.2.1] - 2026-01-02

### Fixed
- **CRITICAL:** Position sizing bypass via manual orders
- Added dual cap enforcement (2% risk + 20% notional)
- See `docs/CRITICAL_FIX_20260102.md`

---

## [2.2.0] - 2026-01-01

### Added
- Bounce analysis database (10Y + 5Y, 1M+ events)
- `bounce/` module for streak analysis
- `tools/build_bounce_db.py` script
- Historical pattern integration

### Changed
- Quality gate threshold lowered to 55 (ML models not trained)

---

## [2.1.0] - 2025-12-31

### Added
- Pre-Game Blueprint system
- `analysis/historical_patterns.py`
- `analysis/options_expected_move.py`
- `explainability/trade_thesis_builder.py`
- `scripts/generate_pregame_blueprint.py`

### Changed
- ONE Scanner System - `scan.py` is the only scanner
- Deleted all other scan scripts

---

## [2.0.0] - 2025-12-29

### Added
- DualStrategyScanner (IBS+RSI + Turtle Soup combined)
- Professional Execution Flow (kill zones, watchlist)
- Walk-forward validation framework
- ML confidence scoring via `ml_meta/model.py`
- 70 skills in `.claude/skills/`

### Changed
- System audit: Grade A+ (100/100)
- 942 tests, 0 critical issues

### Fixed
- Lookahead bias eliminated (all indicators use `.shift(1)`)

---

## [1.5.0] - 2025-12-15

### Added
- Cognitive brain architecture (`cognitive/`)
- LSTM confidence model (`ml_advanced/lstm_confidence/`)
- HMM regime detector (`ml_advanced/hmm_regime_detector.py`)
- Ensemble predictor (`ml_advanced/ensemble/`)

### Changed
- Enhanced risk gates (PolicyGate, KillZoneGate, ExposureGate)

---

## [1.4.0] - 2025-12-01

### Added
- Synthetic options engine (`options/`)
- Frozen data lake (`data/lake/`)
- Multi-source data provider (`data/providers/multi_source.py`)

### Changed
- Improved position sizing with dual caps

---

## [1.3.0] - 2025-11-15

### Added
- ICT Turtle Soup strategy with sweep filter
- Kill zone gate for time-based blocking
- Weekly exposure gate

### Fixed
- Turtle Soup false signals (added 0.3 ATR filter)

---

## [1.2.0] - 2025-11-01

### Added
- IBS+RSI strategy
- Walk-forward backtesting
- Monte Carlo simulation

### Changed
- Refactored strategy interface

---

## [1.1.0] - 2025-10-15

### Added
- Alpaca broker integration
- IOC LIMIT order execution
- Kill switch mechanism
- Idempotency store

---

## [1.0.0] - 2025-10-01

### Added
- Initial release
- Backtesting engine
- Polygon.io data provider
- Basic risk management

---

## Version Format

- MAJOR: Breaking changes or significant architecture updates
- MINOR: New features, non-breaking changes
- PATCH: Bug fixes, documentation updates

---

## Related Documentation

- [STATUS.md](STATUS.md) - Current system status
- [WORKLOG.md](WORKLOG.md) - Detailed work notes
