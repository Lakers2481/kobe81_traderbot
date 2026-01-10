# PHASE 3: COMPONENT DISCOVERY - 6 HUNTS

**Generated:** 2026-01-05 20:30 ET
**Auditor:** Claude SUPER AUDIT
**Status:** COMPLETE

---

## EXECUTIVE SUMMARY

| Hunt | Target | Found |
|------|--------|-------|
| 1 | Class Definitions | 1,438 |
| 2 | Import Statements | 6,914 |
| 3 | Registries & Factories | 56 |
| 4 | Config References | 418 |
| 5 | Artifact Outputs | 241,376 |
| 6 | Dead Code Indicators | 35 |

---

## HUNT 1: CLASS DEFINITIONS (1,438 classes)

### Key Strategy Classes
| Class | File | Purpose |
|-------|------|---------|
| DualStrategyScanner | strategies/dual_strategy/combined.py | THE scanner |
| DualStrategyParams | strategies/dual_strategy/params.py | Parameters |
| TurtleSoupStrategy | strategies/ict/turtle_soup.py | ICT strategy |
| IbsRsiStrategy | strategies/ibs_rsi/strategy.py | Mean reversion |

### Key Execution Classes
| Class | File | Purpose |
|-------|------|---------|
| AlpacaBroker | execution/broker_alpaca.py | Live broker |
| PaperBroker | execution/broker_paper.py | Paper broker |
| OrderManager | execution/order_manager.py | Order mgmt |

### Key Risk Classes
| Class | File | Purpose |
|-------|------|---------|
| PolicyGate | risk/policy_gate.py | Budget limits |
| SignalQualityGate | risk/signal_quality_gate.py | Quality gate |
| KillZoneGate | risk/kill_zone_gate.py | Time blocking |

### Key Cognitive Classes
| Class | File | Purpose |
|-------|------|---------|
| CognitiveBrain | cognitive/cognitive_brain.py | Main brain |
| MetacognitiveGovernor | cognitive/metacognitive_governor.py | System 1/2 |
| ReflectionEngine | cognitive/reflection_engine.py | Learning |

**Full list:** `AUDITS/03_CLASSES.json`

---

## HUNT 2: IMPORT GRAPH (6,914 imports)

### Files with imports: 702
### External dependencies: 144

**Key External Dependencies:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `alpaca` - Broker interface
- `polygon` - Data provider
- `torch` - ML (PyTorch)
- `sklearn` - ML (scikit-learn)
- `hmmlearn` - HMM regime detection
- `stable_baselines3` - RL agent
- `anthropic` - LLM interface
- `ta` - Technical analysis
- `pytz` - Timezone handling

---

## HUNT 3: REGISTRIES & FACTORIES (56 found)

### Critical Registries
| Registry | File | Purpose |
|----------|------|---------|
| Strategy Registry | strategies/registry.py | Get production scanner |
| Data Registry | data_exploration/data_registry.py | Dataset registry |
| Evolution Registry | evolution/registry.py | Evolution tracking |
| Provider Registry | data/providers/multi_source.py | Data providers |

### Factory Functions
| Function | File | Purpose |
|----------|------|---------|
| get_production_scanner() | strategies/registry.py | THE scanner |
| get_provider() | data/providers/multi_source.py | Data provider |
| get_strategy() | analytics/attribution/strategy_attribution.py | Strategy lookup |

**Full list:** `AUDITS/03_REGISTRIES.json`

---

## HUNT 4: CONFIG REFERENCES (418 found)

### By Type
| Pattern | Count | Examples |
|---------|-------|----------|
| .json | 171 | frozen_strategy_params_v2.2.json |
| .env | 146 | .env for API keys |
| state/ | 47 | state/watchlist/next_day.json |
| .csv | 38 | universe/optionable_liquid_800.csv |
| .yaml | 15 | config/base.yaml |
| config/ | 1 | Configuration directory |

### Critical Config Files
| File | Purpose |
|------|---------|
| config/frozen_strategy_params_v2.2.json | Strategy parameters |
| data/universe/optionable_liquid_800.csv | 900 stock universe |
| state/watchlist/next_day.json | Tomorrow's Top 5 |
| state/autonomous/heartbeat.json | Robot alive status |
| .env | API keys (not in repo) |

**Full list:** `AUDITS/03_CONFIG_REFS.json`

---

## HUNT 5: ARTIFACT OUTPUTS (241,376 references)

High count due to generic pattern matching on 'write'. Key artifact locations:
- `logs/` - Event logs, trade logs
- `state/` - State files
- `reports/` - Generated reports
- `wf_outputs/` - Walk-forward results
- `data/polygon_cache/` - Cached data

---

## HUNT 6: DEAD CODE INDICATORS (35 found)

### Pass-Only Functions: 33
Functions with only `pass` body. May be stubs or intentional placeholders.

### TODO/FIXME Markers: 1
Remaining work to be done.

### Deprecated Markers: 1
Deprecated code identified.

**Full list:** `AUDITS/03_DEAD_CODE.json`

---

## NEXT: PHASE 4 - REALITY CHECK

Verify which classes are stubs vs real implementations.

**Signature:** SUPER_AUDIT_PHASE3_2026-01-05_COMPLETE
