# Enhanced Brain Integration Guide

**Created:** 2026-01-07
**Purpose:** Document the integration of alpha mining components into Kobe's autonomous brain

---

## Overview

The enhanced brain integration brings together **9 new components** into a unified autonomous system:

### New Components Integrated

| Component | File | Purpose |
|-----------|------|---------|
| **VectorBTMiner** | `research/vectorbt_miner.py` | Fast alpha mining (10,000+ variants in seconds) |
| **AlphaLibrary** | `research/alpha_library.py` | 91 alpha factors organized by category |
| **FactorValidator** | `research/factor_validator.py` | Alphalens IC analysis for alpha validation |
| **AlphaFactory** | `research/alpha_factory.py` | Qlib-style workflows for alpha research |
| **VectorizedBacktester** | `backtest/vectorized_fast.py` | Bulk testing infrastructure |
| **BrainState** | `cognitive/states.py` | TypedDict definitions for state machine |
| **KobeBrainGraph** | `cognitive/brain_graph.py` | LangGraph formal state machine |
| **RAGEvaluator** | `cognitive/rag_evaluator.py` | LLM reasoning quality tracking |
| **AlphaResearchIntegration** | `research/alpha_research_integration.py` | Unified integration layer |

### Integration Files Created

| File | Purpose |
|------|---------|
| `autonomous/enhanced_research.py` | EnhancedResearchEngine - wires alpha mining to brain |
| `autonomous/enhanced_brain.py` | EnhancedAutonomousBrain - unified brain with all components |
| `autonomous/__init__.py` | Package exports with feature flags |
| `tests/test_enhanced_brain_integration.py` | Comprehensive integration tests |

---

## Architecture

### Layer 1: Alpha Mining Infrastructure

```
research/
├── alpha_library.py         # 91 alpha factors (momentum, mean_reversion, etc.)
├── vectorbt_miner.py         # Fast parameter sweep (10,000+ variants)
├── factor_validator.py       # Alphalens IC analysis
├── alpha_factory.py          # Qlib-style workflows
└── alpha_research_integration.py  # Unified integration layer
```

**Key Features:**
- **AlphaLibrary**: 91 pre-built alpha factors organized by category
- **VectorBT Mining**: Test 10,000+ parameter combinations in seconds
- **Alphalens Validation**: IC mean, IC Sharpe, quantile spreads, statistical significance
- **AlphaFactory**: Qlib-style workflows for systematic alpha research

### Layer 2: Enhanced Research Engine

```python
# autonomous/enhanced_research.py
class EnhancedResearchEngine(ResearchEngine):
    def run_vectorbt_alpha_sweep(...)     # Run VectorBT mining
    def validate_alpha_with_alphalens(...) # Validate with IC analysis
    def run_alpha_factory_workflow(...)    # Run Qlib workflows
    def get_top_alphas(...)                # Get best performing alphas
    def submit_alpha_hypotheses_to_curiosity_engine(...)  # Brain integration
```

**Extends ResearchEngine with:**
- VectorBT fast alpha mining
- Alphalens factor validation
- AlphaFactory workflow execution
- Integration with CuriosityEngine

### Layer 3: Enhanced Autonomous Brain

```python
# autonomous/enhanced_brain.py
class EnhancedAutonomousBrain(AutonomousBrain):
    def __init__(use_langgraph=False)
    def think_with_langgraph()             # LangGraph state machine
    def _check_alpha_discoveries()         # Alpha mining alerts
    def _check_rag_discoveries()           # LLM quality alerts
    def _run_alpha_mining_sweep()          # Background alpha mining
```

**New Capabilities:**
- **EnhancedResearchEngine** for alpha mining
- **KobeBrainGraph** for formal state machine decisions (optional)
- **RAGEvaluator** for LLM explanation quality tracking
- **Unified discovery system** across all components

### Layer 4: Formal State Machine (Optional)

```python
# cognitive/brain_graph.py
class KobeBrainGraph:
    # Nodes: observe → analyze → decide → execute → reflect
    # Routing: Based on trading phase, kill zones, risk gates
    def run_cycle()    # Run one state machine cycle
    def visualize()    # Generate Mermaid diagram
```

**LangGraph Integration:**
- Formal StateGraph with explicit transitions
- Human-in-the-loop interrupts
- Checkpointing for recovery
- Visualization with Mermaid diagrams

---

## Usage

### Basic Usage (No LangGraph)

```python
from autonomous.enhanced_brain import EnhancedAutonomousBrain

# Initialize enhanced brain
brain = EnhancedAutonomousBrain(use_langgraph=False)

# Run forever (24/7 operation)
brain.run_forever(cycle_seconds=60)
```

### Advanced Usage (With LangGraph)

```python
from autonomous.enhanced_brain import EnhancedAutonomousBrain

# Initialize with LangGraph state machine
brain = EnhancedAutonomousBrain(use_langgraph=True)

# Run forever with formal state machine
brain.run_forever(cycle_seconds=60, use_langgraph=True)
```

### CLI Usage

```bash
# Run enhanced brain (standard mode)
python -m autonomous.enhanced_brain --cycle 60

# Run with LangGraph state machine
python -m autonomous.enhanced_brain --cycle 60 --langgraph

# Get status
python -m autonomous.enhanced_brain --status

# Single cycle (testing)
python -m autonomous.enhanced_brain --once
```

### Programmatic Usage

```python
from autonomous.enhanced_research import EnhancedResearchEngine

# Initialize research engine
engine = EnhancedResearchEngine()

# Run VectorBT alpha sweep
result = engine.run_vectorbt_alpha_sweep(
    min_sharpe=0.5,
    min_trades=30,
)
print(f"Discovered {result['alphas_discovered']} alphas")

# Validate top alpha with Alphalens
top_alphas = engine.get_top_alphas(n=10)
if top_alphas:
    validation = engine.validate_alpha_with_alphalens(
        alpha_name=top_alphas[0].alpha_name
    )
    print(f"IC Mean: {validation['ic_mean']:.4f}")

# Submit hypotheses to CuriosityEngine
submitted = engine.submit_alpha_hypotheses_to_curiosity_engine()
print(f"Submitted {submitted} hypotheses")
```

---

## Discovery Alerting System

The enhanced brain has a **unified discovery system** that alerts on important findings from all sources:

### Discovery Sources

| Source | Discovery Type | Alert Criteria |
|--------|----------------|----------------|
| **Base ResearchEngine** | `parameter_improvement` | Confidence > 0.6, Improvement > 5% |
| **Alpha Mining** | `alpha_discovery` | Sharpe > 1.0, Validated, Significant |
| **LangGraph** | `state_pattern` | Recurring state transitions |
| **RAG Evaluator** | `rag_quality` | Avg score > 85, n >= 10 |
| **External Scrapers** | `external_strategy` | WR > 55%, PF > 1.3 |

### Discovery Flow

```
1. Component discovers something → Creates Discovery object
2. Brain._check_for_discoveries() → Collects from all sources
3. Brain._alert_discovery() → Logs + saves + ALERTS
4. Discovery saved to state/autonomous/discoveries.json
```

### Example Discovery

```json
{
  "type": "alpha_discovery",
  "description": "High-quality alpha 'momentum_cross_rank' discovered and validated",
  "source": "enhanced_research_engine",
  "improvement": 0.85,
  "confidence": 0.92,
  "data": {
    "alpha_id": "vbt_momentum_cross_rank_42",
    "sharpe_ratio": 1.35,
    "win_rate": 0.62,
    "ic_mean": 0.084,
    "statistically_significant": true
  },
  "timestamp": "2026-01-07T15:30:00"
}
```

---

## Background Work Scheduling

The enhanced brain schedules different work based on **work mode**:

| Work Mode | Time (ET) | Enhanced Work |
|-----------|-----------|---------------|
| **DEEP_RESEARCH** | Weekend/Holiday | VectorBT alpha mining sweep |
| **RESEARCH** | Pre-market, Lunch | Alpha mining (every 3rd cycle) |
| **LEARNING** | After hours | Trade analysis + RAG evaluation |
| **OPTIMIZATION** | Night | Feature analysis + hypothesis submission |
| **MONITORING** | Trading hours | Base behavior (scan, trade, monitor) |

### Work Schedule Example

```python
def _do_background_work(self, context):
    work_mode = context.work_mode

    if work_mode == WorkMode.DEEP_RESEARCH:
        # Weekend: Full alpha mining sweep
        return self._run_alpha_mining_sweep()

    elif work_mode == WorkMode.RESEARCH:
        # Weekday research: Mix of alpha mining and experiments
        if self.cycles_completed % 3 == 0:
            return self._run_alpha_mining_sweep()
        else:
            return super()._do_background_work(context)

    # ... other modes
```

---

## State Files

### Enhanced State Files

| File | Purpose |
|------|---------|
| `state/autonomous/alpha_discoveries.json` | All alpha discoveries with metrics |
| `state/autonomous/discoveries.json` | Unified discoveries (all sources) |
| `state/research/alpha_discoveries.json` | Integration layer state |
| `state/rag_evaluation/evaluations.json` | RAG explanation evaluations |

### Example Alpha Discovery State

```json
{
  "updated_at": "2026-01-07T16:00:00",
  "total_discoveries": 15,
  "discoveries": [
    {
      "alpha_id": "vbt_momentum_rank_5",
      "alpha_name": "momentum_rank_5",
      "category": "momentum",
      "parameters": {"window": 5},
      "sharpe_ratio": 1.42,
      "win_rate": 0.63,
      "profit_factor": 1.85,
      "total_trades": 147,
      "ic_mean": 0.092,
      "ic_sharpe": 1.23,
      "statistically_significant": true,
      "discovered_at": "2026-01-07T14:30:00",
      "validated": true,
      "promoted": false
    }
  ]
}
```

---

## Testing

### Run Integration Tests

```bash
# Run all enhanced brain tests
pytest tests/test_enhanced_brain_integration.py -v

# Run specific test class
pytest tests/test_enhanced_brain_integration.py::TestEnhancedResearchEngine -v

# Run with output
pytest tests/test_enhanced_brain_integration.py -v -s
```

### Test Coverage

| Test Class | Tests |
|------------|-------|
| `TestEnhancedResearchEngine` | 6 tests (initialization, VectorBT, Alphalens, summary) |
| `TestEnhancedAutonomousBrain` | 7 tests (init, integration, status, cycle) |
| `TestBrainGraphIntegration` | 2 tests (singleton, cycle with LangGraph) |
| `TestRAGEvaluatorIntegration` | 2 tests (singleton, explanations) |
| `TestAlphaResearchIntegration` | 2 tests (singleton, summary) |
| `TestEndToEndIntegration` | 2 tests (full cycle, discovery alerting) |

**Total:** 21 integration tests

---

## Dependencies

### Required (Base Brain)

```
pandas>=1.5.0
numpy>=1.24.0
```

### Optional (Enhanced Features)

```
# For VectorBT alpha mining
vectorbt>=0.26.0

# For Alphalens validation
alphalens-reloaded>=0.4.3

# For LangGraph state machine
langgraph>=0.0.21
langchain-core>=0.1.0

# For RAG evaluation
scipy>=1.10.0
```

### Install All Enhanced Features

```bash
pip install vectorbt alphalens-reloaded langgraph scipy
```

---

## Feature Flags

The integration uses feature flags to gracefully handle missing dependencies:

```python
from autonomous import (
    HAS_ENHANCED_RESEARCH,  # EnhancedResearchEngine available
    HAS_ENHANCED_BRAIN,     # EnhancedAutonomousBrain available
)

from research import (
    HAS_VBT,                # VectorBT installed
    HAS_ALPHALENS,          # Alphalens installed
    HAS_ALPHA_LIBRARY,      # AlphaLibrary available
    HAS_INTEGRATION,        # AlphaResearchIntegration available
)

from cognitive.brain_graph import HAS_LANGGRAPH  # LangGraph installed
```

**Example Usage:**

```python
if HAS_ENHANCED_BRAIN:
    brain = EnhancedAutonomousBrain()
else:
    brain = AutonomousBrain()  # Fallback

if HAS_VBT:
    result = engine.run_vectorbt_alpha_sweep()
else:
    print("VectorBT not installed - skipping alpha mining")
```

---

## Migration Guide

### From AutonomousBrain to EnhancedAutonomousBrain

```python
# OLD (base brain)
from autonomous import AutonomousBrain
brain = AutonomousBrain()
brain.run_forever()

# NEW (enhanced brain)
from autonomous import EnhancedAutonomousBrain
brain = EnhancedAutonomousBrain(use_langgraph=False)
brain.run_forever()
```

**Changes:**
- Drop-in replacement (extends AutonomousBrain)
- Adds alpha mining background work
- Adds unified discovery alerting
- Optionally enables LangGraph state machine

### From ResearchEngine to EnhancedResearchEngine

```python
# OLD
from autonomous.research import ResearchEngine
engine = ResearchEngine()
engine.backtest_random_params()

# NEW
from autonomous.enhanced_research import EnhancedResearchEngine
engine = EnhancedResearchEngine()

# Old methods still work
engine.backtest_random_params()

# New methods available
engine.run_vectorbt_alpha_sweep()
engine.validate_alpha_with_alphalens("alpha_name")
```

---

## Performance Considerations

### Alpha Mining Performance

| Operation | Time | Scales With |
|-----------|------|-------------|
| VectorBT sweep (100 stocks, 10k variants) | ~30s | Stocks × Variants |
| Alphalens validation | ~5s | Stocks × Periods |
| Single parameter experiment | ~3s | Stocks |
| Alpha hypothesis submission | <1s | Hypotheses |

### Memory Usage

| Component | Memory |
|-----------|--------|
| Base Brain | ~50 MB |
| EnhancedResearchEngine | ~100 MB |
| VectorBT (with data) | ~500 MB |
| LangGraph | ~20 MB |
| Total Enhanced Brain | ~600 MB |

### Recommendations

- Run alpha mining sweeps during low-activity periods (weekends)
- Limit VectorBT to 100-200 stocks for reasonable memory usage
- Use `--cycle 60` or higher to avoid excessive CPU usage
- Enable LangGraph only if formal state machine is needed

---

## Troubleshooting

### VectorBT Not Available

```
Error: VectorBT not installed. Run: pip install vectorbt
```

**Solution:**
```bash
pip install vectorbt
```

**Alternative:** Alpha mining will use fallback mode with AlphaLibrary

### Alphalens Not Available

```
Error: Alphalens not installed. Run: pip install alphalens-reloaded
```

**Solution:**
```bash
pip install alphalens-reloaded
```

**Alternative:** Alpha validation will be skipped

### LangGraph Not Available

```
Warning: LangGraph not installed. Run: pip install langgraph
```

**Solution:**
```bash
pip install langgraph langchain-core
```

**Alternative:** Brain will use base think() method instead of LangGraph

### No Cache Data

```
Error: No cached data available
```

**Solution:** Prefetch data first:
```bash
python scripts/prefetch_polygon_universe.py \
    --universe data/universe/optionable_liquid_800.csv \
    --start 2015-01-01 --end 2024-12-31
```

---

## Next Steps

### For Users

1. **Test the integration:**
   ```bash
   pytest tests/test_enhanced_brain_integration.py -v
   ```

2. **Run enhanced brain once:**
   ```bash
   python -m autonomous.enhanced_brain --once
   ```

3. **Start 24/7 operation:**
   ```bash
   python -m autonomous.enhanced_brain --cycle 60
   ```

### For Developers

1. **Add new alpha factors** to `research/alpha_library.py`
2. **Create custom workflows** in `research/alpha_factory.py`
3. **Extend state machine** in `cognitive/brain_graph.py`
4. **Add RAG retrievers** in `cognitive/rag_evaluator.py`

---

## Summary

The enhanced brain integration successfully wires **9 new components** into a unified autonomous system:

✅ **EnhancedResearchEngine** - VectorBT mining, Alphalens validation, AlphaFactory workflows
✅ **EnhancedAutonomousBrain** - Unified discovery system, optional LangGraph
✅ **KobeBrainGraph** - Formal state machine with visualization
✅ **RAGEvaluator** - LLM explanation quality tracking
✅ **AlphaResearchIntegration** - Unified integration layer

**Result:** Kobe transforms from **3 alphas** to **10,000+ variant testing capability** with formal state machines and LLM quality tracking.
