# Enhanced Brain Architecture

**Visual representation of the enhanced brain integration**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENHANCED AUTONOMOUS BRAIN                            │
│                         (Version 2.0.0)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ EnhancedResearch │  │  KobeBrainGraph  │  │   RAGEvaluator   │
   │     Engine       │  │   (LangGraph)    │  │                  │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
              │                   │                   │
              ▼                   ▼                   ▼
     9 Alpha Mining        State Machine      LLM Quality
       Components            Nodes              Tracking
```

---

## Layer Architecture

### Layer 1: Base Infrastructure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BASE AUTONOMOUS BRAIN                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ContextBuilder  │  Scheduler  │  ResearchEngine  │  LearningEngine    │
├─────────────────────────────────────────────────────────────────────────┤
│  - Time awareness                                                       │
│  - Market calendar                                                      │
│  - Season detection                                                     │
│  - Task scheduling                                                      │
│  - Parameter experiments                                                │
│  - Trade analysis                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 2: Alpha Mining Infrastructure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ALPHA MINING COMPONENTS                            │
├─────────────────────────────────────────────────────────────────────────┤
│  1. VectorBTMiner      │  Fast parameter sweep (10,000+ variants)      │
│  2. AlphaLibrary       │  91 alpha factors across 6 categories         │
│  3. FactorValidator    │  Alphalens IC analysis                        │
│  4. AlphaFactory       │  Qlib-style workflows                         │
│  5. VectorizedBacktest │  Bulk testing infrastructure                  │
├─────────────────────────────────────────────────────────────────────────┤
│                   AlphaResearchIntegration                              │
│                   (Unified Integration Layer)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 3: Cognitive Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      COGNITIVE COMPONENTS                               │
├─────────────────────────────────────────────────────────────────────────┤
│  6. BrainState         │  TypedDict definitions for state machine      │
│  7. KobeBrainGraph     │  LangGraph formal state machine               │
│  8. RAGEvaluator       │  LLM reasoning quality tracking               │
├─────────────────────────────────────────────────────────────────────────┤
│  State Machine Nodes:                                                   │
│  observe → analyze → decide → execute → reflect → research → standby   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 4: Integration Layer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       INTEGRATION LAYER                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  EnhancedResearchEngine                                                 │
│  ├── run_vectorbt_alpha_sweep()                                         │
│  ├── validate_alpha_with_alphalens()                                    │
│  ├── run_alpha_factory_workflow()                                       │
│  ├── get_top_alphas()                                                   │
│  └── submit_alpha_hypotheses_to_curiosity_engine()                      │
├─────────────────────────────────────────────────────────────────────────┤
│  EnhancedAutonomousBrain                                                │
│  ├── Replaces base ResearchEngine with EnhancedResearchEngine           │
│  ├── Optionally enables KobeBrainGraph (LangGraph)                      │
│  ├── Initializes RAGEvaluator                                           │
│  ├── Unified discovery system (all sources)                             │
│  └── Enhanced background work scheduling                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Discovery Flow

```
Component Discovers Something
         │
         ▼
Creates Discovery Object
         │
         ▼
Brain._check_for_discoveries()
         │
         ├─→ Base ResearchEngine experiments
         ├─→ EnhancedResearchEngine alpha mining
         ├─→ LangGraph state patterns
         ├─→ RAGEvaluator quality improvements
         └─→ External scrapers
         │
         ▼
Brain._alert_discovery()
         │
         ├─→ Structured JSON log (events.jsonl)
         ├─→ Human-readable log
         ├─→ Discovery log file
         └─→ Persistent JSON (discoveries.json)
```

### Alpha Mining Flow

```
EnhancedResearchEngine.run_vectorbt_alpha_sweep()
         │
         ▼
AlphaResearchIntegration.run_alpha_mining_sweep()
         │
         ├─→ VectorBTMiner (if available)
         │    └─→ Test 10,000+ variants
         │
         └─→ AlphaLibrary (fallback)
              └─→ 91 pre-built factors
         │
         ▼
Filter by Sharpe > threshold
         │
         ▼
Create AlphaDiscovery records
         │
         ▼
Save to state/research/alpha_discoveries.json
         │
         ▼
Create Discovery objects for brain alerts
```

### Validation Flow

```
EnhancedResearchEngine.validate_alpha_with_alphalens()
         │
         ▼
Load price data
         │
         ▼
Compute alpha factor (AlphaLibrary)
         │
         ▼
FactorValidator.generate_tearsheet()
         │
         ├─→ IC analysis
         ├─→ Quantile spreads
         ├─→ Statistical significance
         └─→ Factor exposure
         │
         ▼
Update AlphaDiscovery with validation metrics
         │
         └─→ ic_mean, ic_sharpe, statistically_significant
```

---

## State Machine (LangGraph)

### Node Graph

```
        START
          │
          ▼
      ┌───────┐
      │OBSERVE│  ← Loop back
      └───────┘
          │
    ┌─────┼─────┐
    │     │     │
    ▼     ▼     ▼
RESEARCH ANALYZE STANDBY
          │       │
          ▼       │
      ┌──────┐    │
      │DECIDE│    │
      └──────┘    │
          │       │
    ┌─────┼─────┐ │
    │     │     │ │
    ▼     ▼     ▼ ▼
EXECUTE REFLECT STANDBY
    │     │
    └──►──┘
          │
          ▼
         END
```

### Routing Logic

```python
def _route_after_observe(state):
    if kill_switch_active:
        return "standby"
    if phase in [PRE_MARKET, LUNCH, AFTER_HOURS]:
        return "research"
    if phase in [MORNING_SESSION, AFTERNOON_SESSION]:
        return "analyze"
    return "standby"

def _route_after_decide(state):
    if decision == TRADE:
        return "execute"
    if decision == STANDBY:
        return "standby"
    return "reflect"
```

---

## Background Work Scheduling

### Work Mode Schedule

```
Time (ET)          Phase           Work Mode        Activity
────────────────────────────────────────────────────────────────────────
Weekend/Holiday    -               DEEP_RESEARCH    VectorBT alpha mining
4:00-7:00 AM       Pre-market      RESEARCH         Alpha mining (every 3rd cycle)
7:00-9:30 AM       Pre-market      MONITORING       Watchlist prep
9:30-10:00 AM      Opening Range   MONITORING       Observe only (NO TRADES)
10:00-11:30 AM     Morning         ACTIVE_TRADING   Scan, trade, monitor
11:30-14:00 PM     Lunch           RESEARCH         Alpha mining experiments
14:00-15:30 PM     Afternoon       ACTIVE_TRADING   Power hour trading
15:30-16:00 PM     Close           MONITORING       Manage positions
16:00-20:00 PM     After Hours     LEARNING         Trade analysis + RAG eval
20:00-4:00 AM      Night           OPTIMIZATION     Feature analysis + hypotheses
```

### Activity by Work Mode

```
DEEP_RESEARCH (Weekend/Holiday)
├── VectorBT alpha mining sweep
├── Full walk-forward backtests
└── Extended research experiments

RESEARCH (Pre-market, Lunch)
├── Alpha mining (every 3rd cycle)
├── Parameter experiments (other cycles)
└── Data quality checks

LEARNING (After Hours)
├── Trade outcome analysis
├── RAG explanation evaluation
├── Episodic memory updates
└── Daily reflections

OPTIMIZATION (Night)
├── Feature importance analysis
├── Hypothesis submission to CuriosityEngine
├── Model retraining
└── Data prefetching

ACTIVE_TRADING (Trading Hours)
├── Signal scanning
├── Order placement
├── Position monitoring
└── Risk management

MONITORING (Opening Range, Close)
├── Market observation
├── Position management
└── No new entries
```

---

## Component Interactions

### EnhancedResearchEngine ↔ CuriosityEngine

```
EnhancedResearchEngine
         │
         │ get_top_alphas(n=10)
         ▼
Filter alphas (Sharpe > 1.0, validated, significant)
         │
         │ generate_alpha_hypotheses_for_curiosity_engine()
         ▼
Create Hypothesis objects
         │
         │ submit_hypotheses_to_curiosity_engine()
         ▼
CuriosityEngine._hypotheses[hyp_id] = hypothesis
         │
         └─→ CuriosityEngine validates and promotes to edges
```

### RAGEvaluator ↔ Brain

```
Trade Decision Needed
         │
         ▼
RAGEvaluator.generate_explanations(trade_context)
         │
         ├─→ Episodic retriever (past trades)
         ├─→ Semantic retriever (rules)
         └─→ Pattern retriever (historical patterns)
         │
         ▼
Generate explanation text for each retriever
         │
         ▼
RAGEvaluator.evaluate_explanation()
         │
         ├─→ Completeness score
         ├─→ Coherence score
         ├─→ Accuracy score
         ├─→ Actionability score
         └─→ Confidence calibration
         │
         ▼
If avg_score > 85 and n >= 10:
    Create Discovery for brain alert
```

---

## State Persistence

### File Hierarchy

```
state/
├── autonomous/
│   ├── brain_state.json              # Base brain state
│   ├── discoveries.json               # Unified discoveries (all sources)
│   ├── alpha_discoveries.json         # Alpha mining discoveries
│   ├── heartbeat.json                 # Liveness monitoring
│   └── research/
│       └── research_state.json        # Parameter experiments
│
├── research/
│   └── alpha_discoveries.json         # AlphaResearchIntegration state
│
├── rag_evaluation/
│   └── evaluations.json               # RAG explanation evaluations
│
└── watchlist/
    ├── next_day.json                  # Overnight watchlist
    ├── today_validated.json           # Premarket validated
    └── opening_range.json             # Opening observations
```

### Discovery JSON Structure

```json
{
  "type": "alpha_discovery",
  "description": "High-quality alpha 'momentum_rank_5' discovered and validated",
  "source": "enhanced_research_engine",
  "improvement": 0.92,
  "confidence": 0.89,
  "data": {
    "alpha_id": "vbt_momentum_rank_5",
    "alpha_name": "momentum_rank_5",
    "category": "momentum",
    "sharpe_ratio": 1.42,
    "win_rate": 0.63,
    "profit_factor": 1.85,
    "total_trades": 147,
    "ic_mean": 0.092,
    "ic_sharpe": 1.23,
    "statistically_significant": true
  },
  "timestamp": "2026-01-07T14:30:00"
}
```

---

## Performance Benchmarks

### Alpha Mining

```
Operation                          Time        Scales With
─────────────────────────────────────────────────────────────
VectorBT sweep (100 stocks, 10k)   ~30s       Stocks × Variants
Alphalens validation               ~5s        Stocks × Periods
Single parameter experiment        ~3s        Stocks
Hypothesis submission              <1s        Hypotheses
```

### Memory Usage

```
Component                Memory      Notes
─────────────────────────────────────────────────────────────
Base Brain               ~50 MB     Context, scheduler, research
EnhancedResearchEngine   ~100 MB    Alpha discoveries, state
VectorBT (with data)     ~500 MB    100-200 stocks recommended
LangGraph                ~20 MB     State machine
RAGEvaluator             ~30 MB     Evaluations, retrievers
─────────────────────────────────────────────────────────────
Total Enhanced Brain     ~600 MB    With all components
```

---

## Dependency Graph

```
EnhancedAutonomousBrain
    │
    ├─→ AutonomousBrain (base)
    │    ├─→ ContextBuilder
    │    ├─→ AutonomousScheduler
    │    ├─→ LearningEngine
    │    └─→ (overridden) ResearchEngine
    │
    ├─→ EnhancedResearchEngine
    │    ├─→ ResearchEngine (base)
    │    └─→ AlphaResearchIntegration
    │         ├─→ VectorBTMiner (optional, HAS_VBT)
    │         ├─→ AlphaLibrary (fallback)
    │         ├─→ FactorValidator (optional, HAS_ALPHALENS)
    │         └─→ AlphaFactory
    │
    ├─→ KobeBrainGraph (optional, HAS_LANGGRAPH)
    │    └─→ BrainState (TypedDicts)
    │
    └─→ RAGEvaluator
         └─→ EpisodicMemory, SemanticMemory (retrievers)
```

---

## Summary Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    KOBE ENHANCED BRAIN v2.0.0                           │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
│  │  VectorBT    │  │  Alphalens   │  │  LangGraph   │                 │
│  │   Mining     │  │  Validation  │  │    State     │                 │
│  │  10k+ vars   │  │   IC, Q5-Q1  │  │   Machine    │                 │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                 │
│         │                 │                 │                          │
│         └────────┬────────┴────────┬────────┘                          │
│                  │                 │                                   │
│         ┌────────▼─────────────────▼────────┐                          │
│         │  ENHANCED RESEARCH ENGINE         │                          │
│         │  - Alpha mining workflows         │                          │
│         │  - Validation pipelines           │                          │
│         │  - Hypothesis submission          │                          │
│         └────────┬──────────────────────────┘                          │
│                  │                                                     │
│         ┌────────▼─────────────────────────┐                           │
│         │  ENHANCED AUTONOMOUS BRAIN       │                           │
│         │  - Unified discovery system      │                           │
│         │  - Background work scheduling    │                           │
│         │  - RAG quality tracking          │                           │
│         │  - 24/7 operation                │                           │
│         └──────────────────────────────────┘                           │
│                                                                         │
│  From 3 alphas → 10,000+ variant testing capability                    │
└─────────────────────────────────────────────────────────────────────────┘
```
