# AI Architecture Prompt: "Project Helios" Upgrade

**Objective:**
Your task is to design a major architectural upgrade for an existing, sophisticated trading bot. This upgrade, codenamed "Project Helios," aims to push the bot's AI capabilities to a new level of autonomous strategy discovery, deep reasoning, and self-explanation.

**Existing Architecture Context:**
The bot already has a layered architecture with a key `cognitive` module. This module includes:
- `CognitiveBrain`: The central decision-making orchestrator.
- `SymbolicReasoner`: A system that validates trades against human-written rules in a YAML file.
- `DynamicPolicyGenerator`: Adjusts risk and trading posture based on market regimes.
- `ReflectionEngine`: Learns from past trades.
- `CuriosityEngine`: Identifies anomalies.
- `SelfModel`: Maintains an understanding of its own performance.
- Various data modules for market data, alternative data (`altdata`), and news (`news_processor`).

You are to design three new pillars that integrate with and enhance this existing structure.

---

### **Pillar 1: The Strategy Foundry**

**Concept:** An offline system that uses Genetic Programming (GP) to autonomously discover new trading strategies from a universe of 800 stocks.

**Requirements:**
1.  **High-Level Design:** Describe how this GP system would work. Detail the five key phases: Initialization, Fitness Evaluation, Selection, Crossover/Mutation, and Repetition.
2.  **Integration:** Propose a file structure (e.g., a new `evolution/foundry.py` module).
3.  **Primitives:** Define a list of "primitive" functions and terminals the GP would use as building blocks (e.g., `SMA(period)`, `RSI(period)`, `sentiment_score`, `market_volume`, operators like `>` and `AND`).
4.  **Output:** Explain how the "fittest" evolved strategies would be exported into a format that the existing `SymbolicReasoner` or a strategy incubator could use.

---

### **Pillar 2: The Metacognitive Loop**

**Concept:** Upgrade the `CognitiveBrain` from a reactive signal-checker to a proactive, hypothesis-driven reasoning engine.

**Requirements:**
1.  **Process Flow:** Detail the new four-stage thought process: Observe & Orient, Hypothesize, Decide & Seek Data, and Act.
2.  **Module Interaction:** Explain how this new loop enhances the roles of the existing `CognitiveBrain`, `CuriosityEngine`, and `DynamicPolicyGenerator`.
3.  **Example Hypothesis:** Provide a concrete example of a hypothesis the bot could generate (e.g., "The tech sector seems overbought...") and the specific data-seeking actions it would take to validate it *before* ever executing a trade.

---

### **Pillar 3: The Socratic Narrator**

**Concept:** A new module that generates hyper-detailed, data-driven narratives explaining *why* a trade was made, reconstructing the entire causal chain of reasoning.

**Requirements:**
1.  **Module Design:** Propose a new module (e.g., `explainability/socratic_narrator.py`) and explain how it would be triggered after a trade.
2.  **Data Sourcing:** List the other modules and data sources it would need to query to build its narrative (e.g., `Journal`, `CognitiveBrain`, config files, `SelfModel`).
3.  **Narrative Generation:** Write a detailed, example trade narrative following the structure below. Be creative and data-rich.
    *   **Action:** What was the trade?
    *   **1. Proximate Cause (The Trigger):** What signal caused it?
    *   **2. Strategic Context (The 'Why Now'):** What was the bot's overall strategy/bias at the time?
    *   **3. Rule-Based Validation (The 'Is it Safe'):** Which key rules did it pass?
    *   **4. Data-Driven Confirmation (The 'What Else Supports This'):** What other data points (sentiment, internal stats) backed the decision?
    *   **5. Considered Alternatives (The 'What I Didn't Do'):** What other actions did it consider and reject, and why?
    *   **Conclusion:** A final summary sentence.

---

**Final Deliverable:**
Produce a comprehensive report that addresses all the requirements above. Flesh out the design, provide pseudo-code for the core loops where appropriate, and clearly show how "Project Helios" would represent a significant leap in the bot's intelligence and autonomy.
