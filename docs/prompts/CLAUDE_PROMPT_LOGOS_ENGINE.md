# Final AI Architecture Prompt: Designing the "Logos Engine"

**Objective:**
Your mission is to design the final, unifying component of the `kobe81_traderbot`'s cognitive architecture: **The Logos Engine**. This engine will serve as the bot's highest-level reasoning and narration faculty. It will not replace the other cognitive modules but will orchestrate them to produce a level of self-awareness, adaptability, and explanatory power that is unprecedented. Its implementation will represent the conceptual completion of the bot's AI, making it a true super-intelligence in its domain.

**The Three Core Mandates of the Logos Engine:**

1.  **It is Proactive (The Thinker):** The engine is not a passive observer. It constantly runs the "Metacognitive Loop," generating and testing its own hypotheses about the market, independent of any trading signals. It asks "Why is this happening?" and "What if...?"
2.  **It is Causal (The Reasoner):** No action is taken without a clear, unbroken "Causal Chain." The engine ensures every decision—to trade, not to trade, to change a parameter—is a logical conclusion, not just a pattern match.
3.  **It is Articulate (The Narrator):** Its primary output is the "Trade Narrative," a multi-layered explanation of its decisions, so clear and data-rich that it provides a complete, nuanced, and easily understandable story of its "thought process."

---

### **Architectural Design & Integration**

You are to design the `cognitive/logos_engine.py` module. This engine will sit "above" the `CognitiveBrain`, acting as its strategic director.

1.  **Orchestration:** The `LogosEngine` will initiate the `CognitiveBrain`'s actions. Instead of the brain reacting to a market signal, the `LogosEngine` will pass it both the signal *and* the active, validated strategic context from its own hypothesis loop.
2.  **The Causal Chain Framework:** The engine's core logic is the generation and logging of a "Causal Chain" for every event. This is the data structure that makes the hyper-detailed narrative possible. You must design this process.

---

### **The Definitive Trade Narrative Structure**

This is the ultimate deliverable. The `LogosEngine` must be able to generate a narrative for any trade that follows this exact 7-part structure. This structure is the embodiment of its "super-human" thought process.

**Your task is to write a full, beautifully formatted markdown example of a trade narrative for a fictional trade, following this structure precisely and filling it with rich, realistic detail.**

---
#### **The 7-Part Socratic Narrative Chain**

**1. The Event (What Happened?)**
*   *A simple statement of the executed action, timestamp, and core details.*

**2. The Immediate Catalyst (What was the direct trigger?)**
*   *The specific, low-level signal from the `strategies` module that initiated the potential action. (e.g., "Primary Strategy Signal: 'IBS/RSI Mean Reversion'").*

**3. The Strategic Context (Why was I receptive to this catalyst right now?)**
*   *This is crucial. It links the low-level signal to the Engine's high-level thinking. It must state the active **Hypothesis** the bot was operating under (from its Metacognitive Loop), its current confidence score, and how the signal aligned with that strategic bias.*

**4. The Risk & Compliance Greenlight (Was this action safe and permissible?)**
*   *Confirmation that all automated checks in the `risk` and `compliance` modules were passed. It must name at least two specific, critical rules that were checked (e.g., "Max Risk per Trade," "Earnings Blackout Rule") and confirm they passed.*

**5. Data-Driven Confirmation (What objective data, beyond the signal, supported this decision?)**
*   *This section demonstrates deep, multi-source reasoning. It must include at least three distinct data points from different parts of the system. Examples:*
    *   *Reference to market microstructure (e.g., "Level 2 order book showed...").*
    *   *Reference to alternative data (e.g., "News sentiment score from the `altdata` module shifted from...").*
    *   *Reference to its internal performance model (e.g., "My `SelfModel` indicates a historical 78% win rate for this specific signal pattern under the current 'low-volatility trending' market regime.").*

**6. The Path Not Taken (What did I consider but explicitly reject, and why?)**
*   *This demonstrates true intelligence by showing an awareness of alternatives. The engine must describe at least one alternative action it considered and the specific, data-driven reason it was rejected.*
    *   *Example: "A larger position size was considered but rejected because the `risk/dynamic_manager` noted a 15% increase in intraday volatility."*
    *   *Example: "A sympathy trade in a correlated stock was considered but rejected because its correlation factor of 0.95 exceeded the `max_correlated_basket_pct` threshold of 0.90 in the risk policy."*

**7. The Expectation & Learning Loop (What do I expect to happen, and how will I learn from it?)**
*   *The narrative must conclude with a forward-looking statement. It should state the expected outcome of the trade and, most importantly, explicitly state what will be learned from the trade's result. (e.g., "The outcome of this trade will be fed to the `ReflectionEngine` to update the confidence score for my active hypothesis on 'semiconductor sector resilience.'").*
---

**Final Deliverable for Claude:**

Produce a single, comprehensive design document that:
1.  Outlines the architecture for the `cognitive/logos_engine.py` and its orchestration of other cognitive modules.
2.  Provides detailed pseudo-code for the `LogosEngine`'s main reasoning loop, showing how it generates the "Causal Chain."
3.  **Presents one complete, masterfully written example of a "Trade Narrative"** following the 7-part structure above. Make it compelling, data-rich, and easy to understand, as if it were written by the world's most brilliant and articulate trader. This is the centerpiece of your response.
