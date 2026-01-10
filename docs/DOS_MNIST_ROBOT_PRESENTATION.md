# DOS Maps + MNIST + Robot: The Connection Story

**Purpose**: Show how one real project (MNIST NN from scratch) sits on the Domain of Science maps, and how the SAME core loop becomes the blueprint for our robot's learning/decision loop.

---

## PART A: SLIDE-BY-SLIDE OUTLINE (8 Slides)

---

### SLIDE 1: "The Big Picture - Where Knowledge Lives"

**On Screen**: The Donut of Knowledge (DOS poster)

**Callout Areas**:
- Point to **Philosophy** (top) - "Where questions start"
- Point to **Mathematics** (inner ring) - "The language"
- Point to **Computer Science** (between math and engineering)
- Point to **Applied Sciences** - "Where we build things"

**Speaker Notes**:
> "This donut shows ALL human knowledge. Math feeds into Computer Science. Computer Science feeds into Engineering. Engineering builds things like our trading robot. Today we'll trace ONE project through this entire loop."

---

### SLIDE 2: "The Four Math Pillars We Need"

**On Screen**: Map of Mathematics (DOS poster)

**Callout Areas** (circle these 4 regions):
1. **Linear Algebra** (matrices, vectors)
2. **Calculus** (derivatives, chain rule)
3. **Probability & Statistics** (distributions, likelihood)
4. **Optimization** (finding minima, convergence)

**Speaker Notes**:
> "Machine Learning sits on exactly FOUR pillars of math. Every ML algorithm - whether it's recognizing handwritten digits or predicting stock moves - uses ALL FOUR. Let's see exactly where."

---

### SLIDE 3: "Where AI/ML Lives on the CS Map"

**On Screen**: Map of Computer Science (DOS poster)

**Callout Areas**:
- Point to **Artificial Intelligence** section
- Show arrow from **Algorithms** into AI
- Show connection to **Machine Learning** subsection

**Speaker Notes**:
> "On the CS map, Machine Learning is a subfield of AI. But notice - it's BUILT ON those math pillars from the previous slide. The Math map is the foundation. The CS map is the application. Now let's see a REAL example."

---

### SLIDE 4: "The MNIST Challenge - Our Example Project"

**On Screen**:
- MNIST digit grid (the 0-9 handwritten samples)
- Simple diagram: `28x28 pixels → [?] → Digit 0-9`

**Callout**:
- "784 input numbers (28 x 28 = 784 pixels)"
- "10 output numbers (probability for each digit)"

**Speaker Notes**:
> "MNIST is THE classic first neural network project. 70,000 images of handwritten digits. The goal: look at pixels, output which digit (0-9). We'll build this from SCRATCH in NumPy - no TensorFlow, no PyTorch. Just raw math."

---

### SLIDE 5: "The Four Pillars IN the Code"

**On Screen**: Split view showing:
- Left: Code snippet from Kaggle notebook
- Right: Math pillar it connects to

```
CODE BLOCK                          MATH PILLAR
─────────────────────────────────────────────────────
z = W @ x + b                       LINEAR ALGEBRA
                                    (matrix multiply)

dW = dz @ x.T                       CALCULUS
                                    (chain rule / backprop)

probs = softmax(z)                  PROBABILITY
loss = cross_entropy(probs, y)      (likelihood, entropy)

W = W - lr * dW                     OPTIMIZATION
                                    (gradient descent)
```

**Speaker Notes**:
> "THIS is the magic. Four lines of math. Four pillars.
> - Line 1: Matrix multiplication. Linear algebra.
> - Line 2: Chain rule derivative. Calculus.
> - Line 3: Softmax gives probabilities. Stats.
> - Line 4: Gradient descent update. Optimization.
> That's it. That's a neural network. That's THE core loop."

---

### SLIDE 6: "Now Watch - The Robot Uses THE SAME Loop"

**On Screen**: Side-by-side comparison diagram

```
┌─────────────────────────────────┬─────────────────────────────────┐
│        MNIST NEURAL NET         │         TRADING ROBOT           │
├─────────────────────────────────┼─────────────────────────────────┤
│                                 │                                 │
│  INPUT: 784 pixel values        │  INPUT: prices, indicators,     │
│         (28x28 image)           │         news, volume (features) │
│                                 │                                 │
│  FORWARD PASS:                  │  FORWARD PASS:                  │
│  z = W @ x + b                  │  score = model(features)        │
│  → predict digit 0-9            │  → predict: BUY / SELL / HOLD   │
│                                 │                                 │
│  LOSS FUNCTION:                 │  LOSS FUNCTION:                 │
│  "How wrong was I?"             │  "How wrong was I?"             │
│  cross_entropy(pred, actual)    │  -PnL, drawdown, regret         │
│                                 │                                 │
│  BACKPROP / UPDATE:             │  BACKPROP / UPDATE:             │
│  dW = gradient of loss          │  Tune params in backtest        │
│  W = W - lr * dW                │  Optimize via walk-forward      │
│                                 │                                 │
│  EVAL:                          │  EVAL:                          │
│  Test accuracy on held-out      │  Walk-forward out-of-sample     │
│  digits (don't cheat!)          │  metrics (don't overfit!)       │
│                                 │                                 │
│  DEPLOY:                        │  DEPLOY:                        │
│  Use model on new images        │  Paper trade first → then LIVE  │
│                                 │  (with explicit approval only)  │
│                                 │                                 │
└─────────────────────────────────┴─────────────────────────────────┘
```

**Speaker Notes**:
> "SAME. EXACT. LOOP. The only difference is:
> - MNIST input = pixels. Robot input = market data.
> - MNIST loss = cross-entropy. Robot loss = negative P&L.
> - MNIST deploy = run on new images. Robot deploy = paper first, then LIVE with gates.
> The math is identical. The structure is identical."

---

### SLIDE 7: "The Robot's Body - Same Structure"

**On Screen**: Robot architecture diagram with labels

```
┌──────────────────────────────────────────────────────────────────┐
│                        TRADING ROBOT                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   SENSORS (Data Ingestion)                                       │
│   ├── Price bars (OHLCV)                                         │
│   ├── Technical indicators (RSI, ATR, etc.)                      │
│   ├── Regime detection (HMM)                                     │
│   └── News / sentiment                                           │
│                           ↓                                      │
│   BRAIN (Decision Layer)        ← This is the "forward pass"     │
│   ├── Signal scoring                                             │
│   ├── Confidence calculation                                     │
│   ├── Kelly position sizing                                      │
│   └── Risk gates (VaR, exposure)                                 │
│                           ↓                                      │
│   MUSCLES (Execution)                                            │
│   ├── Order placement                                            │
│   ├── IOC LIMIT orders                                           │
│   └── Fill tracking                                              │
│                           ↓                                      │
│   MEMORY (State + Learning)     ← This is "backprop over time"   │
│   ├── Episodic memory (trades)                                   │
│   ├── Semantic memory (patterns)                                 │
│   ├── Portfolio state                                            │
│   └── Hash chain audit log                                       │
│                           ↓                                      │
│   HEARTBEAT (Monitoring Loop)                                    │
│   ├── 60-second cycles                                           │
│   ├── Health checks                                              │
│   ├── Self-healing                                               │
│   └── Daily reflections                                          │
│                           ↓                                      │
│   SAFETY GATES                  ← This is "responsible deploy"   │
│   ├── Paper mode vs Live mode                                    │
│   ├── Kill switch                                                │
│   ├── Max risk limits                                            │
│   └── Human approval required                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Speaker Notes**:
> "Sensors = inputs. Brain = forward pass. Memory = learning from outcomes. Safety gates = responsible deployment. It's the MNIST loop, scaled up to handle real money. And SAFETY is built in - paper mode first, kill switch always available, human approval for live."

---

### SLIDE 8: "The Punchline - Why This Matters"

**On Screen**:
- Donut of Knowledge (zoomed out view)
- Arrow path: Math → CS → Engineering → Robot

**Key Message** (large text):
```
"If you understand the MNIST loop,
 you understand the ROBOT loop.

 Same math. Same structure. Same discipline.

 The difference? SAFETY GATES.
 Paper first. Approval required. Always."
```

**Speaker Notes**:
> "The Donut closes the loop. Philosophy asks: can machines learn? Math provides the language. CS builds the algorithms. Engineering deploys them responsibly. Our robot is the end of that chain - but it starts with understanding a simple neural network. Master MNIST, and you have the foundation to build anything."

---

## PART B: CHEAT SHEET - MNIST NN Block → Robot Block

| MNIST Neural Network | Trading Robot | Math Pillar |
|---------------------|---------------|-------------|
| **Input**: 784 pixel values | **Sensors**: prices, indicators, news | Data representation |
| **Weights W**: learned parameters | **Model params**: thresholds, coefficients | Linear Algebra |
| **Forward pass**: `z = Wx + b` | **Scoring**: `conf_score = f(features)` | Linear Algebra |
| **Activation**: ReLU, softmax | **Gating**: regime filter, confidence threshold | Nonlinear transforms |
| **Output**: P(digit=0), P(digit=1), ... | **Output**: BUY signal with confidence | Probability |
| **Loss**: cross-entropy | **Loss**: -P&L, Sharpe, drawdown | Optimization target |
| **Backprop**: `dL/dW` via chain rule | **Tuning**: backtest optimization | Calculus |
| **Gradient descent**: `W -= lr * dW` | **Walk-forward**: update params on new data | Optimization |
| **Batch training**: mini-batches | **Online learning**: incremental updates | Stochastic methods |
| **Regularization**: dropout, L2 | **Risk gates**: position limits, VaR | Preventing overfit |
| **Test accuracy**: held-out digits | **OOS metrics**: walk-forward validation | Generalization |
| **Deploy**: run on new images | **Deploy**: paper → live with gates | Responsible AI |

---

## PART C: ROBOT TERM GLOSSARY

| Robot Term | What It Is | MNIST Equivalent |
|------------|------------|------------------|
| **Sensors** | Data ingestion (prices, indicators, news) | Input layer (pixel values) |
| **Brain** | Decision layer (scoring, confidence, rules) | Hidden layers + output |
| **Muscles** | Execution (placing orders) | Using the prediction |
| **Memory** | State + logs + positions | Training history |
| **Heartbeat** | Monitoring loop (60s cycles) | Training loop iterations |
| **Safety Gates** | Paper vs live, kill switch, max risk | Validation before deploy |
| **Episodic Memory** | Specific trade experiences | Individual training examples |
| **Semantic Memory** | Learned patterns/rules | Learned weights |
| **Reflection Engine** | Learn from outcomes | Backprop / loss analysis |
| **Self-Healing** | Auto-fix issues (reconnect, cleanup) | Error handling in training |

---

## PART D: 30-60 SECOND CLOSING SCRIPT

> "So here's the punchline.
>
> A neural network that recognizes handwritten digits and a trading robot that makes market decisions - they're the SAME LOOP.
>
> Inputs. Forward pass. Loss function. Backprop. Evaluation. Deployment.
>
> The math is identical: linear algebra for the forward pass, calculus for the gradients, probability for the outputs, optimization to improve.
>
> The only difference? When you deploy a digit classifier wrong, you misread a 7 as a 1. When you deploy a trading robot wrong, you lose real money.
>
> That's why we have SAFETY GATES built into every layer:
> - Paper mode before live mode
> - Kill switch always available
> - Human approval required for any live trading
> - Position limits and risk caps enforced automatically
>
> If you can build MNIST from scratch in NumPy, you can understand every component of this robot. Same math. Same loop. Same discipline.
>
> The difference is respect for the consequences.
>
> Master the fundamentals. Build responsibly. That's how you go from a toy project to a production system."

---

## VISUAL REFERENCE: The Core Loop

```
     ┌──────────────────────────────────────────────────────────────┐
     │                    THE UNIVERSAL LEARNING LOOP               │
     └──────────────────────────────────────────────────────────────┘

              MNIST                              ROBOT

         ┌─────────┐                        ┌─────────┐
         │  INPUT  │                        │ SENSORS │
         │ pixels  │                        │  data   │
         └────┬────┘                        └────┬────┘
              │                                  │
              ▼                                  ▼
         ┌─────────┐                        ┌─────────┐
         │ FORWARD │  z = Wx + b            │  BRAIN  │  score = f(features)
         │  PASS   │                        │ scoring │
         └────┬────┘                        └────┬────┘
              │                                  │
              ▼                                  ▼
         ┌─────────┐                        ┌─────────┐
         │  LOSS   │  cross_entropy         │  LOSS   │  -P&L, risk
         │ how off │                        │ how off │
         └────┬────┘                        └────┬────┘
              │                                  │
              ▼                                  ▼
         ┌─────────┐                        ┌─────────┐
         │ BACKPROP│  dW = chain rule       │ TUNING  │  backtest optimize
         │ update  │                        │ update  │
         └────┬────┘                        └────┬────┘
              │                                  │
              ▼                                  ▼
         ┌─────────┐                        ┌─────────┐
         │  EVAL   │  test accuracy         │  EVAL   │  walk-forward OOS
         │ did it  │                        │ did it  │
         │ learn?  │                        │ learn?  │
         └────┬────┘                        └────┬────┘
              │                                  │
              ▼                                  ▼
         ┌─────────┐                        ┌─────────┐
         │ DEPLOY  │  use on new data       │ DEPLOY  │  PAPER FIRST
         │         │                        │         │  then LIVE + GATES
         └─────────┘                        └─────────┘
```

---

## SAFETY REMINDER

**When discussing deployment / live trading:**

1. **Paper mode FIRST** - Always test with simulated money
2. **Live mode ONLY with explicit approval** - Human must authorize
3. **Safety gates always active**:
   - Kill switch (`state/KILL_SWITCH`)
   - Position limits (10% per position, 20% daily)
   - VaR gate (portfolio risk limit)
   - Policy gate ($75/order cap)
4. **No instructions for real-money trading actions** - Keep conceptual

---

## SOURCES

- [Domain of Science Maps](https://dosmaps.com/) - Official site for all DOS posters
- [The Map of Mathematics](https://store.dftba.com/products/map-of-mathematics-poster) - DFTBA Store
- [The Map of Computer Science](https://www.redbubble.com/i/poster/Map-of-Computer-Science-by-DominicWalliman/27929629.LVTDI) - Redbubble
- [The Donut of Knowledge](https://store.dftba.com/products/domain-of-science-the-donut-of-knowledge-poster) - DFTBA Store
- [Dominic Walliman's Site](https://dominicwalliman.com/post/178257063655/the-donut-of-knowledge-summarises-all-of-the) - Creator of DOS maps

---

*Created: 2026-01-08*
*For: Kobe Trading Robot Educational Materials*
