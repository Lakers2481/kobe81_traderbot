"""
FROZEN KOBE STANDARD PIPELINE
=============================

THIS FILE IS FROZEN. DO NOT MODIFY WITHOUT EXPLICIT USER APPROVAL.

Created: 2026-01-07
Purpose: Define the IMMUTABLE trading pipeline that KOBE follows FOREVER.

THE PIPELINE:
=============

    ┌─────────────────────────────────────────────────────────────────┐
    │                    KOBE STANDARD PIPELINE                        │
    │                         800 → 5 → 2                              │
    │                                                                   │
    │   "We scan 800, study 5, trade 2. Every single day. Forever."   │
    └─────────────────────────────────────────────────────────────────┘

                            800 STOCKS
                         (Full Universe)
                               │
                               ▼
                    ┌──────────────────┐
                    │  DualStrategy    │
                    │  Scanner         │
                    │  (IBS+RSI +      │
                    │   TurtleSoup)    │
                    └──────────────────┘
                               │
                               ▼
                        30-50 SIGNALS
                      (Raw, unfiltered)
                               │
                               ▼
                    ┌──────────────────┐
                    │  Quality Gate    │
                    │  Score >= 70     │
                    │  Conf >= 0.60    │
                    │  R:R >= 1.5:1    │
                    └──────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────┐
            │           TOP 5 TO STUDY          │
            │  ─────────────────────────────── │
            │  • Follow these ALL DAY           │
            │  • Paper trade them               │
            │  • Analyze price action           │
            │  • Compare predicted vs actual    │
            │  • Learn from wins AND losses     │
            │  • Update pattern recognition     │
            │  • Feed learnings back to algo    │
            │                                   │
            │  Output: logs/daily_top5.csv      │
            └──────────────────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────┐
            │           TOP 2 TO TRADE          │
            │  ─────────────────────────────── │
            │  • ONLY these get real capital   │
            │  • Highest conviction from Top 5 │
            │  • Full position sizing applied  │
            │  • Risk gates enforced           │
            │  • Execute with discipline       │
            │                                   │
            │  Output: logs/tradeable.csv       │
            └──────────────────────────────────┘


WHY THIS WORKS (Quant Philosophy):
==================================

1. UNIVERSE (800)
   - Liquid stocks (ADV > $1M)
   - Optionable (institutional interest)
   - 10+ years history (backtestable)
   - No penny stocks, no illiquid garbage

2. SCANNER (DualStrategy)
   - Two uncorrelated strategies
   - IBS+RSI: Mean reversion
   - TurtleSoup: Momentum/breakout
   - Combined = diversified edge

3. TOP 5 TO STUDY
   - This is your RESEARCH lab
   - Paper trade ALL 5
   - Win rate doesn't matter here
   - LEARNING is the goal
   - Every signal teaches something

4. TOP 2 TO TRADE
   - This is your EXECUTION
   - Only highest conviction
   - Full risk management
   - Real capital at risk
   - Discipline > frequency


THE LEARNING LOOP:
==================

    TOP 5 signals generated
           │
           ▼
    Follow all day (paper)
           │
           ▼
    End of day: What happened?
           │
           ├──► Did predicted move occur?
           ├──► Was timing right?
           ├──► Was entry level hit?
           ├──► How did stop/target perform?
           │
           ▼
    Feed learnings back
           │
           ├──► Update pattern recognition
           ├──► Refine quality gate
           ├──► Improve confidence scoring
           │
           ▼
    Tomorrow: Better signals


THIS IS HOW QUANTS WORK:
========================

1. They DON'T trade every signal
2. They STUDY every signal
3. They LEARN from every outcome
4. They REFINE continuously
5. They EXECUTE with discipline

"The goal is not to trade more. The goal is to trade BETTER."


FROZEN PARAMETERS:
==================
"""

# FROZEN - DO NOT CHANGE
UNIVERSE_SIZE = 800
SIGNALS_EXPECTED = (30, 50)  # Expected range per day
TOP_N_STUDY = 5              # How many to paper/study
TOP_N_TRADE = 2              # How many to actually trade

# Quality Gate Thresholds
MIN_SCORE = 70
MIN_CONFIDENCE = 0.60
MIN_RR_RATIO = 1.5

# Files
UNIVERSE_FILE = "data/universe/optionable_liquid_800.csv"
TOP5_OUTPUT = "logs/daily_top5.csv"
TRADEABLE_OUTPUT = "logs/tradeable.csv"
SIGNALS_LOG = "logs/signals.jsonl"

# Scanner
SCANNER_CLASS = "DualStrategyScanner"
STRATEGIES = ["IBS+RSI", "TurtleSoup"]

# This pipeline is FROZEN
FROZEN = True
FROZEN_DATE = "2026-01-07"
FROZEN_BY = "User directive: 'FREEZE THIS FOREVER'"


def validate_pipeline():
    """Validate that the pipeline hasn't been tampered with."""
    assert UNIVERSE_SIZE == 800, "Universe must be 800 stocks"
    assert TOP_N_STUDY == 5, "Must study Top 5"
    assert TOP_N_TRADE == 2, "Must trade Top 2"
    assert MIN_SCORE == 70, "Min score must be 70"
    assert FROZEN == True, "Pipeline must remain frozen"
    return True


def get_pipeline_summary() -> str:
    """Return pipeline summary for logging."""
    return f"""
KOBE STANDARD PIPELINE (FROZEN {FROZEN_DATE})
============================================
Universe:     {UNIVERSE_SIZE} stocks
Scanner:      {SCANNER_CLASS}
Strategies:   {', '.join(STRATEGIES)}
Quality Gate: Score >= {MIN_SCORE}, Conf >= {MIN_CONFIDENCE}, R:R >= {MIN_RR_RATIO}:1
Study:        Top {TOP_N_STUDY} (paper, learn, analyze)
Trade:        Top {TOP_N_TRADE} (execute with discipline)
============================================
"""


if __name__ == "__main__":
    print(get_pipeline_summary())
    validate_pipeline()
    print("Pipeline validation: PASSED")
