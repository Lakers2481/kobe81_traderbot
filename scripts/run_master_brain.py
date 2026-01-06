#!/usr/bin/env python3
"""
Run Kobe Master Brain v3.0 - THE REAL 24/7 AUTONOMOUS SYSTEM

This is the ULTIMATE brain that:
1. Cycles through NORMAL, ICT, COMPLEX strategies
2. Uses ALL scrapers (Reddit, GitHub, arXiv)
3. Runs cognitive learning (curiosity, reflection)
4. Follows 150+ scheduled tasks
5. Never stops learning and improving

Usage:
    python scripts/run_master_brain.py              # Run forever
    python scripts/run_master_brain.py --once       # Single cycle
    python scripts/run_master_brain.py --status     # Show status
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonomous.master_brain import run

if __name__ == "__main__":
    run()
