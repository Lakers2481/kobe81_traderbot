#!/usr/bin/env python3
"""
KOBE MASTER BRAIN v4.0 - FULL VISIBILITY
==========================================
Run this to see EVERYTHING working in real-time.

Every task. Every component. Every time. Full logging.
IF YOU DON'T SEE IT, SOMETHING IS WRONG.

Usage:
    python run_brain.py              # Run forever with full visibility
    python run_brain.py --schedule   # Print today's schedule
    python run_brain.py --once       # Run one cycle
    python run_brain.py --simple     # Run simple brain (no scheduled times)
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress TensorFlow/protobuf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kobe Master Brain")
    parser.add_argument("--simple", action="store_true", help="Run simple brain without scheduled times")
    parser.add_argument("--schedule", action="store_true", help="Print today's schedule")
    parser.add_argument("--once", action="store_true", help="Run one cycle")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    args, remaining = parser.parse_known_args()

    # Pass remaining args to the brain
    sys.argv = [sys.argv[0]] + remaining

    if args.simple:
        # Run simple brain (old behavior)
        from autonomous.master_brain import run
        run()
    else:
        # Run FULL VISIBILITY brain with scheduled times
        from autonomous.master_brain_full import run
        run()
