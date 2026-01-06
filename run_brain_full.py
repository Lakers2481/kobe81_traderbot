#!/usr/bin/env python3
"""
KOBE MASTER BRAIN v4.0 - FULL VISIBILITY
==========================================
Run this to see EVERYTHING working in real-time.

Every task. Every component. Every time. Full logging.
IF YOU DON'T SEE IT, SOMETHING IS WRONG.

Usage:
    python run_brain_full.py              # Run forever
    python run_brain_full.py --schedule   # Print today's schedule
    python run_brain_full.py --once       # Run one cycle
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Run the full brain
from autonomous.master_brain_full import run

if __name__ == "__main__":
    run()
