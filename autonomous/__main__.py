"""
Entry point for running autonomous module with python -m autonomous.run

Usage:
    python -m autonomous.run --status
    python -m autonomous.run --demo
    python -m autonomous.run --start
    python -m autonomous.run --weekend
"""

from autonomous.run import main

if __name__ == "__main__":
    main()
