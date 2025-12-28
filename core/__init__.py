"""
Core Infrastructure
====================

Foundational components for the Kobe trading system.

Components:
- hash_chain: Tamper-proof audit trail
- structured_log: JSON event logging
- config_pin: Configuration signature verification
- kill_switch: Emergency trading halt
- regime_filter: Market regime detection
- earnings_filter: Earnings proximity filtering
- rate_limiter: API rate limiting
- journal: Trade journaling
- lineage: Data lineage tracking
"""

from .hash_chain import append_block, verify_chain
from .structured_log import jlog, read_recent_logs
from .config_pin import sha256_file
from .kill_switch import is_kill_switch_active, activate_kill_switch, deactivate_kill_switch

__all__ = [
    # Hash Chain (Audit)
    'append_block',
    'verify_chain',
    # Structured Logging
    'jlog',
    'read_recent_logs',
    # Config Pin
    'sha256_file',
    # Kill Switch
    'is_kill_switch_active',
    'activate_kill_switch',
    'deactivate_kill_switch',
]
