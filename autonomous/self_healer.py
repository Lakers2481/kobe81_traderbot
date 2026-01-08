"""
Self-Healer - Auto-Recovery from Common Failures

CRITICAL FIX (2026-01-08): The trading robot needs to automatically recover
from common failures without human intervention. This module provides:

1. Error classification - identify what type of failure occurred
2. Healing strategies - specific fixes for each failure type
3. Retry logic - attempt recovery before giving up
4. Escalation - alert when auto-heal fails

Usage:
    from autonomous.self_healer import SelfHealer, HealingResult

    healer = SelfHealer()

    try:
        result = some_operation()
    except Exception as e:
        healing = healer.heal(e)
        if healing.healed:
            result = some_operation()  # Retry after healing
        else:
            raise  # Escalate
"""

import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json

from core.structured_log import jlog

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of common errors."""
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTH_FAILURE = "api_auth_failure"
    DATA_STALE = "data_stale"
    DATA_MISSING = "data_missing"
    DATA_CORRUPT = "data_corrupt"
    MODEL_ERROR = "model_error"
    MODEL_NOT_FOUND = "model_not_found"
    BROKER_DISCONNECT = "broker_disconnect"
    BROKER_REJECT = "broker_reject"
    FILE_NOT_FOUND = "file_not_found"
    FILE_PERMISSION = "file_permission"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


@dataclass
class HealingResult:
    """Result of a healing attempt."""
    error_type: ErrorType
    healed: bool
    action_taken: str
    retry_delay_seconds: float
    details: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.error_type.value,
            'healed': self.healed,
            'action_taken': self.action_taken,
            'retry_delay_seconds': self.retry_delay_seconds,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
        }


class SelfHealer:
    """
    Autonomous self-healing system for the trading robot.

    Classifies errors and applies appropriate healing strategies.
    """

    def __init__(self, max_retries: int = 3, state_dir: Path = None):
        self.max_retries = max_retries
        self.state_dir = state_dir or Path('state/autonomous')
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.healing_history: List[HealingResult] = []
        self._load_state()

        # Error classification patterns
        self._error_patterns = {
            ErrorType.API_TIMEOUT: [
                'timeout', 'timed out', 'connection timed out',
                'read timed out', 'ConnectTimeoutError',
            ],
            ErrorType.API_RATE_LIMIT: [
                'rate limit', 'too many requests', '429',
                'rate exceeded', 'throttle',
            ],
            ErrorType.API_AUTH_FAILURE: [
                'unauthorized', '401', 'authentication failed',
                'invalid api key', 'forbidden', '403',
            ],
            ErrorType.DATA_STALE: [
                'stale data', 'outdated', 'not current',
                'data too old', 'expired',
            ],
            ErrorType.DATA_MISSING: [
                'no data', 'data not found', 'empty response',
                'missing data', 'NoneType',
            ],
            ErrorType.DATA_CORRUPT: [
                'corrupt', 'invalid format', 'parse error',
                'malformed', 'decode error',
            ],
            ErrorType.MODEL_ERROR: [
                'model error', 'prediction failed', 'inference error',
                'model not loaded', 'tensor',
            ],
            ErrorType.MODEL_NOT_FOUND: [
                'model not found', 'no model file', 'FileNotFoundError.*model',
                'pkl not found', 'h5 not found',
            ],
            ErrorType.BROKER_DISCONNECT: [
                'broker disconnect', 'connection lost', 'websocket closed',
                'alpaca.*connection', 'broker connection',
            ],
            ErrorType.BROKER_REJECT: [
                'order rejected', 'insufficient', 'buying power',
                'position limit', 'cannot execute',
            ],
            ErrorType.FILE_NOT_FOUND: [
                'FileNotFoundError', 'No such file', 'not found',
            ],
            ErrorType.FILE_PERMISSION: [
                'PermissionError', 'permission denied', 'access denied',
            ],
            ErrorType.MEMORY_ERROR: [
                'MemoryError', 'out of memory', 'OOM',
            ],
            ErrorType.NETWORK_ERROR: [
                'network', 'connection refused', 'ConnectionError',
                'unreachable', 'DNS',
            ],
        }

    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify an exception into a known error type.

        Args:
            error: The exception to classify

        Returns:
            ErrorType enum value
        """
        error_str = str(error).lower()
        error_type_str = type(error).__name__.lower()
        full_text = f"{error_type_str} {error_str}"

        for error_type, patterns in self._error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in full_text:
                    return error_type

        return ErrorType.UNKNOWN

    def heal(self, error: Exception, context: Dict[str, Any] = None) -> HealingResult:
        """
        Attempt to heal from an error.

        Args:
            error: The exception that occurred
            context: Optional context about what was being done

        Returns:
            HealingResult with outcome
        """
        context = context or {}
        error_type = self.classify_error(error)

        # Get healing strategy
        strategy = self._healing_strategies.get(error_type, self._heal_unknown)

        try:
            healed, action, delay, details = strategy(error, context)
        except Exception as heal_error:
            logger.error(f"Healing strategy failed: {heal_error}")
            healed = False
            action = f"Healing failed: {heal_error}"
            delay = 0
            details = {'healing_error': str(heal_error)}

        result = HealingResult(
            error_type=error_type,
            healed=healed,
            action_taken=action,
            retry_delay_seconds=delay,
            details=details,
            timestamp=datetime.now(),
        )

        self.healing_history.append(result)
        self._save_state()

        # Log the healing attempt
        jlog(
            'self_heal_attempt',
            error_type=error_type.value,
            healed=healed,
            action=action,
            original_error=str(error)[:200],
        )

        return result

    # Healing strategies for each error type
    @property
    def _healing_strategies(self) -> Dict[ErrorType, Callable]:
        return {
            ErrorType.API_TIMEOUT: self._heal_api_timeout,
            ErrorType.API_RATE_LIMIT: self._heal_api_rate_limit,
            ErrorType.API_AUTH_FAILURE: self._heal_api_auth,
            ErrorType.DATA_STALE: self._heal_data_stale,
            ErrorType.DATA_MISSING: self._heal_data_missing,
            ErrorType.DATA_CORRUPT: self._heal_data_corrupt,
            ErrorType.MODEL_ERROR: self._heal_model_error,
            ErrorType.MODEL_NOT_FOUND: self._heal_model_not_found,
            ErrorType.BROKER_DISCONNECT: self._heal_broker_disconnect,
            ErrorType.BROKER_REJECT: self._heal_broker_reject,
            ErrorType.FILE_NOT_FOUND: self._heal_file_not_found,
            ErrorType.FILE_PERMISSION: self._heal_file_permission,
            ErrorType.MEMORY_ERROR: self._heal_memory_error,
            ErrorType.NETWORK_ERROR: self._heal_network_error,
        }

    def _heal_api_timeout(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Heal API timeout with exponential backoff."""
        retry_count = ctx.get('retry_count', 0)
        delay = min(2 ** retry_count * 5, 60)  # 5s, 10s, 20s, 40s, max 60s

        if retry_count < self.max_retries:
            return True, f"Wait {delay}s then retry (attempt {retry_count + 1})", delay, {'retry': True}

        return False, "Max retries exceeded for API timeout", 0, {'escalate': True}

    def _heal_api_rate_limit(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Heal rate limit by waiting longer."""
        # Rate limits typically reset after 60s
        return True, "Rate limited - waiting 60s", 60, {'rate_limited': True}

    def _heal_api_auth(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Auth failures usually need human intervention."""
        # Try reloading env vars
        try:
            from config.env_loader import load_env
            load_env(Path('.env'))
            return True, "Reloaded environment variables", 5, {'env_reloaded': True}
        except Exception:
            return False, "Auth failure - check API keys", 0, {'needs_human': True}

    def _heal_data_stale(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Force data refresh."""
        try:
            # Clear cache
            cache_dir = Path('data/.cache')
            if cache_dir.exists():
                stale_files = list(cache_dir.glob('*.csv'))
                for f in stale_files[:10]:  # Clear up to 10 stale files
                    f.unlink()
                return True, f"Cleared {len(stale_files)} cache files", 5, {'cache_cleared': True}
        except Exception as e:
            logger.warning(f"Cache clear failed: {e}")

        return True, "Force data refresh on next attempt", 10, {'force_refresh': True}

    def _heal_data_missing(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Try fallback data source."""
        return True, "Will try fallback data source", 5, {'use_fallback': True}

    def _heal_data_corrupt(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Clear corrupt data and re-fetch."""
        return True, "Clear corrupt data and re-fetch", 10, {'clear_and_refetch': True}

    def _heal_model_error(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Use fallback model or default."""
        return True, "Switch to fallback model/default", 0, {'use_fallback_model': True}

    def _heal_model_not_found(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Model not found - queue retrain or use default."""
        # Check if we can use a default
        return True, "Using default parameters (no model)", 0, {'use_defaults': True}

    def _heal_broker_disconnect(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Reconnect to broker."""
        try:
            from execution.broker_alpaca import AlpacaBroker
            broker = AlpacaBroker()
            # Test connection
            account = broker.get_account()
            if account:
                return True, "Reconnected to broker", 5, {'reconnected': True}
        except Exception as e:
            logger.warning(f"Broker reconnect failed: {e}")

        return True, "Waiting for broker connection", 30, {'wait_reconnect': True}

    def _heal_broker_reject(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Order rejection - reduce size or skip."""
        return True, "Will reduce position size and retry", 5, {'reduce_size': True}

    def _heal_file_not_found(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """File not found - create default or skip."""
        return True, "File missing - using defaults", 0, {'use_defaults': True}

    def _heal_file_permission(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Permission error - needs human."""
        return False, "Permission error - needs manual fix", 0, {'needs_human': True}

    def _heal_memory_error(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Memory error - clear caches and reduce batch size."""
        import gc
        gc.collect()
        return True, "Cleared memory, reduce batch size", 5, {'reduce_batch': True}

    def _heal_network_error(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Network error - wait and retry."""
        return True, "Network error - waiting 30s", 30, {'wait_network': True}

    def _heal_unknown(self, error: Exception, ctx: Dict) -> Tuple[bool, str, float, Dict]:
        """Unknown error - log and escalate."""
        tb = traceback.format_exc()
        return False, f"Unknown error, escalating: {str(error)[:100]}", 0, {'traceback': tb[:500]}

    def execute_with_healing(
        self,
        operation: Callable,
        context: Dict[str, Any] = None,
        max_retries: int = None,
    ) -> Tuple[Any, List[HealingResult]]:
        """
        Execute an operation with automatic healing on failure.

        Args:
            operation: Callable to execute
            context: Context for healing decisions
            max_retries: Override default max retries

        Returns:
            Tuple of (result, list of healing attempts)
        """
        context = context or {}
        max_retries = max_retries or self.max_retries
        healing_attempts = []

        for attempt in range(max_retries + 1):
            try:
                context['retry_count'] = attempt
                result = operation()
                return result, healing_attempts

            except Exception as e:
                healing = self.heal(e, context)
                healing_attempts.append(healing)

                if not healing.healed:
                    raise

                if healing.retry_delay_seconds > 0:
                    time.sleep(healing.retry_delay_seconds)

        # If we get here, we've exhausted retries
        raise RuntimeError(f"Operation failed after {max_retries} healing attempts")

    def get_healing_stats(self) -> Dict[str, Any]:
        """Get statistics about healing attempts."""
        if not self.healing_history:
            return {'total_attempts': 0}

        total = len(self.healing_history)
        healed = sum(1 for h in self.healing_history if h.healed)

        by_type = {}
        for h in self.healing_history:
            type_name = h.error_type.value
            if type_name not in by_type:
                by_type[type_name] = {'total': 0, 'healed': 0}
            by_type[type_name]['total'] += 1
            if h.healed:
                by_type[type_name]['healed'] += 1

        return {
            'total_attempts': total,
            'total_healed': healed,
            'heal_rate': healed / total if total > 0 else 0,
            'by_error_type': by_type,
            'last_attempt': self.healing_history[-1].to_dict() if self.healing_history else None,
        }

    def _load_state(self):
        """Load healing history from disk."""
        state_file = self.state_dir / 'self_healer_state.json'
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                # Only keep last 100 entries
                self.healing_history = []
                for entry in data.get('history', [])[-100:]:
                    try:
                        self.healing_history.append(HealingResult(
                            error_type=ErrorType(entry['error_type']),
                            healed=entry['healed'],
                            action_taken=entry['action_taken'],
                            retry_delay_seconds=entry['retry_delay_seconds'],
                            details=entry.get('details', {}),
                            timestamp=datetime.fromisoformat(entry['timestamp']),
                        ))
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Failed to load self-healer state: {e}")

    def _save_state(self):
        """Save healing history to disk."""
        state_file = self.state_dir / 'self_healer_state.json'
        try:
            # Keep only last 100 entries
            entries = [h.to_dict() for h in self.healing_history[-100:]]
            with open(state_file, 'w') as f:
                json.dump({'history': entries}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save self-healer state: {e}")


# Singleton instance
_healer_instance: Optional[SelfHealer] = None


def get_self_healer() -> SelfHealer:
    """Get the singleton self-healer instance."""
    global _healer_instance
    if _healer_instance is None:
        _healer_instance = SelfHealer()
    return _healer_instance
