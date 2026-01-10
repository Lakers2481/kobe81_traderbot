"""
Resilience Manager - Self-Healing Data Pipeline

Mission 3 Part B: Self-Healing Data Pipeline
--------------------------------------------
This module provides automatic failover for data providers when primary
sources become unavailable. It ensures the trading system stays operational
even when external services fail.

Key Features:
1. Provider status tracking (ONLINE, OFFLINE, DEGRADED)
2. Automatic failover to backup providers
3. Alert notifications on provider changes
4. Recovery detection and automatic restoration

Author: Kobe Trading System
Date: 2026-01-07
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Root directory
ROOT = Path(__file__).resolve().parents[1]

# State file for provider status
PROVIDERS_STATE = ROOT / "state" / "providers.json"


class ProviderStatus(Enum):
    """Status of a data provider."""
    ONLINE = "ONLINE"           # Working normally
    DEGRADED = "DEGRADED"       # Working but slow/partial
    OFFLINE = "OFFLINE"         # Not responding
    RECOVERING = "RECOVERING"   # Was offline, testing recovery


@dataclass
class ProviderState:
    """State of a single data provider."""
    name: str
    status: ProviderStatus = ProviderStatus.ONLINE
    priority: int = 1           # Lower = higher priority
    consecutive_failures: int = 0
    last_success: Optional[str] = None
    last_failure: Optional[str] = None
    failure_reason: Optional[str] = None
    recovery_attempts: int = 0
    is_primary: bool = False


@dataclass
class ResilienceConfig:
    """Configuration for resilience manager."""
    # Failure thresholds
    max_consecutive_failures: int = 3     # Failures before marking offline
    degraded_threshold: int = 2           # Failures before marking degraded

    # Recovery settings
    recovery_check_interval: int = 300    # Seconds between recovery checks
    max_recovery_attempts: int = 5        # Max attempts before giving up
    recovery_cooldown: int = 600          # Seconds to wait after recovery

    # Alert settings
    alert_on_failover: bool = True
    alert_on_recovery: bool = True


class ResilienceManager:
    """
    Manages data provider resilience and automatic failover.

    This class tracks provider health and automatically switches to
    backup providers when primary sources fail.
    """

    def __init__(self, config: Optional[ResilienceConfig] = None):
        """Initialize with configuration."""
        self.config = config or ResilienceConfig()
        self._providers: Dict[str, ProviderState] = {}
        self._load_state()
        logger.info(f"ResilienceManager initialized with {len(self._providers)} providers")

    def _load_state(self) -> None:
        """Load provider state from file."""
        if PROVIDERS_STATE.exists():
            try:
                with open(PROVIDERS_STATE, 'r') as f:
                    data = json.load(f)
                    for name, state in data.get('providers', {}).items():
                        self._providers[name] = ProviderState(
                            name=name,
                            status=ProviderStatus(state.get('status', 'ONLINE')),
                            priority=state.get('priority', 1),
                            consecutive_failures=state.get('consecutive_failures', 0),
                            last_success=state.get('last_success'),
                            last_failure=state.get('last_failure'),
                            failure_reason=state.get('failure_reason'),
                            recovery_attempts=state.get('recovery_attempts', 0),
                            is_primary=state.get('is_primary', False),
                        )
                    logger.info(f"Loaded state for {len(self._providers)} providers")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load provider state: {e}")
                self._init_default_providers()
        else:
            self._init_default_providers()

    def _init_default_providers(self) -> None:
        """Initialize default provider configuration."""
        self._providers = {
            'polygon': ProviderState(
                name='polygon',
                priority=1,
                is_primary=True
            ),
            'stooq': ProviderState(
                name='stooq',
                priority=2
            ),
            'alpaca': ProviderState(
                name='alpaca',
                priority=3
            ),
        }
        self._save_state()

    def _save_state(self) -> None:
        """Save provider state to file."""
        PROVIDERS_STATE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'updated_at': datetime.now().isoformat(),
            'providers': {
                name: {
                    'status': p.status.value,
                    'priority': p.priority,
                    'consecutive_failures': p.consecutive_failures,
                    'last_success': p.last_success,
                    'last_failure': p.last_failure,
                    'failure_reason': p.failure_reason,
                    'recovery_attempts': p.recovery_attempts,
                    'is_primary': p.is_primary,
                }
                for name, p in self._providers.items()
            }
        }

        with open(PROVIDERS_STATE, 'w') as f:
            json.dump(data, f, indent=2)

    def register_provider(
        self,
        name: str,
        priority: int = 5,
        is_primary: bool = False
    ) -> None:
        """Register a new data provider."""
        self._providers[name] = ProviderState(
            name=name,
            priority=priority,
            is_primary=is_primary
        )
        self._save_state()
        logger.info(f"Registered provider: {name} (priority={priority})")

    def record_success(self, provider_name: str) -> None:
        """Record a successful data fetch from a provider."""
        if provider_name not in self._providers:
            self.register_provider(provider_name)

        provider = self._providers[provider_name]
        provider.consecutive_failures = 0
        provider.last_success = datetime.now().isoformat()

        # Check for recovery
        if provider.status in (ProviderStatus.OFFLINE, ProviderStatus.RECOVERING):
            old_status = provider.status
            provider.status = ProviderStatus.ONLINE
            provider.recovery_attempts = 0
            logger.info(f"Provider {provider_name} recovered: {old_status.value} -> ONLINE")

            if self.config.alert_on_recovery:
                self._send_recovery_alert(provider_name)

            # Check if this was the original primary
            if provider.is_primary:
                self._restore_primary(provider_name)

        elif provider.status == ProviderStatus.DEGRADED:
            provider.status = ProviderStatus.ONLINE
            logger.info(f"Provider {provider_name} recovered from DEGRADED -> ONLINE")

        self._save_state()

    def record_failure(self, provider_name: str, reason: str = "Unknown error") -> None:
        """Record a failed data fetch from a provider."""
        if provider_name not in self._providers:
            self.register_provider(provider_name)

        provider = self._providers[provider_name]
        provider.consecutive_failures += 1
        provider.last_failure = datetime.now().isoformat()
        provider.failure_reason = reason

        # Check thresholds
        if provider.consecutive_failures >= self.config.max_consecutive_failures:
            if provider.status != ProviderStatus.OFFLINE:
                old_status = provider.status
                provider.status = ProviderStatus.OFFLINE
                logger.error(
                    f"Provider {provider_name} marked OFFLINE after "
                    f"{provider.consecutive_failures} failures: {reason}"
                )

                # Initiate failover if this was primary
                if provider.is_primary:
                    self.initiate_provider_failover(provider_name)

        elif provider.consecutive_failures >= self.config.degraded_threshold:
            if provider.status == ProviderStatus.ONLINE:
                provider.status = ProviderStatus.DEGRADED
                logger.warning(f"Provider {provider_name} marked DEGRADED: {reason}")

        self._save_state()

    def initiate_provider_failover(self, failed_provider: str) -> Optional[str]:
        """
        Initiate failover from a failed provider to backup.

        Args:
            failed_provider: Name of the failed provider

        Returns:
            Name of the new primary provider, or None if no backup available
        """
        logger.critical(f"FAILOVER INITIATED: {failed_provider} is offline")

        # Mark failed provider as offline
        if failed_provider in self._providers:
            self._providers[failed_provider].status = ProviderStatus.OFFLINE
            self._providers[failed_provider].is_primary = False

        # Find best available backup
        available = [
            p for p in self._providers.values()
            if p.status in (ProviderStatus.ONLINE, ProviderStatus.DEGRADED)
            and p.name != failed_provider
        ]

        if not available:
            logger.critical("NO BACKUP PROVIDERS AVAILABLE - System in degraded mode!")
            self._send_critical_alert(
                f"ALL PROVIDERS OFFLINE - System cannot fetch data!\n"
                f"Failed: {failed_provider}\n"
                f"Please investigate immediately."
            )
            return None

        # Sort by priority and select best
        available.sort(key=lambda p: p.priority)
        new_primary = available[0]

        # Promote to primary
        new_primary.is_primary = True
        self._save_state()

        # Send alert
        if self.config.alert_on_failover:
            self._send_failover_alert(failed_provider, new_primary.name)

        logger.info(f"FAILOVER COMPLETE: {failed_provider} -> {new_primary.name}")
        return new_primary.name

    def _restore_primary(self, provider_name: str) -> None:
        """Restore original primary when it recovers."""
        provider = self._providers.get(provider_name)
        if not provider:
            return

        # Check if another provider is currently primary
        current_primary = self.get_primary_provider()
        if current_primary and current_primary != provider_name:
            # Demote current primary
            self._providers[current_primary].is_primary = False

        # Restore original primary
        provider.is_primary = True
        self._save_state()

        logger.info(f"Primary restored: {provider_name}")
        self._send_recovery_alert(provider_name, restored_primary=True)

    def get_primary_provider(self) -> Optional[str]:
        """Get the current primary provider."""
        for name, provider in self._providers.items():
            if provider.is_primary and provider.status == ProviderStatus.ONLINE:
                return name

        # No online primary - find best available
        available = [
            p for p in self._providers.values()
            if p.status in (ProviderStatus.ONLINE, ProviderStatus.DEGRADED)
        ]

        if available:
            available.sort(key=lambda p: p.priority)
            return available[0].name

        return None

    def get_provider_status(self, provider_name: str) -> Optional[ProviderStatus]:
        """Get the status of a provider."""
        provider = self._providers.get(provider_name)
        return provider.status if provider else None

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available for use."""
        provider = self._providers.get(provider_name)
        if not provider:
            return False
        return provider.status in (ProviderStatus.ONLINE, ProviderStatus.DEGRADED)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        return {
            name: {
                'status': p.status.value,
                'is_primary': p.is_primary,
                'priority': p.priority,
                'consecutive_failures': p.consecutive_failures,
                'last_success': p.last_success,
                'last_failure': p.last_failure,
            }
            for name, p in self._providers.items()
        }

    def _send_failover_alert(self, failed: str, new_primary: str) -> None:
        """Send alert about provider failover."""
        try:
            from notifications.telegram_alerts import send_critical_alert
            message = (
                f"ACTION TAKEN: Data provider '{failed}' is unresponsive.\n"
                f"Switched to '{new_primary}' as primary.\n"
                f"Operating in degraded mode.\n"
                f"Reason: {self._providers.get(failed, {}).failure_reason or 'Unknown'}"
            )
            send_critical_alert(message)
        except Exception as e:
            logger.error(f"Failed to send failover alert: {e}")

    def _send_recovery_alert(self, provider_name: str, restored_primary: bool = False) -> None:
        """Send alert about provider recovery."""
        try:
            from notifications.telegram_alerts import send_alert
            if restored_primary:
                message = f"RECOVERED: Primary provider '{provider_name}' is back online and restored."
            else:
                message = f"RECOVERED: Provider '{provider_name}' is back online."
            send_alert(message)
        except Exception as e:
            logger.warning(f"Failed to send recovery alert: {e}")

    def _send_critical_alert(self, message: str) -> None:
        """Send critical alert."""
        try:
            from notifications.telegram_alerts import send_critical_alert
            send_critical_alert(message)
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")


# Singleton instance
_resilience_manager: Optional[ResilienceManager] = None


def get_resilience_manager(config: Optional[ResilienceConfig] = None) -> ResilienceManager:
    """Get or create the singleton resilience manager."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager(config)
    return _resilience_manager


def initiate_provider_failover(failed_provider: str) -> Optional[str]:
    """
    Convenience function to initiate failover.

    This is the function called by data_validator when a provider fails.
    """
    manager = get_resilience_manager()
    return manager.initiate_provider_failover(failed_provider)


# Example usage
if __name__ == "__main__":
    # Demo usage
    manager = get_resilience_manager()

    # Show current status
    print("Provider Status:")
    for name, status in manager.get_all_statuses().items():
        print(f"  {name}: {status['status']} (primary={status['is_primary']})")

    # Simulate failures
    print("\nSimulating 3 failures on polygon...")
    for i in range(3):
        manager.record_failure('polygon', f"Connection timeout #{i+1}")

    print("\nAfter failures:")
    for name, status in manager.get_all_statuses().items():
        print(f"  {name}: {status['status']} (primary={status['is_primary']})")

    # Simulate recovery
    print("\nSimulating recovery...")
    manager.record_success('polygon')

    print("\nAfter recovery:")
    for name, status in manager.get_all_statuses().items():
        print(f"  {name}: {status['status']} (primary={status['is_primary']})")
