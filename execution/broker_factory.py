"""
Broker Factory for Kobe Trading System.

Creates and manages broker instances based on configuration.
Supports multiple broker types with config-driven selection.

Usage:
    # Get default broker (from config)
    broker = get_broker()

    # Get specific broker type
    broker = get_broker("paper")
    broker = get_broker("alpaca")
    broker = get_broker("crypto", exchange="binance")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Type, TYPE_CHECKING
import yaml

if TYPE_CHECKING:
    from execution.broker_base import BrokerBase

logger = logging.getLogger(__name__)

# Registry of broker implementations
_BROKER_REGISTRY: Dict[str, Type['BrokerBase']] = {}

# Singleton broker instances
_BROKER_INSTANCES: Dict[str, 'BrokerBase'] = {}


def register_broker(name: str, broker_class: Type['BrokerBase']) -> None:
    """
    Register a broker implementation.

    Called automatically when broker modules are imported.

    Args:
        name: Broker name (e.g., "alpaca", "paper", "crypto")
        broker_class: Broker class implementing BrokerBase
    """
    _BROKER_REGISTRY[name.lower()] = broker_class
    logger.debug(f"Registered broker: {name}")


def get_registered_brokers() -> Dict[str, Type['BrokerBase']]:
    """Get all registered broker classes."""
    return _BROKER_REGISTRY.copy()


def get_broker(
    name: Optional[str] = None,
    config_path: Optional[Path] = None,
    use_singleton: bool = True,
    **kwargs,
) -> 'BrokerBase':
    """
    Get a broker instance.

    Args:
        name: Broker name ("alpaca", "paper", "crypto")
               If None, reads from config or env
        config_path: Optional path to config file
        use_singleton: If True, return cached instance
        **kwargs: Broker-specific configuration

    Returns:
        BrokerBase instance

    Raises:
        ValueError: If broker name is unknown
    """
    # Determine broker name
    if name is None:
        name = _get_broker_from_config(config_path)

    name = name.lower()

    # Check singleton cache
    if use_singleton and name in _BROKER_INSTANCES:
        return _BROKER_INSTANCES[name]

    # Ensure brokers are registered
    if name not in _BROKER_REGISTRY:
        _lazy_register_brokers()

    if name not in _BROKER_REGISTRY:
        available = list(_BROKER_REGISTRY.keys())
        raise ValueError(f"Unknown broker: {name}. Available: {available}")

    broker_class = _BROKER_REGISTRY[name]

    # Merge config with kwargs
    config = _load_broker_config(name, config_path)
    config.update(kwargs)

    # Create instance
    broker = broker_class(**config)

    # Cache if singleton
    if use_singleton:
        _BROKER_INSTANCES[name] = broker

    logger.info(f"Created broker: {broker.name}")
    return broker


def get_default_broker() -> 'BrokerBase':
    """
    Get or create the default broker instance.

    Returns:
        Default BrokerBase instance (from config)
    """
    return get_broker()


def set_default_broker(broker: 'BrokerBase') -> None:
    """
    Set the default broker instance.

    Useful for testing or runtime broker switching.

    Args:
        broker: Broker instance to use as default
    """
    name = broker.name.lower()
    _BROKER_INSTANCES[name] = broker
    _BROKER_INSTANCES['default'] = broker


def clear_broker_cache() -> None:
    """Clear all cached broker instances."""
    _BROKER_INSTANCES.clear()


def _get_broker_from_config(config_path: Optional[Path] = None) -> str:
    """Get default broker from config or environment."""
    # Check environment first
    env_broker = os.getenv("KOBE_BROKER", "").lower()
    if env_broker:
        return env_broker

    # Check config file
    if config_path is None:
        config_path = Path("config/base.yaml")

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("execution", {}).get("broker", "alpaca")
        except Exception as e:
            logger.warning(f"Failed to read broker config: {e}")

    return "alpaca"  # Default


def _load_broker_config(name: str, config_path: Optional[Path] = None) -> Dict:
    """Load broker-specific configuration."""
    # Try brokers.yaml first
    brokers_yaml = Path("config/brokers.yaml")
    if brokers_yaml.exists():
        try:
            with open(brokers_yaml) as f:
                all_config = yaml.safe_load(f)
            return all_config.get(name, {})
        except Exception as e:
            logger.warning(f"Failed to load brokers.yaml: {e}")

    # Fall back to base.yaml
    if config_path is None:
        config_path = Path("config/base.yaml")

    if isinstance(config_path, str):
        config_path = Path(config_path)

    if config_path.exists():
        try:
            with open(config_path) as f:
                all_config = yaml.safe_load(f)
            return all_config.get("brokers", {}).get(name, {})
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    return {}


def _lazy_register_brokers() -> None:
    """Lazy register all broker implementations."""
    # Import implementations (they self-register on import)

    # Always available: Paper broker
    try:
        from execution.broker_paper import PaperBroker
        if "paper" not in _BROKER_REGISTRY:
            register_broker("paper", PaperBroker)
    except ImportError as e:
        logger.debug(f"Paper broker not available: {e}")

    # Alpaca broker
    try:
        from execution.broker_alpaca import AlpacaBroker
        if "alpaca" not in _BROKER_REGISTRY:
            register_broker("alpaca", AlpacaBroker)
    except ImportError as e:
        logger.debug(f"Alpaca broker not available: {e}")

    # Crypto broker (requires ccxt)
    try:
        from execution.broker_crypto import CryptoBroker
        if "crypto" not in _BROKER_REGISTRY:
            register_broker("crypto", CryptoBroker)
    except ImportError as e:
        logger.debug(f"Crypto broker not available: {e}")


class BrokerFactory:
    """
    Factory class for creating broker instances.

    Alternative OOP interface to the module-level functions.

    Usage:
        factory = BrokerFactory()
        broker = factory.create("alpaca")
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        _lazy_register_brokers()

    def create(
        self,
        name: Optional[str] = None,
        use_singleton: bool = True,
        **kwargs,
    ) -> 'BrokerBase':
        """Create a broker instance."""
        return get_broker(
            name=name,
            config_path=self.config_path,
            use_singleton=use_singleton,
            **kwargs,
        )

    def get_default(self) -> 'BrokerBase':
        """Get the default broker."""
        return get_default_broker()

    @property
    def available_brokers(self) -> list:
        """Get list of available broker names."""
        _lazy_register_brokers()
        return list(_BROKER_REGISTRY.keys())

    def is_available(self, name: str) -> bool:
        """Check if a broker type is available."""
        _lazy_register_brokers()
        return name.lower() in _BROKER_REGISTRY
