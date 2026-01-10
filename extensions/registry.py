"""
extensions/registry.py - Extension management for Kobe trading robot.

This module provides:
- Extension enable/disable via config
- Runtime extension status checking
- Extension validation before enabling
- Logging of what's active vs inactive

Usage:
    from extensions import is_extension_enabled, get_enabled_extensions

    if is_extension_enabled("ml_markov"):
        # Use Markov chain features
        from ml_advanced.markov_chain import MarkovPredictor

    # Get all enabled extensions
    enabled = get_enabled_extensions()
    print(f"Active extensions: {enabled}")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)

# Root directory
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ExtensionInfo:
    """Information about a single extension."""
    name: str
    enabled: bool
    description: str
    integration: Optional[str]
    files: List[str] = field(default_factory=list)
    tested: bool = False
    impact: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "description": self.description,
            "integration": self.integration,
            "files": self.files,
            "tested": self.tested,
            "impact": self.impact,
        }


class ExtensionRegistry:
    """
    Registry for managing Kobe trading robot extensions.

    Extensions are optional components that enhance the core trading system
    but are not required for basic operation.

    Example:
        registry = ExtensionRegistry()

        # Check if extension is enabled
        if registry.is_enabled("ml_markov"):
            from ml_advanced.markov_chain import MarkovPredictor

        # Enable an extension
        registry.enable("ml_hmm")

        # Disable an extension
        registry.disable("ml_lstm")

        # Get all enabled
        for ext in registry.get_enabled():
            print(f"{ext.name}: {ext.description}")
    """

    def __init__(self, config_path: Optional[Path] = None, manifest_path: Optional[Path] = None):
        """
        Initialize extension registry.

        Args:
            config_path: Path to extensions_enabled.json
            manifest_path: Path to core_manifest.json
        """
        self.config_path = config_path or ROOT / "config" / "extensions_enabled.json"
        self.manifest_path = manifest_path or ROOT / "config" / "core_manifest.json"

        self._extensions: Dict[str, ExtensionInfo] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load extension configuration from files."""
        # Load enabled status from extensions_enabled.json
        enabled_config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                    enabled_config = data.get("extensions", {})
            except Exception as e:
                logger.warning(f"Failed to load extensions config: {e}")

        # Load detailed info from core_manifest.json
        manifest_extensions = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    data = json.load(f)
                    manifest_extensions = data.get("extensions", {})
            except Exception as e:
                logger.warning(f"Failed to load core manifest: {e}")

        # Merge configurations
        all_extensions = set(enabled_config.keys()) | set(manifest_extensions.keys())

        for ext_name in all_extensions:
            enabled_data = enabled_config.get(ext_name, {})
            manifest_data = manifest_extensions.get(ext_name, {})

            if isinstance(enabled_data, bool):
                enabled = enabled_data
                description = ""
                integration = None
            else:
                enabled = enabled_data.get("enabled", False)
                description = enabled_data.get("description", "")
                integration = enabled_data.get("integration")

            # Merge with manifest data
            if isinstance(manifest_data, dict):
                description = description or manifest_data.get("description", "")
                integration = integration or manifest_data.get("integration_point")
                files = manifest_data.get("files", [])
                tested = manifest_data.get("tested", False)
                impact = manifest_data.get("impact")
            else:
                files = []
                tested = False
                impact = None

            self._extensions[ext_name] = ExtensionInfo(
                name=ext_name,
                enabled=enabled,
                description=description,
                integration=integration,
                files=files,
                tested=tested,
                impact=impact,
            )

    def _save_config(self) -> None:
        """Save extension configuration."""
        data = {
            "version": "1.0.0",
            "last_modified": datetime.now().isoformat(),
            "description": "Extension enable/disable configuration for Kobe trading robot",
            "extensions": {},
            "notes": {
                "how_to_enable": "Set 'enabled' to true and ensure integration point is wired",
                "how_to_disable": "Set 'enabled' to false - core will still work",
                "testing_required": "Run verify_extension.py before enabling in production"
            }
        }

        for ext_name, ext_info in self._extensions.items():
            data["extensions"][ext_name] = {
                "enabled": ext_info.enabled,
                "description": ext_info.description,
                "integration": ext_info.integration,
            }

        with open(self.config_path, "w") as f:
            json.dump(data, f, indent=2)

    def is_enabled(self, extension_name: str) -> bool:
        """Check if an extension is enabled."""
        ext = self._extensions.get(extension_name)
        return ext.enabled if ext else False

    def get_extension(self, extension_name: str) -> Optional[ExtensionInfo]:
        """Get extension info by name."""
        return self._extensions.get(extension_name)

    def get_enabled(self) -> List[ExtensionInfo]:
        """Get all enabled extensions."""
        return [ext for ext in self._extensions.values() if ext.enabled]

    def get_disabled(self) -> List[ExtensionInfo]:
        """Get all disabled extensions."""
        return [ext for ext in self._extensions.values() if not ext.enabled]

    def get_all(self) -> List[ExtensionInfo]:
        """Get all extensions."""
        return list(self._extensions.values())

    def enable(self, extension_name: str) -> bool:
        """
        Enable an extension.

        Args:
            extension_name: Name of extension to enable

        Returns:
            True if enabled, False if extension not found
        """
        ext = self._extensions.get(extension_name)
        if ext:
            ext.enabled = True
            self._save_config()
            logger.info(f"Extension enabled: {extension_name}")
            return True
        return False

    def disable(self, extension_name: str) -> bool:
        """
        Disable an extension.

        Args:
            extension_name: Name of extension to disable

        Returns:
            True if disabled, False if extension not found
        """
        ext = self._extensions.get(extension_name)
        if ext:
            ext.enabled = False
            self._save_config()
            logger.info(f"Extension disabled: {extension_name}")
            return True
        return False

    def validate_extension(self, extension_name: str) -> tuple[bool, str]:
        """
        Validate that an extension can be imported.

        Args:
            extension_name: Name of extension to validate

        Returns:
            Tuple of (success, error_message)
        """
        ext = self._extensions.get(extension_name)
        if not ext:
            return False, f"Extension not found: {extension_name}"

        for file_path in ext.files:
            full_path = ROOT / file_path
            if not full_path.exists():
                return False, f"Missing file: {file_path}"

            if file_path.endswith(".py"):
                try:
                    module_name = file_path.replace(".py", "").replace("/", ".").replace("\\", ".")
                    __import__(module_name)
                except Exception as e:
                    return False, f"Import error in {file_path}: {e}"

        return True, "OK"

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of extension status."""
        enabled = self.get_enabled()
        disabled = self.get_disabled()

        return {
            "total": len(self._extensions),
            "enabled": len(enabled),
            "disabled": len(disabled),
            "enabled_names": [e.name for e in enabled],
            "disabled_names": [e.name for e in disabled],
        }


# Singleton instance
_registry_instance: Optional[ExtensionRegistry] = None


def get_extension_registry() -> ExtensionRegistry:
    """Get the singleton extension registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ExtensionRegistry()
    return _registry_instance


def is_extension_enabled(extension_name: str) -> bool:
    """Check if an extension is enabled."""
    # Check for core-only mode (disables ALL extensions)
    if os.environ.get('KOBE_CORE_ONLY', '').lower() == 'true':
        return False

    # Check for explicit extension list
    explicit_list = os.environ.get('KOBE_ENABLED_EXTENSIONS', '')
    if explicit_list:
        enabled_names = [e.strip() for e in explicit_list.split(',')]
        return extension_name in enabled_names

    # Fall back to config-based check
    return get_extension_registry().is_enabled(extension_name)


def get_enabled_extensions() -> List[str]:
    """Get list of enabled extension names."""
    return [e.name for e in get_extension_registry().get_enabled()]


def enable_extension(extension_name: str) -> bool:
    """Enable an extension."""
    return get_extension_registry().enable(extension_name)


def disable_extension(extension_name: str) -> bool:
    """Disable an extension."""
    return get_extension_registry().disable(extension_name)


# For use in conditional imports
def require_extension(extension_name: str):
    """
    Decorator that only runs function if extension is enabled.

    Usage:
        @require_extension("ml_markov")
        def get_markov_prediction():
            from ml_advanced.markov_chain import MarkovPredictor
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_extension_enabled(extension_name):
                return func(*args, **kwargs)
            return None
        return wrapper
    return decorator


if __name__ == "__main__":
    # Quick test
    registry = ExtensionRegistry()

    print("Extension Status:")
    print("-" * 50)

    for ext in registry.get_all():
        status = "[ON]" if ext.enabled else "[OFF]"
        print(f"{status} {ext.name}: {ext.description}")

    print("\n" + "-" * 50)
    summary = registry.get_status_summary()
    print(f"Total: {summary['total']}, Enabled: {summary['enabled']}, Disabled: {summary['disabled']}")
