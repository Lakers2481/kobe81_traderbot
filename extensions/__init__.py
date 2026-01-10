"""
Extensions module for Kobe trading robot.

This module provides the extension registry and management system
for enabling/disabling optional components without breaking core.
"""

from .registry import (
    ExtensionRegistry,
    get_extension_registry,
    is_extension_enabled,
    get_enabled_extensions,
    enable_extension,
    disable_extension,
)

__all__ = [
    "ExtensionRegistry",
    "get_extension_registry",
    "is_extension_enabled",
    "get_enabled_extensions",
    "enable_extension",
    "disable_extension",
]
