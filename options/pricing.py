from __future__ import annotations

"""
Options Pricing Facade.

Thin wrapper that re-exports Black-Scholes pricing and Greeks so callers can
import from options.pricing instead of options.black_scholes.
"""

from .black_scholes import BlackScholes, OptionType, OptionPricing

__all__ = ["BlackScholes", "OptionType", "OptionPricing"]

