"""
Kobe Trading System - Tax Lot Accounting Module.

Provides IRS-compliant tax lot tracking:
- FIFO/LIFO/Specific ID cost basis methods
- Wash sale detection (61-day window)
- Short-term vs long-term capital gains
- Tax lot ledger with full audit trail
"""

from .lot_accounting import (
    TaxLotAccountant,
    TaxLot,
    CostBasisMethod,
    WashSaleResult,
    get_tax_accountant,
)

__all__ = [
    'TaxLotAccountant',
    'TaxLot',
    'CostBasisMethod',
    'WashSaleResult',
    'get_tax_accountant',
]
