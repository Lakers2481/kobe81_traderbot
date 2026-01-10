"""
Simulation Module - Hive Mind Market Simulator

Multi-agent market simulation for stress testing trading strategies
against synthetic but realistic market scenarios.

Components:
- market_agents.py: Trader personas (Value, Momentum, HFT, etc.)
- market_simulator.py: Order book simulation and price discovery
- scenario_generator.py: Market scenario generation (crash, mania, etc.)
"""

from simulation.market_agents import (
    MarketAgent,
    ValueInvestorAgent,
    MomentumFollowerAgent,
    HFTScalperAgent,
    RiskAversePensionFundAgent,
    RetailTraderAgent,
    MarketMakerAgent,
)

from simulation.market_simulator import (
    MarketSimulator,
    OrderBook,
    SimulatedOrder,
    SimulationResult,
)

__all__ = [
    'MarketAgent',
    'ValueInvestorAgent',
    'MomentumFollowerAgent',
    'HFTScalperAgent',
    'RiskAversePensionFundAgent',
    'RetailTraderAgent',
    'MarketMakerAgent',
    'MarketSimulator',
    'OrderBook',
    'SimulatedOrder',
    'SimulationResult',
]
