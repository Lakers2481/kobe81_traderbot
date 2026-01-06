#!/usr/bin/env python3
"""
KOBE KNOWLEDGE INTEGRATOR
==========================
This is the BRAIN that takes all discoveries and INTEGRATES them into Kobe.

NOT just collecting - INTEGRATING, LEARNING, GROWING.

WHAT IT DOES:
1. DISCOVERS - Finds quant strategies, safety improvements, better code
2. EVALUATES - Checks if backed by data, proven, useful
3. CATEGORIZES - Organizes by type (strategy, safety, code, architecture)
4. INTEGRATES - Adds to Kobe's knowledge base
5. REPORTS - Creates structured reports of learnings
6. TRACKS GROWTH - Measures how much smarter Kobe is getting

NOTHING IS RANDOM. EVERYTHING IS PURPOSEFUL.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class KnowledgeIntegrator:
    """
    The intelligent brain that integrates all learnings into Kobe.
    Makes Kobe smarter, safer, more accurate with every discovery.
    """

    # Categories of knowledge Kobe needs
    KNOWLEDGE_CATEGORIES = {
        "quant_strategies": {
            "description": "Quantitative trading strategies with backtested proof",
            "requirements": ["win_rate", "profit_factor", "sample_size", "backtest_period"],
            "priority": 1,
        },
        "swing_strategies": {
            "description": "Swing trading strategies (2-10 day holds)",
            "requirements": ["entry_rules", "exit_rules", "holding_period"],
            "priority": 1,
        },
        "mean_reversion": {
            "description": "Mean reversion strategies like IBS, RSI oversold",
            "requirements": ["indicator", "threshold", "historical_performance"],
            "priority": 1,
        },
        "ict_patterns": {
            "description": "ICT patterns - Order Blocks, FVG, Turtle Soup, Smart Money",
            "requirements": ["pattern_type", "detection_rules", "success_rate"],
            "priority": 1,
        },
        "risk_management": {
            "description": "Position sizing, stop losses, drawdown control",
            "requirements": ["method", "parameters", "risk_reduction"],
            "priority": 2,
        },
        "safety_improvements": {
            "description": "Kill switches, circuit breakers, error handling",
            "requirements": ["protection_type", "implementation"],
            "priority": 2,
        },
        "ml_models": {
            "description": "LSTM, transformers, RL, regime detection",
            "requirements": ["model_type", "accuracy", "use_case"],
            "priority": 2,
        },
        "code_improvements": {
            "description": "Faster, cleaner, better Python code",
            "requirements": ["improvement_type", "before_after"],
            "priority": 3,
        },
        "architecture": {
            "description": "Better system design, modularity, scalability",
            "requirements": ["pattern", "benefit"],
            "priority": 3,
        },
        "reasoning_logic": {
            "description": "Better decision making, thinking, understanding",
            "requirements": ["concept", "application"],
            "priority": 2,
        },
    }

    def __init__(self, state_dir: Optional[Path] = None):
        if state_dir is None:
            state_dir = Path("state/autonomous/learning")
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Knowledge base - organized by category
        self.knowledge_base: Dict[str, List[Dict]] = {cat: [] for cat in self.KNOWLEDGE_CATEGORIES}

        # Growth tracking
        self.growth_metrics = {
            "total_learnings": 0,
            "strategies_found": 0,
            "safety_improvements": 0,
            "code_improvements": 0,
            "integrations_applied": 0,
            "started_at": datetime.now(ET).isoformat(),
        }

        self._load_state()

    def _load_state(self):
        """Load knowledge base and metrics."""
        kb_file = self.state_dir / "knowledge_base.json"
        if kb_file.exists():
            try:
                data = json.loads(kb_file.read_text())
                self.knowledge_base = data.get("knowledge_base", self.knowledge_base)
                self.growth_metrics = data.get("growth_metrics", self.growth_metrics)
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")

    def save_state(self):
        """Save knowledge base and metrics."""
        kb_file = self.state_dir / "knowledge_base.json"
        data = {
            "knowledge_base": self.knowledge_base,
            "growth_metrics": self.growth_metrics,
            "last_updated": datetime.now(ET).isoformat(),
        }
        kb_file.write_text(json.dumps(data, indent=2))

    def evaluate_discovery(self, discovery: Dict) -> Dict[str, Any]:
        """
        Evaluate if a discovery is useful and backed by data.
        Returns evaluation with score and reasoning.
        """
        evaluation = {
            "is_useful": False,
            "score": 0.0,
            "category": None,
            "reasoning": [],
            "data_backed": False,
            "actionable": False,
        }

        title = discovery.get("title", "").lower()
        description = discovery.get("description", "").lower()
        content = f"{title} {description}"

        # Check for data-backed proof
        proof_indicators = [
            "backtest", "win rate", "profit factor", "sharpe",
            "tested", "validated", "proven", "results show",
            "sample size", "statistical", "p-value", "significant",
            "%", "accuracy", "performance", "returns",
        ]

        proof_count = sum(1 for ind in proof_indicators if ind in content)
        if proof_count >= 2:
            evaluation["data_backed"] = True
            evaluation["score"] += 0.3
            evaluation["reasoning"].append(f"Data-backed: {proof_count} proof indicators found")

        # Categorize
        category = self._categorize_discovery(content)
        if category:
            evaluation["category"] = category
            evaluation["score"] += 0.2
            evaluation["reasoning"].append(f"Category: {category}")

        # Check if actionable
        action_indicators = [
            "strategy", "rule", "when", "if", "entry", "exit",
            "buy", "sell", "position", "stop", "target",
            "code", "implement", "algorithm", "function",
        ]

        action_count = sum(1 for ind in action_indicators if ind in content)
        if action_count >= 2:
            evaluation["actionable"] = True
            evaluation["score"] += 0.3
            evaluation["reasoning"].append(f"Actionable: {action_count} action indicators")

        # Check for specific metrics
        if any(x in content for x in ["win rate", "winrate"]):
            evaluation["score"] += 0.1
            evaluation["reasoning"].append("Has win rate metric")

        if any(x in content for x in ["profit factor", "pf"]):
            evaluation["score"] += 0.1
            evaluation["reasoning"].append("Has profit factor metric")

        # Final decision
        evaluation["is_useful"] = evaluation["score"] >= 0.5 and evaluation["data_backed"]

        return evaluation

    def _categorize_discovery(self, content: str) -> Optional[str]:
        """Categorize discovery into knowledge category."""
        content = content.lower()

        # Priority order - check most important first
        if any(kw in content for kw in ["quant", "quantitative", "backtest", "systematic"]):
            return "quant_strategies"
        elif any(kw in content for kw in ["swing", "multi-day", "hold", "overnight"]):
            return "swing_strategies"
        elif any(kw in content for kw in ["mean reversion", "oversold", "overbought", "rsi", "ibs"]):
            return "mean_reversion"
        elif any(kw in content for kw in ["ict", "order block", "fvg", "turtle soup", "liquidity", "smart money"]):
            return "ict_patterns"
        elif any(kw in content for kw in ["risk", "position size", "stop loss", "drawdown", "kelly"]):
            return "risk_management"
        elif any(kw in content for kw in ["safety", "kill switch", "circuit breaker", "error", "exception"]):
            return "safety_improvements"
        elif any(kw in content for kw in ["lstm", "neural", "machine learning", "ml", "transformer", "rl", "regime"]):
            return "ml_models"
        elif any(kw in content for kw in ["code", "python", "faster", "optimize", "refactor", "clean"]):
            return "code_improvements"
        elif any(kw in content for kw in ["architecture", "design", "pattern", "modular", "scalable"]):
            return "architecture"
        elif any(kw in content for kw in ["reason", "logic", "decision", "think", "understand"]):
            return "reasoning_logic"

        return None

    def integrate_discovery(self, discovery: Dict, evaluation: Dict) -> bool:
        """
        Integrate a useful discovery into the knowledge base.
        Returns True if integrated.
        """
        if not evaluation.get("is_useful"):
            return False

        category = evaluation.get("category")
        if not category or category not in self.knowledge_base:
            return False

        # Create knowledge entry
        knowledge_entry = {
            "id": f"kb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.knowledge_base[category])}",
            "title": discovery.get("title", ""),
            "source": discovery.get("source", ""),
            "url": discovery.get("url", ""),
            "category": category,
            "score": evaluation.get("score", 0),
            "data_backed": evaluation.get("data_backed", False),
            "actionable": evaluation.get("actionable", False),
            "reasoning": evaluation.get("reasoning", []),
            "integrated_at": datetime.now(ET).isoformat(),
            "applied": False,
            "impact": None,
        }

        # Check for duplicates
        existing_urls = [k.get("url") for k in self.knowledge_base[category]]
        if knowledge_entry["url"] in existing_urls:
            return False

        # Add to knowledge base
        self.knowledge_base[category].append(knowledge_entry)

        # Update metrics
        self.growth_metrics["total_learnings"] += 1
        if category in ["quant_strategies", "swing_strategies", "mean_reversion", "ict_patterns"]:
            self.growth_metrics["strategies_found"] += 1
        elif category in ["risk_management", "safety_improvements"]:
            self.growth_metrics["safety_improvements"] += 1
        elif category in ["code_improvements", "architecture"]:
            self.growth_metrics["code_improvements"] += 1

        self.save_state()
        return True

    def process_scraped_discoveries(self) -> Dict[str, Any]:
        """
        Process all scraped discoveries and integrate useful ones.
        Returns summary of what was learned.
        """
        scraper_dir = Path("state/autonomous/scrapers")

        results = {
            "processed": 0,
            "useful": 0,
            "integrated": 0,
            "by_category": {},
        }

        if not scraper_dir.exists():
            return results

        # Process each discovery file
        for discovery_file in scraper_dir.glob("discoveries_*.json"):
            try:
                discoveries = json.loads(discovery_file.read_text())

                for discovery in discoveries:
                    results["processed"] += 1

                    # Evaluate
                    evaluation = self.evaluate_discovery(discovery)

                    if evaluation["is_useful"]:
                        results["useful"] += 1

                        # Integrate
                        if self.integrate_discovery(discovery, evaluation):
                            results["integrated"] += 1
                            cat = evaluation["category"]
                            results["by_category"][cat] = results["by_category"].get(cat, 0) + 1

            except Exception as e:
                logger.warning(f"Error processing {discovery_file}: {e}")

        return results

    def generate_learning_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of all learnings.
        Everything organized and structured.
        """
        report = {
            "generated_at": datetime.now(ET).isoformat(),
            "growth_metrics": self.growth_metrics,
            "knowledge_summary": {},
            "top_discoveries": [],
            "actionable_items": [],
            "integration_status": {},
        }

        # Summarize by category
        for category, items in self.knowledge_base.items():
            if items:
                report["knowledge_summary"][category] = {
                    "count": len(items),
                    "data_backed": sum(1 for i in items if i.get("data_backed")),
                    "actionable": sum(1 for i in items if i.get("actionable")),
                    "applied": sum(1 for i in items if i.get("applied")),
                    "latest": items[-1]["title"] if items else None,
                }

        # Top discoveries by score
        all_items = []
        for items in self.knowledge_base.values():
            all_items.extend(items)

        all_items.sort(key=lambda x: x.get("score", 0), reverse=True)
        report["top_discoveries"] = [
            {
                "title": item["title"][:80],
                "category": item["category"],
                "score": item["score"],
                "data_backed": item["data_backed"],
                "url": item["url"],
            }
            for item in all_items[:10]
        ]

        # Actionable items not yet applied
        for items in self.knowledge_base.values():
            for item in items:
                if item.get("actionable") and not item.get("applied"):
                    report["actionable_items"].append({
                        "id": item["id"],
                        "title": item["title"][:60],
                        "category": item["category"],
                        "url": item["url"],
                    })

        # Integration status
        total = sum(len(items) for items in self.knowledge_base.values())
        applied = sum(
            sum(1 for i in items if i.get("applied"))
            for items in self.knowledge_base.values()
        )
        report["integration_status"] = {
            "total_knowledge": total,
            "applied": applied,
            "pending": total - applied,
            "integration_rate": f"{(applied/total*100):.1f}%" if total > 0 else "0%",
        }

        # Save report
        report_file = self.state_dir / "learning_report.json"
        report_file.write_text(json.dumps(report, indent=2))

        return report

    def get_knowledge_for_improvement(self, category: str) -> List[Dict]:
        """
        Get actionable knowledge items for a specific improvement area.
        Returns items that haven't been applied yet.
        """
        items = self.knowledge_base.get(category, [])
        return [
            item for item in items
            if item.get("actionable") and not item.get("applied")
        ]

    def mark_as_applied(self, knowledge_id: str, impact: str = None):
        """Mark a knowledge item as applied with optional impact note."""
        for category, items in self.knowledge_base.items():
            for item in items:
                if item.get("id") == knowledge_id:
                    item["applied"] = True
                    item["applied_at"] = datetime.now(ET).isoformat()
                    item["impact"] = impact
                    self.growth_metrics["integrations_applied"] += 1
                    self.save_state()
                    return True
        return False

    def get_growth_summary(self) -> Dict[str, Any]:
        """Get summary of how Kobe is growing."""
        total_knowledge = sum(len(items) for items in self.knowledge_base.values())

        return {
            "total_learnings": self.growth_metrics["total_learnings"],
            "total_knowledge": total_knowledge,
            "strategies_found": self.growth_metrics["strategies_found"],
            "safety_improvements": self.growth_metrics["safety_improvements"],
            "code_improvements": self.growth_metrics["code_improvements"],
            "integrations_applied": self.growth_metrics["integrations_applied"],
            "started_at": self.growth_metrics["started_at"],
            "growth_rate": f"{total_knowledge} items learned",
        }
