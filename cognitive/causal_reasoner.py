"""
Causal Narrative Engine - LLM-Based Causal Reasoning

Builds knowledge graphs of cause-and-effect relationships from:
- Earnings call transcripts
- SEC filings (10-K, 8-K)
- News articles
- Research papers

Then validates trading signals against these causal narratives,
boosting signals that align with the narrative and vetoing contradictions.

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)

# State file for causal knowledge
ROOT = Path(__file__).resolve().parents[1]
CAUSAL_STATE = ROOT / "state" / "causal_knowledge.json"


class RelationType(Enum):
    """Types of causal relationships."""
    CAUSES = "causes"              # A causes B
    CORRELATES = "correlates"      # A correlates with B
    PREVENTS = "prevents"          # A prevents B
    ENABLES = "enables"            # A enables B
    INCREASES = "increases"        # A increases B
    DECREASES = "decreases"        # A decreases B
    DEPENDS_ON = "depends_on"      # A depends on B
    CONFLICTS = "conflicts"        # A conflicts with B


class EntityType(Enum):
    """Types of entities in causal graph."""
    COMPANY = "company"
    SECTOR = "sector"
    MACRO_INDICATOR = "macro_indicator"
    POLICY = "policy"
    EVENT = "event"
    METRIC = "metric"
    SENTIMENT = "sentiment"


@dataclass
class CausalEntity:
    """Entity in the causal knowledge graph."""
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalRelation:
    """Relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 0.5  # 0 to 1
    confidence: float = 0.5  # 0 to 1
    evidence: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CausalNarrative:
    """A coherent narrative explaining market dynamics."""
    narrative_id: str
    title: str
    summary: str
    entities: List[str]  # Entity IDs involved
    relations: List[str]  # Relation keys involved
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float  # 0 to 1
    symbols_affected: List[str]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: Optional[str] = None


@dataclass
class SignalAlignment:
    """Result of checking signal against causal narratives."""
    symbol: str
    signal_direction: str  # "LONG" or "SHORT"
    alignment_score: float  # -1 (contradiction) to +1 (strong alignment)
    supporting_narratives: List[str]
    contradicting_narratives: List[str]
    confidence_boost: float  # How much to adjust signal confidence
    reasoning: str


class CausalKnowledgeGraph:
    """
    Knowledge graph of causal relationships.

    Stores entities and their relationships, supports querying
    for causal chains and narrative extraction.
    """

    def __init__(self):
        self.entities: Dict[str, CausalEntity] = {}
        self.relations: Dict[str, CausalRelation] = {}
        self._entity_by_name: Dict[str, str] = {}  # name -> entity_id

    def add_entity(self, entity: CausalEntity) -> None:
        """Add or update an entity."""
        self.entities[entity.entity_id] = entity
        self._entity_by_name[entity.name.lower()] = entity.entity_id
        for alias in entity.aliases:
            self._entity_by_name[alias.lower()] = entity.entity_id

    def add_relation(self, relation: CausalRelation) -> None:
        """Add or update a relation."""
        key = f"{relation.source_id}:{relation.relation_type.value}:{relation.target_id}"
        self.relations[key] = relation

    def find_entity(self, name: str) -> Optional[CausalEntity]:
        """Find entity by name or alias."""
        entity_id = self._entity_by_name.get(name.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None

    def get_relations_for(self, entity_id: str) -> List[CausalRelation]:
        """Get all relations involving an entity."""
        results = []
        for rel in self.relations.values():
            if rel.source_id == entity_id or rel.target_id == entity_id:
                results.append(rel)
        return results

    def find_causal_chain(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> List[List[CausalRelation]]:
        """Find causal chains between two entities (BFS)."""
        if start_id not in self.entities or end_id not in self.entities:
            return []

        chains = []
        queue = [(start_id, [])]
        visited = {start_id}

        while queue and len(chains) < 10:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            for rel in self.get_relations_for(current):
                if rel.source_id == current:
                    next_id = rel.target_id
                elif rel.target_id == current:
                    continue  # Only follow forward edges
                else:
                    continue

                new_path = path + [rel]

                if next_id == end_id:
                    chains.append(new_path)
                elif next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, new_path))

        return chains

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "entities": {
                k: {
                    "entity_id": v.entity_id,
                    "name": v.name,
                    "entity_type": v.entity_type.value,
                    "aliases": v.aliases,
                    "metadata": v.metadata,
                }
                for k, v in self.entities.items()
            },
            "relations": {
                k: {
                    "source_id": v.source_id,
                    "target_id": v.target_id,
                    "relation_type": v.relation_type.value,
                    "strength": v.strength,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                    "last_updated": v.last_updated,
                }
                for k, v in self.relations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalKnowledgeGraph":
        """Deserialize from dict."""
        graph = cls()

        for entity_data in data.get("entities", {}).values():
            graph.add_entity(CausalEntity(
                entity_id=entity_data["entity_id"],
                name=entity_data["name"],
                entity_type=EntityType(entity_data["entity_type"]),
                aliases=entity_data.get("aliases", []),
                metadata=entity_data.get("metadata", {}),
            ))

        for rel_data in data.get("relations", {}).values():
            graph.add_relation(CausalRelation(
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                relation_type=RelationType(rel_data["relation_type"]),
                strength=rel_data.get("strength", 0.5),
                confidence=rel_data.get("confidence", 0.5),
                evidence=rel_data.get("evidence", []),
                last_updated=rel_data.get("last_updated", ""),
            ))

        return graph


class CausalReasoner:
    """
    Causal Narrative Engine for trading signal validation.

    Uses LLM to:
    1. Extract causal relationships from text
    2. Build knowledge graph
    3. Validate trading signals against narratives
    """

    def __init__(self, llm_provider: Optional[Any] = None):
        """
        Initialize reasoner.

        Args:
            llm_provider: Optional LLM provider for extraction
        """
        self.llm = llm_provider
        self.knowledge_graph = CausalKnowledgeGraph()
        self.narratives: Dict[str, CausalNarrative] = {}
        self._load_state()
        self._init_base_knowledge()

    def _load_state(self) -> None:
        """Load persisted state."""
        if CAUSAL_STATE.exists():
            try:
                with open(CAUSAL_STATE, 'r') as f:
                    data = json.load(f)
                    self.knowledge_graph = CausalKnowledgeGraph.from_dict(
                        data.get("knowledge_graph", {})
                    )
                    for narr_data in data.get("narratives", {}).values():
                        self.narratives[narr_data["narrative_id"]] = CausalNarrative(**narr_data)
                    logger.info(f"Loaded causal state: {len(self.knowledge_graph.entities)} entities")
            except Exception as e:
                logger.warning(f"Failed to load causal state: {e}")

    def _save_state(self) -> None:
        """Persist state."""
        CAUSAL_STATE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "knowledge_graph": self.knowledge_graph.to_dict(),
            "narratives": {k: v.__dict__ for k, v in self.narratives.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(CAUSAL_STATE, 'w') as f:
            json.dump(data, f, indent=2)

    def _init_base_knowledge(self) -> None:
        """Initialize base macroeconomic knowledge."""
        if self.knowledge_graph.entities:
            return  # Already initialized

        # Core macro entities
        entities = [
            CausalEntity("fed_rate", "Federal Funds Rate", EntityType.MACRO_INDICATOR,
                        aliases=["interest rate", "fed rate", "rates"]),
            CausalEntity("inflation", "Inflation", EntityType.MACRO_INDICATOR,
                        aliases=["CPI", "PCE", "price level"]),
            CausalEntity("gdp", "GDP Growth", EntityType.MACRO_INDICATOR,
                        aliases=["economic growth", "GDP"]),
            CausalEntity("unemployment", "Unemployment", EntityType.MACRO_INDICATOR,
                        aliases=["jobless rate", "jobs"]),
            CausalEntity("usd", "US Dollar", EntityType.MACRO_INDICATOR,
                        aliases=["USD", "DXY", "dollar"]),
            CausalEntity("tech_sector", "Technology Sector", EntityType.SECTOR,
                        aliases=["tech", "technology", "tech stocks"]),
            CausalEntity("fin_sector", "Financial Sector", EntityType.SECTOR,
                        aliases=["financials", "banks", "banking"]),
            CausalEntity("energy_sector", "Energy Sector", EntityType.SECTOR,
                        aliases=["energy", "oil", "O&G"]),
            CausalEntity("consumer_sentiment", "Consumer Sentiment", EntityType.SENTIMENT,
                        aliases=["consumer confidence"]),
        ]

        for entity in entities:
            self.knowledge_graph.add_entity(entity)

        # Core macro relationships
        relations = [
            CausalRelation("fed_rate", "tech_sector", RelationType.DECREASES, 0.7, 0.8,
                          ["Higher rates increase discount rates, hurting growth stock valuations"]),
            CausalRelation("fed_rate", "fin_sector", RelationType.INCREASES, 0.6, 0.7,
                          ["Higher rates improve net interest margins for banks"]),
            CausalRelation("fed_rate", "usd", RelationType.INCREASES, 0.7, 0.8,
                          ["Higher rates attract foreign capital, strengthening dollar"]),
            CausalRelation("inflation", "fed_rate", RelationType.CAUSES, 0.8, 0.9,
                          ["Fed raises rates to combat inflation"]),
            CausalRelation("usd", "tech_sector", RelationType.DECREASES, 0.4, 0.6,
                          ["Strong dollar hurts foreign revenue for multinationals"]),
            CausalRelation("gdp", "consumer_sentiment", RelationType.INCREASES, 0.6, 0.7,
                          ["Economic growth improves consumer confidence"]),
        ]

        for rel in relations:
            self.knowledge_graph.add_relation(rel)

        self._save_state()
        logger.info("Initialized base macroeconomic knowledge")

    def extract_relations_from_text(self, text: str, source: str = "unknown") -> List[CausalRelation]:
        """
        Extract causal relations from text using pattern matching or LLM.

        Args:
            text: Input text (news, filing, transcript)
            source: Source identifier for evidence

        Returns:
            List of extracted relations
        """
        relations = []

        # Pattern-based extraction (fallback when no LLM)
        patterns = [
            (r"(\w+) (?:will |may |could )?(cause|lead to|result in) (\w+)", RelationType.CAUSES),
            (r"(?:rising|higher|increased?) (\w+) (?:is |are )?(?:hurting|damaging|weighing on) (\w+)", RelationType.DECREASES),
            (r"(?:rising|higher|increased?) (\w+) (?:is |are )?(?:helping|boosting|lifting) (\w+)", RelationType.INCREASES),
            (r"(\w+) (?:depends on|relies on|is driven by) (\w+)", RelationType.DEPENDS_ON),
        ]

        for pattern, rel_type in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if len(match) >= 2:
                    source_name = match[0]
                    target_name = match[-1]

                    source_entity = self.knowledge_graph.find_entity(source_name)
                    target_entity = self.knowledge_graph.find_entity(target_name)

                    if source_entity and target_entity:
                        relations.append(CausalRelation(
                            source_id=source_entity.entity_id,
                            target_id=target_entity.entity_id,
                            relation_type=rel_type,
                            strength=0.5,
                            confidence=0.4,  # Lower confidence for pattern-based
                            evidence=[f"Extracted from {source}: '{text[:100]}...'"],
                        ))

        # If LLM available, use for more sophisticated extraction
        if self.llm and len(text) > 50:
            try:
                llm_relations = self._llm_extract_relations(text, source)
                relations.extend(llm_relations)
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        return relations

    def _llm_extract_relations(self, text: str, source: str) -> List[CausalRelation]:
        """Use LLM to extract causal relations."""
        # This would integrate with the LLM provider
        # For now, return empty - would need actual LLM call
        return []

    def build_narrative(
        self,
        symbol: str,
        context: Dict[str, Any]
    ) -> Optional[CausalNarrative]:
        """
        Build a causal narrative for a symbol.

        Args:
            symbol: Trading symbol
            context: Market context (news, filings, etc.)

        Returns:
            CausalNarrative if one can be constructed
        """
        # Find entities related to symbol
        symbol_entity = self.knowledge_graph.find_entity(symbol)

        # Get sector for symbol
        sector_map = {
            "AAPL": "tech_sector", "MSFT": "tech_sector", "NVDA": "tech_sector",
            "GOOGL": "tech_sector", "META": "tech_sector", "AMZN": "tech_sector",
            "JPM": "fin_sector", "BAC": "fin_sector", "GS": "fin_sector",
            "XOM": "energy_sector", "CVX": "energy_sector",
        }

        sector_id = sector_map.get(symbol, "tech_sector")

        # Build narrative from macro factors
        relevant_relations = []
        for rel in self.knowledge_graph.relations.values():
            if rel.target_id == sector_id or rel.source_id == sector_id:
                relevant_relations.append(rel)

        if not relevant_relations:
            return None

        # Determine narrative direction
        bullish_strength = 0.0
        bearish_strength = 0.0

        for rel in relevant_relations:
            if rel.relation_type in (RelationType.INCREASES, RelationType.ENABLES):
                bullish_strength += rel.strength * rel.confidence
            elif rel.relation_type in (RelationType.DECREASES, RelationType.PREVENTS):
                bearish_strength += rel.strength * rel.confidence

        if bullish_strength > bearish_strength * 1.2:
            direction = "BULLISH"
            strength = bullish_strength / (bullish_strength + bearish_strength)
        elif bearish_strength > bullish_strength * 1.2:
            direction = "BEARISH"
            strength = bearish_strength / (bullish_strength + bearish_strength)
        else:
            direction = "NEUTRAL"
            strength = 0.5

        # Create narrative
        narrative = CausalNarrative(
            narrative_id=f"NARR_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=f"Causal analysis for {symbol}",
            summary=self._generate_narrative_summary(symbol, sector_id, relevant_relations, direction),
            entities=[r.source_id for r in relevant_relations] + [r.target_id for r in relevant_relations],
            relations=[f"{r.source_id}:{r.relation_type.value}:{r.target_id}" for r in relevant_relations],
            direction=direction,
            strength=strength,
            symbols_affected=[symbol],
            expires_at=(datetime.now() + timedelta(days=1)).isoformat(),
        )

        self.narratives[narrative.narrative_id] = narrative
        self._save_state()

        return narrative

    def _generate_narrative_summary(
        self,
        symbol: str,
        sector_id: str,
        relations: List[CausalRelation],
        direction: str
    ) -> str:
        """Generate human-readable narrative summary."""
        parts = []

        for rel in relations[:3]:  # Top 3 most relevant
            source = self.knowledge_graph.entities.get(rel.source_id)
            target = self.knowledge_graph.entities.get(rel.target_id)
            if source and target:
                if rel.relation_type == RelationType.INCREASES:
                    parts.append(f"{source.name} is boosting {target.name}")
                elif rel.relation_type == RelationType.DECREASES:
                    parts.append(f"{source.name} is weighing on {target.name}")
                elif rel.relation_type == RelationType.CAUSES:
                    parts.append(f"{source.name} is causing {target.name}")

        if parts:
            return f"Analysis for {symbol}: {'. '.join(parts)}. Overall outlook: {direction}."
        return f"Limited causal data available for {symbol}. Outlook: {direction}."

    def validate_signal(
        self,
        symbol: str,
        signal_direction: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SignalAlignment:
        """
        Validate a trading signal against causal narratives.

        Args:
            symbol: Trading symbol
            signal_direction: "LONG" or "SHORT"
            context: Optional additional context

        Returns:
            SignalAlignment with alignment score and reasoning
        """
        # Build or get narrative for symbol
        narrative = self.build_narrative(symbol, context or {})

        supporting = []
        contradicting = []
        alignment_score = 0.0

        if narrative:
            # Check alignment
            if signal_direction == "LONG":
                if narrative.direction == "BULLISH":
                    supporting.append(narrative.narrative_id)
                    alignment_score = narrative.strength
                elif narrative.direction == "BEARISH":
                    contradicting.append(narrative.narrative_id)
                    alignment_score = -narrative.strength
            else:  # SHORT
                if narrative.direction == "BEARISH":
                    supporting.append(narrative.narrative_id)
                    alignment_score = narrative.strength
                elif narrative.direction == "BULLISH":
                    contradicting.append(narrative.narrative_id)
                    alignment_score = -narrative.strength

        # Calculate confidence boost
        if alignment_score > 0.3:
            confidence_boost = 0.1  # +10% confidence
            reasoning = f"Signal ALIGNS with causal narrative: {narrative.summary if narrative else 'N/A'}"
        elif alignment_score < -0.3:
            confidence_boost = -0.15  # -15% confidence (heavier penalty for contradiction)
            reasoning = f"Signal CONTRADICTS causal narrative: {narrative.summary if narrative else 'N/A'}"
        else:
            confidence_boost = 0.0
            reasoning = "No strong causal alignment or contradiction detected."

        return SignalAlignment(
            symbol=symbol,
            signal_direction=signal_direction,
            alignment_score=alignment_score,
            supporting_narratives=supporting,
            contradicting_narratives=contradicting,
            confidence_boost=confidence_boost,
            reasoning=reasoning,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the causal reasoner."""
        return {
            "entities": len(self.knowledge_graph.entities),
            "relations": len(self.knowledge_graph.relations),
            "narratives": len(self.narratives),
            "active_narratives": len([
                n for n in self.narratives.values()
                if n.expires_at and n.expires_at > datetime.now().isoformat()
            ]),
        }


# Singleton instance
_causal_reasoner: Optional[CausalReasoner] = None


def get_causal_reasoner() -> CausalReasoner:
    """Get or create singleton causal reasoner."""
    global _causal_reasoner
    if _causal_reasoner is None:
        _causal_reasoner = CausalReasoner()
    return _causal_reasoner


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    reasoner = get_causal_reasoner()

    # Validate a signal
    print("Testing signal validation...")

    # LONG signal on tech stock during rate hike environment
    result = reasoner.validate_signal("NVDA", "LONG")
    print(f"\nNVDA LONG Signal:")
    print(f"  Alignment Score: {result.alignment_score:.2f}")
    print(f"  Confidence Boost: {result.confidence_boost:+.1%}")
    print(f"  Reasoning: {result.reasoning}")

    # SHORT signal on financials
    result2 = reasoner.validate_signal("JPM", "SHORT")
    print(f"\nJPM SHORT Signal:")
    print(f"  Alignment Score: {result2.alignment_score:.2f}")
    print(f"  Confidence Boost: {result2.confidence_boost:+.1%}")
    print(f"  Reasoning: {result2.reasoning}")

    print(f"\nStatus: {reasoner.get_status()}")
