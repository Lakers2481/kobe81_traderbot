"""
Political Sentiment Analyzer - Legislation Impact Prediction

Goes beyond congressional trades to analyze:
- Text of proposed legislation
- Regulatory announcements
- Political rhetoric about sectors
- Policy implications for stocks

Author: Kobe Trading System
Date: 2026-01-07
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)

# Cache directory
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache" / "political"
STATE_FILE = ROOT / "state" / "political_sentiment.json"


@dataclass
class LegislationAnalysis:
    """Analysis of a piece of legislation."""
    bill_id: str
    title: str
    summary: str
    status: str  # introduced, passed_house, passed_senate, enacted
    sectors_affected: List[str]
    impact_direction: Dict[str, str]  # sector -> "positive" | "negative" | "neutral"
    impact_magnitude: Dict[str, float]  # sector -> 0-1 impact strength
    key_provisions: List[str]
    sponsors_party: str  # "D", "R", "bipartisan"
    passage_probability: float  # 0-1
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RegulatoryUpdate:
    """Regulatory announcement or update."""
    agency: str  # SEC, FTC, DOJ, FDA, etc.
    action_type: str  # investigation, ruling, guidance, fine
    target_sector: str
    target_companies: List[str]
    sentiment: str  # positive, negative, neutral
    impact_score: float  # 0-1
    summary: str
    date: str


@dataclass
class PoliticalSignal:
    """Trading signal derived from political analysis."""
    symbol: str
    sector: str
    direction: str  # BULLISH, BEARISH, NEUTRAL
    strength: float  # 0-1
    confidence: float  # 0-1
    sources: List[str]  # What generated this signal
    reasoning: str
    expires_at: str


# Sector keyword mapping for legislation analysis
SECTOR_KEYWORDS = {
    "technology": {
        "keywords": ["tech", "technology", "software", "internet", "data", "privacy",
                    "cybersecurity", "AI", "artificial intelligence", "algorithm",
                    "platform", "social media", "digital", "semiconductor", "chip"],
        "symbols": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "AMZN", "CRM"],
    },
    "healthcare": {
        "keywords": ["health", "healthcare", "medical", "pharmaceutical", "drug",
                    "medicare", "medicaid", "hospital", "insurance", "FDA",
                    "prescription", "biotech", "vaccine"],
        "symbols": ["JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT"],
    },
    "financial": {
        "keywords": ["bank", "banking", "financial", "credit", "loan", "mortgage",
                    "interest rate", "fed", "federal reserve", "SEC", "regulation",
                    "capital", "investment", "trading"],
        "symbols": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
    },
    "energy": {
        "keywords": ["energy", "oil", "gas", "petroleum", "renewable", "solar",
                    "wind", "electric", "grid", "pipeline", "drilling", "climate",
                    "carbon", "emission", "green"],
        "symbols": ["XOM", "CVX", "COP", "SLB", "EOG", "NEE", "DUK", "SO"],
    },
    "defense": {
        "keywords": ["defense", "military", "pentagon", "weapon", "contractor",
                    "security", "veteran", "army", "navy", "air force"],
        "symbols": ["LMT", "RTX", "NOC", "GD", "BA", "HII"],
    },
    "consumer": {
        "keywords": ["consumer", "retail", "tariff", "trade", "import", "export",
                    "walmart", "amazon", "spending", "inflation"],
        "symbols": ["WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX"],
    },
}

# Keywords indicating positive/negative impact
POSITIVE_INDICATORS = {
    "subsidy", "tax credit", "incentive", "support", "protect", "invest",
    "funding", "grant", "deregulate", "streamline", "approve", "authorize",
    "benefit", "expand", "grow",
}

NEGATIVE_INDICATORS = {
    "ban", "restrict", "regulate", "fine", "penalty", "investigate",
    "antitrust", "breakup", "limit", "prohibit", "tax increase", "tariff",
    "sanction", "enforce", "crack down", "scrutiny",
}


class PoliticalSentimentAnalyzer:
    """
    Analyze political developments for trading signals.

    Monitors:
    - Proposed legislation text
    - Regulatory announcements
    - Political rhetoric about sectors
    """

    def __init__(self):
        self.legislation_cache: Dict[str, LegislationAnalysis] = {}
        self.regulatory_cache: Dict[str, RegulatoryUpdate] = {}
        self.active_signals: Dict[str, PoliticalSignal] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state."""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    # Load active signals
                    for sig_data in data.get("active_signals", {}).values():
                        self.active_signals[sig_data["symbol"]] = PoliticalSignal(**sig_data)
                    logger.info(f"Loaded {len(self.active_signals)} active political signals")
            except Exception as e:
                logger.warning(f"Failed to load political state: {e}")

    def _save_state(self) -> None:
        """Persist state."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "active_signals": {k: v.__dict__ for k, v in self.active_signals.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)

    def analyze_legislation_text(self, text: str, bill_id: str, title: str) -> LegislationAnalysis:
        """
        Analyze legislation text to determine sector impacts.

        Args:
            text: Full text or summary of legislation
            bill_id: Bill identifier
            title: Bill title

        Returns:
            LegislationAnalysis with sector impacts
        """
        text_lower = text.lower()
        title_lower = title.lower()
        combined = text_lower + " " + title_lower

        # Identify affected sectors
        sectors_affected = []
        impact_direction = {}
        impact_magnitude = {}

        for sector, config in SECTOR_KEYWORDS.items():
            keyword_matches = sum(1 for kw in config["keywords"] if kw in combined)
            if keyword_matches >= 2:  # Require at least 2 keyword matches
                sectors_affected.append(sector)

                # Determine direction based on positive/negative indicators
                positive_count = sum(1 for ind in POSITIVE_INDICATORS if ind in combined)
                negative_count = sum(1 for ind in NEGATIVE_INDICATORS if ind in combined)

                if positive_count > negative_count * 1.5:
                    impact_direction[sector] = "positive"
                elif negative_count > positive_count * 1.5:
                    impact_direction[sector] = "negative"
                else:
                    impact_direction[sector] = "neutral"

                # Magnitude based on keyword density
                impact_magnitude[sector] = min(1.0, keyword_matches / 10)

        # Extract key provisions (simple extraction)
        key_provisions = []
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences[:10]:
            if any(kw in sentence.lower() for sector in sectors_affected
                  for kw in SECTOR_KEYWORDS.get(sector, {}).get("keywords", [])):
                if len(sentence.strip()) > 20:
                    key_provisions.append(sentence.strip()[:200])

        analysis = LegislationAnalysis(
            bill_id=bill_id,
            title=title,
            summary=text[:500] + "..." if len(text) > 500 else text,
            status="introduced",
            sectors_affected=sectors_affected,
            impact_direction=impact_direction,
            impact_magnitude=impact_magnitude,
            key_provisions=key_provisions[:5],
            sponsors_party="unknown",
            passage_probability=0.3,  # Default low probability
        )

        self.legislation_cache[bill_id] = analysis
        return analysis

    def analyze_regulatory_action(
        self,
        agency: str,
        action_type: str,
        description: str,
        target_companies: Optional[List[str]] = None,
    ) -> RegulatoryUpdate:
        """
        Analyze a regulatory action for trading signals.

        Args:
            agency: Regulatory agency (SEC, FTC, etc.)
            action_type: Type of action
            description: Description of the action
            target_companies: Specific companies mentioned

        Returns:
            RegulatoryUpdate with analysis
        """
        desc_lower = description.lower()

        # Identify target sector
        target_sector = "unknown"
        for sector, config in SECTOR_KEYWORDS.items():
            if any(kw in desc_lower for kw in config["keywords"]):
                target_sector = sector
                break

        # Determine sentiment
        if action_type in ["investigation", "fine", "enforcement"]:
            sentiment = "negative"
            impact_score = 0.6
        elif action_type in ["approval", "guidance_positive"]:
            sentiment = "positive"
            impact_score = 0.5
        else:
            positive_count = sum(1 for ind in POSITIVE_INDICATORS if ind in desc_lower)
            negative_count = sum(1 for ind in NEGATIVE_INDICATORS if ind in desc_lower)

            if negative_count > positive_count:
                sentiment = "negative"
                impact_score = min(0.8, 0.3 + negative_count * 0.1)
            elif positive_count > negative_count:
                sentiment = "positive"
                impact_score = min(0.7, 0.3 + positive_count * 0.1)
            else:
                sentiment = "neutral"
                impact_score = 0.2

        update = RegulatoryUpdate(
            agency=agency,
            action_type=action_type,
            target_sector=target_sector,
            target_companies=target_companies or [],
            sentiment=sentiment,
            impact_score=impact_score,
            summary=description[:300],
            date=datetime.now().isoformat(),
        )

        key = f"{agency}_{action_type}_{datetime.now().strftime('%Y%m%d')}"
        self.regulatory_cache[key] = update

        # Generate trading signals
        self._generate_signals_from_regulatory(update)

        return update

    def _generate_signals_from_regulatory(self, update: RegulatoryUpdate) -> None:
        """Generate trading signals from regulatory update."""
        if update.target_sector == "unknown":
            return

        sector_config = SECTOR_KEYWORDS.get(update.target_sector, {})
        symbols = sector_config.get("symbols", [])

        for symbol in symbols:
            # Specific company targeting has higher impact
            if symbol in update.target_companies:
                strength = update.impact_score * 1.5
            else:
                strength = update.impact_score * 0.5

            direction = "BEARISH" if update.sentiment == "negative" else "BULLISH"
            if update.sentiment == "neutral":
                direction = "NEUTRAL"
                strength = 0.0

            signal = PoliticalSignal(
                symbol=symbol,
                sector=update.target_sector,
                direction=direction,
                strength=min(1.0, strength),
                confidence=0.5,  # Moderate confidence for regulatory signals
                sources=[f"{update.agency} {update.action_type}"],
                reasoning=update.summary,
                expires_at=(datetime.now() + timedelta(days=7)).isoformat(),
            )

            self.active_signals[symbol] = signal

        self._save_state()

    def get_signal_for_symbol(self, symbol: str) -> Optional[PoliticalSignal]:
        """Get active political signal for a symbol."""
        signal = self.active_signals.get(symbol)
        if signal:
            # Check expiration
            if signal.expires_at and signal.expires_at < datetime.now().isoformat():
                del self.active_signals[symbol]
                return None
        return signal

    def get_sector_sentiment(self, sector: str) -> Dict[str, Any]:
        """
        Get aggregated political sentiment for a sector.

        Args:
            sector: Sector name

        Returns:
            Dict with sentiment summary
        """
        signals = [
            s for s in self.active_signals.values()
            if s.sector == sector
        ]

        if not signals:
            return {
                "sector": sector,
                "sentiment": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "active_signals": 0,
            }

        # Aggregate signals
        bullish = sum(1 for s in signals if s.direction == "BULLISH")
        bearish = sum(1 for s in signals if s.direction == "BEARISH")
        avg_strength = sum(s.strength for s in signals) / len(signals)
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        if bullish > bearish:
            sentiment = "BULLISH"
        elif bearish > bullish:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return {
            "sector": sector,
            "sentiment": sentiment,
            "strength": round(avg_strength, 2),
            "confidence": round(avg_confidence, 2),
            "active_signals": len(signals),
            "bullish_count": bullish,
            "bearish_count": bearish,
        }

    def validate_trade_signal(
        self,
        symbol: str,
        trade_direction: str
    ) -> Dict[str, Any]:
        """
        Validate a trade signal against political sentiment.

        Args:
            symbol: Trading symbol
            trade_direction: "LONG" or "SHORT"

        Returns:
            Validation result with adjustment recommendation
        """
        signal = self.get_signal_for_symbol(symbol)

        if not signal:
            return {
                "symbol": symbol,
                "has_political_signal": False,
                "alignment": "NEUTRAL",
                "confidence_adjustment": 0.0,
                "reasoning": "No active political signal for this symbol",
            }

        # Check alignment
        aligned = (
            (trade_direction == "LONG" and signal.direction == "BULLISH") or
            (trade_direction == "SHORT" and signal.direction == "BEARISH")
        )
        contradicts = (
            (trade_direction == "LONG" and signal.direction == "BEARISH") or
            (trade_direction == "SHORT" and signal.direction == "BULLISH")
        )

        if aligned:
            alignment = "ALIGNED"
            confidence_adjustment = signal.strength * signal.confidence * 0.1  # +10% max
        elif contradicts:
            alignment = "CONTRADICTS"
            confidence_adjustment = -signal.strength * signal.confidence * 0.15  # -15% max
        else:
            alignment = "NEUTRAL"
            confidence_adjustment = 0.0

        return {
            "symbol": symbol,
            "has_political_signal": True,
            "political_direction": signal.direction,
            "political_strength": signal.strength,
            "alignment": alignment,
            "confidence_adjustment": round(confidence_adjustment, 3),
            "reasoning": signal.reasoning,
            "sources": signal.sources,
        }


# Singleton instance
_analyzer: Optional[PoliticalSentimentAnalyzer] = None


def get_political_analyzer() -> PoliticalSentimentAnalyzer:
    """Get or create singleton analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = PoliticalSentimentAnalyzer()
    return _analyzer


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = get_political_analyzer()

    # Test legislation analysis
    print("Analyzing sample legislation...")
    sample_bill = """
    The Technology Privacy and Consumer Protection Act aims to regulate how
    technology companies collect and use personal data. Key provisions include
    requiring explicit consent for data collection, restricting the sale of
    user data to third parties, and imposing fines for data breaches.
    """

    analysis = analyzer.analyze_legislation_text(
        sample_bill,
        "HR-1234",
        "Technology Privacy and Consumer Protection Act"
    )

    print(f"\nBill: {analysis.title}")
    print(f"Sectors: {analysis.sectors_affected}")
    print(f"Impact: {analysis.impact_direction}")

    # Test regulatory action
    print("\nAnalyzing regulatory action...")
    update = analyzer.analyze_regulatory_action(
        agency="FTC",
        action_type="investigation",
        description="FTC launches antitrust investigation into major technology platform's acquisitions",
        target_companies=["META"]
    )

    print(f"Agency: {update.agency}")
    print(f"Sentiment: {update.sentiment}")
    print(f"Impact: {update.impact_score}")

    # Validate trade signal
    validation = analyzer.validate_trade_signal("META", "LONG")
    print(f"\nTrade Validation for META LONG:")
    print(f"  Alignment: {validation['alignment']}")
    print(f"  Adjustment: {validation['confidence_adjustment']:+.1%}")
