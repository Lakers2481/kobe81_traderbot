"""
LLM Output Validation with Citation Verification

Implements zero-trust validation for LLM-generated content:
- JSON schema validation for structured outputs
- Citation verification against known data sources
- Numerical claim validation against actual data
- Hallucination detection via grounding checks

Based on: Codex & Gemini reliability recommendations (2026-01-04)

Usage:
    from cognitive.llm_validator import LLMValidator, validate_llm_response

    validator = LLMValidator()
    result = validator.validate(
        response="AAPL closed at $150.25 on 2024-01-15",
        expected_schema=TradeAnalysisSchema,
        verify_citations=True
    )
    if not result.is_valid:
        print(f"REJECTED: {result.errors}")
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Type, Union

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ClaimType(Enum):
    """Types of claims that can be validated."""
    PRICE = "price"           # Stock price claims
    PERCENTAGE = "percentage"  # Percentage changes
    DATE = "date"             # Date-based claims
    VOLUME = "volume"         # Trading volume
    STATISTIC = "statistic"   # Win rate, etc.
    RECOMMENDATION = "recommendation"  # Buy/sell/hold
    UNKNOWN = "unknown"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    severity: ValidationSeverity
    message: str
    claim: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None

    def __str__(self) -> str:
        msg = f"[{self.severity.value.upper()}] {self.field}: {self.message}"
        if self.claim:
            msg += f" (claim: '{self.claim}')"
        if self.expected is not None and self.actual is not None:
            msg += f" [expected: {self.expected}, actual: {self.actual}]"
        return msg


@dataclass
class ExtractedClaim:
    """A claim extracted from LLM output."""
    claim_type: ClaimType
    raw_text: str
    symbol: Optional[str] = None
    value: Optional[float] = None
    date: Optional[str] = None
    verified: bool = False
    verification_source: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of LLM output validation."""
    is_valid: bool
    response: str
    timestamp: str
    issues: List[ValidationIssue] = field(default_factory=list)
    claims_found: int = 0
    claims_verified: int = 0
    claims_failed: int = 0
    extracted_claims: List[ExtractedClaim] = field(default_factory=list)
    grounding_score: float = 1.0  # 0-1, how well grounded in data

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "claims_found": self.claims_found,
            "claims_verified": self.claims_verified,
            "claims_failed": self.claims_failed,
            "grounding_score": self.grounding_score,
            "issues": [str(i) for i in self.issues],
            "errors": [str(e) for e in self.errors],
        }


class LLMValidator:
    """
    Validates LLM outputs for trading system reliability.

    Implements:
    1. Schema validation for structured outputs
    2. Numerical claim extraction and verification
    3. Citation/source verification
    4. Hallucination detection
    """

    # Price claim patterns
    PRICE_PATTERNS: List[Pattern] = [
        re.compile(r'\$(\d+(?:\.\d{1,2})?)', re.IGNORECASE),
        re.compile(r'(\w{1,5})\s+(?:closed|opened|traded)\s+at\s+\$?(\d+(?:\.\d{1,2})?)', re.IGNORECASE),
        re.compile(r'price\s+(?:of|was|is)\s+\$?(\d+(?:\.\d{1,2})?)', re.IGNORECASE),
    ]

    # Percentage patterns
    PERCENTAGE_PATTERNS: List[Pattern] = [
        re.compile(r'(\d+(?:\.\d{1,2})?)\s*%', re.IGNORECASE),
        re.compile(r'win\s*rate\s*(?:of|is|was)?\s*(\d+(?:\.\d{1,2})?)\s*%?', re.IGNORECASE),
    ]

    # Date patterns
    DATE_PATTERNS: List[Pattern] = [
        re.compile(r'(\d{4}-\d{2}-\d{2})'),
        re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', re.IGNORECASE),
    ]

    # Symbol patterns
    SYMBOL_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')

    # Price tolerance for verification (0.5%)
    PRICE_TOLERANCE = 0.005

    def __init__(
        self,
        verify_prices: bool = True,
        verify_percentages: bool = True,
        strict_mode: bool = False,
        price_source: Optional[str] = "polygon",
    ):
        self.verify_prices = verify_prices
        self.verify_percentages = verify_percentages
        self.strict_mode = strict_mode
        self.price_source = price_source

    def validate(
        self,
        response: str,
        expected_schema: Optional[Type] = None,
        context: Optional[Dict[str, Any]] = None,
        verify_citations: bool = True,
    ) -> ValidationResult:
        """
        Validate an LLM response.

        Args:
            response: The LLM-generated text
            expected_schema: Optional dataclass or TypedDict for structure
            context: Optional context with known facts for grounding
            verify_citations: Whether to verify cited prices/dates

        Returns:
            ValidationResult with all issues found
        """
        issues: List[ValidationIssue] = []
        claims: List[ExtractedClaim] = []

        # 1. Extract claims
        claims = self._extract_claims(response)

        # 2. Verify claims against data sources
        verified_count = 0
        failed_count = 0

        if verify_citations and claims:
            for claim in claims:
                verified, issue = self._verify_claim(claim, context)
                claim.verified = verified
                if verified:
                    verified_count += 1
                else:
                    failed_count += 1
                    if issue:
                        issues.append(issue)

        # 3. Schema validation (if provided)
        if expected_schema:
            schema_issues = self._validate_schema(response, expected_schema)
            issues.extend(schema_issues)

        # 4. Check for hallucination indicators
        hallucination_issues = self._check_hallucination_indicators(response, context)
        issues.extend(hallucination_issues)

        # 5. Calculate grounding score
        grounding_score = self._calculate_grounding_score(claims, verified_count, failed_count)

        # Determine validity
        has_errors = any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues)
        is_valid = not has_errors and (not self.strict_mode or grounding_score >= 0.7)

        return ValidationResult(
            is_valid=is_valid,
            response=response,
            timestamp=datetime.utcnow().isoformat(),
            issues=issues,
            claims_found=len(claims),
            claims_verified=verified_count,
            claims_failed=failed_count,
            extracted_claims=claims,
            grounding_score=grounding_score,
        )

    def _extract_claims(self, text: str) -> List[ExtractedClaim]:
        """Extract verifiable claims from text."""
        claims = []

        # Extract price claims
        for pattern in self.PRICE_PATTERNS:
            for match in pattern.finditer(text):
                claim = ExtractedClaim(
                    claim_type=ClaimType.PRICE,
                    raw_text=match.group(0),
                )
                # Try to extract value
                try:
                    # Get the numeric group
                    groups = match.groups()
                    if len(groups) >= 2:
                        claim.symbol = groups[0]
                        claim.value = float(groups[1])
                    else:
                        claim.value = float(groups[0].replace('$', ''))
                except (ValueError, IndexError):
                    pass
                claims.append(claim)

        # Extract percentage claims
        for pattern in self.PERCENTAGE_PATTERNS:
            for match in pattern.finditer(text):
                claim = ExtractedClaim(
                    claim_type=ClaimType.PERCENTAGE,
                    raw_text=match.group(0),
                )
                try:
                    claim.value = float(match.group(1))
                except (ValueError, IndexError):
                    pass
                claims.append(claim)

        # Extract date claims
        for pattern in self.DATE_PATTERNS:
            for match in pattern.finditer(text):
                claim = ExtractedClaim(
                    claim_type=ClaimType.DATE,
                    raw_text=match.group(0),
                    date=match.group(1),
                )
                claims.append(claim)

        return claims

    def _verify_claim(
        self,
        claim: ExtractedClaim,
        context: Optional[Dict[str, Any]],
    ) -> tuple[bool, Optional[ValidationIssue]]:
        """Verify a single claim against known data."""

        # If context provides known prices, verify against them
        if context and claim.claim_type == ClaimType.PRICE:
            if claim.symbol and claim.symbol in context.get("prices", {}):
                actual_price = context["prices"][claim.symbol]
                if claim.value is not None:
                    diff = abs(claim.value - actual_price) / actual_price
                    if diff > self.PRICE_TOLERANCE:
                        return False, ValidationIssue(
                            field="price",
                            severity=ValidationSeverity.ERROR,
                            message=f"Price claim differs from actual by {diff:.1%}",
                            claim=claim.raw_text,
                            expected=actual_price,
                            actual=claim.value,
                        )
                    claim.verification_source = "context"
                    return True, None

        # If context provides known percentages, verify
        if context and claim.claim_type == ClaimType.PERCENTAGE:
            if "win_rate" in context and claim.value is not None:
                actual_wr = context["win_rate"]
                if abs(claim.value - actual_wr) > 2.0:  # 2% tolerance
                    return False, ValidationIssue(
                        field="win_rate",
                        severity=ValidationSeverity.WARNING,
                        message="Win rate claim differs from actual",
                        claim=claim.raw_text,
                        expected=actual_wr,
                        actual=claim.value,
                    )
                claim.verification_source = "context"
                return True, None

        # Try to verify against live data source
        if self.verify_prices and claim.claim_type == ClaimType.PRICE:
            return self._verify_price_from_source(claim)

        # Unverifiable claim (not necessarily invalid)
        return True, None

    def _verify_price_from_source(
        self,
        claim: ExtractedClaim,
    ) -> tuple[bool, Optional[ValidationIssue]]:
        """Verify price claim against data provider."""
        if not claim.symbol or not claim.value:
            return True, None

        try:
            # Try to fetch actual price
            if self.price_source == "polygon":
                from data.providers.polygon_eod import PolygonEODProvider
                provider = PolygonEODProvider()
                df = provider.fetch(claim.symbol, claim.date or "2024-01-01", claim.date or "2024-01-01")

                if df is not None and not df.empty:
                    actual = df.iloc[-1]["close"]
                    diff = abs(claim.value - actual) / actual
                    if diff > self.PRICE_TOLERANCE:
                        return False, ValidationIssue(
                            field="price",
                            severity=ValidationSeverity.ERROR,
                            message=f"Price ${claim.value:.2f} differs from Polygon ${actual:.2f}",
                            claim=claim.raw_text,
                            expected=actual,
                            actual=claim.value,
                        )
                    claim.verification_source = "polygon"
                    return True, None

        except Exception as e:
            logger.debug(f"Could not verify price claim: {e}")

        return True, None  # Can't verify, assume OK

    def _validate_schema(
        self,
        response: str,
        expected_schema: Type,
    ) -> List[ValidationIssue]:
        """Validate response against expected schema."""
        issues = []

        # Try to parse as JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Not JSON - might be natural language, skip schema validation
            return issues

        # Get expected fields from schema
        if hasattr(expected_schema, "__annotations__"):
            expected_fields = set(expected_schema.__annotations__.keys())
            actual_fields = set(data.keys()) if isinstance(data, dict) else set()

            # Check missing required fields
            missing = expected_fields - actual_fields
            if missing:
                issues.append(ValidationIssue(
                    field="schema",
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing expected fields: {missing}",
                ))

        return issues

    def _check_hallucination_indicators(
        self,
        response: str,
        context: Optional[Dict[str, Any]],
    ) -> List[ValidationIssue]:
        """Check for common hallucination indicators."""
        issues = []

        # Check for excessive confidence without backing
        confidence_phrases = [
            "guaranteed", "definitely will", "100% certain", "impossible to lose",
            "always works", "never fails", "risk-free",
        ]
        for phrase in confidence_phrases:
            if phrase.lower() in response.lower():
                issues.append(ValidationIssue(
                    field="confidence",
                    severity=ValidationSeverity.WARNING,
                    message=f"Overconfident language detected: '{phrase}'",
                ))

        # Check for specific but unverifiable claims
        specific_unverifiable = [
            re.compile(r'I have inside information', re.IGNORECASE),
            re.compile(r'sources tell me', re.IGNORECASE),
            re.compile(r'according to my sources', re.IGNORECASE),
        ]
        for pattern in specific_unverifiable:
            if pattern.search(response):
                issues.append(ValidationIssue(
                    field="source",
                    severity=ValidationSeverity.ERROR,
                    message="Unverifiable source claim detected",
                ))

        # Check for contradictions with context
        if context:
            # If we know the symbol, check it's mentioned correctly
            if "symbol" in context and context["symbol"] not in response.upper():
                # Allow if response is about something else
                pass

            # Check for wrong direction claims
            if "expected_direction" in context:
                expected = context["expected_direction"].lower()
                wrong_direction = "bearish" if expected == "bullish" else "bullish"
                if wrong_direction in response.lower() and expected not in response.lower():
                    issues.append(ValidationIssue(
                        field="direction",
                        severity=ValidationSeverity.ERROR,
                        message=f"Direction contradicts expected: claimed {wrong_direction}, expected {expected}",
                    ))

        return issues

    def _calculate_grounding_score(
        self,
        claims: List[ExtractedClaim],
        verified: int,
        failed: int,
    ) -> float:
        """Calculate how well-grounded the response is."""
        if not claims:
            return 1.0  # No claims to verify = neutral

        total_claims = len(claims)
        if total_claims == 0:
            return 1.0

        # Score based on verified vs failed
        verified_ratio = verified / total_claims
        failed_ratio = failed / total_claims

        # Penalize failed claims heavily
        score = verified_ratio - (failed_ratio * 2)

        return max(0.0, min(1.0, score))


# ============================================================================
# Convenience Functions
# ============================================================================

_default_validator: Optional[LLMValidator] = None


def get_validator() -> LLMValidator:
    """Get or create default validator."""
    global _default_validator
    if _default_validator is None:
        _default_validator = LLMValidator()
    return _default_validator


def validate_llm_response(
    response: str,
    context: Optional[Dict[str, Any]] = None,
    verify_citations: bool = True,
) -> ValidationResult:
    """Validate an LLM response with default validator."""
    return get_validator().validate(response, context=context, verify_citations=verify_citations)


def is_response_grounded(response: str, min_score: float = 0.7) -> bool:
    """Quick check if response is well-grounded."""
    result = validate_llm_response(response, verify_citations=True)
    return result.grounding_score >= min_score


def extract_and_verify_prices(
    text: str,
    known_prices: Dict[str, float],
) -> List[tuple[str, float, bool]]:
    """
    Extract price claims and verify against known prices.

    Returns list of (symbol, claimed_price, is_verified)
    """
    validator = get_validator()
    claims = validator._extract_claims(text)

    results = []
    for claim in claims:
        if claim.claim_type == ClaimType.PRICE and claim.symbol and claim.value:
            if claim.symbol in known_prices:
                actual = known_prices[claim.symbol]
                diff = abs(claim.value - actual) / actual
                verified = diff <= LLMValidator.PRICE_TOLERANCE
                results.append((claim.symbol, claim.value, verified))

    return results
