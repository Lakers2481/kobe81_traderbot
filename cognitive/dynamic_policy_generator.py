"""
Dynamic Policy Generator - Adaptive Trading Policy Management
==============================================================

This module manages trading policies that dynamically adjust system behavior
based on market conditions, LLM critiques, and discovered edges.

Key features:
1. **Policy Activation**: Automatically activate/deactivate policies based on conditions
2. **LLM Policy Generation**: Generate new policies from LLM critiques
3. **Edge-Based Policies**: Create policies from CuriosityEngine discoveries
4. **Semantic Rule Injection**: Add dynamic rules to SymbolicReasoner

Usage:
    from cognitive import DynamicPolicyGenerator

    generator = DynamicPolicyGenerator()

    # Check if a policy should activate
    active_policy = generator.evaluate_policy_activation(
        market_context=context,
        mood_score=-0.7,
        regime="BEAR"
    )

    if active_policy:
        # Apply policy modifications
        position_multiplier = active_policy.risk_modifications['max_position_size_multiplier']
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional YAML dependency
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


class PolicyType(Enum):
    """Types of trading policies."""
    CRISIS = "crisis"
    RISK_OFF = "risk_off"
    CAUTIOUS = "cautious"
    NEUTRAL = "neutral"
    BULL_MARKET_AGGRESSION = "bull_market_aggression"
    OPPORTUNITY = "opportunity"
    EARNINGS_SEASON = "earnings_season"
    LEARNING = "learning"


class PolicyStatus(Enum):
    """Status of a trading policy."""
    INACTIVE = "inactive"
    ACTIVATING = "activating"  # In grace period
    ACTIVE = "active"
    DEACTIVATING = "deactivating"  # In grace period
    SUSPENDED = "suspended"  # Manually suspended


@dataclass
class TradingPolicy:
    """Represents a complete trading policy configuration."""
    policy_id: str
    policy_type: PolicyType
    description: str
    priority: int = 100

    # Activation/deactivation conditions
    activation_conditions: Dict[str, Any] = field(default_factory=dict)
    deactivation_conditions: Dict[str, Any] = field(default_factory=dict)

    # Risk modifications
    risk_modifications: Dict[str, float] = field(default_factory=lambda: {
        'max_position_size_multiplier': 1.0,
        'max_daily_trades': 10,
        'max_open_positions': 8,
        'max_sector_concentration': 0.30,
    })

    # Execution modifications
    execution_modifications: Dict[str, Any] = field(default_factory=lambda: {
        'slippage_buffer': 0.001,
        'max_spread_pct': 0.005,
    })

    # Cognitive modifications
    cognitive_modifications: Dict[str, float] = field(default_factory=lambda: {
        'fast_path_threshold': 0.75,
        'slow_path_confidence_boost': 0.10,
        'require_llm_above_uncertainty': 0.60,
    })

    # Dynamic semantic rules to add when policy is active
    semantic_rules_to_add: List[Dict[str, Any]] = field(default_factory=list)

    # Status tracking
    status: PolicyStatus = PolicyStatus.INACTIVE
    activated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    activation_count: int = 0
    source: str = "config"  # 'config', 'llm', 'edge', 'manual'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'policy_id': self.policy_id,
            'policy_type': self.policy_type.value,
            'description': self.description,
            'priority': self.priority,
            'risk_modifications': self.risk_modifications,
            'execution_modifications': self.execution_modifications,
            'cognitive_modifications': self.cognitive_modifications,
            'status': self.status.value,
            'activated_at': self.activated_at.isoformat() if self.activated_at else None,
            'activation_count': self.activation_count,
            'source': self.source,
        }


@dataclass
class PolicyTransition:
    """Records a policy state transition."""
    policy_id: str
    from_status: PolicyStatus
    to_status: PolicyStatus
    timestamp: datetime
    reason: str
    market_context_snapshot: Dict[str, Any] = field(default_factory=dict)


class DynamicPolicyGenerator:
    """
    Manages dynamic trading policies based on market conditions and cognitive insights.

    The generator:
    1. Loads policy definitions from YAML configuration
    2. Evaluates activation conditions against market state
    3. Generates new policies from LLM critiques and edge discoveries
    4. Injects dynamic semantic rules into the SymbolicReasoner
    """

    def __init__(
        self,
        policies_file: Optional[str] = None,
        enabled: bool = True,
        min_policy_duration_minutes: int = 30,
    ):
        """
        Initialize the policy generator.

        Args:
            policies_file: Path to trading policies YAML. Defaults to config/trading_policies.yaml
            enabled: Whether dynamic policies are active
            min_policy_duration_minutes: Minimum time before policy can change
        """
        self.enabled = enabled
        self.min_policy_duration = timedelta(minutes=min_policy_duration_minutes)
        self._policies: Dict[str, TradingPolicy] = {}
        self._active_policy: Optional[TradingPolicy] = None
        self._transition_history: List[PolicyTransition] = []
        self._llm_generated_policies: Dict[str, TradingPolicy] = {}

        # LLM policy generation constraints
        self._llm_constraints = {
            'max_position_size_multiplier_range': [0.25, 1.50],
            'max_daily_trades_range': [1, 25],
            'max_open_positions_range': [1, 15],
            'protected_policies': ['POLICY_CRISIS', 'POLICY_DEFAULT'],
        }

        # Determine policies file path
        if policies_file:
            self._policies_file = Path(policies_file)
        else:
            project_root = Path(__file__).parent.parent
            self._policies_file = project_root / "config" / "trading_policies.yaml"

        # Load policies
        self._load_policies()
        logger.info(f"DynamicPolicyGenerator initialized with {len(self._policies)} policies")

    def _load_policies(self) -> None:
        """Load policies from YAML configuration file."""
        if not HAS_YAML:
            logger.warning("PyYAML not available. Using default policies.")
            self._load_default_policies()
            return

        if not self._policies_file.exists():
            logger.warning(f"Policies file not found: {self._policies_file}. Using defaults.")
            self._load_default_policies()
            return

        try:
            with open(self._policies_file, 'r') as f:
                config = yaml.safe_load(f)

            self._policies = {}

            # Load default policy
            if 'default_policy' in config:
                default = self._parse_policy(config['default_policy'])
                if default:
                    self._policies[default.policy_id] = default

            # Load other policies
            if 'policies' in config:
                for policy_dict in config['policies']:
                    policy = self._parse_policy(policy_dict)
                    if policy:
                        self._policies[policy.policy_id] = policy

            # Load LLM constraints
            if 'llm_policy_constraints' in config:
                self._llm_constraints.update(config['llm_policy_constraints'])

            # Load transition rules
            if 'transition_rules' in config:
                min_duration = config['transition_rules'].get('min_policy_duration_minutes', 30)
                self.min_policy_duration = timedelta(minutes=min_duration)

            logger.info(f"Loaded {len(self._policies)} policies from {self._policies_file}")

        except Exception as e:
            logger.error(f"Error loading policies: {e}. Using defaults.")
            self._load_default_policies()

    def _parse_policy(self, policy_dict: Dict[str, Any]) -> Optional[TradingPolicy]:
        """Parse a policy dictionary into a TradingPolicy object."""
        try:
            policy_type_str = policy_dict.get('type', 'neutral')
            try:
                policy_type = PolicyType(policy_type_str)
            except ValueError:
                policy_type = PolicyType.NEUTRAL

            return TradingPolicy(
                policy_id=policy_dict.get('id', 'UNKNOWN'),
                policy_type=policy_type,
                description=policy_dict.get('description', ''),
                priority=int(policy_dict.get('priority', 100)),
                activation_conditions=policy_dict.get('activation_conditions', {}),
                deactivation_conditions=policy_dict.get('deactivation_conditions', {}),
                risk_modifications=policy_dict.get('risk_modifications', {}),
                execution_modifications=policy_dict.get('execution_modifications', {}),
                cognitive_modifications=policy_dict.get('cognitive_modifications', {}),
                semantic_rules_to_add=policy_dict.get('semantic_rules_to_add', []),
                source='config',
            )
        except Exception as e:
            logger.error(f"Error parsing policy {policy_dict.get('id', 'UNKNOWN')}: {e}")
            return None

    def _load_default_policies(self) -> None:
        """Load minimal default policies."""
        self._policies = {
            'POLICY_DEFAULT': TradingPolicy(
                policy_id='POLICY_DEFAULT',
                policy_type=PolicyType.NEUTRAL,
                description='Standard operating mode',
                priority=100,
            ),
            'POLICY_CRISIS': TradingPolicy(
                policy_id='POLICY_CRISIS',
                policy_type=PolicyType.CRISIS,
                description='Crisis mode - maximum defensive',
                priority=1,
                activation_conditions={'vix_min': 40, 'market_mood_max': -0.6},
                deactivation_conditions={'vix_max': 30, 'market_mood_min': -0.3},
                risk_modifications={
                    'max_position_size_multiplier': 0.25,
                    'max_daily_trades': 2,
                    'max_open_positions': 2,
                },
                cognitive_modifications={
                    'fast_path_threshold': 0.95,
                    'stand_down_on_uncertainty': True,
                },
            ),
        }
        logger.info(f"Loaded {len(self._policies)} default policies")

    def evaluate_policy_activation(
        self,
        market_context: Dict[str, Any],
        mood_score: float,
        regime: str,
    ) -> Optional[TradingPolicy]:
        """
        Evaluate which policy should be active based on current conditions.

        Args:
            market_context: Current market state
            mood_score: Market mood score (-1 to 1)
            regime: Current market regime (BULL, BEAR, NEUTRAL, CHOPPY)

        Returns:
            The policy that should be active, or None for default
        """
        if not self.enabled:
            return None

        # Check minimum duration since last change
        if self._active_policy and self._active_policy.activated_at:
            elapsed = datetime.now() - self._active_policy.activated_at
            if elapsed < self.min_policy_duration:
                return self._active_policy

        # Build evaluation context
        context = {
            'vix': market_context.get('vix', 20.0),
            'regime': regime,
            'market_mood_score': mood_score,
            'earnings_season': market_context.get('earnings_season', False),
            'earnings_density': market_context.get('earnings_density', 0.0),
            'pending_hypotheses': market_context.get('pending_hypotheses', 0),
            'hypothesis_validation_rate': market_context.get('hypothesis_validation_rate', 0.5),
        }

        # Check deactivation of current policy first
        if self._active_policy:
            if self._should_deactivate(self._active_policy, context):
                self._deactivate_policy(self._active_policy, context)

        # Evaluate policies by priority (lower = higher priority)
        candidates = []
        for policy in self._policies.values():
            if policy.status == PolicyStatus.SUSPENDED:
                continue
            if self._check_activation_conditions(policy, context):
                candidates.append(policy)

        # Also check LLM-generated policies
        for policy in self._llm_generated_policies.values():
            if policy.status == PolicyStatus.SUSPENDED:
                continue
            if self._check_activation_conditions(policy, context):
                candidates.append(policy)

        if not candidates:
            # Return to default
            return self._policies.get('POLICY_DEFAULT')

        # Select highest priority (lowest number)
        candidates.sort(key=lambda p: p.priority)
        best_policy = candidates[0]

        # Activate if different from current
        if self._active_policy != best_policy:
            self._activate_policy(best_policy, context)

        return self._active_policy

    def _check_activation_conditions(
        self,
        policy: TradingPolicy,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a policy's activation conditions are met."""
        conditions = policy.activation_conditions
        if not conditions:
            return False

        # Check for manual_only flag
        if conditions.get('manual_only', False):
            return False

        # Check VIX conditions
        if 'vix_min' in conditions:
            if context.get('vix', 0) < conditions['vix_min']:
                return False
        if 'vix_max' in conditions:
            if context.get('vix', 100) > conditions['vix_max']:
                return False

        # Check mood conditions
        if 'market_mood_min' in conditions:
            if context.get('market_mood_score', 0) < conditions['market_mood_min']:
                return False
        if 'market_mood_max' in conditions:
            if context.get('market_mood_score', 0) > conditions['market_mood_max']:
                return False

        # Check regime
        if 'regime' in conditions:
            if context.get('regime') != conditions['regime']:
                return False

        # Check earnings conditions
        if 'earnings_season' in conditions:
            if context.get('earnings_season') != conditions['earnings_season']:
                return False

        # Check OR conditions (any one triggers)
        if 'or_conditions' in conditions:
            or_met = False
            for or_cond in conditions['or_conditions']:
                if self._check_simple_conditions(or_cond, context):
                    or_met = True
                    break
            if not or_met:
                return False

        return True

    def _check_simple_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check simple condition dict against context."""
        for key, value in conditions.items():
            if key.endswith('_min'):
                var_name = key[:-4]
                if context.get(var_name, 0) < value:
                    return False
            elif key.endswith('_max'):
                var_name = key[:-4]
                if context.get(var_name, 0) > value:
                    return False
            elif key.endswith('_range'):
                var_name = key[:-6]
                var_value = context.get(var_name, 0)
                if not (value[0] <= var_value <= value[1]):
                    return False
            else:
                if context.get(key) != value:
                    return False
        return True

    def _should_deactivate(
        self,
        policy: TradingPolicy,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a policy should be deactivated."""
        conditions = policy.deactivation_conditions
        if not conditions:
            return False

        # All deactivation conditions must be met
        return self._check_simple_conditions(conditions, context)

    def _activate_policy(
        self,
        policy: TradingPolicy,
        context: Dict[str, Any]
    ) -> None:
        """Activate a policy."""
        old_policy = self._active_policy

        # Record transition
        if old_policy:
            self._transition_history.append(PolicyTransition(
                policy_id=old_policy.policy_id,
                from_status=PolicyStatus.ACTIVE,
                to_status=PolicyStatus.INACTIVE,
                timestamp=datetime.now(),
                reason="New policy activated",
                market_context_snapshot=context.copy(),
            ))
            old_policy.status = PolicyStatus.INACTIVE
            old_policy.deactivated_at = datetime.now()

        # Activate new policy
        policy.status = PolicyStatus.ACTIVE
        policy.activated_at = datetime.now()
        policy.activation_count += 1
        self._active_policy = policy

        self._transition_history.append(PolicyTransition(
            policy_id=policy.policy_id,
            from_status=PolicyStatus.INACTIVE,
            to_status=PolicyStatus.ACTIVE,
            timestamp=datetime.now(),
            reason="Conditions met",
            market_context_snapshot=context.copy(),
        ))

        logger.info(f"Activated policy: {policy.policy_id} ({policy.policy_type.value})")

    def _deactivate_policy(
        self,
        policy: TradingPolicy,
        context: Dict[str, Any]
    ) -> None:
        """Deactivate the current policy."""
        policy.status = PolicyStatus.INACTIVE
        policy.deactivated_at = datetime.now()

        self._transition_history.append(PolicyTransition(
            policy_id=policy.policy_id,
            from_status=PolicyStatus.ACTIVE,
            to_status=PolicyStatus.INACTIVE,
            timestamp=datetime.now(),
            reason="Deactivation conditions met",
            market_context_snapshot=context.copy(),
        ))

        self._active_policy = None
        logger.info(f"Deactivated policy: {policy.policy_id}")

    def generate_policy_from_critique(
        self,
        llm_critique: str,
        reflection: Any,
    ) -> Optional[TradingPolicy]:
        """
        Generate a new policy from an LLM critique.

        Args:
            llm_critique: The LLM's critique text
            reflection: The Reflection object

        Returns:
            New policy if generated, None otherwise
        """
        if not llm_critique:
            return None

        # Look for specific policy suggestions in critique
        policy_keywords = {
            'reduce position size': ('REDUCE_SIZE', 0.75),
            'reduce risk': ('RISK_OFF', 0.80),
            'increase caution': ('CAUTIOUS', 0.75),
            'defensive': ('RISK_OFF', 0.70),
            'more conservative': ('CAUTIOUS', 0.70),
        }

        critique_lower = llm_critique.lower()
        for keyword, (policy_type, multiplier) in policy_keywords.items():
            if keyword in critique_lower:
                policy_id = f"LLM_GEN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                policy = TradingPolicy(
                    policy_id=policy_id,
                    policy_type=PolicyType(policy_type.lower()) if policy_type.lower() in [p.value for p in PolicyType] else PolicyType.CAUTIOUS,
                    description=f"LLM-generated: {keyword}",
                    priority=50,  # Medium priority
                    activation_conditions={'manual_only': False},  # Auto-activate
                    risk_modifications={
                        'max_position_size_multiplier': self._constrain_multiplier(multiplier),
                    },
                    source='llm',
                )

                self._llm_generated_policies[policy_id] = policy
                logger.info(f"Generated policy from LLM critique: {policy_id}")
                return policy

        return None

    def generate_policy_from_edge(
        self,
        edge: Dict[str, Any],
    ) -> Optional[TradingPolicy]:
        """
        Generate a policy from a discovered edge in CuriosityEngine.

        Args:
            edge: Edge discovery dictionary with conditions and performance

        Returns:
            New policy if generated, None otherwise
        """
        if not edge or edge.get('confidence', 0) < 0.7:
            return None

        edge_type = edge.get('edge_type', 'unknown')
        conditions = edge.get('conditions', {})
        performance = edge.get('performance', {})

        policy_id = f"EDGE_{edge_type.upper()}_{datetime.now().strftime('%Y%m%d')}"

        # Determine policy type based on edge
        if 'bull' in edge_type.lower():
            policy_type = PolicyType.BULL_MARKET_AGGRESSION
            multiplier = min(1.25, 1.0 + performance.get('excess_return', 0) * 2)
        elif 'bear' in edge_type.lower() or 'defensive' in edge_type.lower():
            policy_type = PolicyType.CAUTIOUS
            multiplier = max(0.5, 1.0 - performance.get('max_drawdown', 0))
        else:
            policy_type = PolicyType.OPPORTUNITY
            multiplier = 1.0

        policy = TradingPolicy(
            policy_id=policy_id,
            policy_type=policy_type,
            description=f"Edge-based: {edge_type}",
            priority=40,
            activation_conditions=conditions,
            risk_modifications={
                'max_position_size_multiplier': self._constrain_multiplier(multiplier),
            },
            source='edge',
        )

        self._llm_generated_policies[policy_id] = policy
        logger.info(f"Generated policy from edge discovery: {policy_id}")
        return policy

    def _constrain_multiplier(self, value: float) -> float:
        """Constrain a multiplier to allowed range."""
        min_val, max_val = self._llm_constraints.get(
            'max_position_size_multiplier_range', [0.25, 1.50]
        )
        return max(min_val, min(max_val, value))

    def activate_policy_manually(self, policy_id: str) -> bool:
        """Manually activate a specific policy."""
        policy = self._policies.get(policy_id) or self._llm_generated_policies.get(policy_id)
        if not policy:
            logger.warning(f"Policy not found: {policy_id}")
            return False

        self._activate_policy(policy, {})
        return True

    def deactivate_current_policy(self) -> bool:
        """Deactivate the current active policy."""
        if not self._active_policy:
            return False

        self._deactivate_policy(self._active_policy, {})
        return True

    def suspend_policy(self, policy_id: str) -> bool:
        """Suspend a policy (prevent it from activating)."""
        policy = self._policies.get(policy_id) or self._llm_generated_policies.get(policy_id)
        if not policy:
            return False

        policy.status = PolicyStatus.SUSPENDED
        logger.info(f"Suspended policy: {policy_id}")
        return True

    def resume_policy(self, policy_id: str) -> bool:
        """Resume a suspended policy."""
        policy = self._policies.get(policy_id) or self._llm_generated_policies.get(policy_id)
        if not policy:
            return False

        if policy.status == PolicyStatus.SUSPENDED:
            policy.status = PolicyStatus.INACTIVE
            logger.info(f"Resumed policy: {policy_id}")
            return True
        return False

    def get_active_policy(self) -> Optional[TradingPolicy]:
        """Get the currently active policy."""
        return self._active_policy

    def get_all_policies(self) -> List[TradingPolicy]:
        """Get all policies (config + LLM-generated)."""
        return list(self._policies.values()) + list(self._llm_generated_policies.values())

    def get_transition_history(self, limit: int = 50) -> List[PolicyTransition]:
        """Get recent policy transitions."""
        return self._transition_history[-limit:]

    def get_semantic_rules_for_active_policy(self) -> List[Dict[str, Any]]:
        """Get semantic rules to inject for the active policy."""
        if not self._active_policy:
            return []
        return self._active_policy.semantic_rules_to_add

    def reload_policies(self) -> int:
        """Reload policies from configuration. Returns count loaded."""
        self._load_policies()
        return len(self._policies)

    def introspect(self) -> str:
        """Generate human-readable description of policy state."""
        lines = [
            "--- Dynamic Policy Generator ---",
            f"Enabled: {self.enabled}",
            f"Policies file: {self._policies_file}",
            f"Config policies: {len(self._policies)}",
            f"LLM-generated policies: {len(self._llm_generated_policies)}",
            f"Min policy duration: {self.min_policy_duration}",
            "",
            f"Active policy: {self._active_policy.policy_id if self._active_policy else 'None'}",
            "",
            "Policy summary:",
        ]

        for policy in self.get_all_policies():
            status = "ACTIVE" if policy == self._active_policy else policy.status.value
            lines.append(f"  - {policy.policy_id}: {policy.policy_type.value} [{status}]")

        return "\n".join(lines)


# --- Singleton Implementation ---
_policy_generator: Optional[DynamicPolicyGenerator] = None
_lock = threading.Lock()


def get_policy_generator() -> DynamicPolicyGenerator:
    """Factory function to get the singleton DynamicPolicyGenerator instance."""
    global _policy_generator
    if _policy_generator is None:
        with _lock:
            if _policy_generator is None:
                _policy_generator = DynamicPolicyGenerator()
    return _policy_generator
