"""
Self-Learner - Autonomous Learning and Self-Improvement

Tracks every change, learns from outcomes, and enables self-fixing.

Features:
- Change tracking with WHY and OUTCOME
- Error pattern detection
- Self-diagnosis and fix suggestions
- Learning from every trade/decision
- Bug pattern recognition

Author: Kobe Trading System
Created: 2026-01-04
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import hashlib
import traceback

# Load environment
from dotenv import load_dotenv
load_dotenv()

from core.structured_log import get_logger

logger = get_logger(__name__)


class ChangeType(Enum):
    """Types of changes tracked."""
    CODE_FIX = "code_fix"
    CONFIG_CHANGE = "config_change"
    PARAMETER_TUNE = "parameter_tune"
    BUG_FIX = "bug_fix"
    FEATURE_ADD = "feature_add"
    ERROR_RESOLUTION = "error_resolution"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"


class OutcomeType(Enum):
    """Outcome of a change."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"
    REVERTED = "reverted"


@dataclass
class ChangeRecord:
    """Record of a single change."""
    change_id: str
    change_type: ChangeType
    description: str
    why: str                          # WHY this change was made
    what_changed: str                 # WHAT specifically changed
    files_affected: List[str]
    before_state: Dict[str, Any]      # State before change
    after_state: Dict[str, Any]       # State after change
    outcome: OutcomeType
    outcome_details: str
    lessons_learned: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "description": self.description,
            "why": self.why,
            "what_changed": self.what_changed,
            "files_affected": self.files_affected,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "outcome": self.outcome.value,
            "outcome_details": self.outcome_details,
            "lessons_learned": self.lessons_learned,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ErrorPattern:
    """Pattern of recurring errors."""
    error_type: str
    error_message: str
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    files_involved: List[str]
    fix_attempts: List[str]
    successful_fix: Optional[str]
    root_cause: Optional[str]
    prevention_steps: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "files_involved": self.files_involved,
            "fix_attempts": self.fix_attempts,
            "successful_fix": self.successful_fix,
            "root_cause": self.root_cause,
            "prevention_steps": self.prevention_steps,
        }


class SelfLearner:
    """
    Autonomous learning and self-improvement system.

    Tracks:
    - Every code change with WHY and WHAT
    - Error patterns and fixes
    - Successful strategies
    - Failed approaches (to avoid repeating)
    """

    CHANGES_FILE = Path("state/guardian/change_history.jsonl")
    ERRORS_FILE = Path("state/guardian/error_patterns.json")
    LESSONS_FILE = Path("state/guardian/lessons_learned.json")
    FIXES_FILE = Path("state/guardian/known_fixes.json")

    def __init__(self):
        """Initialize self-learner."""
        self._change_history: List[ChangeRecord] = []
        self._error_patterns: Dict[str, ErrorPattern] = {}
        self._lessons: List[Dict[str, Any]] = []
        self._known_fixes: Dict[str, str] = {}

        # Ensure directories
        self.CHANGES_FILE.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()

    def _load_state(self) -> None:
        """Load learning state."""
        # Load error patterns
        if self.ERRORS_FILE.exists():
            try:
                with open(self.ERRORS_FILE, "r") as f:
                    data = json.load(f)
                    for key, pattern in data.items():
                        self._error_patterns[key] = ErrorPattern(
                            error_type=pattern["error_type"],
                            error_message=pattern["error_message"],
                            occurrence_count=pattern["occurrence_count"],
                            first_seen=datetime.fromisoformat(pattern["first_seen"]),
                            last_seen=datetime.fromisoformat(pattern["last_seen"]),
                            files_involved=pattern["files_involved"],
                            fix_attempts=pattern["fix_attempts"],
                            successful_fix=pattern.get("successful_fix"),
                            root_cause=pattern.get("root_cause"),
                            prevention_steps=pattern.get("prevention_steps", []),
                        )
            except Exception as e:
                logger.warning(f"Failed to load error patterns: {e}")

        # Load lessons
        if self.LESSONS_FILE.exists():
            try:
                with open(self.LESSONS_FILE, "r") as f:
                    self._lessons = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load lessons: {e}")

        # Load known fixes
        if self.FIXES_FILE.exists():
            try:
                with open(self.FIXES_FILE, "r") as f:
                    self._known_fixes = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load known fixes: {e}")

    def _save_state(self) -> None:
        """Save learning state."""
        try:
            # Save error patterns
            with open(self.ERRORS_FILE, "w") as f:
                json.dump({k: v.to_dict() for k, v in self._error_patterns.items()}, f, indent=2)

            # Save lessons
            with open(self.LESSONS_FILE, "w") as f:
                json.dump(self._lessons, f, indent=2)

            # Save known fixes
            with open(self.FIXES_FILE, "w") as f:
                json.dump(self._known_fixes, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    def record_change(
        self,
        change_type: ChangeType,
        description: str,
        why: str,
        what_changed: str,
        files_affected: List[str],
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a change with full context.

        Args:
            change_type: Type of change
            description: Brief description
            why: WHY this change was made (critical for learning)
            what_changed: WHAT specifically changed
            files_affected: List of files changed
            before_state: State before change
            after_state: State after change

        Returns:
            change_id for future reference
        """
        change_id = hashlib.md5(
            f"{datetime.now().isoformat()}:{description}".encode()
        ).hexdigest()[:12]

        record = ChangeRecord(
            change_id=change_id,
            change_type=change_type,
            description=description,
            why=why,
            what_changed=what_changed,
            files_affected=files_affected,
            before_state=before_state or {},
            after_state=after_state or {},
            outcome=OutcomeType.PENDING,
            outcome_details="Awaiting verification",
            lessons_learned=[],
        )

        self._change_history.append(record)

        # Append to file
        try:
            with open(self.CHANGES_FILE, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to record change: {e}")

        logger.info(f"Change recorded: {change_id} - {description}")
        return change_id

    def record_outcome(
        self,
        change_id: str,
        outcome: OutcomeType,
        details: str,
        lessons: Optional[List[str]] = None,
    ) -> None:
        """Record the outcome of a change."""
        for record in self._change_history:
            if record.change_id == change_id:
                record.outcome = outcome
                record.outcome_details = details
                record.lessons_learned = lessons or []

                # Add lessons to global lessons
                for lesson in record.lessons_learned:
                    self._lessons.append({
                        "lesson": lesson,
                        "change_id": change_id,
                        "change_type": record.change_type.value,
                        "timestamp": datetime.now().isoformat(),
                    })

                self._save_state()
                break

    def record_error(
        self,
        error_type: str,
        error_message: str,
        file_path: Optional[str] = None,
        stack_trace: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record an error and check for known fixes.

        Returns:
            Known fix if available, None otherwise
        """
        # Create error key
        error_key = hashlib.md5(
            f"{error_type}:{error_message[:100]}".encode()
        ).hexdigest()[:12]

        if error_key in self._error_patterns:
            # Known error - update count
            pattern = self._error_patterns[error_key]
            pattern.occurrence_count += 1
            pattern.last_seen = datetime.now()
            if file_path and file_path not in pattern.files_involved:
                pattern.files_involved.append(file_path)
        else:
            # New error
            self._error_patterns[error_key] = ErrorPattern(
                error_type=error_type,
                error_message=error_message[:500],
                occurrence_count=1,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                files_involved=[file_path] if file_path else [],
                fix_attempts=[],
                successful_fix=None,
                root_cause=None,
                prevention_steps=[],
            )

        self._save_state()

        # Check for known fix
        return self._known_fixes.get(error_key)

    def record_fix(
        self,
        error_type: str,
        error_message: str,
        fix_description: str,
        root_cause: str,
        prevention_steps: List[str],
    ) -> None:
        """Record a successful fix for an error."""
        error_key = hashlib.md5(
            f"{error_type}:{error_message[:100]}".encode()
        ).hexdigest()[:12]

        # Store fix
        self._known_fixes[error_key] = fix_description

        # Update error pattern
        if error_key in self._error_patterns:
            pattern = self._error_patterns[error_key]
            pattern.successful_fix = fix_description
            pattern.root_cause = root_cause
            pattern.prevention_steps = prevention_steps
            pattern.fix_attempts.append(fix_description)

        self._save_state()

        logger.info(f"Fix recorded for: {error_type}")

    def suggest_fix(self, error_type: str, error_message: str) -> Optional[Dict[str, Any]]:
        """
        Suggest a fix based on learned patterns.

        Returns:
            Dict with fix suggestion and confidence, or None
        """
        error_key = hashlib.md5(
            f"{error_type}:{error_message[:100]}".encode()
        ).hexdigest()[:12]

        if error_key in self._error_patterns:
            pattern = self._error_patterns[error_key]

            if pattern.successful_fix:
                return {
                    "fix": pattern.successful_fix,
                    "root_cause": pattern.root_cause,
                    "prevention": pattern.prevention_steps,
                    "confidence": min(0.9, 0.5 + (pattern.occurrence_count * 0.1)),
                    "times_seen": pattern.occurrence_count,
                }

        # Check similar errors
        for key, pattern in self._error_patterns.items():
            if error_type == pattern.error_type and pattern.successful_fix:
                return {
                    "fix": pattern.successful_fix,
                    "root_cause": pattern.root_cause,
                    "prevention": pattern.prevention_steps,
                    "confidence": 0.3,  # Lower confidence for similar but not exact match
                    "times_seen": pattern.occurrence_count,
                    "note": "Similar error pattern",
                }

        return None

    def diagnose_system(self) -> Dict[str, Any]:
        """
        Run self-diagnosis on the system.

        Returns:
            Diagnosis report with issues and suggestions
        """
        issues = []
        suggestions = []

        # Check for recurring errors
        for key, pattern in self._error_patterns.items():
            if pattern.occurrence_count >= 3 and not pattern.successful_fix:
                issues.append({
                    "type": "recurring_error",
                    "error": pattern.error_type,
                    "count": pattern.occurrence_count,
                    "files": pattern.files_involved,
                })
                suggestions.append(f"Investigate recurring {pattern.error_type} in {pattern.files_involved}")

        # Check for failed changes
        failed_count = sum(1 for r in self._change_history if r.outcome == OutcomeType.FAILED)
        if failed_count > 0:
            issues.append({
                "type": "failed_changes",
                "count": failed_count,
            })

        # Check critical files
        critical_files = [
            Path("state/KILL_SWITCH"),
            Path("state/guardian/emergency.json"),
        ]

        for f in critical_files:
            if f.exists():
                if f.name == "KILL_SWITCH":
                    issues.append({
                        "type": "kill_switch_active",
                        "file": str(f),
                    })
                    suggestions.append("Kill switch is active - check emergency status")

        return {
            "diagnosis_time": datetime.now().isoformat(),
            "issues_found": len(issues),
            "issues": issues,
            "suggestions": suggestions,
            "error_patterns_tracked": len(self._error_patterns),
            "lessons_learned": len(self._lessons),
            "known_fixes": len(self._known_fixes),
        }

    def get_recent_lessons(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get lessons learned in the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            l for l in self._lessons
            if datetime.fromisoformat(l["timestamp"]) > cutoff
        ]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error patterns."""
        total_errors = sum(p.occurrence_count for p in self._error_patterns.values())
        fixed_errors = sum(1 for p in self._error_patterns.values() if p.successful_fix)

        top_errors = sorted(
            self._error_patterns.values(),
            key=lambda p: p.occurrence_count,
            reverse=True
        )[:5]

        return {
            "total_error_types": len(self._error_patterns),
            "total_occurrences": total_errors,
            "fixed_count": fixed_errors,
            "fix_rate": fixed_errors / max(len(self._error_patterns), 1),
            "top_errors": [
                {
                    "type": e.error_type,
                    "count": e.occurrence_count,
                    "has_fix": e.successful_fix is not None,
                }
                for e in top_errors
            ],
        }


# Record the fix we just made
def record_dotenv_fix():
    """Record the dotenv fix we just made."""
    learner = SelfLearner()

    # Record the change
    change_id = learner.record_change(
        change_type=ChangeType.BUG_FIX,
        description="Add dotenv loading to all modules that use os.getenv()",
        why="API keys in .env file were not being loaded because os.getenv() only reads "
            "actual environment variables, not .env files. The python-dotenv library "
            "must explicitly load .env first.",
        what_changed="Added 'from dotenv import load_dotenv; load_dotenv()' to: "
                     "guardian/__init__.py, guardian/system_monitor.py, data/providers/polygon_eod.py",
        files_affected=[
            "guardian/__init__.py",
            "guardian/system_monitor.py",
            "data/providers/polygon_eod.py",
        ],
        before_state={
            "guardian_status": "EMERGENCY",
            "api_keys_loaded": False,
            "data_feeds": "UNHEALTHY",
            "broker": "UNHEALTHY",
        },
        after_state={
            "guardian_status": "MONITORING",
            "api_keys_loaded": True,
            "data_feeds": "DEGRADED (stale cache only)",
            "broker": "HEALTHY",
        },
    )

    # Record outcome
    learner.record_outcome(
        change_id=change_id,
        outcome=OutcomeType.SUCCESS,
        details="All 8 API keys now loading correctly. Guardian no longer in EMERGENCY mode.",
        lessons=[
            "Always add load_dotenv() at the top of any module that uses os.getenv()",
            "The .env file is not automatically loaded - must use python-dotenv",
            "When adding new modules, check if they need environment variables",
            "Test API key loading with direct os.getenv() check after import",
        ],
    )

    # Record the fix pattern
    learner.record_fix(
        error_type="EnvironmentVariableNotFound",
        error_message="API keys not set / POLYGON_API_KEY not set",
        fix_description="Add 'from dotenv import load_dotenv; load_dotenv()' at module top",
        root_cause="python-dotenv was not loading .env file before os.getenv() calls",
        prevention_steps=[
            "Always import and call load_dotenv() in modules using environment variables",
            "Add dotenv loading to __init__.py files for packages",
            "Test environment variable access after module import",
        ],
    )

    return learner


# Singleton
_learner: Optional[SelfLearner] = None


def get_self_learner() -> SelfLearner:
    """Get or create singleton learner."""
    global _learner
    if _learner is None:
        _learner = SelfLearner()
    return _learner


if __name__ == "__main__":
    # Demo and record the dotenv fix
    print("=== Self-Learner Demo ===\n")

    learner = record_dotenv_fix()

    print("Recorded dotenv fix. Running diagnosis...\n")

    diagnosis = learner.diagnose_system()
    print(f"Issues found: {diagnosis['issues_found']}")
    print(f"Error patterns tracked: {diagnosis['error_patterns_tracked']}")
    print(f"Lessons learned: {diagnosis['lessons_learned']}")
    print(f"Known fixes: {diagnosis['known_fixes']}")

    print("\nRecent lessons:")
    for lesson in learner.get_recent_lessons(1):
        print(f"  - {lesson['lesson']}")

    print("\nError summary:")
    summary = learner.get_error_summary()
    print(f"  Total error types: {summary['total_error_types']}")
    print(f"  Fix rate: {summary['fix_rate']:.0%}")
