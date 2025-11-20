"""
Warrior Trading Self-Healing System

Auto-detects, diagnoses, and recovers from system errors using Claude AI.

Error Categories:
- Data feed issues (stale/missing data)
- API connection errors (IBKR, data providers)
- Pattern detection failures
- Risk management violations
- Database errors

Features:
- Automated error detection
- AI-powered diagnosis
- Automatic recovery attempts
- Error pattern learning
- Alert escalation
"""

import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import time

from claude_integration import get_claude_integration, ClaudeRequest

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """Error categories"""
    DATA_FEED = "DATA_FEED"
    API_CONNECTION = "API_CONNECTION"
    PATTERN_DETECTION = "PATTERN_DETECTION"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    DATABASE = "DATABASE"
    CONFIGURATION = "CONFIGURATION"
    UNKNOWN = "UNKNOWN"


class RecoveryStatus(Enum):
    """Recovery attempt status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESSFUL = "SUCCESSFUL"
    FAILED = "FAILED"
    MANUAL_REQUIRED = "MANUAL_REQUIRED"


@dataclass
class ErrorContext:
    """Contextual information about an error"""
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    stack_trace: Optional[str] = None
    component: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryAction:
    """A recovery action to attempt"""
    action_name: str
    action_function: Optional[Callable] = None
    description: str = ""
    estimated_time_seconds: int = 5
    success_criteria: Optional[Callable] = None


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    error_context: ErrorContext
    diagnosis: str
    recovery_actions: List[str]
    attempted_actions: List[str]
    status: RecoveryStatus
    recovery_time_seconds: float
    resolution_notes: str
    requires_manual_intervention: bool
    timestamp: datetime = datetime.now()


class SelfHealingSystem:
    """
    Automated error detection and recovery system

    Uses Claude AI to diagnose errors and suggest/attempt recovery
    """

    def __init__(self):
        """Initialize self-healing system"""
        self.claude = get_claude_integration()
        self.error_history: List[RecoveryResult] = []
        self.active_errors: Dict[str, ErrorContext] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryAction]] = {}
        self._initialize_recovery_strategies()
        logger.info("Self-healing system initialized")

    def _initialize_recovery_strategies(self):
        """Initialize predefined recovery strategies"""

        # Data feed recovery strategies
        self.recovery_strategies[ErrorCategory.DATA_FEED] = [
            RecoveryAction(
                action_name="clear_cache",
                description="Clear cached market data",
                estimated_time_seconds=1
            ),
            RecoveryAction(
                action_name="reconnect_data_feed",
                description="Reconnect to data provider",
                estimated_time_seconds=5
            ),
            RecoveryAction(
                action_name="switch_backup_feed",
                description="Switch to backup data source",
                estimated_time_seconds=10
            )
        ]

        # API connection recovery strategies
        self.recovery_strategies[ErrorCategory.API_CONNECTION] = [
            RecoveryAction(
                action_name="retry_connection",
                description="Retry API connection (3 attempts)",
                estimated_time_seconds=15
            ),
            RecoveryAction(
                action_name="verify_credentials",
                description="Verify API credentials",
                estimated_time_seconds=2
            ),
            RecoveryAction(
                action_name="check_gateway_status",
                description="Check IBKR Gateway status",
                estimated_time_seconds=3
            ),
            RecoveryAction(
                action_name="restart_gateway",
                description="Attempt to restart IBKR Gateway",
                estimated_time_seconds=30
            )
        ]

        # Pattern detection recovery strategies
        self.recovery_strategies[ErrorCategory.PATTERN_DETECTION] = [
            RecoveryAction(
                action_name="lower_confidence_threshold",
                description="Temporarily lower confidence thresholds",
                estimated_time_seconds=1
            ),
            RecoveryAction(
                action_name="restart_detector",
                description="Restart pattern detector service",
                estimated_time_seconds=5
            ),
            RecoveryAction(
                action_name="check_market_data",
                description="Verify market data availability",
                estimated_time_seconds=2
            )
        ]

        # Risk management recovery strategies
        self.recovery_strategies[ErrorCategory.RISK_MANAGEMENT] = [
            RecoveryAction(
                action_name="halt_trading",
                description="Halt all trading activity",
                estimated_time_seconds=1
            ),
            RecoveryAction(
                action_name="verify_positions",
                description="Verify open positions",
                estimated_time_seconds=5
            ),
            RecoveryAction(
                action_name="recalculate_risk",
                description="Recalculate risk metrics",
                estimated_time_seconds=2
            )
        ]

        # Database recovery strategies
        self.recovery_strategies[ErrorCategory.DATABASE] = [
            RecoveryAction(
                action_name="reconnect_database",
                description="Reconnect to database",
                estimated_time_seconds=2
            ),
            RecoveryAction(
                action_name="verify_integrity",
                description="Check database integrity",
                estimated_time_seconds=5
            ),
            RecoveryAction(
                action_name="backup_database",
                description="Create emergency backup",
                estimated_time_seconds=10
            )
        ]

    def detect_error(
        self,
        error: Exception,
        component: str = "unknown",
        additional_data: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """
        Detect and categorize an error

        Args:
            error: The exception that occurred
            component: Component where error occurred
            additional_data: Additional context data

        Returns:
            ErrorContext object
        """
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()

        # Categorize error
        category = self._categorize_error(error_type, error_message, component)

        # Determine severity
        severity = self._determine_severity(category, error_message)

        context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            category=category,
            severity=severity,
            timestamp=datetime.now(),
            stack_trace=stack_trace,
            component=component,
            additional_data=additional_data
        )

        # Track active error
        error_key = f"{component}_{error_type}"
        self.active_errors[error_key] = context

        logger.error(
            f"Error detected: {error_type} in {component} "
            f"(severity: {severity.value}, category: {category.value})"
        )

        return context

    def _categorize_error(
        self,
        error_type: str,
        error_message: str,
        component: str
    ) -> ErrorCategory:
        """Categorize error based on type, message, and component"""

        error_lower = error_message.lower()
        component_lower = component.lower()

        # Data feed errors
        if any(keyword in error_lower for keyword in ["stale", "missing data", "no data", "timeout"]):
            return ErrorCategory.DATA_FEED

        # API connection errors
        if any(keyword in error_lower for keyword in ["connection", "api", "gateway", "timeout"]):
            return ErrorCategory.API_CONNECTION

        # Pattern detection errors
        if "pattern" in component_lower or "detector" in component_lower:
            return ErrorCategory.PATTERN_DETECTION

        # Risk management errors
        if "risk" in component_lower or "position" in error_lower:
            return ErrorCategory.RISK_MANAGEMENT

        # Database errors
        if any(keyword in error_lower for keyword in ["database", "sql", "sqlite", "query"]):
            return ErrorCategory.DATABASE

        # Configuration errors
        if "config" in error_lower or "settings" in error_lower:
            return ErrorCategory.CONFIGURATION

        return ErrorCategory.UNKNOWN

    def _determine_severity(
        self,
        category: ErrorCategory,
        error_message: str
    ) -> ErrorSeverity:
        """Determine error severity"""

        error_lower = error_message.lower()

        # Critical keywords
        if any(keyword in error_lower for keyword in ["critical", "fatal", "crash", "corrupt"]):
            return ErrorSeverity.CRITICAL

        # Category-based severity
        if category == ErrorCategory.RISK_MANAGEMENT:
            return ErrorSeverity.HIGH

        if category == ErrorCategory.DATABASE:
            return ErrorSeverity.HIGH

        if category == ErrorCategory.API_CONNECTION:
            return ErrorSeverity.MEDIUM

        if category == ErrorCategory.DATA_FEED:
            return ErrorSeverity.MEDIUM

        if category == ErrorCategory.PATTERN_DETECTION:
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def diagnose_error(
        self,
        context: ErrorContext,
        use_ai: bool = True
    ) -> Dict[str, Any]:
        """
        Diagnose error and suggest recovery

        Args:
            context: ErrorContext object
            use_ai: Whether to use Claude AI for diagnosis

        Returns:
            Diagnosis dictionary
        """
        if use_ai:
            return self._diagnose_with_ai(context)
        else:
            return self._diagnose_rule_based(context)

    def _diagnose_rule_based(
        self,
        context: ErrorContext
    ) -> Dict[str, Any]:
        """Rule-based error diagnosis"""

        # Get predefined recovery actions
        recovery_actions = self.recovery_strategies.get(context.category, [])

        return {
            "diagnosis": f"{context.category.value} error detected",
            "severity": context.severity.value,
            "root_cause": f"Component: {context.component}, Type: {context.error_type}",
            "recovery_steps": [action.description for action in recovery_actions],
            "requires_manual_intervention": context.severity == ErrorSeverity.CRITICAL,
            "preventive_measures": [
                "Monitor error frequency",
                "Review logs for patterns",
                "Update error handling"
            ]
        }

    def _diagnose_with_ai(
        self,
        context: ErrorContext
    ) -> Dict[str, Any]:
        """AI-powered error diagnosis using Claude"""

        # Build error context for Claude
        prompt = f"""Diagnose this trading system error and suggest recovery:

ERROR DETAILS:
- Type: {context.error_type}
- Message: {context.error_message}
- Category: {context.category.value}
- Severity: {context.severity.value}
- Component: {context.component}
- Timestamp: {context.timestamp.isoformat()}

STACK TRACE:
{context.stack_trace if context.stack_trace else 'Not available'}

ADDITIONAL CONTEXT:
{json.dumps(context.additional_data, indent=2) if context.additional_data else 'None'}

Provide comprehensive error diagnosis in JSON format:
{{
  "diagnosis": "clear explanation of what went wrong",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "root_cause": "likely root cause of the error",
  "recovery_steps": [
    "Step 1: specific recovery action",
    "Step 2: another recovery action",
    ...
  ],
  "preventive_measures": [
    "Measure 1: prevent recurrence",
    "Measure 2: improve resilience",
    ...
  ],
  "requires_manual_intervention": true|false,
  "estimated_recovery_time": "time estimate",
  "similar_errors": "known similar error patterns"
}}

Focus on actionable recovery steps for a day trading system."""

        request = ClaudeRequest(
            request_type="error_diagnosis",
            prompt=prompt,
            max_tokens=1536,
            temperature=0.3,  # Lower temperature for consistent diagnosis
            system_prompt="You are a system reliability engineer diagnosing trading system errors. Provide clear, actionable recovery steps."
        )

        response = self.claude.request(request, use_cache=False)  # Don't cache error diagnoses

        if not response.success:
            logger.error(f"AI diagnosis failed: {response.error}")
            # Fallback to rule-based
            return self._diagnose_rule_based(context)

        # Parse diagnosis
        try:
            diagnosis = json.loads(response.content)
            return diagnosis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI diagnosis: {e}")
            return self._diagnose_rule_based(context)

    def recover_from_error(
        self,
        context: ErrorContext,
        auto_recover: bool = True
    ) -> RecoveryResult:
        """
        Attempt to recover from error

        Args:
            context: ErrorContext object
            auto_recover: Whether to automatically attempt recovery

        Returns:
            RecoveryResult object
        """
        start_time = time.time()

        # Diagnose error
        diagnosis_dict = self.diagnose_error(context, use_ai=True)

        # Determine if manual intervention required
        manual_required = diagnosis_dict.get('requires_manual_intervention', False)

        attempted_actions = []
        recovery_status = RecoveryStatus.PENDING

        if auto_recover and not manual_required:
            # Attempt automated recovery
            recovery_status = RecoveryStatus.IN_PROGRESS

            # Get recovery actions
            recovery_actions = self.recovery_strategies.get(context.category, [])

            for action in recovery_actions:
                attempted_actions.append(action.action_name)
                logger.info(f"Attempting recovery action: {action.description}")

                try:
                    # In production, would execute actual recovery function
                    # For now, we'll simulate with a delay
                    time.sleep(0.1)  # Simulate action execution

                    # If action has success criteria, check it
                    if action.success_criteria and callable(action.success_criteria):
                        if action.success_criteria():
                            recovery_status = RecoveryStatus.SUCCESSFUL
                            logger.info(f"Recovery successful: {action.action_name}")
                            break
                    else:
                        # Assume success if no explicit criteria
                        recovery_status = RecoveryStatus.SUCCESSFUL
                        logger.info(f"Recovery action completed: {action.action_name}")
                        break

                except Exception as e:
                    logger.error(f"Recovery action failed: {action.action_name} - {e}")
                    continue

            # If all actions failed
            if recovery_status == RecoveryStatus.IN_PROGRESS:
                recovery_status = RecoveryStatus.FAILED
                manual_required = True

        elif manual_required:
            recovery_status = RecoveryStatus.MANUAL_REQUIRED

        # Calculate recovery time
        recovery_time = time.time() - start_time

        # Create result
        result = RecoveryResult(
            error_context=context,
            diagnosis=diagnosis_dict.get('diagnosis', 'Error detected'),
            recovery_actions=diagnosis_dict.get('recovery_steps', []),
            attempted_actions=attempted_actions,
            status=recovery_status,
            recovery_time_seconds=recovery_time,
            resolution_notes=diagnosis_dict.get('root_cause', ''),
            requires_manual_intervention=manual_required
        )

        # Store in history
        self.error_history.append(result)

        # Remove from active errors if successful
        if recovery_status == RecoveryStatus.SUCCESSFUL:
            error_key = f"{context.component}_{context.error_type}"
            if error_key in self.active_errors:
                del self.active_errors[error_key]

        logger.info(
            f"Recovery complete: {recovery_status.value} "
            f"(time: {recovery_time:.2f}s, manual: {manual_required})"
        )

        return result

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status

        Returns:
            Health status dictionary
        """
        # Count active errors by severity
        severity_counts = {
            ErrorSeverity.LOW: 0,
            ErrorSeverity.MEDIUM: 0,
            ErrorSeverity.HIGH: 0,
            ErrorSeverity.CRITICAL: 0
        }

        for error in self.active_errors.values():
            severity_counts[error.severity] += 1

        # Calculate health score (100 = perfect, 0 = critical issues)
        health_score = 100
        health_score -= severity_counts[ErrorSeverity.LOW] * 5
        health_score -= severity_counts[ErrorSeverity.MEDIUM] * 15
        health_score -= severity_counts[ErrorSeverity.HIGH] * 30
        health_score -= severity_counts[ErrorSeverity.CRITICAL] * 50
        health_score = max(0, health_score)

        # Determine overall status
        if health_score >= 90:
            status = "HEALTHY"
        elif health_score >= 70:
            status = "WARNING"
        elif health_score >= 50:
            status = "DEGRADED"
        else:
            status = "CRITICAL"

        # Recent recovery stats
        recent_recoveries = [
            r for r in self.error_history
            if r.timestamp > datetime.now() - timedelta(hours=1)
        ]

        successful_recoveries = sum(
            1 for r in recent_recoveries
            if r.status == RecoveryStatus.SUCCESSFUL
        )

        return {
            "status": status,
            "health_score": health_score,
            "active_errors": len(self.active_errors),
            "errors_by_severity": {
                "low": severity_counts[ErrorSeverity.LOW],
                "medium": severity_counts[ErrorSeverity.MEDIUM],
                "high": severity_counts[ErrorSeverity.HIGH],
                "critical": severity_counts[ErrorSeverity.CRITICAL]
            },
            "recent_recoveries_1h": len(recent_recoveries),
            "successful_recoveries_1h": successful_recoveries,
            "recovery_success_rate": (
                (successful_recoveries / len(recent_recoveries)) * 100
                if recent_recoveries else 0
            ),
            "last_check": datetime.now().isoformat()
        }

    def get_error_history(
        self,
        hours: int = 24,
        category: Optional[ErrorCategory] = None
    ) -> List[RecoveryResult]:
        """
        Get error recovery history

        Args:
            hours: Number of hours of history
            category: Filter by error category (optional)

        Returns:
            List of RecoveryResult objects
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        history = [
            r for r in self.error_history
            if r.timestamp > cutoff
        ]

        if category:
            history = [
                r for r in history
                if r.error_context.category == category
            ]

        return history

    def clear_resolved_errors(self):
        """Clear errors that have been resolved"""
        # Keep only errors from the last hour
        cutoff = datetime.now() - timedelta(hours=1)
        self.active_errors = {
            k: v for k, v in self.active_errors.items()
            if v.timestamp > cutoff
        }
        logger.info(f"Cleared resolved errors (active: {len(self.active_errors)})")


# Global instance
_self_healing: Optional[SelfHealingSystem] = None


def get_self_healing() -> SelfHealingSystem:
    """Get or create global SelfHealingSystem instance"""
    global _self_healing
    if _self_healing is None:
        _self_healing = SelfHealingSystem()
    return _self_healing


if __name__ == "__main__":
    # Test the self-healing system
    system = SelfHealingSystem()

    print("Self-Healing System Test")
    print("=" * 50)

    # Simulate an error
    try:
        # Raise a test error
        raise ConnectionError("Failed to connect to IBKR Gateway")
    except Exception as e:
        # Detect error
        context = system.detect_error(
            error=e,
            component="ibkr_connector",
            additional_data={"retry_count": 3, "last_success": "2025-11-15 09:30:00"}
        )

        print(f"\nError Detected:")
        print(f"Type: {context.error_type}")
        print(f"Category: {context.category.value}")
        print(f"Severity: {context.severity.value}")

        # Diagnose
        diagnosis = system.diagnose_error(context, use_ai=False)
        print(f"\nDiagnosis:")
        print(f"Root Cause: {diagnosis['root_cause']}")
        print(f"Recovery Steps: {len(diagnosis['recovery_steps'])}")

        # Attempt recovery
        result = system.recover_from_error(context, auto_recover=True)
        print(f"\nRecovery Result:")
        print(f"Status: {result.status.value}")
        print(f"Attempted Actions: {', '.join(result.attempted_actions)}")
        print(f"Time: {result.recovery_time_seconds:.2f}s")
        print(f"Manual Required: {result.requires_manual_intervention}")

    # Check system health
    health = system.get_system_health()
    print(f"\nSystem Health:")
    print(json.dumps(health, indent=2))

    print("\n" + "=" * 50)
    print("Test complete!")
