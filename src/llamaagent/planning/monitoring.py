"""
Monitoring and Performance Tracking for Task Planning and Execution

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for task execution."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_duration: float = 0.0
    success_rate: float = 0.0
    throughput: float = 0.0  # tasks per hour
    peak_concurrent_tasks: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionReport:
    """Comprehensive execution report."""

    plan_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    task_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    task_details: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "plan_id": self.plan_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": (
                self.total_duration.total_seconds() if self.total_duration else None
            ),
            "task_count": self.task_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": (
                self.success_count / self.task_count if self.task_count > 0 else 0.0
            ),
            "task_details": self.task_details,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class AlertManager:
    """Manages alerts and notifications for execution issues."""

    def __init__(self):
        self.alert_handlers: List[Callable] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {
            "high_failure_rate": {"threshold": 0.3, "enabled": True},
            "long_execution_time": {"threshold": 3600, "enabled": True},  # 1 hour
            "memory_threshold": {"threshold": 1000, "enabled": True},  # 1GB
            "task_timeout": {"threshold": 1800, "enabled": True},  # 30 minutes
        }
        self.active_alerts: Dict[str, Dict[str, Any]] = {}

    def add_alert_handler(self, handler: Callable[[str, Dict[str, Any]], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def check_failure_rate(self, success_rate: float, plan_id: str):
        """Check if failure rate exceeds threshold."""
        rule = self.alert_rules.get("high_failure_rate", {})
        if rule.get("enabled") and success_rate < (1 - rule.get("threshold", 0.3)):
            self._trigger_alert(
                "high_failure_rate",
                {
                    "plan_id": plan_id,
                    "success_rate": success_rate,
                    "threshold": rule["threshold"],
                    "message": f"High failure rate detected: {(1 - success_rate) * 100:.1f}%",
                },
            )

    def check_execution_time(self, duration: timedelta, plan_id: str):
        """Check if execution time exceeds threshold."""
        rule = self.alert_rules.get("long_execution_time", {})
        if rule.get("enabled") and duration.total_seconds() > rule.get(
            "threshold", 3600
        ):
            self._trigger_alert(
                "long_execution_time",
                {
                    "plan_id": plan_id,
                    "duration": duration.total_seconds(),
                    "threshold": rule["threshold"],
                    "message": f"Long execution time: {duration}",
                },
            )

    def check_memory_usage(self, memory_mb: float, plan_id: str):
        """Check if memory usage exceeds threshold."""
        rule = self.alert_rules.get("memory_threshold", {})
        if rule.get("enabled") and memory_mb > rule.get("threshold", 1000):
            self._trigger_alert(
                "memory_threshold",
                {
                    "plan_id": plan_id,
                    "memory_mb": memory_mb,
                    "threshold": rule["threshold"],
                    "message": f"High memory usage: {memory_mb:.1f}MB",
                },
            )

    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger an alert."""
        alert_id = f"{alert_type}_{alert_data.get('plan_id', 'unknown')}_{datetime.now().isoformat()}"

        self.active_alerts[alert_id] = {
            "type": alert_type,
            "data": alert_data,
            "timestamp": datetime.now(),
            "acknowledged": False,
        }

        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["acknowledged"] = True

    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """Get all active alerts."""
        return self.active_alerts.copy()


class CheckpointManager:
    """Manages checkpoints for plan execution state."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, plan_id: str, execution_state: Dict[str, Any]) -> str:
        """Save execution state checkpoint."""
        checkpoint_data = {
            "plan_id": plan_id,
            "timestamp": datetime.now().isoformat(),
            "execution_state": execution_state,
        }

        checkpoint_file = (
            self.checkpoint_dir
            / f"checkpoint_{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )

        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)

            logger.info(f"Checkpoint saved: {checkpoint_file}")
            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load execution state from checkpoint."""
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)

            logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return checkpoint_data

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    def list_checkpoints(self, plan_id: Optional[str] = None) -> List[str]:
        """List available checkpoints."""
        pattern = f"checkpoint_{plan_id}_*.pkl" if plan_id else "checkpoint_*.pkl"
        return [str(f) for f in self.checkpoint_dir.glob(pattern)]

    def cleanup_old_checkpoints(self, max_age_days: int = 7):
        """Clean up old checkpoint files."""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            if checkpoint_file.stat().st_mtime < cutoff_time.timestamp():
                try:
                    checkpoint_file.unlink()
                    logger.info(f"Deleted old checkpoint: {checkpoint_file}")
                except Exception as e:
                    logger.error(f"Failed to delete checkpoint {checkpoint_file}: {e}")


class ExecutionMonitor:
    """Comprehensive monitoring for task execution."""

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self.metrics = PerformanceMetrics()
        self.alert_manager = AlertManager()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.execution_history: List[ExecutionReport] = []
        self.active_executions: Dict[str, ExecutionReport] = {}

        # Set up default alert handler
        self.alert_manager.add_alert_handler(self._default_alert_handler)

    def start_execution_monitoring(self, plan_id: str) -> ExecutionReport:
        """Start monitoring a plan execution."""
        report = ExecutionReport(plan_id=plan_id, start_time=datetime.now())

        self.active_executions[plan_id] = report
        logger.info(f"Started monitoring execution for plan: {plan_id}")

        return report

    def update_task_metrics(self, plan_id: str, task_result: Dict[str, Any]):
        """Update metrics with task execution result."""
        if plan_id not in self.active_executions:
            return

        report = self.active_executions[plan_id]

        # Update task details
        report.task_details.append(
            {
                "task_id": task_result.get("task_id"),
                "success": task_result.get("success", False),
                "duration": task_result.get("duration", 0),
                "error": task_result.get("error"),
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Update counters
        report.task_count += 1
        if task_result.get("success", False):
            report.success_count += 1
        else:
            report.failure_count += 1
            if task_result.get("error"):
                report.errors.append(task_result["error"])

        # Update metrics
        self._update_performance_metrics(report)

        # Check for alerts
        success_rate = (
            report.success_count / report.task_count if report.task_count > 0 else 1.0
        )
        self.alert_manager.check_failure_rate(success_rate, plan_id)

    def finish_execution_monitoring(self, plan_id: str) -> ExecutionReport:
        """Finish monitoring and generate final report."""
        if plan_id not in self.active_executions:
            raise ValueError(f"No active monitoring for plan: {plan_id}")

        report = self.active_executions[plan_id]
        report.end_time = datetime.now()
        report.total_duration = report.end_time - report.start_time

        # Final metric updates
        self._update_performance_metrics(report)

        # Check execution time alert
        if report.total_duration:
            self.alert_manager.check_execution_time(report.total_duration, plan_id)

        # Move to history
        self.execution_history.append(report)
        del self.active_executions[plan_id]

        logger.info(f"Finished monitoring execution for plan: {plan_id}")
        return report

    def _update_performance_metrics(self, report: ExecutionReport):
        """Update performance metrics based on execution report."""
        if report.task_count == 0:
            return

        # Calculate averages
        total_duration = sum(task.get("duration", 0) for task in report.task_details)

        report.metrics.total_tasks = report.task_count
        report.metrics.completed_tasks = report.success_count
        report.metrics.failed_tasks = report.failure_count
        report.metrics.success_rate = report.success_count / report.task_count
        report.metrics.average_duration = total_duration / report.task_count

        # Calculate throughput (tasks per hour)
        if report.end_time:
            duration_hours = report.total_duration.total_seconds() / 3600
            report.metrics.throughput = (
                report.task_count / duration_hours if duration_hours > 0 else 0
            )

        report.metrics.last_updated = datetime.now()

    def save_execution_state(self, plan_id: str) -> str:
        """Save current execution state as checkpoint."""
        if plan_id not in self.active_executions:
            raise ValueError(f"No active execution for plan: {plan_id}")

        execution_state = {
            "report": self.active_executions[plan_id].to_dict(),
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "success_rate": self.metrics.success_rate,
            },
        }

        return self.checkpoint_manager.save_checkpoint(plan_id, execution_state)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate overall summary report."""
        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {"message": "No executions to report"}

        # Aggregate metrics
        total_tasks = sum(report.task_count for report in self.execution_history)
        total_success = sum(report.success_count for report in self.execution_history)
        total_failures = sum(report.failure_count for report in self.execution_history)

        avg_duration = (
            sum(
                report.total_duration.total_seconds() if report.total_duration else 0
                for report in self.execution_history
            )
            / total_executions
        )

        return {
            "total_executions": total_executions,
            "total_tasks": total_tasks,
            "overall_success_rate": (
                total_success / total_tasks if total_tasks > 0 else 0
            ),
            "total_failures": total_failures,
            "average_execution_duration": avg_duration,
            "active_executions": len(self.active_executions),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "last_execution": (
                self.execution_history[-1].to_dict() if self.execution_history else None
            ),
        }

    def _default_alert_handler(self, alert_type: str, alert_data: Dict[str, Any]):
        """Default alert handler that logs alerts."""
        logger.warning(
            f"ALERT [{alert_type}]: {alert_data.get('message', 'No message')}"
        )

        # Could extend to send emails, Slack notifications, etc.

    def export_metrics(self, output_file: str):
        """Export metrics to JSON file."""
        try:
            export_data = {
                "summary": self.generate_summary_report(),
                "execution_history": [
                    report.to_dict() for report in self.execution_history
                ],
                "active_alerts": self.alert_manager.get_active_alerts(),
                "export_timestamp": datetime.now().isoformat(),
            }

            with open(output_file, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Metrics exported to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
