"""
Advanced alerting system for LlamaAgent.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Optional imports
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from slack_sdk.webhook import WebhookClient

    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""

    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class Alert:
    """Alert definition"""

    id: str
    name: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class AlertRule:
    """Alert rule definition"""

    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    message_template: str
    channels: List[AlertChannel]
    cooldown_minutes: int = 5
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class ChannelConfig:
    """Configuration for alert channels"""

    # Email config
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)
    # Slack config
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None

    # Webhook config
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    # PagerDuty config
    pagerduty_routing_key: Optional[str] = None
    pagerduty_service_key: Optional[str] = None


class AlertManager:
    """Advanced alert management system"""

    def __init__(
        self,
        channel_config: Optional[ChannelConfig] = None,
        rate_limit_per_hour: int = 100,
        aggregate_window_minutes: int = 5,
    ) -> None:
        self.channel_config = channel_config or ChannelConfig()
        self.rate_limit_per_hour = rate_limit_per_hour
        self.aggregate_window_minutes = aggregate_window_minutes

        # Alert storage
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}

        # Rate limiting
        self.sent_alerts: List[datetime] = []

        # Alert aggregation
        self.pending_alerts: Dict[str, List[Alert]] = {}
        self.cooldown_until: Dict[str, datetime] = {}

        # Handlers
        self.channel_handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.LOG: self._send_log_alert,
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SLACK: self._send_slack_alert,
            AlertChannel.WEBHOOK: self._send_webhook_alert,
            AlertChannel.PAGERDUTY: self._send_pagerduty_alert,
        }

        # Start background aggregation task
        self._aggregation_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Start the alert manager"""
        if self._aggregation_task is None:
            self._aggregation_task = asyncio.create_task(self._aggregation_loop())

    def stop(self) -> None:
        """Stop the alert manager"""
        if self._aggregation_task:
            self._aggregation_task.cancel()
            self._aggregation_task = None

    def add_rule(self, rule_name: str, rule: AlertRule) -> None:
        """Add an alert rule"""
        self.alert_rules[rule_name] = rule

    def remove_rule(self, rule_name: str) -> None:
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]

    async def trigger_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Trigger an alert"""
        # Create alert
        alert = Alert(
            id=f"{name}_{datetime.now().timestamp()}",
            name=name,
            severity=severity,
            message=message,
            details=details or {},
            tags=tags or [],
        )

        # For critical alerts, send immediately
        if severity == AlertSeverity.CRITICAL:
            channels = self._get_channels_for_severity(severity)
            await self._send_alert_to_channels(alert, channels)
            return

        # Store alert
        self.alerts.append(alert)
        # Get channels based on severity
        channels = self._get_channels_for_severity(alert.severity)
        # Add to pending alerts for aggregation
        key = f"{alert.name}:{alert.severity.value}"
        if key not in self.pending_alerts:
            self.pending_alerts[key] = []
        self.pending_alerts[key].append(alert)

    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Get appropriate channels for alert severity"""
        if severity == AlertSeverity.CRITICAL:
            return [
                AlertChannel.LOG,
                AlertChannel.EMAIL,
                AlertChannel.SLACK,
                AlertChannel.PAGERDUTY,
            ]
        elif severity == AlertSeverity.ERROR:
            return [AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.SLACK]
        elif severity == AlertSeverity.WARNING:
            return [AlertChannel.LOG, AlertChannel.SLACK]
        else:
            return [AlertChannel.LOG]

    async def _aggregation_loop(self) -> None:
        """Background aggregation loop"""
        while True:
            try:
                await asyncio.sleep(self.aggregate_window_minutes * 60)
                await self._process_pending_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")

    async def _process_pending_alerts(self) -> None:
        """Process pending alerts for aggregation"""
        if not self.pending_alerts:
            return

        for key, alerts in self.pending_alerts.items():
            if not alerts:
                continue

            # Skip critical alerts (already sent)
            if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
                continue

            if len(alerts) == 1:
                # Single alert
                alert = alerts[0]
                channels = self._get_channels_for_severity(alert.severity)
                await self._send_alert_to_channels(alert, channels)
            else:
                # Aggregate multiple alerts
                aggregated = self._aggregate_alerts(alerts)
                channels = self._get_channels_for_severity(aggregated.severity)
                await self._send_alert_to_channels(aggregated, channels)
        # Clear pending alerts
        self.pending_alerts.clear()

    def _aggregate_alerts(self, alerts: List[Alert]) -> Alert:
        """Aggregate multiple alerts into one"""
        # Use highest severity
        max_severity = max(
            alerts, key=lambda a: list(AlertSeverity).index(a.severity)
        ).severity

        # Combine messages
        messages = [a.message for a in alerts]
        unique_messages = list(
            dict.fromkeys(messages)
        )  # Preserve order, remove duplicates

        combined_message = f"Aggregated {len(alerts)} alerts:\n" + "\n".join(
            f"- {msg}" for msg in unique_messages[:5]
        )
        if len(unique_messages) > 5:
            combined_message += f"\n... and {len(unique_messages) - 5} more"

        # Combine tags
        all_tags = set()
        for alert in alerts:
            all_tags.update(alert.tags)
        return Alert(
            id=f"aggregated_{datetime.now().timestamp()}",
            name=f"Aggregated: {alerts[0].name}",
            severity=max_severity,
            message=combined_message,
            details={
                "aggregated_count": len(alerts),
                "original_alerts": [a.id for a in alerts],
            },
            tags=list(all_tags),
        )

    async def _send_alert_to_channels(
        self, alert: Alert, channels: List[AlertChannel]
    ) -> None:
        """Send alert to specified channels"""
        tasks: List[asyncio.Task] = []
        for channel in channels:
            if channel in self.channel_handlers:
                handler = self.channel_handlers[channel]
                tasks.append(asyncio.create_task(handler(alert)))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_log_alert(self, alert: Alert) -> None:
        """Send alert to logs"""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.info)
        log_method(
            f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}"
        )

    async def _send_email_alert(self, alert: Alert) -> None:
        """Send alert via email"""
        if not all(
            [
                self.channel_config.smtp_host,
                self.channel_config.smtp_username,
                self.channel_config.smtp_password,
                self.channel_config.email_from,
                self.channel_config.email_to,
            ]
        ):
            logger.warning("Email not configured")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.channel_config.email_from
            msg['To'] = ", ".join(self.channel_config.email_to)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"

            # Email body
            body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Time: {alert.timestamp.isoformat()}
Message: {alert.message}

Details:
{json.dumps(alert.details, indent=2)}

Tags: {', '.join(alert.tags)}
"""

            msg.attach(MIMEText(body, 'plain'))
            # Send email
            server = smtplib.SMTP(
                self.channel_config.smtp_host, self.channel_config.smtp_port
            )
            server.starttls()
            server.login(
                self.channel_config.smtp_username, self.channel_config.smtp_password
            )
            server.send_message(msg)
            server.quit()

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_slack_alert(self, alert: Alert) -> None:
        """Send alert to Slack"""
        if not SLACK_AVAILABLE or not self.channel_config.slack_webhook_url:
            logger.warning("Slack not configured")
            return

        try:
            webhook = WebhookClient(self.channel_config.slack_webhook_url)
            # Color based on severity
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "danger",
                AlertSeverity.CRITICAL: "danger",
            }.get(alert.severity, "good")
            payload = {
                "text": f"Alert: {alert.name}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True,
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.isoformat(),
                                "short": True,
                            },
                            {
                                "title": "Message",
                                "value": alert.message,
                                "short": False,
                            },
                        ],
                    }
                ],
            }

            if alert.tags:
                payload["attachments"][0]["fields"].append(
                    {"title": "Tags", "value": ", ".join(alert.tags), "short": False}
                )

            webhook.send(**payload)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_webhook_alert(self, alert: Alert) -> None:
        """Send alert to webhook"""
        if not AIOHTTP_AVAILABLE or not self.channel_config.webhook_url:
            logger.warning("Webhook not configured")
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.channel_config.webhook_url,
                    json=alert.to_dict(),
                    headers=self.channel_config.webhook_headers,
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def _send_pagerduty_alert(self, alert: Alert) -> None:
        """Send alert to PagerDuty"""
        if not all([AIOHTTP_AVAILABLE, self.channel_config.pagerduty_routing_key]):
            logger.warning("PagerDuty not configured")
            return

        try:
            payload = {
                "routing_key": self.channel_config.pagerduty_routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"{alert.name}: {alert.message}",
                    "severity": alert.severity.value,
                    "source": "llamaagent",
                    "custom_details": alert.details,
                },
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue", json=payload
                ) as response:
                    if response.status >= 400:
                        logger.error(f"PagerDuty alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")

    async def check_conditions(self, metrics: Dict[str, Any]) -> None:
        """Check alert conditions against metrics"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule_name in self.cooldown_until:
                if datetime.now(timezone.utc) < self.cooldown_until[rule_name]:
                    continue

            # Check condition
            try:
                if rule.condition(metrics):
                    # Create alert
                    alert = Alert(
                        id=f"{rule_name}_{datetime.now().timestamp()}",
                        name=rule_name,
                        severity=rule.severity,
                        message=rule.message_template.format(**metrics),
                        details=metrics,
                        tags=rule.tags,
                    )

                    await self.trigger_alert(
                        alert.name,
                        alert.severity,
                        alert.message,
                        alert.details,
                        alert.tags,
                    )
                    # Set cooldown
                    self.cooldown_until[rule_name] = datetime.now(
                        timezone.utc
                    ) + timedelta(minutes=rule.cooldown_minutes)
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        # Clean old entries
        self.sent_alerts = [t for t in self.sent_alerts if t > hour_ago]

        # Check limit
        if len(self.sent_alerts) >= self.rate_limit_per_hour:
            return False

        self.sent_alerts.append(now)
        return True

    def get_alert_history(self) -> List[Alert]:
        """Get alert history"""
        return self.alerts.copy()


# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# Convenience functions
async def send_alert(
    name: str,
    severity: AlertSeverity,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> None:
    """Send an alert using the global alert manager"""
    manager = get_alert_manager()
    await manager.trigger_alert(name, severity, message, details, tags)
