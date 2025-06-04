"""
Alert Handling Module for Trading Strategy Optimization API

This module provides webhook endpoints for receiving alerts from monitoring systems
and implements alert processing, notification routing, and alert acknowledgment.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Create alerts router
alerts_router = APIRouter()


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class Alert:
    """Alert data structure."""
    alertname: str
    severity: AlertSeverity
    status: AlertStatus
    summary: str
    description: str
    service: str
    category: str
    timestamp: datetime
    runbook_url: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AlertManager:
    """
    Manages alert processing, storage, and notification routing.
    """
    
    def __init__(self):
        # In-memory storage for alerts (in production, use a database)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # Alert statistics
        self.stats = {
            "total_alerts": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "info_alerts": 0,
            "resolved_alerts": 0,
            "acknowledged_alerts": 0
        }
        
        logger.info("AlertManager initialized")
    
    def process_alertmanager_webhook(self, payload: Dict[str, Any]) -> List[Alert]:
        """Process webhook payload from Alertmanager."""
        
        processed_alerts = []
        alerts_data = payload.get('alerts', [])
        
        for alert_data in alerts_data:
            try:
                alert = self._parse_alertmanager_alert(alert_data)
                processed_alert = self._process_alert(alert)
                processed_alerts.append(processed_alert)
                
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
                continue
        
        return processed_alerts
    
    def process_grafana_webhook(self, payload: Dict[str, Any]) -> List[Alert]:
        """Process webhook payload from Grafana."""
        
        processed_alerts = []
        
        try:
            alert = self._parse_grafana_alert(payload)
            processed_alert = self._process_alert(alert)
            processed_alerts.append(processed_alert)
            
        except Exception as e:
            logger.error(f"Error processing Grafana alert: {e}")
        
        return processed_alerts
    
    def _parse_alertmanager_alert(self, alert_data: Dict[str, Any]) -> Alert:
        """Parse Alertmanager alert format."""
        
        labels = alert_data.get('labels', {})
        annotations = alert_data.get('annotations', {})
        
        # Determine status
        status = AlertStatus.RESOLVED if alert_data.get('status') == 'resolved' else AlertStatus.FIRING
        
        return Alert(
            alertname=labels.get('alertname', 'Unknown'),
            severity=AlertSeverity(labels.get('severity', 'info')),
            status=status,
            summary=annotations.get('summary', 'No summary'),
            description=annotations.get('description', 'No description'),
            service=labels.get('service', 'unknown'),
            category=labels.get('category', 'general'),
            timestamp=datetime.fromisoformat(alert_data.get('startsAt', datetime.now().isoformat()).replace('Z', '+00:00')),
            runbook_url=annotations.get('runbook_url'),
            labels=labels,
            annotations=annotations
        )
    
    def _parse_grafana_alert(self, payload: Dict[str, Any]) -> Alert:
        """Parse Grafana alert format."""
        
        # Grafana sends different format
        state = payload.get('state', 'alerting')
        status = AlertStatus.RESOLVED if state == 'ok' else AlertStatus.FIRING
        
        # Extract severity from tags or rule name
        tags = payload.get('tags', {})
        severity = AlertSeverity(tags.get('severity', 'warning'))
        
        return Alert(
            alertname=payload.get('ruleName', 'Grafana Alert'),
            severity=severity,
            status=status,
            summary=payload.get('title', 'Grafana Alert'),
            description=payload.get('message', 'No description'),
            service=tags.get('service', 'trading-api'),
            category=tags.get('category', 'grafana'),
            timestamp=datetime.now(),
            runbook_url=payload.get('ruleUrl'),
            labels=tags,
            annotations={}
        )
    
    def _process_alert(self, alert: Alert) -> Alert:
        """Process and store alert."""
        
        alert_key = f"{alert.alertname}:{alert.service}"
        
        # Update statistics
        self.stats["total_alerts"] += 1
        if alert.severity == AlertSeverity.CRITICAL:
            self.stats["critical_alerts"] += 1
        elif alert.severity == AlertSeverity.WARNING:
            self.stats["warning_alerts"] += 1
        else:
            self.stats["info_alerts"] += 1
        
        if alert.status == AlertStatus.RESOLVED:
            self.stats["resolved_alerts"] += 1
            # Remove from active alerts
            self.active_alerts.pop(alert_key, None)
        else:
            # Add/update active alert
            self.active_alerts[alert_key] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        logger.info(f"Processed alert: {alert.alertname} ({alert.severity.value}) - {alert.status.value}")
        
        return alert
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        
        return {
            **self.stats,
            "active_alerts": len(self.active_alerts),
            "critical_active": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            "warning_active": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.WARNING]),
            "info_active": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.INFO])
        }
    
    def acknowledge_alert(self, alertname: str, service: str) -> bool:
        """Acknowledge an active alert."""
        
        alert_key = f"{alertname}:{service}"
        
        if alert_key in self.active_alerts:
            self.active_alerts[alert_key].status = AlertStatus.ACKNOWLEDGED
            self.stats["acknowledged_alerts"] += 1
            logger.info(f"Alert acknowledged: {alertname} for {service}")
            return True
        
        return False


# Global alert manager instance
alert_manager = AlertManager()


@alerts_router.post("/webhook/alertmanager")
async def alertmanager_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook endpoint for Alertmanager notifications.
    """
    try:
        payload = await request.json()
        
        # Process alerts in background
        background_tasks.add_task(process_alertmanager_alerts, payload)
        
        return JSONResponse(
            status_code=200,
            content={"message": "Alertmanager webhook received", "alerts_count": len(payload.get('alerts', []))}
        )
        
    except Exception as e:
        logger.error(f"Alertmanager webhook error: {e}")
        raise HTTPException(status_code=400, detail=f"Webhook processing error: {str(e)}")


@alerts_router.post("/webhook/grafana")
async def grafana_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Webhook endpoint for Grafana notifications.
    """
    try:
        payload = await request.json()
        
        # Process alert in background
        background_tasks.add_task(process_grafana_alert, payload)
        
        return JSONResponse(
            status_code=200,
            content={"message": "Grafana webhook received", "alert": payload.get('title', 'Unknown')}
        )
        
    except Exception as e:
        logger.error(f"Grafana webhook error: {e}")
        raise HTTPException(status_code=400, detail=f"Webhook processing error: {str(e)}")


@alerts_router.get("/alerts/active")
async def get_active_alerts(severity: Optional[str] = None):
    """
    Get currently active alerts.
    """
    try:
        severity_filter = AlertSeverity(severity) if severity else None
        alerts = alert_manager.get_active_alerts(severity_filter)
        
        return JSONResponse(
            content={
                "active_alerts": [alert.to_dict() for alert in alerts],
                "count": len(alerts),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving alerts")


@alerts_router.get("/alerts/stats")
async def get_alert_stats():
    """
    Get alert statistics.
    """
    try:
        stats = alert_manager.get_alert_stats()
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting alert stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving alert statistics")


@alerts_router.post("/alerts/{alertname}/acknowledge")
async def acknowledge_alert(alertname: str, service: str = "trading-api"):
    """
    Acknowledge an active alert.
    """
    try:
        success = alert_manager.acknowledge_alert(alertname, service)
        
        if success:
            return JSONResponse(
                content={"message": f"Alert {alertname} acknowledged", "alertname": alertname, "service": service}
            )
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alertname} not found or already resolved")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail="Error acknowledging alert")


async def process_alertmanager_alerts(payload: Dict[str, Any]):
    """Background task to process Alertmanager alerts."""
    try:
        alerts = alert_manager.process_alertmanager_webhook(payload)
        logger.info(f"Processed {len(alerts)} alerts from Alertmanager")
        
        # Here you could add additional processing like:
        # - Sending custom notifications
        # - Integrating with ticketing systems
        # - Triggering automated remediation
        
    except Exception as e:
        logger.error(f"Error in background alert processing: {e}")


async def process_grafana_alert(payload: Dict[str, Any]):
    """Background task to process Grafana alert."""
    try:
        alerts = alert_manager.process_grafana_webhook(payload)
        logger.info(f"Processed Grafana alert: {payload.get('title', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"Error in background Grafana alert processing: {e}")


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    return alert_manager 