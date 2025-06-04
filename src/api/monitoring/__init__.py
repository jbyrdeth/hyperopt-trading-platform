"""
Monitoring and Observability Module

This module provides comprehensive monitoring capabilities including:
- Prometheus metrics collection
- Custom API instrumentation
- Performance monitoring
- Health checks
- Alerting integration
- Health monitoring automation
- Structured logging and log aggregation
"""

from .metrics import (
    PrometheusMetrics,
    MetricsCollector,
    MetricsMiddleware,
    get_metrics_collector
)

from .health import (
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    get_health_checker
)

from .alerts import (
    AlertManager,
    get_alert_manager
)

from .health_automation import (
    HealthAutomation,
    get_health_automation
)

from .logging import (
    LogAggregator,
    LoggingMiddleware,
    LogLevel,
    LogCategory,
    LogContext,
    StructuredJsonFormatter,
    get_log_aggregator,
    configure_logging,
    log_context,
    log_with_context
)

__all__ = [
    "PrometheusMetrics",
    "MetricsCollector", 
    "MetricsMiddleware",
    "get_metrics_collector",
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "get_health_checker",
    "AlertManager",
    "get_alert_manager",
    "HealthAutomation",
    "get_health_automation",
    "LogAggregator",
    "LoggingMiddleware",
    "LogLevel",
    "LogCategory",
    "LogContext",
    "StructuredJsonFormatter",
    "get_log_aggregator",
    "configure_logging",
    "log_context",
    "log_with_context"
] 