"""
Structured Logging System for Trading Strategy Optimization API

This module provides comprehensive logging capabilities including:
- Structured JSON logging
- Log aggregation and analysis
- Log correlation with metrics
- Performance and security logging
- Error tracking and alerting
"""

import logging
import logging.config
import json
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import os
import sys
from contextlib import contextmanager

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import psutil


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log category enumeration for structured classification."""
    API = "api"
    HEALTH = "health"
    METRICS = "metrics"
    ALERTS = "alerts"
    OPTIMIZATION = "optimization"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    DATABASE = "database"
    EXTERNAL = "external"


@dataclass
class LogContext:
    """Context information for structured logging."""
    timestamp: str
    level: str
    category: str
    component: str
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    # Performance metrics
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    
    # API specific
    method: Optional[str] = None
    endpoint: Optional[str] = None
    status_code: Optional[int] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Business context
    strategy_name: Optional[str] = None
    job_id: Optional[str] = None
    symbol: Optional[str] = None
    
    # Error context
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None


class StructuredJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add custom context if present
        if hasattr(record, 'context') and record.context:
            context_dict = asdict(record.context) if isinstance(record.context, LogContext) else record.context
            log_data["context"] = context_dict
            
            # Extract key context fields to top level for easier querying
            if isinstance(record.context, LogContext):
                if record.context.category:
                    log_data["category"] = record.context.category
                if record.context.component:
                    log_data["component"] = record.context.component
                if record.context.request_id:
                    log_data["request_id"] = record.context.request_id
                if record.context.trace_id:
                    log_data["trace_id"] = record.context.trace_id
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_') and key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'message', 'context'
            ]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class LogAggregator:
    """
    Log aggregation and analysis system.
    
    Collects logs from various sources and provides analysis capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logs_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = self.config.get("max_buffer_size", 1000)
        self.flush_interval = self.config.get("flush_interval", 60)  # seconds
        self.analysis_patterns = self._load_analysis_patterns()
        
        # Log statistics
        self.stats = {
            "total_logs": 0,
            "logs_by_level": {},
            "logs_by_category": {},
            "logs_by_component": {},
            "error_patterns": {},
            "performance_metrics": []
        }
        
    def _load_analysis_patterns(self) -> Dict[str, Any]:
        """Load log analysis patterns for anomaly detection."""
        return {
            "error_patterns": [
                {"pattern": r".*timeout.*", "severity": "warning", "category": "performance"},
                {"pattern": r".*connection.*failed.*", "severity": "critical", "category": "connectivity"},
                {"pattern": r".*authentication.*failed.*", "severity": "critical", "category": "security"},
                {"pattern": r".*optimization.*failed.*", "severity": "warning", "category": "business"},
                {"pattern": r".*memory.*error.*", "severity": "critical", "category": "system"},
                {"pattern": r".*disk.*full.*", "severity": "critical", "category": "system"}
            ],
            "performance_thresholds": {
                "slow_request": 5000,  # ms
                "high_memory": 1000,   # MB
                "high_cpu": 80         # percent
            },
            "security_patterns": [
                {"pattern": r".*unauthorized.*", "severity": "warning"},
                {"pattern": r".*forbidden.*", "severity": "warning"},
                {"pattern": r".*sql.*injection.*", "severity": "critical"},
                {"pattern": r".*xss.*", "severity": "critical"}
            ]
        }
    
    def add_log(self, log_data: Dict[str, Any]):
        """Add a log entry to the aggregation buffer."""
        self.logs_buffer.append(log_data)
        self._update_stats(log_data)
        
        # Analyze log for patterns
        self._analyze_log(log_data)
        
        # Flush buffer if needed
        if len(self.logs_buffer) >= self.max_buffer_size:
            self.flush_logs()
    
    def _update_stats(self, log_data: Dict[str, Any]):
        """Update log statistics."""
        self.stats["total_logs"] += 1
        
        # Count by level
        level = log_data.get("level", "UNKNOWN")
        self.stats["logs_by_level"][level] = self.stats["logs_by_level"].get(level, 0) + 1
        
        # Count by category
        category = log_data.get("category", "unknown")
        self.stats["logs_by_category"][category] = self.stats["logs_by_category"].get(category, 0) + 1
        
        # Count by component
        component = log_data.get("component", "unknown")
        self.stats["logs_by_component"][component] = self.stats["logs_by_component"].get(component, 0) + 1
        
        # Track performance metrics
        context = log_data.get("context", {})
        if isinstance(context, dict):
            duration_ms = context.get("duration_ms")
            memory_mb = context.get("memory_mb")
            cpu_percent = context.get("cpu_percent")
            
            if any([duration_ms, memory_mb, cpu_percent]):
                self.stats["performance_metrics"].append({
                    "timestamp": log_data.get("timestamp"),
                    "duration_ms": duration_ms,
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent
                })
                
                # Keep only recent performance metrics
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                cutoff_str = cutoff_time.isoformat() + "Z"
                self.stats["performance_metrics"] = [
                    m for m in self.stats["performance_metrics"]
                    if m["timestamp"] and m["timestamp"] > cutoff_str
                ]
    
    def _analyze_log(self, log_data: Dict[str, Any]):
        """Analyze log entry for patterns and anomalies."""
        import re
        
        message = log_data.get("message", "").lower()
        level = log_data.get("level", "")
        
        # Check error patterns
        if level in ["ERROR", "CRITICAL"]:
            for pattern_config in self.analysis_patterns["error_patterns"]:
                pattern = pattern_config["pattern"]
                if re.search(pattern, message, re.IGNORECASE):
                    pattern_key = f"{pattern_config['category']}_{pattern_config['severity']}"
                    self.stats["error_patterns"][pattern_key] = self.stats["error_patterns"].get(pattern_key, 0) + 1
        
        # Check security patterns
        for pattern_config in self.analysis_patterns["security_patterns"]:
            pattern = pattern_config["pattern"]
            if re.search(pattern, message, re.IGNORECASE):
                # Security patterns should trigger immediate attention
                print(f"SECURITY ALERT: {pattern_config['severity']} - {message}")
        
        # Check performance thresholds
        context = log_data.get("context", {})
        if isinstance(context, dict):
            thresholds = self.analysis_patterns["performance_thresholds"]
            
            duration_ms = context.get("duration_ms")
            if duration_ms and duration_ms > thresholds["slow_request"]:
                print(f"PERFORMANCE ALERT: Slow request detected - {duration_ms}ms")
            
            memory_mb = context.get("memory_mb")
            if memory_mb and memory_mb > thresholds["high_memory"]:
                print(f"PERFORMANCE ALERT: High memory usage - {memory_mb}MB")
            
            cpu_percent = context.get("cpu_percent")
            if cpu_percent and cpu_percent > thresholds["high_cpu"]:
                print(f"PERFORMANCE ALERT: High CPU usage - {cpu_percent}%")
    
    def flush_logs(self):
        """Flush logs to storage/external systems."""
        if not self.logs_buffer:
            return
        
        # In a real implementation, this would send to Loki, ELK stack, etc.
        # For now, we'll write to a file
        log_file = Path("logs/aggregated.jsonl")
        log_file.parent.mkdir(exist_ok=True)
        
        with log_file.open("a") as f:
            for log_entry in self.logs_buffer:
                f.write(json.dumps(log_entry, default=str) + "\n")
        
        print(f"Flushed {len(self.logs_buffer)} logs to {log_file}")
        self.logs_buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current log statistics."""
        return self.stats.copy()
    
    def get_recent_logs(self, limit: int = 100, level: Optional[str] = None, 
                       category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent logs with optional filtering."""
        logs = self.logs_buffer.copy()
        
        # Apply filters
        if level:
            logs = [log for log in logs if log.get("level") == level]
        
        if category:
            logs = [log for log in logs if log.get("category") == category]
        
        # Return most recent logs
        return logs[-limit:] if logs else []
    
    def analyze_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze log patterns over a time window."""
        # This would typically analyze stored logs
        # For now, return current pattern analysis
        return {
            "time_window_hours": time_window_hours,
            "error_patterns": self.stats["error_patterns"],
            "performance_alerts": len([
                m for m in self.stats["performance_metrics"]
                if m.get("duration_ms", 0) > self.analysis_patterns["performance_thresholds"]["slow_request"]
            ]),
            "total_errors": sum(
                count for level, count in self.stats["logs_by_level"].items()
                if level in ["ERROR", "CRITICAL"]
            ),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on log analysis."""
        recommendations = []
        
        # Check error rates
        total_logs = self.stats["total_logs"]
        error_count = sum(
            count for level, count in self.stats["logs_by_level"].items()
            if level in ["ERROR", "CRITICAL"]
        )
        
        if total_logs > 0:
            error_rate = (error_count / total_logs) * 100
            if error_rate > 5:
                recommendations.append(f"High error rate detected: {error_rate:.1f}%. Consider investigating error patterns.")
        
        # Check performance issues
        slow_requests = len([
            m for m in self.stats["performance_metrics"]
            if m.get("duration_ms", 0) > self.analysis_patterns["performance_thresholds"]["slow_request"]
        ])
        
        if slow_requests > 0:
            recommendations.append(f"Found {slow_requests} slow requests. Consider optimizing performance.")
        
        # Check component health
        component_errors = {}
        for pattern, count in self.stats["error_patterns"].items():
            if "_critical" in pattern or "_warning" in pattern:
                component = pattern.split("_")[0]
                component_errors[component] = component_errors.get(component, 0) + count
        
        for component, error_count in component_errors.items():
            if error_count > 5:
                recommendations.append(f"Component '{component}' has {error_count} errors. Check component health.")
        
        return recommendations


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request/response logging.
    
    Logs all API requests with structured data including performance metrics.
    """
    
    def __init__(self, app, log_aggregator: LogAggregator = None):
        super().__init__(app)
        self.log_aggregator = log_aggregator
        self.logger = logging.getLogger("api.requests")
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Collect request info
        request_info = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        # Log request start
        context = LogContext(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level="INFO",
            category=LogCategory.API.value,
            component="request_middleware",
            request_id=request_id,
            method=request.method,
            endpoint=request.url.path,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if request.client else None
        )
        
        self._log_with_context("API request started", context, extra=request_info)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            # Update context with response info
            context.status_code = response.status_code
            context.duration_ms = duration_ms
            context.memory_mb = memory_info.used / 1024 / 1024
            context.cpu_percent = cpu_percent
            
            # Log response
            response_info = {
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "response_headers": dict(response.headers)
            }
            
            log_level = "INFO"
            if response.status_code >= 400:
                log_level = "WARNING"
            if response.status_code >= 500:
                log_level = "ERROR"
            
            context.level = log_level
            
            self._log_with_context(
                f"API request completed - {response.status_code}",
                context,
                extra=response_info
            )
            
            return response
            
        except Exception as e:
            # Calculate metrics for error case
            duration_ms = (time.time() - start_time) * 1000
            
            # Update context with error info
            context.status_code = 500
            context.duration_ms = duration_ms
            context.error_type = type(e).__name__
            context.stack_trace = traceback.format_exc()
            context.level = "ERROR"
            
            # Log error
            self._log_with_context(
                f"API request failed - {str(e)}",
                context,
                extra={"exception": str(e), "traceback": traceback.format_exc()}
            )
            
            raise
    
    def _log_with_context(self, message: str, context: LogContext, extra: Dict[str, Any] = None):
        """Log message with structured context."""
        # Create log record
        log_data = {
            "message": message,
            "context": context,
            **(extra or {})
        }
        
        # Log to standard logger
        self.logger.info(message, extra={"context": context})
        
        # Add to aggregator if available
        if self.log_aggregator:
            log_entry = {
                "timestamp": context.timestamp,
                "level": context.level,
                "category": context.category,
                "component": context.component,
                "message": message,
                "context": asdict(context),
                **(extra or {})
            }
            self.log_aggregator.add_log(log_entry)


# Global log aggregator instance
_log_aggregator: Optional[LogAggregator] = None


def get_log_aggregator() -> LogAggregator:
    """Get the global log aggregator instance."""
    global _log_aggregator
    
    if _log_aggregator is None:
        _log_aggregator = LogAggregator()
    
    return _log_aggregator


def configure_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_aggregation: bool = True
) -> Dict[str, Any]:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json, text)
        log_file: Optional log file path
        enable_aggregation: Enable log aggregation
    
    Returns:
        Logging configuration dictionary
    """
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure formatters
    formatters = {
        "json": {
            "()": StructuredJsonFormatter,
            "include_context": True
        },
        "text": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    }
    
    # Configure handlers
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": log_format,
            "stream": "ext://sys.stdout"
        }
    }
    
    # Add file handler if specified
    if log_file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": log_format,
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    
    # Add aggregation handler if enabled
    if enable_aggregation:
        handlers["aggregation"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": str(log_dir / "aggregated.log"),
            "maxBytes": 52428800,  # 50MB
            "backupCount": 10,
            "encoding": "utf8"
        }
    
    # Configure loggers
    loggers = {
        "": {  # Root logger
            "level": log_level,
            "handlers": list(handlers.keys()),
            "propagate": False
        },
        "api": {
            "level": log_level,
            "handlers": list(handlers.keys()),
            "propagate": False
        },
        "api.requests": {
            "level": "INFO",
            "handlers": list(handlers.keys()),
            "propagate": False
        },
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": False
        },
        "uvicorn.access": {
            "level": "INFO", 
            "handlers": ["console"],
            "propagate": False
        }
    }
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": handlers,
        "loggers": loggers
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Initialize log aggregator if enabled
    if enable_aggregation:
        aggregator = get_log_aggregator()
        print(f"Log aggregation enabled - logs will be stored in {log_dir}")
    
    return config


@contextmanager
def log_context(category: str, component: str, **kwargs):
    """Context manager for structured logging with additional context."""
    
    # Create context
    context = LogContext(
        timestamp=datetime.utcnow().isoformat() + "Z",
        level="INFO",
        category=category,
        component=component,
        **kwargs
    )
    
    # Store in thread-local storage or similar
    # For this implementation, we'll yield the context
    yield context


def log_with_context(logger: logging.Logger, level: str, message: str, 
                    context: LogContext, **kwargs):
    """Log a message with structured context."""
    
    # Get log level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Log with context
    logger.log(log_level, message, extra={"context": context, **kwargs})
    
    # Add to aggregator
    aggregator = get_log_aggregator()
    log_entry = {
        "timestamp": context.timestamp,
        "level": level.upper(),
        "category": context.category,
        "component": context.component,
        "message": message,
        "context": asdict(context),
        **kwargs
    }
    aggregator.add_log(log_entry) 