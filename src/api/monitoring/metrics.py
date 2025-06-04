"""
Prometheus Metrics Collection for Trading Strategy Optimization API

This module provides comprehensive metrics collection including:
- API performance metrics (response times, request rates, error rates)
- Optimization job metrics (success rates, completion times, queue depths)
- System resource metrics (CPU, memory, disk usage)
- Business metrics (strategy performance, usage patterns)
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass, field

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    Enum,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class MetricConfig:
    """Configuration for metrics collection."""
    
    # Prometheus settings
    metrics_path: str = "/metrics"
    enable_default_metrics: bool = True
    
    # Collection intervals
    system_metrics_interval: float = 30.0  # seconds
    job_metrics_interval: float = 10.0     # seconds
    
    # Histogram buckets
    response_time_buckets: tuple = (
        0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
    )
    
    # Resource thresholds
    cpu_warning_threshold: float = 80.0    # percent
    memory_warning_threshold: float = 85.0  # percent
    disk_warning_threshold: float = 90.0    # percent


class PrometheusMetrics:
    """
    Prometheus metrics definitions and collection.
    
    Provides comprehensive metrics for API monitoring including:
    - Request/response metrics
    - System resource metrics
    - Business logic metrics
    - Job processing metrics
    """
    
    def __init__(self, config: MetricConfig = None, registry: CollectorRegistry = None):
        self.config = config or MetricConfig()
        self.registry = registry or REGISTRY
        
        # Initialize all metrics
        self._init_api_metrics()
        self._init_system_metrics()
        self._init_job_metrics()
        self._init_business_metrics()
        
        # Internal state
        self._start_time = time.time()
        self._request_start_times: Dict[str, float] = {}
        
    def _init_api_metrics(self):
        """Initialize API performance metrics."""
        
        # Request counters
        self.requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.requests_in_progress = Gauge(
            'api_requests_in_progress',
            'Number of API requests currently being processed',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Response time histograms
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            buckets=self.config.response_time_buckets,
            registry=self.registry
        )
        
        # Error tracking
        self.errors_total = Counter(
            'api_errors_total',
            'Total number of API errors',
            ['method', 'endpoint', 'error_type'],
            registry=self.registry
        )
        
        # Rate limiting
        self.rate_limit_hits = Counter(
            'api_rate_limit_hits_total',
            'Total number of rate limit hits',
            ['endpoint'],
            registry=self.registry
        )
        
    def _init_system_metrics(self):
        """Initialize system resource metrics."""
        
        # System info
        self.system_info = Info(
            'system_info',
            'System information',
            registry=self.registry
        )
        
        # CPU metrics
        self.cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.cpu_count = Gauge(
            'system_cpu_count',
            'Number of CPU cores',
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # total, available, used, free
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage = Gauge(
            'system_disk_usage_bytes',
            'Disk usage in bytes',
            ['mountpoint', 'type'],  # total, used, free
            registry=self.registry
        )
        
        self.disk_total_bytes = Gauge(
            'system_disk_total_bytes',
            'Total disk space in bytes',
            ['mountpoint'],
            registry=self.registry
        )
        
        self.disk_usage_percent = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            ['mountpoint'],
            registry=self.registry
        )
        
        # Application uptime
        self.uptime_seconds = Gauge(
            'application_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        # Health check metrics
        self.health_check_status = Gauge(
            'health_check_status',
            'Health check status (1 = healthy, 0 = unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        self.health_check_duration = Gauge(
            'health_check_duration_seconds',
            'Health check execution time in seconds',
            ['component'],
            registry=self.registry
        )
        
    def _init_job_metrics(self):
        """Initialize optimization job metrics."""
        
        # Job counters
        self.jobs_total = Counter(
            'optimization_jobs_total',
            'Total number of optimization jobs',
            ['strategy', 'status'],  # submitted, running, completed, failed, cancelled
            registry=self.registry
        )
        
        self.jobs_in_progress = Gauge(
            'optimization_jobs_in_progress',
            'Number of optimization jobs currently running',
            ['strategy'],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_size = Gauge(
            'optimization_queue_size',
            'Number of jobs waiting in queue',
            ['priority'],
            registry=self.registry
        )
        
        # Job duration
        self.job_duration = Histogram(
            'optimization_job_duration_seconds',
            'Optimization job duration in seconds',
            ['strategy', 'status'],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600),
            registry=self.registry
        )
        
        # Job progress
        self.job_progress = Gauge(
            'optimization_job_progress_percent',
            'Current job progress percentage',
            ['job_id', 'strategy'],
            registry=self.registry
        )
        
        # Worker metrics
        self.worker_pool_size = Gauge(
            'optimization_worker_pool_size',
            'Number of worker threads in pool',
            registry=self.registry
        )
        
        self.worker_busy_count = Gauge(
            'optimization_worker_busy_count',
            'Number of busy worker threads',
            registry=self.registry
        )
        
    def _init_business_metrics(self):
        """Initialize business logic metrics."""
        
        # Strategy metrics
        self.strategy_optimizations = Counter(
            'strategy_optimizations_total',
            'Total optimizations per strategy',
            ['strategy_name'],
            registry=self.registry
        )
        
        self.strategy_performance = Gauge(
            'strategy_performance_score',
            'Latest performance score for strategy',
            ['strategy_name', 'metric'],  # sharpe_ratio, total_return, max_drawdown
            registry=self.registry
        )
        
        # Export metrics
        self.exports_total = Counter(
            'exports_total',
            'Total number of exports generated',
            ['export_type'],  # pine_script, pdf_report
            registry=self.registry
        )
        
        self.export_duration = Histogram(
            'export_duration_seconds',
            'Export generation duration in seconds',
            ['export_type'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry
        )
        
        # File storage metrics
        self.stored_files_count = Gauge(
            'stored_files_count',
            'Number of files currently stored',
            ['file_type'],
            registry=self.registry
        )
        
        self.stored_files_size_bytes = Gauge(
            'stored_files_size_bytes',
            'Total size of stored files in bytes',
            ['file_type'],
            registry=self.registry
        )


class MetricsCollector:
    """
    Centralized metrics collection and management.
    
    Handles periodic collection of system metrics and provides
    methods for recording application events.
    """
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
        self.config = metrics.config
        
        # Background collection state
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_collection = threading.Event()
        self._last_collection_time = time.time()
        
        # Cache for expensive operations
        self._system_info_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Initialize system info
        self._update_system_info()
        
    def start_background_collection(self):
        """Start background thread for periodic metrics collection."""
        if self._collection_thread is not None:
            return
            
        self._stop_collection.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        self._collection_thread.start()
        
    def stop_background_collection(self):
        """Stop background metrics collection."""
        if self._collection_thread is None:
            return
            
        self._stop_collection.set()
        self._collection_thread.join(timeout=5.0)
        self._collection_thread = None
        
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while not self._stop_collection.is_set():
            try:
                self.collect_system_metrics()
                
                # Sleep with shorter intervals to allow responsive shutdown
                for _ in range(int(self.config.system_metrics_interval)):
                    if self._stop_collection.wait(1.0):
                        break
                        
            except Exception as e:
                # Log error but continue collection
                print(f"Error in metrics collection: {e}")
                self._stop_collection.wait(5.0)  # Brief pause on error
                
    def collect_system_metrics(self):
        """Collect system resource metrics."""
        
        # CPU metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.cpu_usage.set(cpu_percent)
            self.metrics.cpu_count.set(psutil.cpu_count())
        except Exception as e:
            print(f"Error collecting CPU metrics: {e}")
            
        # Memory metrics
        try:
            memory = psutil.virtual_memory()
            self.metrics.memory_usage.labels(type='total').set(memory.total)
            self.metrics.memory_usage.labels(type='available').set(memory.available)
            self.metrics.memory_usage.labels(type='used').set(memory.used)
            self.metrics.memory_usage.labels(type='free').set(memory.free)
            self.metrics.memory_usage_percent.set(memory.percent)
        except Exception as e:
            print(f"Error collecting memory metrics: {e}")
            
        # Disk metrics
        try:
            disk = psutil.disk_usage('/')
            self.metrics.disk_usage.labels(mountpoint='/', type='total').set(disk.total)
            self.metrics.disk_usage.labels(mountpoint='/', type='used').set(disk.used)
            self.metrics.disk_usage.labels(mountpoint='/', type='free').set(disk.free)
            self.metrics.disk_total_bytes.labels(mountpoint='/').set(disk.total)
            self.metrics.disk_usage_percent.labels(mountpoint='/').set(
                (disk.used / disk.total) * 100
            )
        except Exception as e:
            print(f"Error collecting disk metrics: {e}")
            
        # Application uptime
        uptime = time.time() - self.metrics._start_time
        self.metrics.uptime_seconds.set(uptime)
        
    def _update_system_info(self):
        """Update cached system information."""
        try:
            import platform
            import sys
            
            info = {
                'python_version': sys.version,
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0]
            }
            
            self.metrics.system_info.info(info)
            self._system_info_cache = info
            
        except Exception as e:
            print(f"Error updating system info: {e}")
            
    # API Event Recording Methods
    
    def record_request_start(self, method: str, endpoint: str, request_id: str):
        """Record the start of an API request."""
        self.metrics.requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        self.metrics._request_start_times[request_id] = time.time()
        
    def record_request_end(self, method: str, endpoint: str, status_code: int, request_id: str):
        """Record the end of an API request."""
        # Update counters
        self.metrics.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.metrics.requests_in_progress.labels(method=method, endpoint=endpoint).dec()
        
        # Record duration
        if request_id in self.metrics._request_start_times:
            duration = time.time() - self.metrics._request_start_times[request_id]
            self.metrics.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            del self.metrics._request_start_times[request_id]
            
    def record_error(self, method: str, endpoint: str, error_type: str):
        """Record an API error."""
        self.metrics.errors_total.labels(
            method=method,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
        
    def record_rate_limit_hit(self, endpoint: str):
        """Record a rate limit hit."""
        self.metrics.rate_limit_hits.labels(endpoint=endpoint).inc()
        
    # Job Event Recording Methods
    
    def record_job_submitted(self, strategy: str):
        """Record a new optimization job submission."""
        self.metrics.jobs_total.labels(strategy=strategy, status='submitted').inc()
        self.metrics.strategy_optimizations.labels(strategy_name=strategy).inc()
        
    def record_job_started(self, job_id: str, strategy: str):
        """Record an optimization job start."""
        self.metrics.jobs_total.labels(strategy=strategy, status='running').inc()
        self.metrics.jobs_in_progress.labels(strategy=strategy).inc()
        
    def record_job_progress(self, job_id: str, strategy: str, progress_percent: float):
        """Record optimization job progress."""
        self.metrics.job_progress.labels(job_id=job_id, strategy=strategy).set(progress_percent)
        
    def record_job_completed(self, job_id: str, strategy: str, duration: float, status: str):
        """Record optimization job completion."""
        # Update counters
        self.metrics.jobs_total.labels(strategy=strategy, status=status).inc()
        self.metrics.jobs_in_progress.labels(strategy=strategy).dec()
        
        # Record duration
        self.metrics.job_duration.labels(strategy=strategy, status=status).observe(duration)
        
        # Clear progress metric
        self.metrics.job_progress.remove(job_id, strategy)
        
    def record_queue_size(self, priority: str, size: int):
        """Record optimization queue size."""
        self.metrics.queue_size.labels(priority=priority).set(size)
        
    def record_worker_metrics(self, pool_size: int, busy_count: int):
        """Record worker pool metrics."""
        self.metrics.worker_pool_size.set(pool_size)
        self.metrics.worker_busy_count.set(busy_count)
        
    # Business Event Recording Methods
    
    def record_strategy_performance(self, strategy_name: str, metric: str, value: float):
        """Record strategy performance metrics."""
        self.metrics.strategy_performance.labels(
            strategy_name=strategy_name,
            metric=metric
        ).set(value)
        
    def record_export_start(self, export_type: str):
        """Record the start of an export operation."""
        return time.time()  # Return start time for duration calculation
        
    def record_export_completed(self, export_type: str, start_time: float):
        """Record export completion."""
        self.metrics.exports_total.labels(export_type=export_type).inc()
        duration = time.time() - start_time
        self.metrics.export_duration.labels(export_type=export_type).observe(duration)
        
    def record_file_storage(self, file_type: str, file_count: int, total_size_bytes: int):
        """Record file storage metrics."""
        self.metrics.stored_files_count.labels(file_type=file_type).set(file_count)
        self.metrics.stored_files_size_bytes.labels(file_type=file_type).set(total_size_bytes)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic API metrics collection.
    
    Automatically tracks all HTTP requests including:
    - Request counts by method, endpoint, and status code
    - Response times
    - Error rates
    - Concurrent request tracking
    """
    
    def __init__(self, app, metrics_collector: MetricsCollector):
        super().__init__(app)
        self.metrics_collector = metrics_collector
        
    async def dispatch(self, request: Request, call_next):
        # Generate request ID for tracking
        request_id = f"{id(request)}_{time.time()}"
        
        # Extract endpoint info
        method = request.method
        endpoint = self._get_endpoint_path(request)
        
        # Skip metrics endpoint to avoid self-monitoring
        if endpoint == "/metrics":
            return await call_next(request)
            
        # Record request start
        self.metrics_collector.record_request_start(method, endpoint, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record successful completion
            self.metrics_collector.record_request_end(
                method, endpoint, response.status_code, request_id
            )
            
            return response
            
        except Exception as e:
            # Record error
            error_type = type(e).__name__
            self.metrics_collector.record_error(method, endpoint, error_type)
            
            # Clean up request tracking
            if request_id in self.metrics_collector.metrics._request_start_times:
                del self.metrics_collector.metrics._request_start_times[request_id]
                
            raise
            
    def _get_endpoint_path(self, request: Request) -> str:
        """Extract normalized endpoint path for metrics."""
        path = request.url.path
        
        # Normalize dynamic segments
        # Example: /api/v1/optimize/jobs/123 -> /api/v1/optimize/jobs/{id}
        normalized_patterns = [
            (r'/optimize/jobs/[^/]+', '/optimize/jobs/{id}'),
            (r'/optimize/status/[^/]+', '/optimize/status/{id}'),
            (r'/optimize/results/[^/]+', '/optimize/results/{id}'),
            (r'/strategies/[^/]+', '/strategies/{name}'),
            (r'/export/download/[^/]+', '/export/download/{id}'),
            (r'/export/files/[^/]+', '/export/files/{id}'),
        ]
        
        import re
        for pattern, replacement in normalized_patterns:
            path = re.sub(pattern, replacement, path)
            
        return path


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        config = MetricConfig()
        metrics = PrometheusMetrics(config)
        _metrics_collector = MetricsCollector(metrics)
        _metrics_collector.start_background_collection()
        
    return _metrics_collector


def get_metrics_response() -> Response:
    """Generate Prometheus metrics response."""
    collector = get_metrics_collector()
    
    # Generate metrics output
    output = generate_latest(collector.metrics.registry)
    
    return Response(
        content=output,
        media_type=CONTENT_TYPE_LATEST,
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    ) 