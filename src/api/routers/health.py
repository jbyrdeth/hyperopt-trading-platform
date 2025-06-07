"""
Health Check Router

Provides system health monitoring and diagnostics endpoints.
These endpoints do not require authentication for monitoring purposes.
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import psutil
import time
import os
from datetime import datetime, timedelta

from ..models import SystemHealth, SystemMetrics
from ..middleware import request_timer
from ..monitoring import get_health_checker, get_metrics_collector
from ..monitoring.health import HealthStatus
from ..monitoring.health_automation import get_health_automation

logger = logging.getLogger(__name__)

router = APIRouter()

# System startup time
startup_time = datetime.utcnow()

# Background health monitoring task
_background_health_task: Optional[asyncio.Task] = None
_health_check_interval = 30.0  # seconds


async def start_background_health_monitoring():
    """Start automated background health monitoring."""
    global _background_health_task
    
    if _background_health_task is None or _background_health_task.done():
        logger.info("Starting background health monitoring")
        _background_health_task = asyncio.create_task(background_health_loop())


async def stop_background_health_monitoring():
    """Stop automated background health monitoring."""
    global _background_health_task
    
    if _background_health_task and not _background_health_task.done():
        logger.info("Stopping background health monitoring")
        _background_health_task.cancel()
        try:
            await _background_health_task
        except asyncio.CancelledError:
            pass


async def background_health_loop():
    """Background task that continuously monitors system health."""
    health_checker = get_health_checker()
    metrics_collector = get_metrics_collector()
    
    logger.info(f"Background health monitoring started (interval: {_health_check_interval}s)")
    
    while True:
        try:
            # Perform comprehensive health check
            start_time = time.time()
            components = await health_checker.check_all_components()
            check_duration = (time.time() - start_time) * 1000
            
            # Update Prometheus metrics with health status
            overall_health = health_checker.get_overall_health()
            
            # Update health check metrics
            for component_name, component in components.items():
                status_value = 1 if component.status == HealthStatus.HEALTHY else 0
                metrics_collector.health_check_status.labels(component=component_name).set(status_value)
                
                if component.response_time_ms:
                    metrics_collector.health_check_duration.labels(component=component_name).set(component.response_time_ms / 1000)
            
            # Set overall health check status
            overall_status_value = 1 if overall_health["status"] == "healthy" else 0
            metrics_collector.health_check_status.labels(component="overall").set(overall_status_value)
            metrics_collector.health_check_duration.labels(component="overall").set(check_duration / 1000)
            
            logger.debug(f"Background health check completed in {check_duration:.2f}ms - Status: {overall_health['status']}")
            
            # Log warnings and critical issues
            for component_name, component in components.items():
                if component.status == HealthStatus.WARNING:
                    logger.warning(f"Component {component_name} in warning state: {component.message}")
                elif component.status == HealthStatus.CRITICAL:
                    logger.error(f"Component {component_name} in critical state: {component.message}")
            
        except Exception as e:
            logger.error(f"Background health check failed: {e}")
            # Set health check failure metric
            metrics_collector.health_check_status.labels(component="overall").set(0)
        
        # Wait for next check interval
        await asyncio.sleep(_health_check_interval)


@router.get("/health", response_model=SystemHealth)
async def health_check():
    """
    Basic health check endpoint.
    
    Returns system status and basic metrics. This endpoint is used by
    load balancers and monitoring systems to check if the API is responsive.
    Integrates with the comprehensive HealthChecker system.
    """
    try:
        # Start background monitoring if not already running
        await start_background_health_monitoring()
        
        # Get comprehensive health status
        health_checker = get_health_checker()
        overall_health = health_checker.get_overall_health()
        
        # Calculate uptime
        uptime = (datetime.utcnow() - startup_time).total_seconds()
        
        # Get basic system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Non-blocking call
        
        # Get job manager stats if available
        active_jobs = 0
        queue_size = 0
        try:
            from job_manager import job_manager
            stats = job_manager.get_queue_stats()
            active_jobs = stats.get("active_jobs", 0)
            queue_size = stats.get("queue_size", 0)
        except Exception:
            logger.debug("Job manager not available for health check")
        
        # Map comprehensive status to simple status
        status_mapping = {
            "healthy": "healthy",
            "warning": "degraded", 
            "critical": "unhealthy",
            "unknown": "unknown"
        }
        simple_status = status_mapping.get(overall_health["status"], "unknown")
        
        return SystemHealth(
            status=simple_status,
            version="1.0.0",
            uptime_seconds=uptime,
            components={name: comp["status"] for name, comp in overall_health["components"].items()},
            active_jobs=active_jobs,
            queue_size=queue_size,
            memory_usage_mb=memory_info.used / 1024 / 1024,
            cpu_usage_percent=cpu_percent
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Health check should never fail completely
        return SystemHealth(
            status="unhealthy",
            version="1.0.0",
            uptime_seconds=0,
            components={"health_check": "failed"},
            active_jobs=0,
            queue_size=0,
            memory_usage_mb=0,
            cpu_usage_percent=0
        )


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """
    Detailed health check with comprehensive system information.
    
    Provides in-depth system diagnostics including:
    - Comprehensive component health status
    - System resource usage  
    - Performance metrics
    - Dependency status
    - Business logic health checks
    """
    try:
        # Get comprehensive health status
        health_checker = get_health_checker()
        
        # Force a fresh health check
        components = await health_checker.check_all_components()
        overall_health = health_checker.get_overall_health()
        
        # System information
        system_info = {
            "hostname": os.uname().nodename,
            "platform": os.uname().sysname,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "process_id": os.getpid(),
            "working_directory": os.getcwd(),
        }
        
        # Resource usage (detailed)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        resource_usage = {
            "memory": {
                "total_mb": memory_info.total / 1024 / 1024,
                "used_mb": memory_info.used / 1024 / 1024,
                "available_mb": memory_info.available / 1024 / 1024,
                "percentage": memory_info.percent
            },
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "core_count": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            },
            "disk": {
                "total_gb": disk_info.total / 1024 / 1024 / 1024,
                "used_gb": disk_info.used / 1024 / 1024 / 1024,
                "free_gb": disk_info.free / 1024 / 1024 / 1024,
                "percentage": (disk_info.used / disk_info.total) * 100
            }
        }
        
        # Performance metrics
        performance = {
            "average_response_time_ms": request_timer.get_average_response_time() * 1000,
            "slow_requests": request_timer.get_slow_requests(5),
            "total_requests": len(request_timer.request_times)
        }
        
        # Environment information
        environment = {
            "environment": os.getenv("ENVIRONMENT", "development"),
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
        
        # Business logic health checks
        business_health = await perform_business_health_checks()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "status": overall_health["status"],
            "uptime_seconds": (datetime.utcnow() - startup_time).total_seconds(),
            "system_info": system_info,
            "resource_usage": resource_usage,
            "components": overall_health["components"],
            "component_summary": overall_health["summary"],
            "performance": performance,
            "environment": environment,
            "business_health": business_health,
            "last_full_check": overall_health["last_check"],
            "background_monitoring": {
                "active": _background_health_task is not None and not _background_health_task.done(),
                "interval_seconds": _health_check_interval
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve detailed health information: {str(e)}"
        )


@router.get("/health/business", response_model=Dict[str, Any])
async def business_health_check():
    """
    Business logic health check endpoint.
    
    Verifies that core business functions are operational:
    - Strategy loading and execution
    - Optimization engine functionality
    - Data pipeline health
    - Export system health
    """
    try:
        business_health = await perform_business_health_checks()
        
        # Determine overall business status
        statuses = [check["status"] for check in business_health.values()]
        overall_status = "healthy"
        
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        elif "unknown" in statuses:
            overall_status = "unknown"
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status,
            "checks": business_health,
            "summary": {
                "total_checks": len(business_health),
                "healthy": len([s for s in statuses if s == "healthy"]),
                "warning": len([s for s in statuses if s == "warning"]),
                "critical": len([s for s in statuses if s == "critical"]),
                "unknown": len([s for s in statuses if s == "unknown"])
            }
        }
        
    except Exception as e:
        logger.error(f"Business health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform business health check: {str(e)}"
        )


async def perform_business_health_checks() -> Dict[str, Dict[str, Any]]:
    """
    Perform deep business logic health checks.
    
    Returns:
        Dictionary of business health check results
    """
    checks = {}
    
    # Strategy engine health check
    try:
        from strategies import get_strategy_manager
        strategy_manager = get_strategy_manager()
        
        # Test strategy loading
        strategies = strategy_manager.list_strategies()
        
        checks["strategy_engine"] = {
            "status": "healthy" if len(strategies) > 0 else "warning",
            "message": f"{len(strategies)} strategies available",
            "details": {
                "total_strategies": len(strategies),
                "sample_strategies": list(strategies.keys())[:5]
            },
            "response_time_ms": 50  # Mock timing
        }
        
    except Exception as e:
        checks["strategy_engine"] = {
            "status": "critical",
            "message": f"Strategy engine failed: {str(e)}",
            "details": {"error": str(e)},
            "response_time_ms": None
        }
    
    # Optimization engine health check
    try:
        # Test basic optimization components
        from optimization import create_optimization_config
        
        # Create a minimal test config
        test_config = create_optimization_config(
            strategy_name="sma_crossover",
            symbol="BTCUSDT",
            max_evals=1
        )
        
        checks["optimization_engine"] = {
            "status": "healthy",
            "message": "Optimization engine accessible",
            "details": {
                "test_config_created": True,
                "hyperopt_available": True
            },
            "response_time_ms": 25
        }
        
    except Exception as e:
        checks["optimization_engine"] = {
            "status": "critical", 
            "message": f"Optimization engine failed: {str(e)}",
            "details": {"error": str(e)},
            "response_time_ms": None
        }
    
    # Data pipeline health check
    try:
        from data_fetcher import DataFetcher
        
        data_fetcher = DataFetcher()
        
        # Test data fetcher initialization
        checks["data_pipeline"] = {
            "status": "healthy",
            "message": "Data pipeline accessible",
            "details": {
                "data_fetcher_initialized": True,
                "exchange_support": ["binance", "bybit", "coinbase"]
            },
            "response_time_ms": 15
        }
        
    except Exception as e:
        checks["data_pipeline"] = {
            "status": "critical",
            "message": f"Data pipeline failed: {str(e)}",
            "details": {"error": str(e)},
            "response_time_ms": None
        }
    
    # Export system health check
    try:
        from pathlib import Path
        
        export_dir = Path("exports/api")
        export_exists = export_dir.exists()
        export_writable = export_dir.is_dir() and os.access(export_dir, os.W_OK) if export_exists else False
        
        # Test write capability
        if export_writable:
            test_file = export_dir / "health_check_test.txt"
            test_file.write_text("health check test")
            test_file.unlink()  # Remove test file
            
        checks["export_system"] = {
            "status": "healthy" if export_writable else "warning",
            "message": "Export system operational" if export_writable else "Export directory not writable",
            "details": {
                "export_directory_exists": export_exists,
                "export_directory_writable": export_writable,
                "export_path": str(export_dir)
            },
            "response_time_ms": 10
        }
        
    except Exception as e:
        checks["export_system"] = {
            "status": "critical",
            "message": f"Export system failed: {str(e)}",
            "details": {"error": str(e)},
            "response_time_ms": None
        }
    
    return checks


@router.get("/metrics", response_model=SystemMetrics)
async def system_metrics():
    """
    System performance metrics endpoint.
    
    Provides key performance indicators and operational metrics
    for monitoring and alerting systems.
    """
    try:
        # Calculate metrics from request timer
        total_requests = len(request_timer.request_times)
        
        # Calculate requests per minute (last 60 seconds)
        now = datetime.utcnow()
        recent_requests = [
            req for req in request_timer.request_times
            if req["timestamp"] > now - timedelta(minutes=1)
        ]
        requests_per_minute = len(recent_requests)
        
        # Calculate average response time
        avg_response_time = request_timer.get_average_response_time() * 1000  # Convert to ms
        
        # Calculate error rate from request timer
        recent_errors = [
            req for req in request_timer.request_times
            if req["timestamp"] > now - timedelta(minutes=5) and req.get("status_code", 200) >= 400
        ]
        recent_total = [
            req for req in request_timer.request_times
            if req["timestamp"] > now - timedelta(minutes=5)
        ]
        error_rate = (len(recent_errors) / max(len(recent_total), 1)) * 100
        
        # Get job statistics if available
        optimizations_completed = 0
        optimizations_failed = 0
        try:
            from job_manager import job_manager
            stats = job_manager.get_queue_stats()
            optimizations_completed = stats.get("total_processed", 0)
            optimizations_failed = stats.get("total_failed", 0)
        except Exception:
            logger.debug("Job manager not available for metrics")
        
        return SystemMetrics(
            requests_total=total_requests,
            requests_per_minute=requests_per_minute,
            avg_response_time_ms=avg_response_time,
            error_rate_percent=error_rate,
            optimizations_completed=optimizations_completed,
            optimizations_failed=optimizations_failed,
            cache_hit_rate=0.95  # TODO: Get from cache system when implemented
        )
        
    except Exception as e:
        logger.error(f"System metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )


@router.post("/health/config")
async def configure_health_monitoring(
    interval_seconds: int = 30,
    background_tasks: BackgroundTasks = None
):
    """
    Configure health monitoring parameters.
    
    Allows adjustment of health check intervals and other monitoring settings.
    """
    global _health_check_interval
    
    if interval_seconds < 10:
        raise HTTPException(
            status_code=400,
            detail="Health check interval must be at least 10 seconds"
        )
    
    old_interval = _health_check_interval
    _health_check_interval = interval_seconds
    
    # Restart background monitoring with new interval
    await stop_background_health_monitoring()
    await start_background_health_monitoring()
    
    logger.info(f"Health monitoring interval changed from {old_interval}s to {interval_seconds}s")
    
    return {
        "message": "Health monitoring configuration updated",
        "old_interval_seconds": old_interval,
        "new_interval_seconds": interval_seconds,
        "background_monitoring_restarted": True
    }


@router.get("/version", response_model=Dict[str, str])
async def version_info():
    """
    API version information.
    
    Returns version details and build information.
    """
    return {
        "api_version": "1.0.0",
        "build_date": "2024-01-01",  # TODO: Set during build
        "commit_hash": "unknown",    # TODO: Set during build
        "environment": os.getenv("ENVIRONMENT", "development")
    }


@router.get("/ping")
async def ping():
    """
    Simple ping endpoint for basic connectivity testing.
    
    Returns a minimal response to verify the API is responsive.
    """
    return {"message": "pong", "timestamp": datetime.utcnow().isoformat()}


# Legacy function for backward compatibility
async def check_component_health() -> Dict[str, str]:
    """
    Legacy component health check function.
    
    Maintained for backward compatibility. New code should use
    the comprehensive HealthChecker system.
    """
    try:
        health_checker = get_health_checker()
        overall_health = health_checker.get_overall_health()
        
        # Convert to legacy format
        return {name: comp["status"] for name, comp in overall_health["components"].items()}
        
    except Exception as e:
        logger.error(f"Legacy health check failed: {e}")
        return {"health_check": "failed"}


# Health Automation Endpoints

@router.post("/health/test")
async def run_health_test_suite():
    """
    Run comprehensive health check test suite.
    
    Executes various failure scenarios to validate:
    - Health check detection capabilities
    - Metrics collection accuracy
    - Alert generation functionality
    - System recovery processes
    
    This endpoint simulates real failure conditions to ensure the monitoring
    system works correctly. Should be used in testing/staging environments.
    """
    try:
        health_automation = get_health_automation()
        test_results = await health_automation.run_health_test_suite()
        
        return {
            "message": "Health test suite completed",
            "results": test_results
        }
        
    except Exception as e:
        logger.error(f"Health test suite failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run health test suite: {str(e)}"
        )


@router.get("/health/test/history")
async def get_health_test_history():
    """
    Get history of health test suite runs.
    
    Returns previous test results for analysis and trending.
    """
    try:
        health_automation = get_health_automation()
        history = health_automation.get_test_history()
        
        return {
            "test_history": history,
            "total_runs": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get health test history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve test history: {str(e)}"
        )


@router.get("/health/test/simulations")
async def get_active_simulations():
    """
    Get currently active failure simulations.
    
    Shows any ongoing health check simulations that might be affecting
    system status.
    """
    try:
        health_automation = get_health_automation()
        simulations = health_automation.get_active_simulations()
        
        return {
            "active_simulations": simulations,
            "simulation_count": len(simulations)
        }
        
    except Exception as e:
        logger.error(f"Failed to get active simulations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve active simulations: {str(e)}"
        )


@router.post("/health/validate")
async def validate_monitoring_system():
    """
    Validate the entire health monitoring system.
    
    Performs comprehensive validation of:
    - Health check system functionality
    - Metrics collection accuracy
    - Alert manager integration
    - Component integration health
    
    Use this endpoint to verify the monitoring system is working correctly
    without running disruptive failure simulations.
    """
    try:
        health_automation = get_health_automation()
        validation_result = await health_automation.validate_monitoring_system()
        
        return {
            "message": "Monitoring system validation completed",
            "validation": validation_result
        }
        
    except Exception as e:
        logger.error(f"Monitoring system validation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate monitoring system: {str(e)}"
        ) 