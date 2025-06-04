"""
Performance Monitoring and Optimization Router

This router provides endpoints for monitoring platform performance,
accessing optimization reports, and managing performance-related operations.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import asyncio

from ..auth import verify_api_key
from ..performance_middleware import get_performance_middleware, get_async_processor
from ..utils.performance_optimizer import get_performance_optimizer
from ..models import (
    StandardResponse,
    PerformanceMetricsResponse,
    OptimizationReportResponse,
    SystemHealthResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/performance",
    tags=["Performance"],
    dependencies=[Depends(verify_api_key)]
)

@router.get("/metrics", 
           response_model=PerformanceMetricsResponse,
           summary="Get Real-time Performance Metrics",
           description="Retrieve comprehensive real-time performance metrics including API response times, cache hit rates, memory usage, and system performance indicators.")
async def get_performance_metrics(request: Request) -> PerformanceMetricsResponse:
    """
    Get comprehensive real-time performance metrics.
    
    Returns detailed performance information including:
    - API response time statistics
    - Cache performance metrics
    - Memory usage patterns
    - System resource utilization
    - Active background tasks
    """
    try:
        performance_middleware = get_performance_middleware()
        performance_optimizer = get_performance_optimizer()
        
        # Get middleware statistics
        middleware_stats = performance_middleware.get_performance_stats()
        
        # Get optimizer report
        optimizer_report = performance_optimizer.get_optimization_report()
        
        # Get system performance summary
        system_summary = performance_optimizer.performance_monitor.get_performance_summary()
        
        # Get active background tasks
        async_processor = get_async_processor()
        active_tasks = async_processor.get_active_tasks()
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'api_performance': {
                'total_requests': middleware_stats['total_requests'],
                'average_response_time': middleware_stats['average_response_time'],
                'cache_hit_rate': middleware_stats['cache_hit_rate'],
                'error_rate': middleware_stats['error_rate'],
                'cached_responses': middleware_stats['cached_responses']
            },
            'system_performance': system_summary,
            'cache_performance': optimizer_report['cache_performance'],
            'background_tasks': {
                'active_count': len(active_tasks),
                'tasks': active_tasks
            },
            'optimization_recommendations': optimizer_report.get('optimization_recommendations', [])
        }
        
        return PerformanceMetricsResponse(
            success=True,
            message="Performance metrics retrieved successfully",
            data=metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )

@router.get("/optimization-report",
           response_model=OptimizationReportResponse,
           summary="Get Performance Optimization Report",
           description="Generate a comprehensive performance optimization report with recommendations and detailed analysis.")
async def get_optimization_report(
    include_recommendations: bool = True,
    window_minutes: int = 60
) -> OptimizationReportResponse:
    """
    Generate comprehensive performance optimization report.
    
    Parameters:
    - include_recommendations: Include optimization recommendations
    - window_minutes: Time window for performance analysis (default: 60 minutes)
    
    Returns detailed analysis including:
    - Performance bottlenecks identification
    - Resource utilization analysis
    - Optimization recommendations
    - Benchmark comparisons
    """
    try:
        performance_optimizer = get_performance_optimizer()
        
        # Generate comprehensive report
        report = performance_optimizer.get_optimization_report()
        
        # Add time-windowed analysis
        performance_summary = performance_optimizer.performance_monitor.get_performance_summary(window_minutes)
        report['time_window_analysis'] = performance_summary
        
        # Add cache statistics
        cache_stats = performance_optimizer.cache_manager.get_stats()
        report['detailed_cache_stats'] = cache_stats
        
        # Add memory analysis
        memory_snapshots = len(performance_optimizer.memory_tracker.snapshots)
        if memory_snapshots > 0:
            latest_snapshot = performance_optimizer.memory_tracker.snapshots[-1]
            memory_diff = performance_optimizer.memory_tracker.get_memory_diff()
            report['memory_analysis'] = {
                'snapshots_count': memory_snapshots,
                'latest_snapshot_time': latest_snapshot['timestamp'].isoformat(),
                'memory_differences': [str(diff) for diff in memory_diff[:5]]  # Top 5
            }
        
        # Add database optimization if available
        if performance_optimizer.db_optimizer:
            slow_queries = performance_optimizer.db_optimizer.get_slow_queries(5)
            report['database_analysis'] = {
                'slow_queries_count': len(slow_queries),
                'slowest_queries': slow_queries
            }
        
        if not include_recommendations:
            report.pop('optimization_recommendations', None)
        
        return OptimizationReportResponse(
            success=True,
            message="Optimization report generated successfully",
            data=report
        )
        
    except Exception as e:
        logger.error(f"Failed to generate optimization report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate optimization report: {str(e)}"
        )

@router.get("/health",
           response_model=SystemHealthResponse,
           summary="Get System Health Status",
           description="Check overall system health including performance indicators, resource availability, and service status.")
async def get_system_health() -> SystemHealthResponse:
    """
    Get comprehensive system health status.
    
    Returns health information including:
    - System resource availability
    - Performance indicator status
    - Service health checks
    - Alert status
    """
    try:
        performance_optimizer = get_performance_optimizer()
        
        # Collect current system metrics
        current_metrics = performance_optimizer.performance_monitor.collect_system_metrics()
        
        # Determine health status
        health_status = "healthy"
        health_issues = []
        
        # Check memory usage
        if current_metrics['memory_usage_percent'] > 90:
            health_status = "critical"
            health_issues.append("Critical memory usage (>90%)")
        elif current_metrics['memory_usage_percent'] > 80:
            health_status = "warning"
            health_issues.append("High memory usage (>80%)")
        
        # Check CPU usage
        if current_metrics['cpu_usage_percent'] > 90:
            health_status = "critical"
            health_issues.append("Critical CPU usage (>90%)")
        elif current_metrics['cpu_usage_percent'] > 80:
            if health_status != "critical":
                health_status = "warning"
            health_issues.append("High CPU usage (>80%)")
        
        # Check cache performance
        cache_stats = performance_optimizer.cache_manager.get_stats()
        if cache_stats['hit_rate'] < 0.3:
            if health_status == "healthy":
                health_status = "warning"
            health_issues.append("Low cache hit rate (<30%)")
        
        health_data = {
            'status': health_status,
            'timestamp': datetime.utcnow().isoformat(),
            'system_metrics': current_metrics,
            'cache_stats': cache_stats,
            'health_issues': health_issues,
            'uptime_info': {
                'memory_available_gb': current_metrics['memory_available_gb'],
                'cpu_usage_percent': current_metrics['cpu_usage_percent'],
                'memory_usage_percent': current_metrics['memory_usage_percent']
            }
        }
        
        return SystemHealthResponse(
            success=True,
            message=f"System health: {health_status}",
            data=health_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system health: {str(e)}"
        )

@router.post("/cleanup",
            response_model=StandardResponse,
            summary="Trigger Performance Cleanup",
            description="Manually trigger performance cleanup operations including garbage collection and cache optimization.")
async def trigger_cleanup(background_tasks: BackgroundTasks) -> StandardResponse:
    """
    Trigger manual performance cleanup operations.
    
    Performs:
    - Garbage collection
    - Cache cleanup
    - Memory optimization
    - Resource cleanup
    """
    try:
        performance_optimizer = get_performance_optimizer()
        
        # Add cleanup to background tasks
        background_tasks.add_task(performance_optimizer.cleanup_resources)
        
        logger.info("Performance cleanup triggered")
        
        return StandardResponse(
            success=True,
            message="Performance cleanup initiated in background"
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger cleanup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger cleanup: {str(e)}"
        )

@router.get("/cache/stats",
           response_model=Dict[str, Any],
           summary="Get Cache Statistics",
           description="Get detailed cache performance statistics and configuration information.")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get detailed cache performance statistics.
    
    Returns comprehensive cache information including:
    - Hit/miss rates
    - Cache size and usage
    - Performance metrics
    - Configuration details
    """
    try:
        performance_optimizer = get_performance_optimizer()
        cache_stats = performance_optimizer.cache_manager.get_stats()
        
        return {
            'success': True,
            'message': "Cache statistics retrieved successfully",
            'data': cache_stats,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )

@router.post("/cache/clear",
            response_model=StandardResponse,
            summary="Clear Cache",
            description="Clear all cached data to force fresh data retrieval.")
async def clear_cache() -> StandardResponse:
    """
    Clear all cached data.
    
    This will:
    - Clear all memory cache entries
    - Reset cache statistics
    - Force fresh data retrieval for subsequent requests
    """
    try:
        performance_optimizer = get_performance_optimizer()
        
        # Clear memory cache
        cache_size_before = len(performance_optimizer.cache_manager.memory_cache)
        performance_optimizer.cache_manager.memory_cache.clear()
        
        # Reset cache statistics
        performance_optimizer.cache_manager.cache_stats = {
            'hits': 0, 'misses': 0, 'sets': 0
        }
        
        logger.info(f"Cache cleared - removed {cache_size_before} entries")
        
        return StandardResponse(
            success=True,
            message=f"Cache cleared successfully - removed {cache_size_before} entries"
        )
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@router.get("/background-tasks",
           response_model=Dict[str, Any],
           summary="Get Background Tasks Status",
           description="Get information about active background tasks and processing queue.")
async def get_background_tasks() -> Dict[str, Any]:
    """
    Get status of background processing tasks.
    
    Returns information about:
    - Active background tasks
    - Task execution times
    - Processing queue status
    - Task completion rates
    """
    try:
        async_processor = get_async_processor()
        active_tasks = async_processor.get_active_tasks()
        
        task_summary = {
            'total_active': len(active_tasks),
            'cpu_intensive_tasks': len([t for t in active_tasks.values() if t['type'] == 'cpu_intensive']),
            'io_tasks': len([t for t in active_tasks.values() if t['type'] == 'io_task']),
            'tasks': active_tasks
        }
        
        return {
            'success': True,
            'message': "Background tasks status retrieved successfully",
            'data': task_summary,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get background tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve background tasks: {str(e)}"
        )

@router.get("/benchmark",
           response_model=Dict[str, Any],
           summary="Run Performance Benchmark",
           description="Execute a quick performance benchmark to assess current system performance.")
async def run_benchmark(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Run a quick performance benchmark.
    
    This endpoint triggers a background performance benchmark that will:
    - Test API response times
    - Measure memory efficiency
    - Check cache performance
    - Assess system resource utilization
    """
    try:
        # Import benchmark here to avoid circular imports
        from ...scripts.performance_benchmark import PerformanceBenchmark
        
        benchmark_id = f"benchmark_{int(datetime.utcnow().timestamp())}"
        
        async def run_benchmark_task():
            try:
                benchmark = PerformanceBenchmark()
                results = benchmark.run_complete_benchmark()
                
                # Store results (in production, you might store in database)
                logger.info(f"Benchmark {benchmark_id} completed with grade: {results.get('summary', {}).get('overall_grade', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Benchmark {benchmark_id} failed: {e}")
        
        # Add benchmark to background tasks
        background_tasks.add_task(run_benchmark_task)
        
        return {
            'success': True,
            'message': "Performance benchmark started in background",
            'data': {
                'benchmark_id': benchmark_id,
                'status': 'started',
                'estimated_duration': '2-5 minutes'
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start benchmark: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start performance benchmark: {str(e)}"
        ) 