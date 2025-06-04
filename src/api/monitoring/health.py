"""
Health Check System for Trading Strategy Optimization API

This module provides comprehensive health monitoring including:
- Component health checks
- System resource monitoring
- Dependency validation
- Service availability checks
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a system component."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    last_check: datetime
    response_time_ms: Optional[float] = None


class HealthChecker:
    """
    Comprehensive health checking system.
    
    Monitors various system components and provides health status
    information for monitoring and alerting systems.
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.check_interval = 30.0  # seconds
        self.last_full_check = datetime.utcnow()
        
        # Health thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 90.0
        self.disk_critical_threshold = 98.0
        
        logger.info("HealthChecker initialized")
    
    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """
        Perform health checks on all system components.
        
        Returns:
            Dictionary of component health status
        """
        start_time = time.time()
        
        # Run all health checks
        await asyncio.gather(
            self._check_system_resources(),
            self._check_api_components(),
            self._check_job_manager(),
            self._check_storage_systems(),
            return_exceptions=True
        )
        
        self.last_full_check = datetime.utcnow()
        
        check_duration = (time.time() - start_time) * 1000
        logger.debug(f"Full health check completed in {check_duration:.2f}ms")
        
        return self.components.copy()
    
    async def _check_system_resources(self):
        """Check system resource health."""
        start_time = time.time()
        
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._determine_resource_status(
                cpu_percent, self.cpu_warning_threshold, self.cpu_critical_threshold
            )
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_status = self._determine_resource_status(
                memory.percent, self.memory_warning_threshold, self.memory_critical_threshold
            )
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._determine_resource_status(
                disk_percent, self.disk_warning_threshold, self.disk_critical_threshold
            )
            
            # Overall system status
            overall_status = max(cpu_status, memory_status, disk_status, key=lambda x: x.value)
            
            response_time = (time.time() - start_time) * 1000
            
            self.components["system_resources"] = ComponentHealth(
                name="System Resources",
                status=overall_status,
                message=f"CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Disk: {disk_percent:.1f}%",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                last_check=datetime.utcnow(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.components["system_resources"] = ComponentHealth(
                name="System Resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                details={"error": str(e)},
                last_check=datetime.utcnow()
            )
    
    async def _check_api_components(self):
        """Check API component health."""
        start_time = time.time()
        
        try:
            # Check if we can import key modules
            from job_manager import job_manager
            from monitoring import get_metrics_collector
            
            # Basic functionality test
            metrics_collector = get_metrics_collector()
            queue_stats = job_manager.get_queue_stats()
            
            response_time = (time.time() - start_time) * 1000
            
            self.components["api_core"] = ComponentHealth(
                name="API Core",
                status=HealthStatus.HEALTHY,
                message="API components operational",
                details={
                    "job_queue_size": queue_stats.get("queue_size", 0),
                    "active_jobs": queue_stats.get("active_jobs", 0),
                    "metrics_collector_active": metrics_collector is not None,
                    "total_jobs_processed": queue_stats.get("total_processed", 0)
                },
                last_check=datetime.utcnow(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.components["api_core"] = ComponentHealth(
                name="API Core",
                status=HealthStatus.CRITICAL,
                message=f"API components failed: {str(e)}",
                details={"error": str(e)},
                last_check=datetime.utcnow()
            )
    
    async def _check_job_manager(self):
        """Check job manager health."""
        start_time = time.time()
        
        try:
            from job_manager import job_manager
            
            stats = job_manager.get_queue_stats()
            
            # Determine status based on queue health
            status = HealthStatus.HEALTHY
            message = "Job manager operational"
            
            if stats.get("queue_size", 0) > 50:
                status = HealthStatus.WARNING
                message = "High queue size detected"
            elif stats.get("total_failed", 0) > stats.get("total_processed", 1) * 0.1:
                status = HealthStatus.WARNING
                message = "High failure rate detected"
            
            response_time = (time.time() - start_time) * 1000
            
            self.components["job_manager"] = ComponentHealth(
                name="Job Manager",
                status=status,
                message=message,
                details=stats,
                last_check=datetime.utcnow(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.components["job_manager"] = ComponentHealth(
                name="Job Manager",
                status=HealthStatus.CRITICAL,
                message=f"Job manager check failed: {str(e)}",
                details={"error": str(e)},
                last_check=datetime.utcnow()
            )
    
    async def _check_storage_systems(self):
        """Check storage system health."""
        start_time = time.time()
        
        try:
            from pathlib import Path
            
            # Check export storage
            export_dir = Path("exports/api")
            export_exists = export_dir.exists()
            export_writable = export_dir.is_dir() and os.access(export_dir, os.W_OK) if export_exists else False
            
            # Check data cache
            cache_dir = Path("data/cache")
            cache_exists = cache_dir.exists()
            
            status = HealthStatus.HEALTHY
            message = "Storage systems operational"
            
            if not export_writable:
                status = HealthStatus.WARNING
                message = "Export directory not writable"
            
            response_time = (time.time() - start_time) * 1000
            
            self.components["storage"] = ComponentHealth(
                name="Storage Systems",
                status=status,
                message=message,
                details={
                    "export_directory_exists": export_exists,
                    "export_directory_writable": export_writable,
                    "cache_directory_exists": cache_exists,
                    "export_path": str(export_dir),
                    "cache_path": str(cache_dir)
                },
                last_check=datetime.utcnow(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            self.components["storage"] = ComponentHealth(
                name="Storage Systems",
                status=HealthStatus.CRITICAL,
                message=f"Storage check failed: {str(e)}",
                details={"error": str(e)},
                last_check=datetime.utcnow()
            )
    
    def _determine_resource_status(self, value: float, warning_threshold: float, critical_threshold: float) -> HealthStatus:
        """Determine health status based on resource usage."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health summary.
        
        Returns:
            Overall health status and summary
        """
        if not self.components:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks performed yet",
                "last_check": None,
                "components": {}
            }
        
        # Determine overall status
        statuses = [comp.status for comp in self.components.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
            message = "Critical issues detected"
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
            message = "Warning conditions present"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All systems operational"
        
        # Component summary
        component_summary = {}
        for name, health in self.components.items():
            component_summary[name] = {
                "status": health.status.value,
                "message": health.message,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms
            }
        
        return {
            "status": overall_status.value,
            "message": message,
            "last_check": self.last_full_check.isoformat(),
            "components": component_summary,
            "summary": {
                "total_components": len(self.components),
                "healthy": len([c for c in self.components.values() if c.status == HealthStatus.HEALTHY]),
                "warning": len([c for c in self.components.values() if c.status == HealthStatus.WARNING]),
                "critical": len([c for c in self.components.values() if c.status == HealthStatus.CRITICAL])
            }
        }


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    
    if _health_checker is None:
        _health_checker = HealthChecker()
    
    return _health_checker 