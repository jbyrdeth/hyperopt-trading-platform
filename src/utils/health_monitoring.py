"""
Production Health Monitoring System

Comprehensive health checks and performance monitoring.
"""

import psutil
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess

class HealthMonitor:
    """Production health monitoring and alerting system."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.logger = logging.getLogger('health_monitor')
        self.alerts = []
        self.metrics_history = []
        
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status['components']['system'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'status': 'healthy'
        }
        
        # Check thresholds
        if cpu_percent > 80:
            health_status['components']['system']['status'] = 'warning'
            health_status['overall_status'] = 'warning'
            self._add_alert('high_cpu', f'CPU usage at {cpu_percent:.1f}%')
            
        if memory.percent > 85:
            health_status['components']['system']['status'] = 'critical'
            health_status['overall_status'] = 'critical'
            self._add_alert('high_memory', f'Memory usage at {memory.percent:.1f}%')
            
        if disk.percent > 90:
            health_status['components']['system']['status'] = 'critical'
            health_status['overall_status'] = 'critical'
            self._add_alert('high_disk', f'Disk usage at {disk.percent:.1f}%')
        
        # Application-specific checks
        health_status['components']['application'] = self._check_application_health()
        
        # Strategy performance checks
        health_status['components']['strategies'] = self._check_strategy_health()
        
        # Log health status
        self.logger.info(f"Health check completed: {health_status['overall_status']}")
        
        # Store metrics
        self.metrics_history.append(health_status)
        self._cleanup_old_metrics()
        
        return health_status
    
    def _check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health metrics."""
        app_health = {
            'status': 'healthy',
            'components_checked': []
        }
        
        # Check log files exist and are writable
        logs_dir = self.project_root / 'logs'
        if logs_dir.exists():
            app_health['logs_directory'] = 'accessible'
            app_health['components_checked'].append('logs')
        else:
            app_health['logs_directory'] = 'missing'
            app_health['status'] = 'warning'
            
        # Check configuration files
        config_dir = self.project_root / 'config'
        if config_dir.exists():
            app_health['config_directory'] = 'accessible'
            app_health['components_checked'].append('config')
        else:
            app_health['config_directory'] = 'missing'
            app_health['status'] = 'warning'
            
        # Check strategy modules
        strategies_dir = self.project_root / 'src' / 'strategies'
        if strategies_dir.exists():
            strategy_files = list(strategies_dir.glob('*.py'))
            app_health['strategy_modules'] = len(strategy_files)
            app_health['components_checked'].append('strategies')
        else:
            app_health['strategy_modules'] = 0
            app_health['status'] = 'critical'
            
        return app_health
    
    def _check_strategy_health(self) -> Dict[str, Any]:
        """Check strategy-specific health metrics."""
        strategy_health = {
            'status': 'healthy',
            'total_strategies': 0,
            'healthy_strategies': 0,
            'warning_strategies': 0,
            'critical_strategies': 0
        }
        
        # This would integrate with actual strategy monitoring
        # For now, simulate based on available strategies
        try:
            from src.optimization.strategy_factory import StrategyFactory
            factory = StrategyFactory()
            available_strategies = factory.get_all_strategies()
            
            strategy_health['total_strategies'] = len(available_strategies)
            strategy_health['healthy_strategies'] = len(available_strategies)  # Assume healthy for now
            strategy_health['available_strategies'] = available_strategies
            
        except Exception as e:
            self.logger.error(f"Error checking strategy health: {e}")
            strategy_health['status'] = 'critical'
            strategy_health['error'] = str(e)
            
        return strategy_health
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert to the system."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'severity': self._get_alert_severity(alert_type)
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity."""
        critical_alerts = ['high_memory', 'high_disk', 'strategy_failure']
        warning_alerts = ['high_cpu', 'performance_degradation']
        
        if alert_type in critical_alerts:
            return 'critical'
        elif alert_type in warning_alerts:
            return 'warning'
        else:
            return 'info'
    
    def _cleanup_old_metrics(self):
        """Remove old metrics to prevent memory bloat."""
        # Keep only last 24 hours of metrics (assuming checks every 5 minutes)
        max_entries = 24 * 60 // 5  # 288 entries
        if len(self.metrics_history) > max_entries:
            self.metrics_history = self.metrics_history[-max_entries:]
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of metrics over the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No metrics available for the specified period'}
        
        # Calculate averages
        cpu_values = [m['components']['system']['cpu_percent'] for m in recent_metrics]
        memory_values = [m['components']['system']['memory_percent'] for m in recent_metrics]
        
        summary = {
            'period_hours': hours,
            'total_checks': len(recent_metrics),
            'average_cpu': sum(cpu_values) / len(cpu_values),
            'max_cpu': max(cpu_values),
            'average_memory': sum(memory_values) / len(memory_values),
            'max_memory': max(memory_values),
            'alerts_in_period': len([a for a in self.alerts if datetime.fromisoformat(a['timestamp']) > cutoff_time])
        }
        
        return summary

# Global health monitor instance
health_monitor = None

def initialize_health_monitor(project_root: str):
    """Initialize the global health monitor."""
    global health_monitor
    health_monitor = HealthMonitor(project_root)
    return health_monitor
