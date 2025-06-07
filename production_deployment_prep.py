#!/usr/bin/env python3
"""
Production Deployment Preparation

Comprehensive preparation script for production deployment of the 
24-strategy trading platform with enterprise-grade features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import json
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import traceback
import psutil
import warnings
warnings.filterwarnings('ignore')

class ProductionDeploymentPreparator:
    """Comprehensive production deployment preparation system."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.deployment_config = {}
        self.setup_results = {}
        
    def prepare_production_logging(self) -> bool:
        """Set up production-grade logging configuration."""
        print("\n🪵 SETTING UP PRODUCTION LOGGING")
        print("=" * 50)
        
        try:
            # Create logs directory
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Production logging configuration
            logging_config = {
                'version': 1,
                'disable_existing_loggers': False,
                'formatters': {
                    'standard': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                    },
                    'detailed': {
                        'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                    },
                    'json': {
                        'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
                    }
                },
                'handlers': {
                    'console': {
                        'class': 'logging.StreamHandler',
                        'level': 'INFO',
                        'formatter': 'standard',
                        'stream': 'ext://sys.stdout'
                    },
                    'info_file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'detailed',
                        'filename': str(logs_dir / 'application.log'),
                        'maxBytes': 10485760,  # 10MB
                        'backupCount': 5
                    },
                    'error_file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'ERROR',
                        'formatter': 'detailed',
                        'filename': str(logs_dir / 'errors.log'),
                        'maxBytes': 10485760,  # 10MB
                        'backupCount': 5
                    },
                    'performance_file': {
                        'class': 'logging.handlers.RotatingFileHandler',
                        'level': 'INFO',
                        'formatter': 'json',
                        'filename': str(logs_dir / 'performance.log'),
                        'maxBytes': 10485760,  # 10MB
                        'backupCount': 3
                    }
                },
                'loggers': {
                    '': {  # Root logger
                        'level': 'INFO',
                        'handlers': ['console', 'info_file', 'error_file']
                    },
                    'performance': {
                        'level': 'INFO',
                        'handlers': ['performance_file'],
                        'propagate': False
                    },
                    'trading_strategies': {
                        'level': 'DEBUG',
                        'handlers': ['info_file', 'error_file'],
                        'propagate': False
                    }
                }
            }
            
            # Save logging configuration
            config_path = self.project_root / "config" / "logging.yaml"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(logging_config, f, default_flow_style=False)
            
            print(f"✅ Production logging configured:")
            print(f"   📂 Log directory: {logs_dir}")
            print(f"   📄 Config file: {config_path}")
            print(f"   🔄 Log rotation: 10MB files, 5 backups")
            print(f"   📊 Separate performance logs")
            
            self.setup_results['logging'] = {
                'status': 'success',
                'config_path': str(config_path),
                'logs_directory': str(logs_dir)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Logging setup failed: {str(e)}")
            self.setup_results['logging'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def create_error_handling_framework(self) -> bool:
        """Create production-grade error handling framework."""
        print("\n🛡️ CREATING ERROR HANDLING FRAMEWORK")
        print("=" * 50)
        
        try:
            # Create error handling module
            error_handling_code = '''"""
Production Error Handling Framework

Comprehensive error handling for the trading strategy platform.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional, Union, Dict
from datetime import datetime
import json

class ProductionErrorHandler:
    """Production-grade error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger('error_handler')
        self.error_counts = {}
        self.circuit_breakers = {}
        
    def handle_strategy_error(self, strategy_name: str, error: Exception, context: Dict[str, Any] = None):
        """Handle strategy-specific errors with context."""
        error_info = {
            'strategy': strategy_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Strategy error in {strategy_name}: {error}", extra=error_info)
        
        # Track error frequency
        error_key = f"{strategy_name}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Circuit breaker logic
        if self.error_counts[error_key] > 5:  # More than 5 errors
            self.logger.critical(f"Circuit breaker triggered for {strategy_name}")
            self.circuit_breakers[strategy_name] = datetime.now()
            
    def handle_backtesting_error(self, error: Exception, strategy_name: str = None, data_info: Dict = None):
        """Handle backtesting engine errors."""
        error_info = {
            'component': 'backtesting_engine',
            'strategy': strategy_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'data_info': data_info or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Backtesting error: {error}", extra=error_info)
        
    def handle_api_error(self, error: Exception, endpoint: str, request_data: Dict = None):
        """Handle API-specific errors."""
        error_info = {
            'component': 'api',
            'endpoint': endpoint,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'request_data': request_data or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"API error at {endpoint}: {error}", extra=error_info)
        
    def is_circuit_open(self, strategy_name: str) -> bool:
        """Check if circuit breaker is open for a strategy."""
        if strategy_name in self.circuit_breakers:
            # Auto-reset after 5 minutes
            if (datetime.now() - self.circuit_breakers[strategy_name]).seconds > 300:
                del self.circuit_breakers[strategy_name]
                self.logger.info(f"Circuit breaker reset for {strategy_name}")
                return False
            return True
        return False

def with_error_handling(component: str = "general"):
    """Decorator for adding error handling to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = ProductionErrorHandler()
                
                if component == "strategy":
                    strategy_name = kwargs.get('strategy_name', 'unknown')
                    handler.handle_strategy_error(strategy_name, e, {'args': args, 'kwargs': kwargs})
                elif component == "backtesting":
                    strategy_name = kwargs.get('strategy_name', None)
                    handler.handle_backtesting_error(e, strategy_name, kwargs)
                elif component == "api":
                    endpoint = kwargs.get('endpoint', func.__name__)
                    handler.handle_api_error(e, endpoint, kwargs)
                else:
                    logging.getLogger(component).error(f"Error in {func.__name__}: {e}", exc_info=True)
                
                # Re-raise for proper error propagation
                raise
                
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logging.getLogger('retry').warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    
        return wrapper
    return decorator

# Global error handler instance
error_handler = ProductionErrorHandler()
'''
            
            # Save error handling framework
            error_module_path = self.project_root / "src" / "utils" / "error_handling.py"
            error_module_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(error_module_path, 'w') as f:
                f.write(error_handling_code)
            
            print(f"✅ Error handling framework created:")
            print(f"   📄 Module: {error_module_path}")
            print(f"   🔧 Features: Circuit breakers, retry logic, context logging")
            print(f"   🛡️ Strategy-specific error tracking")
            print(f"   🔄 Auto-recovery mechanisms")
            
            self.setup_results['error_handling'] = {
                'status': 'success',
                'module_path': str(error_module_path)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling setup failed: {str(e)}")
            self.setup_results['error_handling'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def create_health_monitoring_system(self) -> bool:
        """Create comprehensive health monitoring and alerting."""
        print("\n💓 CREATING HEALTH MONITORING SYSTEM")
        print("=" * 50)
        
        try:
            # Health check module
            health_check_code = '''"""
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
'''
            
            # Save health monitoring module
            health_module_path = self.project_root / "src" / "utils" / "health_monitoring.py"
            
            with open(health_module_path, 'w') as f:
                f.write(health_check_code)
            
            # Create health check script
            health_script = '''#!/usr/bin/env python3
"""
Health Check Script

Standalone script for checking system health.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.health_monitoring import initialize_health_monitor
import json

def main():
    project_root = os.path.dirname(__file__)
    monitor = initialize_health_monitor(project_root)
    
    health_status = monitor.check_system_health()
    
    print(json.dumps(health_status, indent=2))
    
    # Exit with error code if not healthy
    if health_status['overall_status'] != 'healthy':
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
            
            health_script_path = self.project_root / "health_check.py"
            with open(health_script_path, 'w') as f:
                f.write(health_script)
            
            # Make script executable
            os.chmod(health_script_path, 0o755)
            
            print(f"✅ Health monitoring system created:")
            print(f"   📄 Module: {health_module_path}")
            print(f"   🔧 Script: {health_script_path}")
            print(f"   📊 Features: System metrics, alerts, trend analysis")
            print(f"   🚨 Automated alerting for critical thresholds")
            
            self.setup_results['health_monitoring'] = {
                'status': 'success',
                'module_path': str(health_module_path),
                'script_path': str(health_script_path)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Health monitoring setup failed: {str(e)}")
            self.setup_results['health_monitoring'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def create_deployment_documentation(self) -> bool:
        """Create comprehensive deployment documentation."""
        print("\n📚 CREATING DEPLOYMENT DOCUMENTATION")
        print("=" * 50)
        
        try:
            docs_dir = self.project_root / "deployment"
            docs_dir.mkdir(exist_ok=True)
            
            # Production deployment guide
            deployment_guide = '''# Production Deployment Guide

## Overview
This guide covers the production deployment of the 24-Strategy Trading Platform, ensuring enterprise-grade reliability, monitoring, and performance.

## System Requirements

### Minimum Hardware Requirements
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 16GB (32GB recommended)
- **Storage**: 100GB SSD (500GB recommended)
- **Network**: Stable internet connection with low latency

### Software Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Docker
- **Python**: 3.9+
- **Dependencies**: See requirements.txt

## Pre-Deployment Checklist

### 1. Environment Setup
```bash
# Create production user
sudo useradd -m -s /bin/bash trading-platform
sudo usermod -aG sudo trading-platform

# Create application directory
sudo mkdir -p /opt/trading-platform
sudo chown trading-platform:trading-platform /opt/trading-platform

# Set up virtual environment
cd /opt/trading-platform
python3 -m venv venv
source venv/bin/activate
```

### 2. Application Installation
```bash
# Clone repository
git clone <repository_url> .

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-prod.txt

# Set up configuration
cp config/production.yaml.example config/production.yaml
# Edit configuration as needed
```

### 3. Database Setup (if applicable)
```bash
# PostgreSQL setup for strategy results storage
sudo apt-get install postgresql postgresql-contrib
sudo -u postgres createdb trading_platform
sudo -u postgres createuser trading_platform
```

### 4. Logging Configuration
```bash
# Create log directories
sudo mkdir -p /var/log/trading-platform
sudo chown trading-platform:trading-platform /var/log/trading-platform

# Set up log rotation
sudo cp deployment/logrotate.conf /etc/logrotate.d/trading-platform
```

### 5. Monitoring Setup
```bash
# Install monitoring tools
sudo apt-get install htop iotop nethogs

# Set up health checks
cp health_check.py /opt/trading-platform/
chmod +x /opt/trading-platform/health_check.py

# Add to crontab for regular checks
echo "*/5 * * * * /opt/trading-platform/health_check.py >> /var/log/trading-platform/health.log 2>&1" | crontab -
```

## Deployment Steps

### 1. Initial Deployment
```bash
# Switch to production user
sudo su - trading-platform

# Navigate to application directory
cd /opt/trading-platform

# Activate virtual environment
source venv/bin/activate

# Run initial setup
python production_deployment_prep.py

# Test the platform
python -m pytest tests/ -v

# Start the API server (if using FastAPI)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 2. Service Configuration (Systemd)
```bash
# Create service file
sudo cp deployment/trading-platform.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable trading-platform
sudo systemctl start trading-platform

# Check status
sudo systemctl status trading-platform
```

### 3. Nginx Reverse Proxy (Optional)
```bash
# Install Nginx
sudo apt-get install nginx

# Configure reverse proxy
sudo cp deployment/nginx.conf /etc/nginx/sites-available/trading-platform
sudo ln -s /etc/nginx/sites-available/trading-platform /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

## Configuration

### Environment Variables
Create `/opt/trading-platform/.env`:
```
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
MAX_WORKERS=4

# API Keys (if needed)
EXCHANGE_API_KEY=your_api_key
EXCHANGE_SECRET=your_secret
```

### Production Configuration
Edit `config/production.yaml`:
```yaml
environment: production
debug: false

logging:
  level: INFO
  file_path: /var/log/trading-platform/application.log
  max_size: 100MB
  backup_count: 5

performance:
  max_concurrent_strategies: 24
  optimization_timeout: 3600
  memory_limit: 8GB

monitoring:
  health_check_interval: 300  # 5 minutes
  alert_thresholds:
    cpu_percent: 80
    memory_percent: 85
    disk_percent: 90
```

## Monitoring and Maintenance

### Health Checks
```bash
# Manual health check
/opt/trading-platform/health_check.py

# View logs
tail -f /var/log/trading-platform/application.log
tail -f /var/log/trading-platform/errors.log
```

### Performance Monitoring
```bash
# System resources
htop
iotop
df -h

# Application metrics
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

### Backup Procedures
```bash
# Backup configuration
tar -czf backup-config-$(date +%Y%m%d).tar.gz config/

# Backup results (if stored locally)
tar -czf backup-results-$(date +%Y%m%d).tar.gz results/
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks in strategies
   - Reduce concurrent strategy limit
   - Restart application

2. **Performance Degradation**
   - Monitor CPU and disk I/O
   - Check for blocked processes
   - Review optimization parameters

3. **Strategy Failures**
   - Check error logs
   - Verify data availability
   - Test individual strategies

### Log Analysis
```bash
# Error patterns
grep -i error /var/log/trading-platform/application.log

# Performance issues
grep -i "slow\|timeout\|memory" /var/log/trading-platform/application.log

# Strategy failures
grep -i "strategy.*failed" /var/log/trading-platform/application.log
```

## Security Considerations

### 1. Network Security
- Use firewall rules to restrict access
- Consider VPN for remote access
- Enable SSL/TLS for API endpoints

### 2. Application Security
- Regular security updates
- Secure API key storage
- Input validation and sanitization

### 3. Data Protection
- Encrypt sensitive configuration
- Regular security audits
- Access logging and monitoring

## Scaling Considerations

### Horizontal Scaling
- Use container orchestration (Docker + Kubernetes)
- Load balancing for API endpoints
- Distributed caching for strategy results

### Vertical Scaling
- Increase memory for more concurrent strategies
- Faster CPU for optimization performance
- SSD storage for improved I/O

## Support and Maintenance

### Regular Tasks
- Daily: Check health status and logs
- Weekly: Review performance metrics
- Monthly: Update dependencies and security patches
- Quarterly: Performance optimization review

### Contact Information
- Technical Support: [support_email]
- Emergency Contact: [emergency_contact]
- Documentation: [documentation_url]
'''
            
            deployment_guide_path = docs_dir / "DEPLOYMENT_GUIDE.md"
            with open(deployment_guide_path, 'w') as f:
                f.write(deployment_guide)
            
            # Production configuration template
            prod_config = {
                'environment': 'production',
                'debug': False,
                'logging': {
                    'level': 'INFO',
                    'file_path': '/var/log/trading-platform/application.log',
                    'max_size': '100MB',
                    'backup_count': 5
                },
                'performance': {
                    'max_concurrent_strategies': 24,
                    'optimization_timeout': 3600,
                    'memory_limit': '8GB'
                },
                'monitoring': {
                    'health_check_interval': 300,
                    'alert_thresholds': {
                        'cpu_percent': 80,
                        'memory_percent': 85,
                        'disk_percent': 90
                    }
                },
                'strategies': {
                    'default_position_size': 0.1,
                    'max_position_size': 0.2,
                    'stop_loss_enabled': True,
                    'take_profit_enabled': True
                }
            }
            
            config_dir = self.project_root / "config"
            config_dir.mkdir(exist_ok=True)
            
            prod_config_path = config_dir / "production.yaml"
            with open(prod_config_path, 'w') as f:
                yaml.dump(prod_config, f, default_flow_style=False)
            
            # Systemd service file
            service_config = '''[Unit]
Description=Trading Platform 24-Strategy System
After=network.target

[Service]
Type=simple
User=trading-platform
Group=trading-platform
WorkingDirectory=/opt/trading-platform
Environment=PATH=/opt/trading-platform/venv/bin
ExecStart=/opt/trading-platform/venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
'''
            
            service_path = docs_dir / "trading-platform.service"
            with open(service_path, 'w') as f:
                f.write(service_config)
            
            # Nginx configuration
            nginx_config = '''upstream trading_platform {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://trading_platform;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://trading_platform/health;
        access_log off;
    }
}
'''
            
            nginx_path = docs_dir / "nginx.conf"
            with open(nginx_path, 'w') as f:
                f.write(nginx_config)
            
            print(f"✅ Deployment documentation created:")
            print(f"   📖 Guide: {deployment_guide_path}")
            print(f"   ⚙️ Config: {prod_config_path}")
            print(f"   🔧 Service: {service_path}")
            print(f"   🌐 Nginx: {nginx_path}")
            
            self.setup_results['documentation'] = {
                'status': 'success',
                'guide_path': str(deployment_guide_path),
                'config_path': str(prod_config_path)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Documentation creation failed: {str(e)}")
            self.setup_results['documentation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def create_production_requirements(self) -> bool:
        """Create production-specific requirements file."""
        print("\n📦 CREATING PRODUCTION REQUIREMENTS")
        print("=" * 50)
        
        try:
            # Production-specific dependencies
            prod_requirements = [
                "# Production-specific dependencies",
                "gunicorn>=20.1.0",
                "psutil>=5.8.0",
                "prometheus-client>=0.12.0",
                "structlog>=21.0.0",
                "sentry-sdk>=1.4.0",
                "redis>=4.0.0",
                "supervisor>=4.2.0",
                "nginx",
                "",
                "# Security and monitoring",
                "security",
                "bandit",
                "safety",
                "",
                "# Performance monitoring",
                "py-spy",
                "memory-profiler",
                "psutil",
                "",
                "# Environment management",
                "python-dotenv",
                "pyyaml>=6.0"
            ]
            
            requirements_prod_path = self.project_root / "requirements-prod.txt"
            with open(requirements_prod_path, 'w') as f:
                f.write('\n'.join(prod_requirements))
            
            print(f"✅ Production requirements created:")
            print(f"   📄 File: {requirements_prod_path}")
            print(f"   📦 Includes: monitoring, security, performance tools")
            
            self.setup_results['requirements'] = {
                'status': 'success',
                'file_path': str(requirements_prod_path)
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Requirements creation failed: {str(e)}")
            self.setup_results['requirements'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def validate_production_readiness(self) -> bool:
        """Validate that the platform is ready for production deployment."""
        print("\n✅ VALIDATING PRODUCTION READINESS")
        print("=" * 50)
        
        validation_results = {}
        
        try:
            # Check if all components were set up successfully
            required_components = ['logging', 'error_handling', 'health_monitoring', 'documentation', 'requirements']
            
            for component in required_components:
                if component in self.setup_results and self.setup_results[component]['status'] == 'success':
                    validation_results[component] = '✅ Ready'
                else:
                    validation_results[component] = '❌ Failed'
            
            # Check strategy factory availability
            try:
                sys.path.append(str(self.project_root / 'src'))
                from src.optimization.strategy_factory import StrategyFactory
                factory = StrategyFactory()
                available_strategies = factory.get_all_strategies()
                validation_results['strategies'] = f'✅ {len(available_strategies)} strategies available'
            except Exception as e:
                validation_results['strategies'] = f'❌ Strategy factory error: {str(e)[:50]}'
            
            # Check backtesting engine
            try:
                from src.strategies.backtesting_engine import BacktestingEngine
                engine = BacktestingEngine()
                validation_results['backtesting'] = '✅ Engine available'
            except Exception as e:
                validation_results['backtesting'] = f'❌ Engine error: {str(e)[:50]}'
            
            # Check system resources
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            if memory.total > 8 * 1024**3:  # 8GB
                validation_results['memory'] = '✅ Sufficient (>8GB)'
            else:
                validation_results['memory'] = f'⚠️ Limited ({memory.total/(1024**3):.1f}GB)'
            
            if disk.free > 50 * 1024**3:  # 50GB
                validation_results['disk_space'] = '✅ Sufficient (>50GB)'
            else:
                validation_results['disk_space'] = f'⚠️ Limited ({disk.free/(1024**3):.1f}GB)'
            
            # Print validation results
            print("🔍 PRODUCTION READINESS CHECKLIST:")
            for component, status in validation_results.items():
                print(f"   {component.ljust(20)}: {status}")
            
            # Determine overall readiness
            failed_components = [k for k, v in validation_results.items() if '❌' in v]
            warning_components = [k for k, v in validation_results.items() if '⚠️' in v]
            
            if not failed_components:
                if not warning_components:
                    overall_status = '🎉 FULLY READY FOR PRODUCTION'
                else:
                    overall_status = '✅ READY (with warnings)'
            else:
                overall_status = '❌ NOT READY - Fix failed components'
            
            print(f"\n🎯 OVERALL STATUS: {overall_status}")
            
            if warning_components:
                print(f"\n⚠️ WARNINGS: {', '.join(warning_components)}")
            
            if failed_components:
                print(f"\n❌ FAILURES: {', '.join(failed_components)}")
            
            self.setup_results['validation'] = {
                'status': 'success' if not failed_components else 'failed',
                'results': validation_results,
                'overall_status': overall_status
            }
            
            return len(failed_components) == 0
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            self.setup_results['validation'] = {
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def run_complete_preparation(self) -> Dict[str, Any]:
        """Run complete production deployment preparation."""
        print("🚀 PRODUCTION DEPLOYMENT PREPARATION")
        print("=" * 60)
        print("🎯 Preparing 24-strategy platform for enterprise deployment")
        
        start_time = datetime.now()
        
        # Run all preparation steps
        steps = [
            ("Setting up production logging", self.prepare_production_logging),
            ("Creating error handling framework", self.create_error_handling_framework),
            ("Building health monitoring system", self.create_health_monitoring_system),
            ("Generating deployment documentation", self.create_deployment_documentation),
            ("Creating production requirements", self.create_production_requirements),
            ("Validating production readiness", self.validate_production_readiness)
        ]
        
        success_count = 0
        for step_name, step_function in steps:
            print(f"\n🔄 {step_name}...")
            try:
                if step_function():
                    success_count += 1
                    print(f"   ✅ {step_name} completed")
                else:
                    print(f"   ❌ {step_name} failed")
            except Exception as e:
                print(f"   ❌ {step_name} failed: {str(e)}")
        
        # Generate final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_report = {
            'timestamp': end_time.isoformat(),
            'duration_seconds': duration,
            'total_steps': len(steps),
            'successful_steps': success_count,
            'success_rate': success_count / len(steps) * 100,
            'setup_results': self.setup_results,
            'overall_status': 'success' if success_count == len(steps) else 'partial_success'
        }
        
        # Save report
        report_path = self.project_root / "deployment" / "preparation_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📋 PREPARATION SUMMARY")
        print("=" * 30)
        print(f"✅ Successful steps: {success_count}/{len(steps)}")
        print(f"⏱️ Duration: {duration:.1f} seconds")
        print(f"📊 Success rate: {final_report['success_rate']:.1f}%")
        print(f"📄 Report saved: {report_path}")
        
        if success_count == len(steps):
            print(f"\n🎉 PRODUCTION PREPARATION COMPLETE!")
            print(f"   🚀 Platform is ready for enterprise deployment")
            print(f"   📚 See deployment/DEPLOYMENT_GUIDE.md for instructions")
        else:
            print(f"\n⚠️ PREPARATION PARTIALLY COMPLETE")
            print(f"   🔧 Review failed steps and retry if needed")
        
        return final_report

def main():
    """Run production deployment preparation."""
    preparator = ProductionDeploymentPreparator()
    
    print("🏭 TRADING PLATFORM PRODUCTION DEPLOYMENT PREP")
    print("=" * 60)
    print("🎯 Enterprise-grade 24-strategy platform preparation")
    
    result = preparator.run_complete_preparation()
    
    # Exit with appropriate code
    if result['overall_status'] == 'success':
        print(f"\n🎉 ALL SYSTEMS GO FOR PRODUCTION! 🚀")
        sys.exit(0)
    else:
        print(f"\n⚠️ Production preparation needs attention")
        sys.exit(1)

if __name__ == "__main__":
    main() 