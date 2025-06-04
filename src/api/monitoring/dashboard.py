"""
Performance Monitoring Dashboard for Trading Strategy Optimization API

This module provides dashboard capabilities including:
- Real-time metrics visualization
- System health monitoring
- Performance analytics
- Grafana integration
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from .metrics import get_metrics_collector
from .health import get_health_checker

logger = logging.getLogger(__name__)

# Create dashboard router
dashboard_router = APIRouter()


class DashboardManager:
    """
    Manages monitoring dashboard functionality including data collection,
    visualization, and Grafana integration.
    """
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.health_checker = get_health_checker()
        
        # Dashboard configuration
        self.refresh_interval = 30  # seconds
        self.data_retention_hours = 24
        
        # Historical data storage (in-memory for simplicity)
        self.historical_data = {
            "timestamps": [],
            "api_requests": [],
            "response_times": [],
            "error_rates": [],
            "cpu_usage": [],
            "memory_usage": [],
            "active_jobs": [],
            "queue_size": []
        }
        
        logger.info("DashboardManager initialized")
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics for dashboard display."""
        
        try:
            # Get health status
            health_status = self.health_checker.get_overall_health()
            
            # Get metrics from collector
            current_time = datetime.now()
            
            # Basic system metrics
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # API metrics (simplified - in production you'd query Prometheus)
            api_metrics = {
                "total_requests": getattr(self.metrics_collector, '_request_count', 0),
                "error_count": getattr(self.metrics_collector, '_error_count', 0),
                "avg_response_time": getattr(self.metrics_collector, '_avg_response_time', 0.0),
                "requests_per_minute": getattr(self.metrics_collector, '_requests_per_minute', 0)
            }
            
            # Job metrics
            job_metrics = {
                "active_jobs": getattr(self.metrics_collector, '_active_jobs', 0),
                "queue_size": getattr(self.metrics_collector, '_queue_size', 0),
                "completed_jobs": getattr(self.metrics_collector, '_completed_jobs', 0),
                "failed_jobs": getattr(self.metrics_collector, '_failed_jobs', 0)
            }
            
            return {
                "timestamp": current_time.isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_used_gb": disk.used / (1024**3),
                    "disk_total_gb": disk.total / (1024**3)
                },
                "api": api_metrics,
                "jobs": job_metrics,
                "health": {
                    "status": health_status.get("status", "unknown"),
                    "components": health_status.get("components", {}),
                    "uptime": health_status.get("uptime_seconds", 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "system": {},
                "api": {},
                "jobs": {},
                "health": {"status": "error"}
            }
    
    def update_historical_data(self, current_metrics: Dict[str, Any]):
        """Update historical data for trend analysis."""
        
        try:
            # Add current timestamp
            self.historical_data["timestamps"].append(current_metrics["timestamp"])
            
            # Add system metrics
            self.historical_data["cpu_usage"].append(
                current_metrics.get("system", {}).get("cpu_percent", 0)
            )
            self.historical_data["memory_usage"].append(
                current_metrics.get("system", {}).get("memory_percent", 0)
            )
            
            # Add API metrics
            api_data = current_metrics.get("api", {})
            self.historical_data["api_requests"].append(api_data.get("total_requests", 0))
            self.historical_data["response_times"].append(api_data.get("avg_response_time", 0))
            self.historical_data["error_rates"].append(api_data.get("error_count", 0))
            
            # Add job metrics
            job_data = current_metrics.get("jobs", {})
            self.historical_data["active_jobs"].append(job_data.get("active_jobs", 0))
            self.historical_data["queue_size"].append(job_data.get("queue_size", 0))
            
            # Trim old data (keep last 24 hours worth)
            max_points = int((self.data_retention_hours * 3600) / self.refresh_interval)
            for key in self.historical_data:
                if len(self.historical_data[key]) > max_points:
                    self.historical_data[key] = self.historical_data[key][-max_points:]
                    
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data including current and historical metrics."""
        
        current_metrics = self.collect_current_metrics()
        self.update_historical_data(current_metrics)
        
        return {
            "current": current_metrics,
            "historical": self.historical_data,
            "config": {
                "refresh_interval": self.refresh_interval,
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def generate_grafana_config(self) -> Dict[str, Any]:
        """Generate Grafana dashboard configuration."""
        
        return {
            "dashboard": {
                "id": None,
                "title": "Trading Strategy Optimization API",
                "tags": ["trading", "optimization", "api"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "API Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(api_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, api_request_duration_seconds_bucket)",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, api_request_duration_seconds_bucket)",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "System Resources",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "system_cpu_usage_percent",
                                "legendFormat": "CPU %"
                            },
                            {
                                "expr": "system_memory_usage_percent",
                                "legendFormat": "Memory %"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Optimization Jobs",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "optimization_queue_size",
                                "legendFormat": "Queue Size"
                            },
                            {
                                "expr": "optimization_jobs_active",
                                "legendFormat": "Active Jobs"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Error Rate",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "rate(api_requests_total{status_code!~\"2..\"}[5m])",
                                "legendFormat": "Error Rate"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
                    },
                    {
                        "id": 6,
                        "title": "Active Jobs",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "optimization_jobs_active",
                                "legendFormat": "Active Jobs"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
                    }
                ]
            }
        }


# Global dashboard manager instance
dashboard_manager = DashboardManager()


@dashboard_router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """
    Main dashboard page with real-time monitoring.
    """
    try:
        dashboard_data = dashboard_manager.get_dashboard_data()
        
        # Create simple HTML dashboard
        html_content = _generate_dashboard_html(dashboard_data)
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")


@dashboard_router.get("/api/data")
async def dashboard_api_data():
    """
    API endpoint for dashboard data (for AJAX updates).
    """
    try:
        return JSONResponse(content=dashboard_manager.get_dashboard_data())
    except Exception as e:
        logger.error(f"Dashboard API error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard API error: {str(e)}")


@dashboard_router.get("/grafana/config")
async def grafana_config():
    """
    Get Grafana dashboard configuration.
    """
    try:
        return JSONResponse(content=dashboard_manager.generate_grafana_config())
    except Exception as e:
        logger.error(f"Grafana config error: {e}")
        raise HTTPException(status_code=500, detail=f"Grafana config error: {str(e)}")


def _generate_dashboard_html(dashboard_data: Dict[str, Any]) -> str:
    """Generate simple HTML dashboard."""
    
    current = dashboard_data.get("current", {})
    system = current.get("system", {})
    api = current.get("api", {})
    jobs = current.get("jobs", {})
    health = current.get("health", {})
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading Strategy Optimization API - Monitoring Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .metric-card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }}
            .metric-title {{
                font-size: 18px;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
                margin-bottom: 5px;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .status-healthy {{ color: #28a745; }}
            .status-warning {{ color: #ffc107; }}
            .status-critical {{ color: #dc3545; }}
            .refresh-info {{
                text-align: center;
                color: #666;
                margin-top: 20px;
                font-size: 14px;
            }}
            .progress-bar {{
                width: 100%;
                height: 8px;
                background-color: #e9ecef;
                border-radius: 4px;
                overflow: hidden;
                margin-top: 10px;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #28a745, #ffc107, #dc3545);
                transition: width 0.3s ease;
            }}
        </style>
        <script>
            function refreshDashboard() {{
                fetch('/api/v1/monitoring/api/data')
                    .then(response => response.json())
                    .then(data => {{
                        // Update timestamp
                        document.getElementById('last-updated').textContent = 
                            new Date(data.current.timestamp).toLocaleString();
                        
                        // Update metrics (simplified - in production you'd update all values)
                        console.log('Dashboard data updated:', data);
                    }})
                    .catch(error => console.error('Error refreshing dashboard:', error));
            }}
            
            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        </script>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Trading Strategy Optimization API</h1>
            <h2>Monitoring Dashboard</h2>
            <p>Real-time system performance and health monitoring</p>
        </div>
        
        <div class="metrics-grid">
            <!-- System Health -->
            <div class="metric-card">
                <div class="metric-title">🏥 System Health</div>
                <div class="metric-value status-{health.get('status', 'unknown').lower()}">
                    {health.get('status', 'Unknown').upper()}
                </div>
                <div class="metric-label">Overall system status</div>
                <div class="metric-label">Uptime: {health.get('uptime', 0):.0f} seconds</div>
            </div>
            
            <!-- CPU Usage -->
            <div class="metric-card">
                <div class="metric-title">💻 CPU Usage</div>
                <div class="metric-value">{system.get('cpu_percent', 0):.1f}%</div>
                <div class="metric-label">Current CPU utilization</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {system.get('cpu_percent', 0)}%"></div>
                </div>
            </div>
            
            <!-- Memory Usage -->
            <div class="metric-card">
                <div class="metric-title">🧠 Memory Usage</div>
                <div class="metric-value">{system.get('memory_percent', 0):.1f}%</div>
                <div class="metric-label">
                    {system.get('memory_used_gb', 0):.1f}GB / {system.get('memory_total_gb', 0):.1f}GB
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {system.get('memory_percent', 0)}%"></div>
                </div>
            </div>
            
            <!-- API Requests -->
            <div class="metric-card">
                <div class="metric-title">📊 API Requests</div>
                <div class="metric-value">{api.get('total_requests', 0)}</div>
                <div class="metric-label">Total requests processed</div>
                <div class="metric-label">Errors: {api.get('error_count', 0)}</div>
            </div>
            
            <!-- Response Time -->
            <div class="metric-card">
                <div class="metric-title">⚡ Response Time</div>
                <div class="metric-value">{api.get('avg_response_time', 0):.3f}s</div>
                <div class="metric-label">Average response time</div>
                <div class="metric-label">Requests/min: {api.get('requests_per_minute', 0)}</div>
            </div>
            
            <!-- Active Jobs -->
            <div class="metric-card">
                <div class="metric-title">🔄 Optimization Jobs</div>
                <div class="metric-value">{jobs.get('active_jobs', 0)}</div>
                <div class="metric-label">Currently active jobs</div>
                <div class="metric-label">Queue: {jobs.get('queue_size', 0)} | Completed: {jobs.get('completed_jobs', 0)}</div>
            </div>
            
            <!-- Disk Usage -->
            <div class="metric-card">
                <div class="metric-title">💾 Disk Usage</div>
                <div class="metric-value">{system.get('disk_percent', 0):.1f}%</div>
                <div class="metric-label">
                    {system.get('disk_used_gb', 0):.1f}GB / {system.get('disk_total_gb', 0):.1f}GB
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {system.get('disk_percent', 0)}%"></div>
                </div>
            </div>
            
            <!-- Component Status -->
            <div class="metric-card">
                <div class="metric-title">🔧 Components</div>
                <div style="font-size: 14px;">
                    {"<br>".join([f"{comp}: {status}" for comp, status in health.get('components', {}).items()])}
                </div>
            </div>
        </div>
        
        <div class="refresh-info">
            Last updated: <span id="last-updated">{current.get('timestamp', 'Unknown')}</span><br>
            Auto-refresh every 30 seconds | 
            <a href="/metrics" target="_blank">Raw Metrics</a> | 
            <a href="/api/v1/monitoring/grafana/config" target="_blank">Grafana Config</a>
        </div>
    </body>
    </html>
    """
    
    return html 