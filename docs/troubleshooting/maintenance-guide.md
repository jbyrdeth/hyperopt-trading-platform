# System Maintenance Guide

## Daily Maintenance Tasks

### Automated Health Checks
```bash
# Add to crontab for daily execution
0 9 * * * /opt/hyperopt/scripts/daily_health_check.sh

# daily_health_check.sh
#!/bin/bash
echo "📊 Daily Health Check - $(date)"

# 1. API Performance Check
response_time=$(curl -w "%{time_total}" -s -o /dev/null http://localhost:8000/health)
echo "API Response Time: ${response_time}s (target: <0.2s)"

# 2. Database Performance
psql -d hyperopt -c "SELECT COUNT(*) as active_strategies FROM strategies WHERE status='active';"

# 3. Memory Usage
free -h | grep Mem

# 4. Disk Space
df -h | grep -E "/$|/opt"

# 5. Recent Error Count
journalctl --since="24 hours ago" --unit=hyperopt-api | grep -c ERROR
```

### Data Quality Monitoring
```python
# data_quality_check.py
import pandas as pd
from datetime import datetime, timedelta

def daily_data_quality_check():
    """Validate data quality for all active trading pairs"""
    
    active_pairs = get_active_trading_pairs()
    quality_report = {}
    
    for pair in active_pairs:
        # Get last 24 hours of data
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        
        data = fetch_market_data(pair, start_time, end_time)
        
        quality_metrics = {
            'total_points': len(data),
            'missing_points': calculate_missing_points(data, start_time, end_time),
            'data_gaps': find_data_gaps(data),
            'price_anomalies': detect_price_anomalies(data),
            'volume_anomalies': detect_volume_anomalies(data)
        }
        
        quality_report[pair] = quality_metrics
        
        # Alert on quality issues
        if quality_metrics['missing_points'] > 100:
            send_alert(f"High missing data count for {pair}: {quality_metrics['missing_points']}")
    
    return quality_report

# Schedule daily execution
if __name__ == "__main__":
    report = daily_data_quality_check()
    save_quality_report(report)
```

---

## Weekly Maintenance Tasks

### Database Maintenance
```sql
-- weekly_db_maintenance.sql
-- Run every Sunday at 2 AM

-- 1. Update table statistics
ANALYZE;

-- 2. Vacuum tables to reclaim space
VACUUM (ANALYZE, VERBOSE);

-- 3. Reindex performance-critical tables
REINDEX INDEX idx_strategies_symbol;
REINDEX INDEX idx_results_timestamp;
REINDEX INDEX idx_trades_strategy_id;

-- 4. Check database size and growth
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 5. Archive old data (older than 1 year)
DELETE FROM optimization_results 
WHERE created_at < NOW() - INTERVAL '1 year';

DELETE FROM backtest_trades 
WHERE timestamp < NOW() - INTERVAL '1 year';
```

### Performance Optimization
```python
# weekly_performance_optimization.py

def optimize_redis_cache():
    """Weekly Redis optimization and cleanup"""
    
    # Get cache statistics
    info = redis_client.info('memory')
    used_memory_mb = info['used_memory'] / 1024 / 1024
    
    print(f"Redis memory usage: {used_memory_mb:.1f} MB")
    
    # Clean expired keys
    redis_client.execute_command('MEMORY PURGE')
    
    # Optimize memory fragmentation if needed
    if info.get('mem_fragmentation_ratio', 1) > 1.5:
        redis_client.execute_command('MEMORY DEFRAG')
        print("Redis memory defragmented")
    
    # Update cache configuration
    redis_client.config_set('maxmemory-policy', 'allkeys-lru')
    redis_client.config_set('save', '900 1 300 10 60 10000')

def optimize_api_performance():
    """Weekly API performance tuning"""
    
    # Check API response times over the week
    logs = get_api_logs(days=7)
    
    performance_metrics = {
        'avg_response_time': calculate_avg_response_time(logs),
        'p95_response_time': calculate_percentile_response_time(logs, 95),
        'error_rate': calculate_error_rate(logs),
        'requests_per_hour': calculate_requests_per_hour(logs)
    }
    
    print("Weekly API Performance:")
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value}")
    
    # Adjust worker configuration if needed
    if performance_metrics['avg_response_time'] > 0.3:  # >300ms
        print("⚠️  High response times detected - consider scaling")
    
    return performance_metrics
```

---

## Monthly Maintenance Tasks

### Comprehensive System Backup
```bash
#!/bin/bash
# monthly_backup.sh

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/monthly/$BACKUP_DATE"
mkdir -p "$BACKUP_DIR"

echo "🗄️  Starting monthly backup - $BACKUP_DATE"

# 1. Database backup
echo "Backing up database..."
pg_dump hyperopt | gzip > "$BACKUP_DIR/database.sql.gz"

# 2. Strategy files backup
echo "Backing up strategies..."
tar -czf "$BACKUP_DIR/strategies.tar.gz" strategies/

# 3. Configuration backup
echo "Backing up configuration..."
cp .env "$BACKUP_DIR/env.backup"
cp docker-compose.yml "$BACKUP_DIR/"
cp -r config/ "$BACKUP_DIR/"

# 4. Documentation backup
echo "Backing up documentation..."
tar -czf "$BACKUP_DIR/docs.tar.gz" docs/

# 5. Logs backup (last 30 days)
echo "Backing up logs..."
journalctl --since="30 days ago" --unit=hyperopt-* > "$BACKUP_DIR/system.log"

# 6. Performance data backup
echo "Backing up performance data..."
python scripts/export_performance_data.py --output "$BACKUP_DIR/performance_data.json"

# 7. Verify backup integrity
echo "Verifying backup integrity..."
if gunzip -t "$BACKUP_DIR/database.sql.gz" && tar -tf "$BACKUP_DIR/strategies.tar.gz" > /dev/null; then
    echo "✅ Backup verification successful"
    
    # Upload to cloud storage (optional)
    # aws s3 cp "$BACKUP_DIR" s3://hyperopt-backups/$BACKUP_DATE/ --recursive
else
    echo "❌ Backup verification failed"
    exit 1
fi

echo "✅ Monthly backup completed: $BACKUP_DIR"
```

### Security Updates
```python
# monthly_security_check.py

def check_security_updates():
    """Monthly security assessment and updates"""
    
    security_checklist = {
        'api_keys_rotation': check_api_key_age(),
        'ssl_certificate': check_ssl_expiry(),
        'dependency_vulnerabilities': scan_dependencies(),
        'access_logs_review': review_access_logs(),
        'password_strength': audit_user_passwords(),
        'firewall_rules': verify_firewall_config()
    }
    
    issues_found = []
    
    for check, result in security_checklist.items():
        if not result['passed']:
            issues_found.append({
                'check': check,
                'issue': result['issue'],
                'recommendation': result['recommendation']
            })
    
    if issues_found:
        print("🔒 Security Issues Found:")
        for issue in issues_found:
            print(f"  ❌ {issue['check']}: {issue['issue']}")
            print(f"     Recommendation: {issue['recommendation']}")
    else:
        print("✅ All security checks passed")
    
    return security_checklist

def update_dependencies():
    """Update system dependencies safely"""
    
    # 1. Create backup before updates
    os.system('cp requirements.txt requirements.txt.backup')
    
    # 2. Update Python packages
    os.system('pip list --outdated --format=json > outdated_packages.json')
    
    # Review critical packages manually
    critical_packages = ['fastapi', 'sqlalchemy', 'redis', 'numpy', 'pandas']
    
    for package in critical_packages:
        current_version = get_package_version(package)
        latest_version = get_latest_version(package)
        
        if current_version != latest_version:
            print(f"Update available: {package} {current_version} -> {latest_version}")
            # Test update in staging first
            test_update_in_staging(package, latest_version)
```

---

## Quarterly Maintenance Tasks

### Performance Review and Optimization
```python
# quarterly_performance_review.py

def quarterly_performance_analysis():
    """Comprehensive quarterly performance analysis"""
    
    # Analyze 3 months of data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=90)
    
    performance_data = {
        'api_metrics': get_api_performance_metrics(start_date, end_date),
        'optimization_metrics': get_optimization_performance(start_date, end_date),
        'database_metrics': get_database_performance(start_date, end_date),
        'user_activity': get_user_activity_metrics(start_date, end_date),
        'resource_utilization': get_resource_utilization(start_date, end_date)
    }
    
    # Generate performance trends
    trends = analyze_performance_trends(performance_data)
    
    # Capacity planning
    capacity_forecast = forecast_capacity_needs(trends)
    
    report = {
        'period': f"{start_date.date()} to {end_date.date()}",
        'performance_data': performance_data,
        'trends': trends,
        'capacity_forecast': capacity_forecast,
        'recommendations': generate_optimization_recommendations(performance_data)
    }
    
    # Save quarterly report
    save_quarterly_report(report)
    
    return report

def system_upgrade_planning():
    """Plan system upgrades and improvements"""
    
    upgrade_candidates = {
        'python_version': check_python_version_updates(),
        'database_version': check_postgresql_updates(),
        'docker_images': check_docker_image_updates(),
        'os_packages': check_system_package_updates(),
        'hardware_requirements': assess_hardware_needs()
    }
    
    # Prioritize upgrades
    priority_upgrades = prioritize_upgrades(upgrade_candidates)
    
    # Create upgrade schedule
    upgrade_schedule = create_upgrade_schedule(priority_upgrades)
    
    return upgrade_schedule
```

### Disaster Recovery Testing
```bash
#!/bin/bash
# quarterly_dr_test.sh

echo "🔄 Disaster Recovery Test - $(date)"

# 1. Test backup restoration
echo "Testing backup restoration..."
LATEST_BACKUP=$(ls -t /backups/monthly/ | head -1)
TEST_DB="hyperopt_dr_test"

# Create test database
createdb "$TEST_DB"

# Restore from backup
gunzip -c "/backups/monthly/$LATEST_BACKUP/database.sql.gz" | psql "$TEST_DB"

# Verify data integrity
echo "Verifying restored data..."
psql "$TEST_DB" -c "SELECT COUNT(*) FROM strategies;"
psql "$TEST_DB" -c "SELECT COUNT(*) FROM optimization_results;"

# Cleanup test database
dropdb "$TEST_DB"

# 2. Test failover procedures
echo "Testing API failover..."
# Simulate primary API failure
docker stop hyperopt_api_1

# Verify secondary API takes over
sleep 10
curl -f http://localhost:8001/health || echo "❌ Failover test failed"

# Restore primary API
docker start hyperopt_api_1

# 3. Test monitoring and alerting
echo "Testing monitoring system..."
python scripts/test_monitoring_alerts.py

echo "✅ Disaster recovery test completed"
```

---

## Continuous Monitoring Setup

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hyperopt-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

### Custom Metrics Collection
```python
# metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define custom metrics
optimization_duration = Histogram('optimization_duration_seconds', 
                                 'Time spent on strategy optimization')
api_requests_total = Counter('api_requests_total', 
                           'Total API requests', ['method', 'endpoint'])
active_strategies = Gauge('active_strategies_count', 
                         'Number of active strategies')
system_memory_usage = Gauge('system_memory_usage_percent', 
                           'System memory usage percentage')

def collect_business_metrics():
    """Collect business-specific metrics"""
    
    # Strategy performance metrics
    active_count = db.query("SELECT COUNT(*) FROM strategies WHERE status='active'")[0][0]
    active_strategies.set(active_count)
    
    # Optimization success rate
    success_rate = calculate_optimization_success_rate()
    optimization_success_rate.set(success_rate)
    
    # Revenue metrics (if applicable)
    monthly_revenue = calculate_monthly_revenue()
    revenue_gauge.set(monthly_revenue)

# Start metrics server
if __name__ == "__main__":
    start_http_server(8080)
    
    # Collect metrics every 30 seconds
    while True:
        collect_business_metrics()
        time.sleep(30)
```

### Alerting Rules
```yaml
# alerting_rules.yml
groups:
  - name: hyperopt_alerts
    rules:
      - alert: HighAPIResponseTime
        expr: api_response_time_seconds > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API response time detected"
          description: "API response time is {{ $value }}s for 5 minutes"

      - alert: OptimizationFailures
        expr: rate(optimization_failures_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High optimization failure rate"
          description: "Optimization failure rate is {{ $value }} per second"

      - alert: DatabaseConnectionFailure
        expr: database_connections_active == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "No active database connections detected"

      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "System memory usage is {{ $value }}%"
```

---

## Maintenance Calendar

### Daily Tasks (Automated)
- ✅ Health checks and monitoring
- ✅ Data quality validation  
- ✅ Error log review
- ✅ Backup verification

### Weekly Tasks (Semi-automated)
- 🔧 Database maintenance and optimization
- 🔧 Cache cleanup and optimization
- 🔧 Performance metrics review
- 🔧 Security log analysis

### Monthly Tasks (Manual Review Required)
- 📋 Comprehensive backup creation
- 📋 Security updates and patches
- 📋 Capacity planning review
- 📋 User access audit

### Quarterly Tasks (Planning Required)
- 📊 Performance analysis and optimization
- 📊 Disaster recovery testing
- 📊 System upgrade planning
- 📊 Architecture review

---

## Emergency Procedures

### System Recovery Checklist
```bash
# emergency_recovery_checklist.sh

echo "🚨 EMERGENCY RECOVERY PROCEDURE"
echo "================================"

# 1. Assess system status
echo "1. Checking system status..."
systemctl status hyperopt-api
systemctl status postgresql
systemctl status redis
systemctl status nginx

# 2. Check recent logs for errors
echo "2. Checking recent errors..."
journalctl --since="1 hour ago" --priority=err

# 3. Verify data integrity
echo "3. Verifying data integrity..."
psql hyperopt -c "SELECT COUNT(*) FROM strategies;"

# 4. Test API connectivity
echo "4. Testing API connectivity..."
curl -f http://localhost:8000/health

# 5. Check disk space
echo "5. Checking disk space..."
df -h

# 6. Monitor system resources
echo "6. System resources:"
free -h
uptime

echo "Emergency assessment complete. Check output above for issues."
```

### Rollback Procedures
```python
# rollback_procedures.py

def emergency_rollback(target_version):
    """Emergency rollback to previous stable version"""
    
    print(f"🔄 Rolling back to version {target_version}")
    
    # 1. Stop current services
    os.system('docker-compose down')
    
    # 2. Backup current state
    backup_current_state()
    
    # 3. Checkout target version
    os.system(f'git checkout {target_version}')
    
    # 4. Restore database to compatible state
    restore_database_to_version(target_version)
    
    # 5. Start services
    os.system('docker-compose up -d')
    
    # 6. Verify rollback success
    if verify_system_health():
        print("✅ Rollback successful")
        send_alert("System rollback completed successfully")
    else:
        print("❌ Rollback failed - manual intervention required")
        send_critical_alert("Emergency rollback failed")
```

---

## Maintenance Automation

### Cron Jobs Setup
```bash
# Add to /etc/crontab

# Daily health check (9 AM)
0 9 * * * hyperopt /opt/hyperopt/scripts/daily_health_check.sh

# Weekly database maintenance (Sunday 2 AM)
0 2 * * 0 hyperopt /opt/hyperopt/scripts/weekly_db_maintenance.sh

# Monthly backup (1st of month, 1 AM)
0 1 1 * * hyperopt /opt/hyperopt/scripts/monthly_backup.sh

# Quarterly DR test (1st of quarter, 3 AM)
0 3 1 1,4,7,10 * hyperopt /opt/hyperopt/scripts/quarterly_dr_test.sh
```

### Monitoring Integration
```python
# monitoring_integration.py

class MaintenanceMonitor:
    def __init__(self):
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.email_config = load_email_config()
    
    def send_maintenance_notification(self, task, status, details=None):
        """Send maintenance task notifications"""
        
        message = {
            'task': task,
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }
        
        # Send to Slack
        if self.slack_webhook:
            self.send_slack_notification(message)
        
        # Send email for critical tasks
        if status in ['failed', 'critical']:
            self.send_email_alert(message)
        
        # Log to monitoring system
        self.log_maintenance_event(message)
    
    def schedule_maintenance_window(self, start_time, duration, description):
        """Schedule maintenance window with notifications"""
        
        # Notify users of upcoming maintenance
        self.send_maintenance_announcement(start_time, duration, description)
        
        # Set system to maintenance mode
        self.enable_maintenance_mode(start_time)
        
        # Schedule automatic exit from maintenance mode
        self.schedule_maintenance_exit(start_time + duration)
```

---

*This maintenance guide should be reviewed and updated quarterly to ensure all procedures remain current and effective.* 