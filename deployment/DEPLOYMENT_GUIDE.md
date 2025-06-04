# Production Deployment Guide

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
