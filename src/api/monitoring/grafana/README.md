# Trading Strategy Optimization API - Monitoring Stack

This directory contains a complete monitoring solution for the Trading Strategy Optimization API using **Grafana** and **Prometheus**.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Trading Strategy Optimization API running on port 8000

### Setup
```bash
# Navigate to the monitoring directory
cd src/api/monitoring/grafana

# Run the setup script
./setup-monitoring.sh
```

### Access Points
- **Grafana Dashboard**: http://localhost:3001
  - Username: `admin`
  - Password: `trading_api_2024`
- **Prometheus**: http://localhost:9090
- **Node Exporter**: http://localhost:9100
- **API Dashboard**: http://localhost:8000/api/v1/monitoring/ (requires API key: `dev_key_123`)

**Note**: This setup works perfectly with **OrbStack**, the fast and lightweight Docker alternative for macOS. OrbStack's excellent networking and resource efficiency make it ideal for running monitoring stacks.

## 📊 Available Dashboards

### Trading Strategy Optimization API - Overview
A comprehensive dashboard featuring:

#### API Performance Metrics
- **Request Rate**: Real-time API request rates by endpoint and method
- **Response Time**: 95th and 50th percentile response times
- **Error Rate**: Percentage of failed requests
- **Requests per Minute**: Current throughput

#### System Resources
- **CPU Usage**: Real-time CPU utilization
- **Memory Usage**: Memory consumption and availability
- **Disk Usage**: Storage utilization

#### Optimization Jobs
- **Queue Size**: Number of pending optimization jobs
- **Active Jobs**: Currently running optimizations
- **Job Success Rate**: Optimization completion statistics

#### Health Monitoring
- **API Health**: Service availability status
- **Component Status**: Individual service health checks

## 🔧 Configuration Files

### Docker Compose (`docker-compose.yml`)
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Node Exporter**: System metrics collection

### Prometheus Configuration (`prometheus.yml`)
- Scrapes API metrics from `/metrics` endpoint every 30s
- Monitors system metrics via Node Exporter
- 200-hour data retention

### Grafana Provisioning
- **Datasources**: Auto-configured Prometheus connection
- **Dashboards**: Pre-loaded Trading API dashboard
- **Plugins**: Clock panel and JSON datasource

## 📈 Metrics Collected

### API Metrics (from `/metrics` endpoint)
```
# Request metrics
api_requests_total{method, endpoint, status_code}
api_request_duration_seconds_bucket{method, endpoint}

# System metrics
system_cpu_usage_percent
system_memory_usage_percent
system_disk_usage_percent

# Optimization metrics
optimization_queue_size
optimization_jobs_active
optimization_jobs_completed_total
optimization_jobs_failed_total

# Export metrics
export_operations_total{type}
export_operation_duration_seconds{type}
```

### System Metrics (from Node Exporter)
- CPU, memory, disk, network usage
- Process and file descriptor counts
- System load averages

## 🛠️ Management Commands

### Start/Stop Services
```bash
# Start monitoring stack
docker-compose up -d

# Stop monitoring stack
docker-compose down

# Restart services
docker-compose restart

# View logs
docker-compose logs -f

# Update images
docker-compose pull && docker-compose up -d
```

### Data Management
```bash
# Remove all data (including volumes)
docker-compose down -v

# Backup Grafana data
docker run --rm -v trading-api-grafana_data:/data -v $(pwd):/backup alpine tar czf /backup/grafana-backup.tar.gz -C /data .

# Restore Grafana data
docker run --rm -v trading-api-grafana_data:/data -v $(pwd):/backup alpine tar xzf /backup/grafana-backup.tar.gz -C /data
```

## 🔍 Troubleshooting

### Common Issues

#### Prometheus Can't Reach API
- Ensure API is running on port 8000
- Check Docker network connectivity
- Verify `host.docker.internal` resolves correctly

#### Grafana Dashboard Not Loading
- Check Prometheus datasource configuration
- Verify metrics are being collected: http://localhost:9090/targets
- Ensure dashboard JSON is valid

#### No Data in Panels
- Confirm API is generating metrics: http://localhost:8000/metrics
- Check Prometheus query syntax
- Verify time range in Grafana

### Logs and Debugging
```bash
# Check container status
docker-compose ps

# View specific service logs
docker-compose logs prometheus
docker-compose logs grafana
docker-compose logs node-exporter

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test API metrics endpoint
curl http://localhost:8000/metrics
```

## 🔐 Security Considerations

### Production Deployment
- Change default Grafana admin password
- Configure proper authentication
- Use HTTPS for external access
- Restrict network access to monitoring ports
- Set up proper backup procedures

### API Key Management
- Use environment variables for API keys
- Rotate keys regularly
- Monitor API key usage

## 📝 Customization

### Adding New Dashboards
1. Create dashboard in Grafana UI
2. Export JSON configuration
3. Place in `grafana/dashboards/` directory
4. Restart Grafana container

### Custom Metrics
1. Add metrics to API code using Prometheus client
2. Update Prometheus configuration if needed
3. Create new dashboard panels

### Alerting
1. Configure notification channels in Grafana
2. Set up alert rules on dashboard panels
3. Test alert delivery

## 🔗 Related Documentation

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Trading Strategy API Documentation](http://localhost:8000/api/docs)
- [API Monitoring Dashboard](http://localhost:8000/api/v1/monitoring/)

## 📞 Support

For issues with the monitoring setup:
1. Check the troubleshooting section above
2. Review container logs
3. Verify API is running and accessible
4. Check Docker and Docker Compose versions 