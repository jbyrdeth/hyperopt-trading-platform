# ⚙️ Configuration Guide

## Complete System Configuration Reference

This guide covers all configuration options for the Trading Strategy Optimization System, including environment variables, API settings, optimization parameters, and deployment configurations.

---

## 🌍 **Environment Variables**

### **Core API Configuration**

```env
# API Server Settings
API_HOST=0.0.0.0                    # Bind address (0.0.0.0 for all interfaces)
API_PORT=8000                       # Server port
API_WORKERS=4                       # Number of worker processes
API_RELOAD=false                    # Auto-reload on code changes (dev only)

# Application Settings
APP_NAME="Trading Strategy Optimizer"
APP_VERSION="1.0.0"
APP_DESCRIPTION="Professional trading strategy optimization platform"
APP_DEBUG=false                     # Debug mode (dev only)
```

### **Security Configuration**

```env
# Authentication
SECRET_KEY="your-super-secret-key-here-change-in-production"
API_KEYS="dev-key-12345,prod-key-67890,enterprise-key-abc123"

# CORS Settings
CORS_ORIGINS="http://localhost:3000,https://your-frontend.com"
CORS_METHODS="GET,POST,PUT,DELETE,OPTIONS"
CORS_HEADERS="*"

# Rate Limiting
RATE_LIMIT_REQUESTS=100             # Requests per window
RATE_LIMIT_WINDOW=60               # Window size in seconds
RATE_LIMIT_ENABLED=true            # Enable/disable rate limiting
```

### **Database Configuration**

```env
# Database Settings
DATABASE_URL="sqlite:///./trading_optimizer.db"
# For PostgreSQL: postgresql://user:password@localhost:5432/trading_optimizer
# For MySQL: mysql://user:password@localhost:3306/trading_optimizer

# Database Pool Settings
DB_POOL_SIZE=10                    # Connection pool size
DB_MAX_OVERFLOW=20                 # Max overflow connections
DB_POOL_TIMEOUT=30                 # Pool timeout in seconds
DB_POOL_RECYCLE=3600              # Connection recycle time
```

### **External Data Sources**

```env
# Market Data APIs
ALPHA_VANTAGE_API_KEY="your-alpha-vantage-key"
POLYGON_API_KEY="your-polygon-key"
BINANCE_API_KEY="your-binance-key"
BINANCE_SECRET_KEY="your-binance-secret"

# Yahoo Finance Settings (no key required)
YAHOO_FINANCE_ENABLED=true
YAHOO_FINANCE_TIMEOUT=30

# Data Cache Settings
DATA_CACHE_TTL=3600               # Cache TTL in seconds
DATA_CACHE_SIZE=1000              # Max cache entries
DATA_CACHE_ENABLED=true           # Enable/disable caching
```

### **Optimization Settings**

```env
# Hyperopt Configuration
HYPEROPT_MAX_EVALS=1000           # Maximum evaluations per optimization
HYPEROPT_TIMEOUT=3600             # Optimization timeout in seconds
HYPEROPT_PARALLEL_JOBS=4          # Number of parallel jobs
HYPEROPT_RANDOM_STATE=42          # Random seed for reproducibility

# Validation Settings
VALIDATION_SPLIT=0.3              # Validation split ratio
VALIDATION_CV_FOLDS=5             # Cross-validation folds
VALIDATION_MIN_SAMPLES=100        # Minimum samples for validation

# Performance Limits
MAX_CONCURRENT_OPTIMIZATIONS=10    # Max concurrent optimization jobs
MAX_OPTIMIZATION_DURATION=7200    # Max duration per optimization (seconds)
MAX_MEMORY_USAGE_MB=2048          # Max memory usage per optimization
```

### **Monitoring & Observability**

```env
# Prometheus Metrics
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=8001
PROMETHEUS_METRICS_PATH="/metrics"

# Logging Configuration
LOG_LEVEL="INFO"                  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT="json"                 # json, text
LOG_FILE="logs/api.log"
LOG_MAX_SIZE_MB=100              # Max log file size
LOG_BACKUP_COUNT=5               # Number of backup files

# Log Aggregation (Loki)
LOKI_ENABLED=true
LOKI_URL="http://localhost:3100"
LOKI_BATCH_SIZE=1000
LOKI_FLUSH_INTERVAL=5

# Health Checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30         # Health check interval in seconds
HEALTH_CHECK_TIMEOUT=10          # Health check timeout in seconds
```

### **Export & Reporting**

```env
# PDF Report Settings
PDF_ENABLED=true
PDF_QUALITY="high"               # low, medium, high
PDF_DPI=300                      # DPI for images
PDF_MAX_SIZE_MB=50              # Max PDF file size

# Pine Script Export
PINESCRIPT_VERSION="v5"          # Pine Script version
PINESCRIPT_TEMPLATE_DIR="templates/pinescript"
PINESCRIPT_VALIDATION=true       # Validate generated scripts

# File Storage
STORAGE_TYPE="local"             # local, s3, gcs
STORAGE_PATH="./storage"
STORAGE_MAX_SIZE_GB=10          # Max storage size
```

---

## 📁 **Configuration Files**

### **Main Configuration File (.env)**

Create a `.env` file in your project root:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings
nano .env
```

**Example production .env:**

```env
# Production Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8
APP_DEBUG=false

# Security (CHANGE THESE!)
SECRET_KEY="prod-secret-key-very-long-and-random-string"
API_KEYS="prod-key-xyz789,admin-key-abc123"

# Database
DATABASE_URL="postgresql://trading_user:secure_password@localhost:5432/trading_optimizer"

# External APIs
ALPHA_VANTAGE_API_KEY="your-real-api-key"
POLYGON_API_KEY="your-real-polygon-key"

# Performance
HYPEROPT_MAX_EVALS=1000
HYPEROPT_PARALLEL_JOBS=8
MAX_CONCURRENT_OPTIMIZATIONS=20

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL="INFO"
LOG_FORMAT="json"

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
```

### **Development Configuration**

**Example development .env:**

```env
# Development Configuration
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=1
APP_DEBUG=true
API_RELOAD=true

# Security (development only)
SECRET_KEY="dev-secret-key"
API_KEYS="dev-key-12345"

# Database
DATABASE_URL="sqlite:///./dev_trading_optimizer.db"

# Performance (reduced for development)
HYPEROPT_MAX_EVALS=50
HYPEROPT_PARALLEL_JOBS=2
MAX_CONCURRENT_OPTIMIZATIONS=2

# Logging
LOG_LEVEL="DEBUG"
LOG_FORMAT="text"

# External APIs (optional for development)
ALPHA_VANTAGE_API_KEY=""
POLYGON_API_KEY=""
```

### **Testing Configuration**

**Example test .env:**

```env
# Testing Configuration
API_HOST=127.0.0.1
API_PORT=8001
API_WORKERS=1
APP_DEBUG=true

# Security
SECRET_KEY="test-secret-key"
API_KEYS="test-key-12345"

# Database (in-memory for tests)
DATABASE_URL="sqlite:///:memory:"

# Performance (minimal for fast tests)
HYPEROPT_MAX_EVALS=10
HYPEROPT_PARALLEL_JOBS=1
MAX_CONCURRENT_OPTIMIZATIONS=1

# Disable external services
PROMETHEUS_ENABLED=false
LOKI_ENABLED=false
ALPHA_VANTAGE_API_KEY=""
POLYGON_API_KEY=""
```

---

## ⚙️ **Application Configuration**

### **Strategy Configuration**

Configure available strategies in `config/strategies.yaml`:

```yaml
strategies:
  moving_average:
    enabled: true
    default_params:
      fast_period: 10
      slow_period: 30
    param_ranges:
      fast_period: [5, 20]
      slow_period: [21, 50]
  
  rsi:
    enabled: true
    default_params:
      period: 14
      overbought: 70
      oversold: 30
    param_ranges:
      period: [10, 21]
      overbought: [65, 80]
      oversold: [20, 35]
  
  bollinger_bands:
    enabled: true
    default_params:
      period: 20
      std_dev: 2.0
    param_ranges:
      period: [15, 30]
      std_dev: [1.5, 2.5]
```

### **Optimization Configuration**

Configure optimization settings in `config/optimization.yaml`:

```yaml
optimization:
  default_algorithm: "tpe"
  algorithms:
    tpe:
      name: "Tree-structured Parzen Estimator"
      max_evals: 1000
      timeout: 3600
    random:
      name: "Random Search"
      max_evals: 500
      timeout: 1800
  
  metrics:
    primary: "sharpe_ratio"
    secondary: ["total_return", "max_drawdown", "calmar_ratio"]
  
  constraints:
    min_trades: 10
    max_drawdown: 0.3
    min_profit_factor: 1.1
```

### **Validation Configuration**

Configure validation settings in `config/validation.yaml`:

```yaml
validation:
  methods:
    cross_validation:
      enabled: true
      folds: 5
      method: "time_series"
    
    walk_forward:
      enabled: true
      training_window: 252  # Trading days
      test_window: 63      # Trading days
      step_size: 21        # Trading days
    
    monte_carlo:
      enabled: true
      iterations: 1000
      confidence_level: 0.95
  
  out_of_sample:
    ratio: 0.3
    method: "chronological"
  
  performance_metrics:
    - "sharpe_ratio"
    - "sortino_ratio"
    - "calmar_ratio"
    - "max_drawdown"
    - "total_return"
    - "win_rate"
    - "profit_factor"
```

---

## 🐳 **Docker Configuration**

### **Docker Compose - Development**

`docker-compose.dev.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - APP_DEBUG=true
      - API_RELOAD=true
    volumes:
      - "./src:/app/src"
      - "./config:/app/config"
      - "./logs:/app/logs"
    depends_on:
      - redis
      - postgres
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  
  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: trading_optimizer
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: dev_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

### **Docker Compose - Production**

`docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - APP_DEBUG=false
      - API_WORKERS=8
    env_file:
      - .env.prod
    volumes:
      - "./logs:/app/logs"
      - "./storage:/app/storage"
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - "./nginx/nginx.conf:/etc/nginx/nginx.conf"
      - "./nginx/ssl:/etc/nginx/ssl"
    depends_on:
      - api
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - "./monitoring/prometheus:/etc/prometheus"
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - "./monitoring/grafana:/etc/grafana/provisioning"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

---

## ☸️ **Kubernetes Configuration**

### **ConfigMap for Application Settings**

`k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-optimizer-config
  namespace: trading-optimizer
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "8"
  APP_DEBUG: "false"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  PROMETHEUS_ENABLED: "true"
  PROMETHEUS_PORT: "8001"
  HYPEROPT_PARALLEL_JOBS: "8"
  MAX_CONCURRENT_OPTIMIZATIONS: "20"
```

### **Secret for Sensitive Data**

`k8s/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-optimizer-secrets
  namespace: trading-optimizer
type: Opaque
data:
  SECRET_KEY: <base64-encoded-secret-key>
  API_KEYS: <base64-encoded-api-keys>
  DATABASE_URL: <base64-encoded-database-url>
  ALPHA_VANTAGE_API_KEY: <base64-encoded-alpha-vantage-key>
  POLYGON_API_KEY: <base64-encoded-polygon-key>
```

### **Deployment Configuration**

`k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-optimizer-api
  namespace: trading-optimizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: trading-optimizer-api
  template:
    metadata:
      labels:
        app: trading-optimizer-api
    spec:
      containers:
      - name: api
        image: trading-optimizer:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: trading-optimizer-config
        - secretRef:
            name: trading-optimizer-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## 🔧 **Advanced Configuration**

### **Nginx Configuration**

`nginx/nginx.conf`:

```nginx
worker_processes auto;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    upstream trading_optimizer_api {
        server api:8000;
        keepalive 32;
    }
    
    server {
        listen 80;
        server_name _;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Security Headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
        
        # API Proxy
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://trading_optimizer_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        # Metrics endpoint (restricted access)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            
            proxy_pass http://trading_optimizer_api;
            proxy_set_header Host $host;
        }
        
        # Health check
        location /health {
            proxy_pass http://trading_optimizer_api/api/v1/health;
        }
    }
}
```

### **Systemd Service Configuration**

`/etc/systemd/system/trading-optimizer.service`:

```ini
[Unit]
Description=Trading Strategy Optimizer API
After=network.target
Wants=network.target

[Service]
Type=exec
User=trading-optimizer
Group=trading-optimizer
WorkingDirectory=/opt/trading-optimizer
Environment=PATH=/opt/trading-optimizer/venv/bin
EnvironmentFile=/opt/trading-optimizer/.env
ExecStart=/opt/trading-optimizer/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 8
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/trading-optimizer/logs /opt/trading-optimizer/storage
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

---

## 🔒 **Security Configuration**

### **API Key Management**

```python
# Generate secure API keys
import secrets

# Generate a new API key
api_key = secrets.token_urlsafe(32)
print(f"New API key: {api_key}")

# Generate multiple keys
api_keys = [secrets.token_urlsafe(32) for _ in range(5)]
api_keys_str = ",".join(api_keys)
print(f"API_KEYS={api_keys_str}")
```

### **Secret Key Generation**

```python
# Generate secure secret key
import secrets

secret_key = secrets.token_urlsafe(64)
print(f"SECRET_KEY={secret_key}")
```

### **SSL Certificate Configuration**

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# For production, use Let's Encrypt:
# certbot --nginx -d your-domain.com
```

---

## 📊 **Monitoring Configuration**

### **Prometheus Configuration**

`monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'trading-optimizer-api'
    static_configs:
      - targets: ['api:8001']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
      
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### **Grafana Datasource Configuration**

`monitoring/grafana/provisioning/datasources/prometheus.yaml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

---

## 🚨 **Troubleshooting Configuration Issues**

### **Common Configuration Problems**

#### **Environment Variables Not Loading**

```bash
# Check if .env file exists and has correct permissions
ls -la .env
chmod 600 .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('API_PORT', 'NOT_SET'))"

# Check for syntax errors in .env
cat .env | grep -E '^[A-Z_]+=.*'
```

#### **Database Connection Issues**

```bash
# Test database connection
python -c "
import os
from sqlalchemy import create_engine
engine = create_engine(os.getenv('DATABASE_URL'))
try:
    engine.connect()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

#### **API Key Authentication Issues**

```bash
# Test API key
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/strategies/list

# Check configured API keys
python -c "
import os
keys = os.getenv('API_KEYS', '').split(',')
print(f'Configured API keys: {len(keys)}')
for i, key in enumerate(keys):
    print(f'Key {i+1}: {key[:8]}...')
"
```

#### **Port Conflicts**

```bash
# Check what's using the port
sudo netstat -tulpn | grep :8000
sudo lsof -i :8000

# Kill conflicting process
sudo kill -9 <PID>

# Use different port
export API_PORT=8001
```

---

## 📝 **Configuration Validation**

### **Configuration Checker Script**

Create `scripts/check_config.py`:

```python
#!/usr/bin/env python3
import os
import sys
from urllib.parse import urlparse

def check_required_env_vars():
    """Check if all required environment variables are set."""
    required_vars = [
        'SECRET_KEY',
        'API_KEYS',
        'DATABASE_URL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def check_database_url():
    """Validate database URL format."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return False
    
    try:
        parsed = urlparse(db_url)
        if not parsed.scheme or not parsed.netloc:
            print(f"❌ Invalid database URL format: {db_url}")
            return False
        
        print(f"✅ Database URL format is valid: {parsed.scheme}://{parsed.netloc}/{parsed.path[1:]}")
        return True
    except Exception as e:
        print(f"❌ Error parsing database URL: {e}")
        return False

def check_api_keys():
    """Validate API keys."""
    api_keys = os.getenv('API_KEYS', '').split(',')
    api_keys = [key.strip() for key in api_keys if key.strip()]
    
    if not api_keys:
        print("❌ No API keys configured")
        return False
    
    for i, key in enumerate(api_keys):
        if len(key) < 16:
            print(f"❌ API key {i+1} is too short (minimum 16 characters)")
            return False
    
    print(f"✅ {len(api_keys)} API keys configured and validated")
    return True

def main():
    """Run all configuration checks."""
    print("🔍 Checking Trading Optimizer configuration...")
    
    checks = [
        check_required_env_vars,
        check_database_url,
        check_api_keys
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
    
    if all_passed:
        print("\n🎉 All configuration checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some configuration checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Run the configuration checker:

```bash
python scripts/check_config.py
```

---

**🎉 Configuration complete! Your Trading Strategy Optimization System is properly configured and ready for use.**

**Next Steps:**
- 📖 [Quick Start Guide](quick-start.md) - Run your first optimization
- 🚀 [Deployment Guide](../deployment/production-setup.md) - Deploy to production
- 📊 [Monitoring Setup](../deployment/monitoring-setup.md) - Set up monitoring 