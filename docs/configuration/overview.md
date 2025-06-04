# ⚙️ **Configuration Guide**

## 🎯 **Professional Configuration Management**

The HyperOpt Strategy Platform uses a comprehensive configuration system with YAML files and environment variables for maximum flexibility and security.

---

## 📁 **Configuration Structure**

```
config/
├── production.yaml      # Production environment settings
├── optimization.yaml    # Hyperopt and tournament configuration  
├── strategies.yaml      # Strategy-specific parameters
├── exchanges.yaml       # Exchange connections and data sources
└── logging.yaml         # Logging and monitoring configuration
```

**Environment Variables**:
- `.env` - Local development environment variables
- System environment - Production deployment variables

---

## 🏭 **Production Configuration**

### **File**: `config/production.yaml`

```yaml
# Environment Settings
debug: false
environment: production

# Logging Configuration
logging:
  level: INFO
  file_path: /var/log/trading-platform/application.log
  max_size: 100MB
  backup_count: 5

# Performance Settings  
performance:
  max_concurrent_strategies: 24
  memory_limit: 8GB
  optimization_timeout: 3600

# Monitoring & Alerts
monitoring:
  health_check_interval: 300
  alert_thresholds:
    cpu_percent: 80
    memory_percent: 85
    disk_percent: 90

# Strategy Defaults
strategies:
  default_position_size: 0.1
  max_position_size: 0.2
  stop_loss_enabled: true
  take_profit_enabled: true
```

### **Key Settings Explained**

#### **Environment**
- `debug`: Enable/disable debug mode (affects logging, error handling)
- `environment`: Current environment (`development`, `production`, `testing`)

#### **Performance**
- `max_concurrent_strategies`: Maximum strategies to optimize simultaneously
- `memory_limit`: RAM limit for the application
- `optimization_timeout`: Maximum time per optimization (seconds)

#### **Monitoring**
- `health_check_interval`: Health check frequency (seconds)
- `alert_thresholds`: CPU, memory, disk usage alert levels

---

## 🔧 **Optimization Configuration**

### **File**: `config/optimization.yaml`

```yaml
# Hyperopt Algorithm Settings
hyperopt:
  algorithm: "tpe"         # Tree-structured Parzen Estimator
  max_evals: 100          # Maximum evaluations per strategy
  timeout: 3600           # 1 hour timeout per strategy
  
  # Multi-objective optimization weights
  objectives:
    annual_return:
      weight: 0.25
      target: "maximize"
      min_threshold: 0.20
    
    sharpe_ratio:
      weight: 0.25
      target: "maximize"
      min_threshold: 1.0
    
    max_drawdown:
      weight: 0.20
      target: "minimize"
      max_threshold: 0.50
    
    profit_factor:
      weight: 0.15
      target: "maximize"
      min_threshold: 1.5
    
    win_rate:
      weight: 0.15
      target: "optimize"
      min_threshold: 0.25
      max_threshold: 0.75

# Tournament System
tournament:
  rounds:
    round_1:
      description: "All strategies with basic optimization"
      strategies: "all"
      max_evals: 50
      timeout: 1800
    
    round_2:
      description: "Top performers with deep optimization"
      strategies: "top_20"
      max_evals: 200
      timeout: 7200
```

### **Hyperopt Settings**

#### **Algorithm Options**
- `tpe`: Tree-structured Parzen Estimator (recommended)
- `random`: Random search (faster, less accurate)
- `adaptive_tpe`: Adaptive TPE (experimental)

#### **Objective Weights**
Configure how different metrics are weighted in optimization:

| Metric | Weight | Purpose |
|--------|--------|---------|
| `annual_return` | 25% | Primary profit metric |
| `sharpe_ratio` | 25% | Risk-adjusted returns |
| `max_drawdown` | 20% | Risk management |
| `profit_factor` | 15% | Trading efficiency |
| `win_rate` | 15% | Consistency measure |

---

## 📊 **Validation Configuration**

```yaml
# Data Splitting
validation:
  train_split: 0.70      # 70% for training
  validation_split: 0.15 # 15% for validation
  test_split: 0.15       # 15% for out-of-sample testing

# Walk-Forward Analysis
  walk_forward:
    enabled: true
    optimization_window: 180  # 6 months
    testing_window: 90       # 3 months
    step_size: 30           # 1 month steps

# Cross-Asset Validation
  cross_asset:
    enabled: true
    primary_asset: "BTC"
    validation_assets: ["ETH", "SOL", "ADA", "DOT"]
    correlation_threshold: 0.3

# Statistical Testing
  statistical_tests:
    enabled: true
    confidence_level: 0.95
    bootstrap_samples: 1000
    permutation_tests: true
```

### **Validation Methods**

#### **Walk-Forward Analysis**
- **Window Size**: 6-month optimization, 3-month testing
- **Step Size**: 1-month forward steps
- **Purpose**: Simulate real trading conditions

#### **Cross-Asset Validation**
- **Primary Asset**: BTC (main optimization target)
- **Validation Assets**: ETH, SOL, ADA, DOT
- **Threshold**: 30% minimum correlation for validation

#### **Statistical Tests**
- **Bootstrap Sampling**: 1000 samples for significance
- **Confidence Level**: 95% confidence intervals
- **Permutation Tests**: Null hypothesis testing

---

## 🚫 **Overfitting Prevention**

```yaml
overfitting_prevention:
  # Trade Frequency Constraints
  trade_frequency:
    min_trades_per_year: 12
    max_trades_per_year: 300
    penalty_factor: 0.5

  # Win Rate Penalties
  win_rate_penalty:
    enabled: true
    threshold: 0.85        # Penalize >85% win rate
    penalty_factor: 0.3    # 30% score reduction

  # Robustness Scoring
  robustness:
    consistency_weight: 0.3
    stability_weight: 0.2
    generalization_weight: 0.5

  # Complexity Penalty
  complexity_penalty:
    max_parameters: 10
    penalty_per_parameter: 0.02  # 2% per extra parameter
```

---

## 📈 **Performance Thresholds**

```yaml
performance_thresholds:
  # Minimum Requirements
  minimum:
    annual_return: 0.15    # 15%
    sharpe_ratio: 0.8
    profit_factor: 1.3
    max_drawdown: 0.60     # 60%
    win_rate: 0.20         # 20%

  # Good Performance
  good:
    annual_return: 0.25    # 25%
    sharpe_ratio: 1.2
    profit_factor: 1.8
    max_drawdown: 0.40     # 40%
    win_rate: 0.45         # 45%

  # Excellent Performance  
  excellent:
    annual_return: 0.40    # 40%
    sharpe_ratio: 2.0
    profit_factor: 2.5
    max_drawdown: 0.25     # 25%
    win_rate: 0.60         # 60%
```

---

## 🔐 **Environment Variables**

### **API Keys** (Required)
```bash
# AI Model Providers
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
GOOGLE_API_KEY=your_google_key_here

# Exchange API Keys (if using live data)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
COINBASE_API_KEY=your_coinbase_key
COINBASE_SECRET_KEY=your_coinbase_secret
```

### **Application Settings**
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your_super_secret_key_here
API_KEY_SALT=random_salt_for_api_keys

# Performance
MAX_WORKERS=8
MEMORY_LIMIT=8GB
OPTIMIZATION_TIMEOUT=3600
```

### **Monitoring & Alerting**
```bash
# Prometheus/Grafana
PROMETHEUS_GATEWAY=http://localhost:9091
GRAFANA_URL=http://localhost:3000

# Alerting
SLACK_WEBHOOK_URL=your_slack_webhook
DISCORD_WEBHOOK_URL=your_discord_webhook
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@example.com
EMAIL_PASSWORD=your_email_password
```

---

## 🔧 **Configuration Management**

### **Loading Priority**
1. **Environment Variables** (highest priority)
2. **Local `.env` file** 
3. **YAML Configuration Files**
4. **Default Values** (lowest priority)

### **Environment-Specific Configs**
```bash
# Development
config/development.yaml

# Testing  
config/testing.yaml

# Production
config/production.yaml
```

### **Dynamic Configuration**
Some settings can be updated without restart:
- Logging levels
- Monitoring thresholds
- Rate limits
- Alert settings

### **Static Configuration**
These require application restart:
- Database connections
- Security keys
- Core algorithm settings
- Memory limits

---

## 📝 **Configuration Examples**

### **Development Setup**
```yaml
# config/development.yaml
debug: true
environment: development

logging:
  level: DEBUG
  file_path: ./logs/app.log

performance:
  max_concurrent_strategies: 4
  memory_limit: 2GB
  optimization_timeout: 300

monitoring:
  health_check_interval: 60
```

### **Docker Environment**
```yaml
# docker-compose.yml environment
environment:
  - ENVIRONMENT=production
  - DEBUG=false
  - LOG_LEVEL=INFO
  - DATABASE_URL=postgresql://postgres:password@db:5432/trading
  - REDIS_URL=redis://redis:6379/0
```

### **Kubernetes ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trading-config
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  MAX_WORKERS: "8"
  OPTIMIZATION_TIMEOUT: "3600"
```

---

## 🛠️ **Configuration Validation**

### **Validation Rules**
- **Required Fields**: Must be present
- **Type Validation**: Correct data types
- **Range Validation**: Values within acceptable ranges
- **Dependency Validation**: Related settings are compatible

### **Validation Example**
```python
from config.validator import ConfigValidator

# Validate configuration on startup
validator = ConfigValidator()
try:
    validator.validate_all()
    print("✅ Configuration valid")
except ConfigValidationError as e:
    print(f"❌ Configuration error: {e}")
    exit(1)
```

### **Common Validation Errors**
- Missing required environment variables
- Invalid timeout values (too high/low)
- Conflicting optimization settings
- Invalid threshold ranges

---

## 🔄 **Configuration Updates**

### **Hot Reload** (No Restart Required)
```bash
# Update logging level
curl -X POST http://localhost:8000/admin/config/logging \
  -d '{"level": "DEBUG"}'

# Update monitoring thresholds
curl -X POST http://localhost:8000/admin/config/monitoring \
  -d '{"cpu_threshold": 90}'
```

### **Cold Updates** (Restart Required)
- Database connection strings
- Security keys and secrets
- Core algorithm parameters
- Memory and CPU limits

---

## 📚 **Configuration Best Practices**

### **Security**
- ✅ Store secrets in environment variables
- ✅ Use secure key generation
- ✅ Rotate API keys regularly
- ❌ Never commit secrets to version control

### **Performance**
- ✅ Set appropriate memory limits
- ✅ Configure optimal timeout values
- ✅ Use parallel processing when possible
- ❌ Don't over-allocate resources

### **Monitoring**
- ✅ Set realistic alert thresholds
- ✅ Monitor key performance metrics
- ✅ Configure proper log rotation
- ❌ Don't ignore system alerts

### **Development**
- ✅ Use different configs per environment
- ✅ Validate configurations on startup
- ✅ Document all configuration options
- ❌ Don't use production configs in development

---

## 🔗 **Related Documentation**

- **[Deployment Guide](../deployment/production-setup.md)**
- **[Monitoring Setup](../monitoring/grafana-setup.md)**
- **[Security Guidelines](../security/best-practices.md)**
- **[Troubleshooting](../reference/troubleshooting.md)**

---

**⚙️ Need help with configuration? Check our [Configuration Troubleshooting Guide](../reference/config-troubleshooting.md)!** 