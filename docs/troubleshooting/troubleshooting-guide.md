# Troubleshooting Guide

## Quick Diagnosis

### System Health Check
```bash
# Run comprehensive system check
./scripts/health_check.sh

# Expected output:
✅ API server responding (200ms)
✅ Database connection active
✅ Redis cache operational
✅ Market data feeds connected
✅ All services healthy
```

### Performance Benchmarks
```python
# Test optimization speed
from benchmark import run_performance_test

results = run_performance_test()
print(f"Optimization time: {results['optimization_time']:.1f}s")
print(f"API response: {results['api_response_time']:.0f}ms")

# Expected results:
# Optimization time: 24.1s (target: <30s)
# API response: 180ms (target: <200ms)
```

---

## Installation Issues

### Docker Environment Problems

**Issue**: Containers fail to start
```bash
# 1. Check Docker resources
docker system df
docker system prune -f  # Clean up if needed

# 2. Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# 3. Monitor startup logs
docker-compose logs -f api
```

**Issue**: Port conflicts
```bash
# Find processes using required ports
lsof -i :8000 -i :6379 -i :5432

# Kill conflicting processes
sudo kill -9 $(lsof -t -i:8000)

# Or change ports in docker-compose.yml
```

### Environment Configuration

**Issue**: Missing API keys
```bash
# 1. Create .env file from template
cp .env.example .env

# 2. Required variables for basic operation:
TRADING_API_KEY=your_key_here
MARKET_DATA_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/hyperopt
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=$(openssl rand -hex 32)

# 3. Validate configuration
python scripts/validate_config.py
```

---

## Optimization Issues

### Slow Performance

**Diagnostic Steps**:
```python
# 1. Check parameter space size
def calculate_search_space(params):
    total = 1
    for param, values in params.items():
        total *= len(values)
    print(f"Parameter combinations: {total:,}")
    
    if total > 10000:
        print("⚠️  Search space too large - consider reducing")

# 2. Profile optimization bottlenecks
import cProfile
cProfile.run('optimize_strategy(params)', 'optimization_profile.prof')

# 3. Monitor resource usage during optimization
htop  # Check CPU/memory usage
```

**Solutions**:
```python
# Optimize parameter space
params_optimized = {
    'lookback': [10, 20, 50],           # Reduced from range(5, 100)
    'threshold': [0.5, 1.0, 1.5, 2.0]  # Reduced precision
}

# Enable parallel processing
optimization_config = {
    'n_jobs': -1,              # Use all CPU cores
    'batch_size': 50,          # Process in batches
    'enable_caching': True,    # Cache intermediate results
    'early_stopping': 25       # Stop if no improvement
}
```

### Memory Issues

**Diagnostic**:
```python
import psutil
import tracemalloc

# Start memory tracking
tracemalloc.start()

def monitor_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 8000:  # > 8GB
        print(f"⚠️  High memory usage: {memory_mb:.1f} MB")
        
        # Get top memory consumers
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1024 / 1024:.1f} MB")
        print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

**Solutions**:
```python
# Data optimization
def optimize_memory_usage():
    # 1. Use efficient data types
    df = df.astype({
        'open': 'float32',
        'high': 'float32', 
        'low': 'float32',
        'close': 'float32',
        'volume': 'int32'
    })
    
    # 2. Process data in chunks
    chunk_size = 10000
    for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
        process_chunk(chunk)
        del chunk  # Explicit cleanup
        gc.collect()
```

---

## API Issues

### Authentication Errors

**401 Unauthorized**:
```python
# Validate API key format
def validate_api_key(key):
    checks = {
        'prefix': key.startswith('hopt_'),
        'length': len(key) == 64,
        'characters': key.replace('hopt_', '').isalnum()
    }
    
    for check, passed in checks.items():
        print(f"{check}: {'✅' if passed else '❌'}")
    
    return all(checks.values())

# Test authentication
response = requests.get(
    'https://api.hyperopt-strat.com/v1/auth/test',
    headers={'Authorization': f'Bearer {api_key}'}
)
print(f"Status: {response.status_code}")
```

### Rate Limiting

**429 Too Many Requests**:
```python
# Implement exponential backoff
import time
import random

def api_call_with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                # Exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {delay:.1f}s...")
                time.sleep(delay)
                continue
            raise
    
    raise Exception("Max retries exceeded")
```

### Request Timeouts

```python
# Configure appropriate timeouts
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

# Retry strategy
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    backoff_factor=1
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Set reasonable timeouts
response = session.post(
    'https://api.hyperopt-strat.com/v1/optimize',
    json=data,
    timeout=(30, 300)  # (connect, read) timeouts
)
```

---

## Data Issues

### Market Data Problems

**Stale Data Detection**:
```python
def validate_data_quality(df):
    issues = []
    
    # Check data freshness
    latest = df.index[-1]
    age_hours = (pd.Timestamp.now(tz='UTC') - latest).total_seconds() / 3600
    
    if age_hours > 1:
        issues.append(f"Data is {age_hours:.1f} hours old")
    
    # Check for gaps
    expected_intervals = len(pd.date_range(df.index[0], df.index[-1], freq='1min'))
    actual_intervals = len(df)
    gap_percentage = (expected_intervals - actual_intervals) / expected_intervals * 100
    
    if gap_percentage > 5:
        issues.append(f"Missing {gap_percentage:.1f}% of expected data points")
    
    # Check OHLC consistency
    ohlc_issues = df[(df['high'] < df['low']) | 
                     (df['close'] > df['high']) | 
                     (df['close'] < df['low'])]
    
    if len(ohlc_issues) > 0:
        issues.append(f"Found {len(ohlc_issues)} OHLC inconsistencies")
    
    return issues
```

**Data Feed Connectivity**:
```python
def test_data_feeds():
    feeds = ['binance', 'coinbase', 'kraken']
    status = {}
    
    for feed in feeds:
        try:
            connector = get_connector(feed)
            test_data = connector.get_latest_price('BTC/USD')
            
            if test_data and 'price' in test_data:
                status[feed] = '✅ Connected'
            else:
                status[feed] = '❌ No data returned'
                
        except Exception as e:
            status[feed] = f'❌ Error: {str(e)[:50]}'
    
    return status
```

---

## Strategy Development Issues

### Backtest Validation

**Unrealistic Results**:
```python
def validate_backtest(results):
    warnings = []
    
    # Check for common red flags
    if results['annual_return'] > 3.0:  # >300%
        warnings.append("Returns too high - possible lookahead bias")
    
    if results['sharpe_ratio'] > 4.0:
        warnings.append("Sharpe ratio too high - possible overfitting")
    
    if results['max_drawdown'] < 0.05:  # <5%
        warnings.append("Drawdown too low - unrealistic risk profile")
    
    if results['win_rate'] > 0.9:  # >90%
        warnings.append("Win rate too high - check data quality")
    
    # Validate sufficient trades
    if results['total_trades'] < 30:
        warnings.append("Too few trades for statistical significance")
    
    return warnings

# Realistic benchmark expectations
realistic_ranges = {
    'annual_return': (0.1, 1.0),     # 10-100%
    'sharpe_ratio': (0.5, 3.0),      # 0.5-3.0
    'max_drawdown': (0.05, 0.4),     # 5-40%
    'win_rate': (0.35, 0.75),        # 35-75%
    'total_trades': (50, 1000)       # 50-1000 trades
}
```

### Pine Script Export Issues

**Common Pine Script Problems**:
```pinescript
// 1. Version compatibility
//@version=5  // Always use latest version

// 2. Variable declaration
var float entry_price = na      // Use 'var' for persistence
var bool in_position = false

// 3. Avoid repainting
sma_20 = ta.sma(close[1], 20)  // Use [1] for confirmed data

// 4. Proper condition handling
long_condition = ta.crossover(fast_ma, slow_ma) and not in_position
if long_condition
    strategy.entry("Long", strategy.long)
    in_position := true

short_condition = ta.crossunder(fast_ma, slow_ma) and in_position
if short_condition
    strategy.close("Long")
    in_position := false
```

---

## Performance Monitoring

### Real-time Health Monitoring

```python
# System health dashboard
def get_system_health():
    health = {}
    
    # API response times
    start = time.time()
    response = requests.get('http://localhost:8000/health')
    health['api_response_time'] = (time.time() - start) * 1000
    
    # Database connectivity
    try:
        db.execute('SELECT 1')
        health['database'] = 'healthy'
    except Exception:
        health['database'] = 'unhealthy'
    
    # Memory usage
    health['memory_usage'] = psutil.virtual_memory().percent
    
    # CPU usage
    health['cpu_usage'] = psutil.cpu_percent(interval=1)
    
    return health

# Alert thresholds
thresholds = {
    'api_response_time': 500,   # ms
    'memory_usage': 85,         # %
    'cpu_usage': 90            # %
}
```

### Performance Optimization

```python
# Database optimization
def optimize_database():
    # Index optimization
    indexes = [
        'CREATE INDEX idx_strategies_symbol ON strategies(symbol)',
        'CREATE INDEX idx_results_timestamp ON results(timestamp)',
        'CREATE INDEX idx_trades_strategy_id ON trades(strategy_id)'
    ]
    
    for index in indexes:
        db.execute(index)
    
    # Query optimization
    db.execute('ANALYZE;')  # Update statistics
    db.execute('VACUUM;')   # Cleanup

# Redis optimization
def optimize_cache():
    redis_config = {
        'maxmemory': '2gb',
        'maxmemory-policy': 'allkeys-lru',
        'save': '900 1 300 10 60 10000'  # Persistence settings
    }
    
    for key, value in redis_config.items():
        redis_client.config_set(key, value)
```

---

## Emergency Procedures

### System Recovery

**Complete System Reset**:
```bash
#!/bin/bash
# emergency_reset.sh

echo "🚨 Emergency system reset starting..."

# 1. Stop all services
docker-compose down -v
systemctl stop nginx
systemctl stop redis

# 2. Backup current data
mkdir -p backups/$(date +%Y%m%d_%H%M%S)
pg_dump hyperopt > backups/$(date +%Y%m%d_%H%M%S)/database.sql

# 3. Clean and rebuild
docker system prune -af
docker-compose build --no-cache

# 4. Restore from known good configuration
git checkout production  # Known stable branch
docker-compose up -d

# 5. Verify system health
sleep 30
python scripts/health_check.py

echo "✅ Emergency reset complete"
```

### Data Recovery

```python
# Backup strategy
def create_backup():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'backups/{timestamp}'
    
    # Database backup
    os.system(f'pg_dump hyperopt > {backup_path}/database.sql')
    
    # Strategy files backup
    shutil.copytree('strategies/', f'{backup_path}/strategies/')
    
    # Configuration backup
    shutil.copy('.env', f'{backup_path}/.env')
    
    print(f"Backup created: {backup_path}")

# Restore from backup
def restore_backup(backup_path):
    print(f"Restoring from {backup_path}...")
    
    # Stop services
    os.system('docker-compose down')
    
    # Restore database
    os.system(f'psql hyperopt < {backup_path}/database.sql')
    
    # Restore files
    shutil.copytree(f'{backup_path}/strategies/', 'strategies/')
    shutil.copy(f'{backup_path}/.env', '.env')
    
    # Restart services
    os.system('docker-compose up -d')
    
    print("✅ Restore complete")
```

---

## Getting Help

### Diagnostic Information Collection

```python
# Generate support bundle
def generate_support_bundle():
    bundle = {
        'timestamp': datetime.now().isoformat(),
        'version': get_version(),
        'system_info': {
            'os': platform.system(),
            'python': platform.python_version(),
            'memory': psutil.virtual_memory()._asdict(),
            'cpu_count': psutil.cpu_count()
        },
        'configuration': get_sanitized_config(),
        'recent_logs': get_recent_logs(hours=24),
        'health_check': get_system_health()
    }
    
    with open('support_bundle.json', 'w') as f:
        json.dump(bundle, f, indent=2, default=str)
    
    print("Support bundle created: support_bundle.json")
    print("Please include this file when requesting support")
```

### Contact Information

- **Emergency Support**: emergency@hyperopt-strat.com
- **Technical Issues**: [GitHub Issues](https://github.com/hyperopt-strat/issues)
- **Community**: [Discord](https://discord.gg/hyperopt-strat)
- **Documentation**: [docs.hyperopt-strat.com](https://docs.hyperopt-strat.com)

---

*For issues not covered in this guide, please run the diagnostic bundle script and contact support with the generated file.* 