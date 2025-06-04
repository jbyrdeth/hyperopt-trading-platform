# 📚 **Complete API Reference**

## 🎯 **Trading Strategy Optimization API - Comprehensive Reference**

**Production-Ready API** • **<200ms Response Times** • **Enterprise Security** • **24 Trading Strategies Available**

---

## 📋 **Table of Contents**

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)  
3. [Core Endpoints](#core-endpoints)
4. [Request/Response Models](#requestresponse-models)
5. [Real-World Examples](#real-world-examples)
6. [Error Handling](#error-handling)
7. [Rate Limits](#rate-limits)
8. [Monitoring](#monitoring)

---

## 🚀 **Quick Start**

### **Base URL**
```
http://localhost:8000/api/v1
```

### **Authentication**
All requests require an API key in the header:
```bash
-H "X-API-Key: your_api_key_here"
```

### **Complete Working Example** 
*Optimize a strategy end-to-end:*

```bash
# 1. List available strategies
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/strategies

# 2. Start optimization
curl -X POST "http://localhost:8000/api/v1/optimize/single" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "MovingAverageCrossover",
    "asset": "BTC",
    "timeframe": "4h",
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:59:59Z",
    "optimization_config": {
      "max_evals": 50,
      "timeout_minutes": 60,
      "objective": "sharpe_ratio"
    }
  }'

# 3. Check optimization status
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/optimize/status/{job_id}

# 4. Generate Pine Script
curl -X POST "http://localhost:8000/api/v1/export/pinescript" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "MovingAverageCrossover",
    "optimization_results": {...},
    "output_format": "strategy",
    "include_alerts": true
  }'
```

---

## 🔐 **Authentication**

### **API Key Management**

#### **Development Keys**
```bash
# Development key (full access)
X-API-Key: dev_key_123

# Test key (read-only access) 
X-API-Key: test_key_456
```

#### **Authentication Test**
```bash
# ✅ Test valid authentication
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/health

# Response: {"status":"healthy","version":"1.0.0",...}
```

#### **Error Responses**
```json
// Missing API key
{
  "success": false,
  "error_code": "MISSING_API_KEY",
  "error_message": "API key is required"
}

// Invalid API key
{
  "success": false, 
  "error_code": "INVALID_API_KEY",
  "error_message": "Invalid or expired API key"
}
```

---

## 🎯 **Core Endpoints**

### **1. System Health & Monitoring**

#### **GET /health** - System Health Check
```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0", 
  "uptime_seconds": 3600.5,
  "components": {
    "system_resources": "healthy",
    "api_core": "healthy",
    "job_manager": "healthy", 
    "storage": "healthy"
  },
  "active_jobs": 2,
  "queue_size": 0,
  "memory_usage_mb": 1024.5,
  "cpu_usage_percent": 15.2
}
```

#### **GET /metrics** - Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

**Sample Output:**
```
# API performance metrics
api_requests_total{method="POST",endpoint="/optimize/single"} 145
api_request_duration_seconds{method="POST"} 0.189

# Optimization metrics
optimization_jobs_total{status="completed"} 42
optimization_jobs_total{status="failed"} 3
optimization_duration_seconds_avg 127.5

# System metrics
system_memory_usage_bytes 1073741824
system_cpu_usage_percent 15.2
```

### **2. Strategy Management**

#### **GET /strategies** - List Available Strategies
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/strategies
```

**Response:**
```json
{
  "success": true,
  "strategies": [
    {
      "name": "MovingAverageCrossover",
      "description": "Moving average crossover with signal filtering",
      "category": "trend_following",
      "parameters": {
        "fast_period": {
          "name": "fast_period",
          "value": 12,
          "type": "int",
          "min_value": 1,
          "max_value": 100,
          "description": "Fast moving average period"
        },
        "slow_period": {
          "name": "slow_period", 
          "value": 26,
          "type": "int",
          "min_value": 2,
          "max_value": 200,
          "description": "Slow moving average period"
        }
      },
      "default_timeframe": "4h",
      "recommended_assets": ["BTC", "ETH"],
      "risk_level": "Medium",
      "complexity_score": 6.5
    }
  ],
  "total_count": 24,
  "categories": {
    "trend_following": 8,
    "mean_reversion": 6,
    "momentum": 5,
    "volume": 3, 
    "volatility": 2
  }
}
```

#### **GET /strategies/{strategy_name}** - Get Strategy Details
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/strategies/MovingAverageCrossover
```

### **3. Optimization Engine**

#### **POST /optimize/single** - Single Strategy Optimization
```bash
curl -X POST "http://localhost:8000/api/v1/optimize/single" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "MovingAverageCrossover",
    "asset": "BTC",
    "timeframe": "4h", 
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:59:59Z",
    "optimization_config": {
      "max_evals": 100,
      "timeout_minutes": 60,
      "algorithm": "tpe",
      "objective": "sharpe_ratio",
      "cross_validation": true
    },
    "validation_config": {
      "out_of_sample_ratio": 0.3,
      "monte_carlo_runs": 100,
      "cross_asset_validation": true
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "job_id": "opt_abc123def456",
  "status": "queued",
  "estimated_duration_minutes": 45,
  "message": "Optimization job queued successfully"
}
```

#### **POST /optimize/batch** - Batch Strategy Optimization
```bash
curl -X POST "http://localhost:8000/api/v1/optimize/batch" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "strategies": ["MovingAverageCrossover", "RSIStrategy", "MACDStrategy"],
    "common_config": {
      "asset": "BTC",
      "timeframe": "4h",
      "start_date": "2023-01-01T00:00:00Z", 
      "end_date": "2023-12-31T23:59:59Z"
    },
    "parallel_jobs": 3
  }'
```

#### **GET /optimize/status/{job_id}** - Check Optimization Status
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/optimize/status/opt_abc123def456
```

**Response:**
```json
{
  "success": true,
  "job_id": "opt_abc123def456",
  "strategy_name": "MovingAverageCrossover",
  "status": "completed",
  "progress": 100.0,
  "best_parameters": {
    "fast_period": 18,
    "slow_period": 42,
    "signal_threshold": 0.025
  },
  "best_score": 1.847,
  "performance_metrics": {
    "total_return": 45.2,
    "sharpe_ratio": 1.847,
    "sortino_ratio": 2.341,
    "max_drawdown": 12.5,
    "win_rate": 0.68,
    "trades_count": 156
  },
  "validation_results": {
    "out_of_sample_score": 8.5,
    "cross_validation_score": 7.8,
    "monte_carlo_score": 8.2
  },
  "created_at": "2024-01-01T10:00:00Z",
  "started_at": "2024-01-01T10:00:30Z", 
  "completed_at": "2024-01-01T10:45:15Z"
}
```

#### **GET /optimize/results/{job_id}** - Get Detailed Results
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/optimize/results/opt_abc123def456
```

#### **DELETE /optimize/cancel/{job_id}** - Cancel Running Job
```bash
curl -X DELETE -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/optimize/cancel/opt_abc123def456
```

### **4. Validation Framework**

#### **POST /validate/cross-asset** - Cross-Asset Validation
```bash
curl -X POST "http://localhost:8000/api/v1/validate/cross-asset" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_result": {...},
    "additional_assets": ["ETH", "BNB", "ADA"],
    "config": {
      "monte_carlo_runs": 200,
      "regime_analysis": true
    }
  }'
```

#### **POST /validate/out-of-sample** - Out-of-Sample Testing
```bash
curl -X POST "http://localhost:8000/api/v1/validate/out-of-sample" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "optimization_result": {...},
    "test_ratio": 0.3,
    "statistical_tests": true
  }'
```

### **5. Export System**

#### **POST /export/pinescript** - Generate Pine Script
```bash
curl -X POST "http://localhost:8000/api/v1/export/pinescript" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "MovingAverageCrossover",
    "optimization_results": {
      "best_parameters": {
        "fast_period": 18,
        "slow_period": 42
      }
    },
    "output_format": "strategy",
    "include_alerts": true,
    "include_debugging": true
  }'
```

**Response:**
```json
{
  "success": true,
  "file_id": "pine_xyz789abc123",
  "filename": "MovingAverageCrossover_optimized.pine",
  "file_size": 4096,
  "download_url": "/api/v1/export/download/pine_xyz789abc123",
  "expires_at": "2024-01-08T10:00:00Z",
  "script_preview": "// This Pine Script was generated by the Trading Strategy Optimization System\n//@version=5\nstrategy(\"MovingAverageCrossover\", overlay=true)\n\n// Optimized Parameters\nfast_period = input.int(18, \"Fast MA Period\")\nslow_period = input.int(42, \"Slow MA Period\")...",
  "generation_time": "2024-01-01T10:00:00Z"
}
```

#### **POST /export/report** - Generate PDF Report
```bash
curl -X POST "http://localhost:8000/api/v1/export/report" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "MovingAverageCrossover", 
    "optimization_results": {...},
    "report_type": "full",
    "include_charts": true,
    "include_detailed_tables": true
  }'
```

#### **GET /export/download/{file_id}** - Download Generated File
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/export/download/pine_xyz789abc123 \
  --output strategy.pine
```

#### **GET /export/files** - List Generated Files
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/export/files
```

### **6. Data Management**

#### **POST /data/fetch** - Fetch Market Data
```bash
curl -X POST "http://localhost:8000/api/v1/data/fetch" \
  -H "X-API-Key: dev_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "BTC",
    "timeframe": "4h",
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-12-31T23:59:59Z",
    "exchange": "binance",
    "use_cache": true
  }'
```

#### **GET /data/info** - Data Availability Info
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/data/info?asset=BTC&timeframe=4h
```

### **7. Job Management**

#### **GET /jobs** - List All Jobs
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/jobs?status=running&limit=10
```

#### **GET /jobs/{job_id}** - Get Job Details
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/jobs/opt_abc123def456
```

### **8. Logging & Debugging**

#### **GET /logs** - System Logs
```bash
curl -H "X-API-Key: dev_key_123" \
  "http://localhost:8000/api/v1/logs?level=ERROR&since=2024-01-01T00:00:00Z&limit=100"
```

#### **GET /logs/job/{job_id}** - Job-Specific Logs
```bash
curl -H "X-API-Key: dev_key_123" \
  http://localhost:8000/api/v1/logs/job/opt_abc123def456
```

---

## 📋 **Request/Response Models**

### **Core Enumerations**
```python
# Strategy Categories
StrategyType = "trend_following" | "mean_reversion" | "momentum" | "volume" | "volatility" | "pattern" | "multi_timeframe"

# Optimization Status
OptimizationStatus = "queued" | "running" | "completed" | "failed" | "cancelled"

# Timeframes
TimeFrame = "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "8h" | "1d" | "1w"

# Assets
Asset = "BTC" | "ETH" | "BNB" | "ADA" | "SOL" | "DOT" | "LINK" | "MATIC"

# Report Types  
ReportType = "full" | "executive" | "technical"
```

### **Request Models**

#### **OptimizationRequest**
```json
{
  "strategy_name": "string (required)",
  "asset": "Asset enum (default: BTC)", 
  "timeframe": "TimeFrame enum (default: 4h)",
  "start_date": "ISO datetime (required)",
  "end_date": "ISO datetime (required)",
  "optimization_config": {
    "max_evals": "int (10-1000, default: 100)",
    "timeout_minutes": "int (5-480, default: 60)",
    "n_startup_jobs": "int (1-50, default: 10)",
    "algorithm": "string (default: tpe)",
    "objective": "string (default: sharpe_ratio)",
    "cross_validation": "bool (default: true)"
  },
  "validation_config": {
    "out_of_sample_ratio": "float (0.1-0.5, default: 0.3)",
    "monte_carlo_runs": "int (10-1000, default: 100)",
    "cross_asset_validation": "bool (default: true)",
    "regime_analysis": "bool (default: true)",
    "statistical_tests": "bool (default: true)"
  },
  "custom_parameters": "object (optional)"
}
```

#### **PineScriptRequest**
```json
{
  "strategy_name": "string (required)",
  "optimization_results": "object (required)",
  "output_format": "strategy | indicator (default: strategy)",
  "include_alerts": "bool (default: true)",
  "include_debugging": "bool (default: true)",
  "include_visualization": "bool (default: true)"
}
```

### **Response Models**

#### **PerformanceMetrics**
```json
{
  "total_return": "float (%)",
  "sharpe_ratio": "float",
  "sortino_ratio": "float", 
  "calmar_ratio": "float",
  "max_drawdown": "float (%)",
  "volatility": "float (%)",
  "win_rate": "float (0-1)",
  "profit_factor": "float",
  "trades_count": "int",
  "avg_trade_return": "float (%)"
}
```

#### **OptimizationResult**
```json
{
  "job_id": "string",
  "strategy_name": "string",
  "status": "OptimizationStatus",
  "progress": "float (0-100)",
  "best_parameters": "object",
  "best_score": "float",
  "performance_metrics": "PerformanceMetrics",
  "validation_results": "object",
  "created_at": "ISO datetime",
  "started_at": "ISO datetime",
  "completed_at": "ISO datetime",
  "error_message": "string (if failed)"
}
```

---

## ⚠️ **Error Handling**

### **Standard Error Response**
```json
{
  "success": false,
  "error_code": "ERROR_CODE",
  "error_message": "Human readable error description",
  "details": {
    "field": "Additional error context",
    "validation_errors": ["List of validation issues"]
  },
  "timestamp": "2024-01-01T10:00:00Z",
  "request_id": "req_abc123def456"
}
```

### **Common Error Codes**
| Code | HTTP Status | Description |
|------|------------|-------------|
| `MISSING_API_KEY` | 401 | API key not provided |
| `INVALID_API_KEY` | 401 | Invalid or expired API key |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `STRATEGY_NOT_FOUND` | 404 | Strategy name not found |
| `INVALID_PARAMETERS` | 400 | Invalid request parameters |
| `OPTIMIZATION_FAILED` | 500 | Optimization job failed |
| `FILE_NOT_FOUND` | 404 | Generated file not found |
| `SYSTEM_ERROR` | 500 | Internal system error |

---

## 🚦 **Rate Limits**

| Endpoint Category | Rate Limit | Burst Limit |
|------------------|------------|-------------|
| **General API** | 100 req/hour | 20 req/min |
| **Optimization** | 10 req/hour | 2 req/min |
| **Data Fetching** | 1000 req/hour | 100 req/min |
| **Export** | 50 req/hour | 10 req/min |
| **Health/Monitoring** | No limit | No limit |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
X-RateLimit-Retry-After: 3600
```

---

## 📊 **Monitoring**

### **Health Check Endpoints**
- `GET /health` - Basic health status
- `GET /api/v1/health` - Detailed health with components
- `GET /metrics` - Prometheus metrics

### **Key Metrics Available**
- **Performance**: Response times, error rates, throughput
- **Optimization**: Job success rates, queue depth, completion times  
- **System**: CPU, memory, disk usage
- **Business**: Strategy performance, user activity

### **Grafana Dashboard**
Available at: `http://localhost:3000` (when monitoring stack is deployed)

---

## 🔗 **Additional Resources**

- **Interactive API Documentation**: `/api/docs` (Swagger UI)
- **ReDoc Documentation**: `/api/redoc` 
- **OpenAPI Schema**: `/api/openapi.json`
- **Health Check**: `/api/v1/health`
- **Metrics**: `/metrics`

---

**📖 For complete examples and tutorials, see our [Examples Documentation](../examples/complete-workflow.md)** 