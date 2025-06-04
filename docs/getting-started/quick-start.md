# 🚀 Quick Start Guide

## Get Your First Trading Strategy Optimization Running in 15 Minutes

This guide will have you running your first strategy optimization, analyzing results, and exporting Pine Script code for TradingView in just 15 minutes.

!!! tip "**Prerequisites**"
    - Python 3.9+ installed
    - Git installed  
    - 8GB+ RAM recommended
    - Internet connection for data fetching

---

## Step 1: Installation & Setup (5 minutes)

### **Clone the Repository**

```bash
git clone https://github.com/trading-optimizer/hyperopt-strat.git
cd hyperopt-strat
```

### **Set Up Python Environment**

=== "Using venv (Recommended)"

    ```bash
    # Create virtual environment
    python -m venv venv
    
    # Activate environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

=== "Using conda"

    ```bash
    # Create conda environment
    conda create -n trading-optimizer python=3.9
    conda activate trading-optimizer
    ```

### **Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, requests; print('✅ Dependencies installed successfully')"
```

### **Start the API Server**

```bash
# Navigate to API directory
cd src/api

# Start the development server
python main.py
```

!!! success "**Server Started!**"
    You should see: `🚀 Trading Strategy Optimization API started on http://0.0.0.0:8000`

---

## Step 2: Configure API Access (2 minutes)

### **Get Your API Key**

The system uses API keys for authentication. For development, you can use the default key:

```bash
export API_KEY="dev-key-12345"
```

!!! warning "**Production Security**"
    For production use, generate secure API keys and configure proper authentication. See our [Security Guide](../deployment/security-hardening.md).

### **Test API Connection**

```bash
# Test health endpoint (no auth required)
curl http://localhost:8000/api/v1/health

# Test authenticated endpoint
curl -H "X-API-Key: dev-key-12345" http://localhost:8000/api/v1/strategies/list
```

---

## Step 3: Run Your First Optimization (5 minutes)

### **Option A: Using Python (Recommended)**

Create a file called `first_optimization.py`:

```python
import requests
import time
import json

# Configuration
API_BASE = "http://localhost:8000/api/v1"
API_KEY = "dev-key-12345"
HEADERS = {"X-API-Key": API_KEY}

def run_optimization():
    """Run a simple moving average crossover optimization."""
    
    print("🚀 Starting optimization...")
    
    # 1. Submit optimization request
    optimization_request = {
        "strategy_name": "MovingAverageCrossover",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "optimization_config": {
            "trials": 50,  # Reduced for quick demo
            "cv_folds": 3,
            "optimization_metric": "sharpe_ratio"
        },
        "strategy_params": {
            "fast_period": {"min": 5, "max": 20},
            "slow_period": {"min": 21, "max": 50}
        }
    }
    
    response = requests.post(
        f"{API_BASE}/optimize/single",
        headers=HEADERS,
        json=optimization_request
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return None
    
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✅ Optimization job started: {job_id}")
    
    # 2. Monitor progress
    print("📊 Monitoring optimization progress...")
    while True:
        status_response = requests.get(
            f"{API_BASE}/optimize/status/{job_id}",
            headers=HEADERS
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            status = status_data["status"]
            progress = status_data.get("progress", 0)
            
            print(f"Status: {status} ({progress}% complete)")
            
            if status == "completed":
                print("🎉 Optimization completed!")
                break
            elif status == "failed":
                print("❌ Optimization failed!")
                return None
            
            time.sleep(10)  # Check every 10 seconds
        else:
            print("❌ Error checking status")
            return None
    
    # 3. Get results
    print("📈 Fetching results...")
    results_response = requests.get(
        f"{API_BASE}/optimize/results/{job_id}",
        headers=HEADERS
    )
    
    if results_response.status_code == 200:
        results = results_response.json()
        
        # Display key metrics
        best_params = results["best_parameters"]
        performance = results["performance_metrics"]
        
        print("\n🏆 Best Parameters Found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A'):.3f}")
        print(f"  Total Return: {performance.get('total_return', 'N/A'):.2%}")
        print(f"  Max Drawdown: {performance.get('max_drawdown', 'N/A'):.2%}")
        print(f"  Win Rate: {performance.get('win_rate', 'N/A'):.2%}")
        
        return job_id
    else:
        print("❌ Error fetching results")
        return None

if __name__ == "__main__":
    job_id = run_optimization()
    if job_id:
        print(f"\n✅ Optimization complete! Job ID: {job_id}")
        print("Next steps:")
        print("  1. Generate Pine Script: Get Pine Script code for TradingView")
        print("  2. Create PDF Report: Generate detailed performance report") 
        print("  3. Run Validation: Perform out-of-sample testing")
```

Run the optimization:

```bash
python first_optimization.py
```

### **Option B: Using cURL**

```bash
# Submit optimization
curl -X POST "http://localhost:8000/api/v1/optimize/single" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "MovingAverageCrossover",
    "symbol": "BTCUSDT", 
    "timeframe": "4h",
    "optimization_config": {
      "trials": 20
    }
  }'

# Check status (replace JOB_ID with actual ID)
curl -H "X-API-Key: dev-key-12345" \
  "http://localhost:8000/api/v1/optimize/status/YOUR_JOB_ID"

# Get results when complete
curl -H "X-API-Key: dev-key-12345" \
  "http://localhost:8000/api/v1/optimize/results/YOUR_JOB_ID"
```

---

## Step 4: Export Results (3 minutes)

### **Generate Pine Script for TradingView**

```python
# Add this to your script after getting results
def export_pine_script(job_id):
    """Export optimized strategy as Pine Script."""
    
    print("📝 Generating Pine Script...")
    
    response = requests.get(
        f"{API_BASE}/export/pinescript/{job_id}",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        pine_data = response.json()
        pine_script = pine_data["pine_script"]
        
        # Save to file
        with open(f"strategy_{job_id}.pine", "w") as f:
            f.write(pine_script)
        
        print(f"✅ Pine Script saved to: strategy_{job_id}.pine")
        print("\n📋 To use in TradingView:")
        print("  1. Open TradingView Pine Script Editor")
        print("  2. Copy and paste the generated code")
        print("  3. Click 'Add to Chart'")
        print("  4. Your optimized strategy is now live!")
        
        return pine_script
    else:
        print("❌ Error generating Pine Script")
        return None

# Use it:
if job_id:
    export_pine_script(job_id)
```

### **Generate PDF Performance Report**

```python
def generate_pdf_report(job_id):
    """Generate comprehensive PDF performance report."""
    
    print("📄 Generating PDF report...")
    
    response = requests.get(
        f"{API_BASE}/export/pdf/{job_id}",
        headers=HEADERS
    )
    
    if response.status_code == 200:
        # Save PDF file
        with open(f"report_{job_id}.pdf", "wb") as f:
            f.write(response.content)
        
        print(f"✅ PDF report saved to: report_{job_id}.pdf")
        print("📊 Report includes:")
        print("  • Complete performance analysis")
        print("  • Equity curve charts")
        print("  • Risk metrics and statistics")
        print("  • Parameter optimization results")
    else:
        print("❌ Error generating PDF report")

# Use it:
if job_id:
    generate_pdf_report(job_id)
```

---

## 🎉 **Congratulations!**

You've successfully:

✅ **Installed** the trading optimization system  
✅ **Configured** API access and authentication  
✅ **Run** your first strategy optimization  
✅ **Analyzed** the performance results  
✅ **Exported** Pine Script for TradingView  
✅ **Generated** a professional PDF report  

---

## 🚀 **What's Next?**

### **Explore More Strategies**

```python
# Get list of all available strategies
response = requests.get(f"{API_BASE}/strategies/list", headers=HEADERS)
strategies = response.json()["strategies"]

print("Available strategies:")
for strategy in strategies[:10]:  # Show first 10
    print(f"  • {strategy['name']}: {strategy['description']}")
```

### **Advanced Optimization Features**

- **[Multi-Asset Validation](../strategies/validation-testing.md)**: Test across different symbols
- **[Walk-Forward Analysis](../strategies/performance-analysis.md)**: Robust time-series validation  
- **[Monte Carlo Testing](../strategies/best-practices.md)**: Statistical significance analysis
- **[Custom Strategies](../strategies/creating-strategies.md)**: Build your own trading strategies

### **Production Deployment**

- **[Docker Setup](../deployment/docker-deployment.md)**: Containerized deployment
- **[Monitoring](../deployment/monitoring-setup.md)**: Prometheus + Grafana monitoring
- **[Security](../deployment/security-hardening.md)**: Production security configuration

### **API Integration**

- **[Complete API Reference](../api/overview.md)**: All endpoints documented
- **[Integration Examples](../api/integration-examples.md)**: Real-world usage patterns
- **[Authentication Setup](../api/authentication.md)**: Secure API configuration

---

## 💡 **Tips for Success**

!!! tip "**Optimization Best Practices**"
    - Start with 50-100 trials for quick testing
    - Use 500+ trials for production strategies
    - Always validate out-of-sample performance
    - Test across multiple market conditions

!!! warning "**Common Pitfalls**"
    - Don't over-optimize on limited data
    - Always use proper validation frameworks
    - Be aware of look-ahead bias
    - Test strategies across different market regimes

!!! info "**Performance Tips**"
    - Use SSD storage for faster data access
    - Increase trials gradually to find optimal balance
    - Monitor system resources during optimization
    - Use async endpoints for long-running optimizations

---

## 🆘 **Need Help?**

- **📖 [Full Documentation](../index.md)**: Complete guides and references
- **🐛 [Troubleshooting](../examples/troubleshooting.md)**: Common issues and solutions
- **💬 [Community Support](https://github.com/trading-optimizer/hyperopt-strat/discussions)**: Ask questions and share insights
- **📧 [Enterprise Support](mailto:contact@trading-optimizer.com)**: Professional support options

**You're now ready to optimize trading strategies like a pro! 🚀** 