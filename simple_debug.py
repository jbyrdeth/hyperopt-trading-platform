#!/usr/bin/env python3
"""
Simple Debug Script for Strategy Testing

Test the API directly to understand the 0 trades issue
"""

import requests
import json
import time
from datetime import datetime

def test_single_strategy():
    """Test a single strategy optimization to debug the issue."""
    
    print("🔍 DEBUGGING STRATEGY OPTIMIZATION")
    print("=" * 50)
    
    # Test configuration with shorter period for faster debugging
    config = {
        "strategy_name": "MovingAverageCrossover",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "start_date": "2023-01-01",
        "end_date": "2023-02-28",  # Shorter period
        "optimization_config": {
            "trials": 5,  # Fewer trials for debugging
            "timeout": 120,
            "optimization_metric": "sharpe_ratio"
        }
    }
    
    print(f"📊 Test Configuration:")
    print(f"   Strategy: {config['strategy_name']}")
    print(f"   Symbol: {config['symbol']}")
    print(f"   Period: {config['start_date']} to {config['end_date']}")
    print(f"   Trials: {config['optimization_config']['trials']}")
    
    # Submit optimization job
    print(f"\n🚀 Submitting optimization job...")
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/optimize/single",
            headers={"X-API-Key": "dev_key_123"},
            json=config,
            timeout=30
        )
        
        if response.status_code == 200:
            job_data = response.json()
            job_id = job_data.get("job_id")
            print(f"   ✅ Job submitted successfully: {job_id}")
            
            # Poll for results
            print(f"\n⏳ Waiting for optimization to complete...")
            max_wait = 180  # 3 minutes max
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                try:
                    status_response = requests.get(
                        f"http://localhost:8000/api/v1/optimize/status/{job_id}",
                        headers={"X-API-Key": "dev_key_123"},
                        timeout=10
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get("status")
                        
                        print(f"   Status: {status}")
                        
                        if status == "completed":
                            print(f"   ✅ Optimization completed!")
                            
                            # Get results
                            results_response = requests.get(
                                f"http://localhost:8000/api/v1/optimize/results/{job_id}",
                                headers={"X-API-Key": "dev_key_123"},
                                timeout=10
                            )
                            
                            if results_response.status_code == 200:
                                results = results_response.json()
                                
                                print(f"\n📊 OPTIMIZATION RESULTS:")
                                print(f"   Job ID: {job_id}")
                                print(f"   Strategy: {results.get('strategy_name', 'N/A')}")
                                print(f"   Symbol: {results.get('symbol', 'N/A')}")
                                
                                # Best parameters
                                best_params = results.get('best_parameters', {})
                                print(f"\n🎯 Best Parameters:")
                                for key, value in best_params.items():
                                    print(f"   {key}: {value}")
                                
                                # Performance metrics
                                metrics = results.get('performance_metrics', {})
                                print(f"\n📈 Performance Metrics:")
                                print(f"   Total Return: {metrics.get('total_return', 0)*100:.2f}%")
                                print(f"   Annual Return: {metrics.get('annual_return', 0)*100:.2f}%")
                                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                                print(f"   Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
                                print(f"   Total Trades: {metrics.get('total_trades', 0)}")
                                print(f"   Win Rate: {metrics.get('win_rate', 0)*100:.2f}%")
                                print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                                
                                # Check for the 0 trades issue
                                total_trades = metrics.get('total_trades', 0)
                                total_return = metrics.get('total_return', 0)
                                
                                if total_trades == 0 and total_return > 0.1:
                                    print(f"\n⚠️ ISSUE DETECTED:")
                                    print(f"   0 trades but {total_return*100:.2f}% return!")
                                    print(f"   This indicates a calculation bug in the backtesting engine.")
                                    
                                    # Additional debugging info
                                    print(f"\n🔍 Additional Debug Info:")
                                    equity_curve = results.get('equity_curve', [])
                                    if equity_curve:
                                        print(f"   Equity curve length: {len(equity_curve)}")
                                        print(f"   Starting equity: {equity_curve[0] if equity_curve else 'N/A'}")
                                        print(f"   Ending equity: {equity_curve[-1] if equity_curve else 'N/A'}")
                                    
                                    trades = results.get('trades', [])
                                    print(f"   Trades list length: {len(trades)}")
                                    
                                elif total_trades > 0:
                                    print(f"\n✅ TRADES DETECTED:")
                                    print(f"   {total_trades} trades executed successfully")
                                    
                                    # Show first few trades
                                    trades = results.get('trades', [])
                                    if trades:
                                        print(f"\n💰 Sample Trades:")
                                        for i, trade in enumerate(trades[:3]):
                                            print(f"   Trade {i+1}: {trade}")
                                
                                return results
                            else:
                                print(f"   ❌ Failed to get results: {results_response.status_code}")
                                return None
                                
                        elif status == "failed":
                            print(f"   ❌ Optimization failed!")
                            error = status_data.get("error", "Unknown error")
                            print(f"   Error: {error}")
                            return None
                        
                        elif status in ["pending", "running"]:
                            print(f"   ⏳ Still {status}... waiting...")
                            time.sleep(10)
                        else:
                            print(f"   ❓ Unknown status: {status}")
                            time.sleep(5)
                    else:
                        print(f"   ❌ Status check failed: {status_response.status_code}")
                        time.sleep(5)
                        
                except requests.exceptions.RequestException as e:
                    print(f"   ⚠️ Request error: {e}")
                    time.sleep(5)
            
            print(f"\n⏰ Timeout reached after {max_wait} seconds")
            return None
            
        else:
            print(f"   ❌ Failed to submit job: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        return None

def check_api_health():
    """Check if the API is healthy and responsive."""
    print("🏥 CHECKING API HEALTH")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ API is healthy")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   Timestamp: {health_data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"   ❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API not accessible: {e}")
        return False

def list_available_strategies():
    """List available strategies from the API."""
    print("\n📋 AVAILABLE STRATEGIES")
    print("=" * 30)
    
    try:
        response = requests.get(
            "http://localhost:8000/api/v1/strategies",
            headers={"X-API-Key": "dev_key_123"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            strategies = data.get("strategies", [])
            print(f"   Found {len(strategies)} strategies:")
            for i, strategy in enumerate(strategies, 1):
                name = strategy.get("name", "Unknown")
                description = strategy.get("description", "No description")
                print(f"   {i}. {name}: {description}")
            return strategies
        else:
            print(f"   ❌ Failed to get strategies: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request error: {e}")
        return []

if __name__ == "__main__":
    print("🚀 STARTING STRATEGY DEBUG SESSION")
    print("=" * 60)
    
    # 1. Check API health
    if not check_api_health():
        print("\n❌ API is not accessible. Please start the server first.")
        exit(1)
    
    # 2. List available strategies
    strategies = list_available_strategies()
    if not strategies:
        print("\n❌ No strategies available. Check API configuration.")
        exit(1)
    
    # 3. Test single strategy
    print(f"\n" + "="*60)
    results = test_single_strategy()
    
    if results:
        print(f"\n🎯 DEBUG SESSION COMPLETE!")
        print(f"   Results obtained successfully.")
    else:
        print(f"\n❌ DEBUG SESSION FAILED!")
        print(f"   No results obtained.") 