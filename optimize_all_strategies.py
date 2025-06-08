#!/usr/bin/env python3
"""
Comprehensive optimization of all available trading strategies
"""

import requests
import json
import time
from datetime import datetime

def optimize_all_strategies():
    """Run optimization on all 24 available strategies."""
    
    # Complete list of all 24 strategies based on the strategy factory
    all_strategies = [
        "MovingAverageCrossover",
        "MACD", 
        "RSI",
        "BollingerBands",
        "Momentum",
        "ROC",
        "Stochastic",
        "WilliamsR", 
        "UltimateOscillator",
        "VWAP",
        "OBV",
        "AD",
        "CMF",
        "ATR",
        "BollingerSqueeze",
        "KeltnerChannel",
        "HistoricalVolatility",
        "SupportResistance",
        "PivotPoints",
        "FibonacciRetracement",
        "DoubleTopBottom",
        "MTFTrendAnalysis",
        "MTFRSI",
        "MTFMACD"
    ]
    
    print("🚀 COMPREHENSIVE STRATEGY OPTIMIZATION")
    print("=" * 60)
    print(f"📊 Optimizing {len(all_strategies)} strategies")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration for comprehensive optimization
    optimization_request = {
        "strategies": all_strategies,
        "common_config": {
            "strategy_name": "BatchOptimization",
            "asset": "BTC",
            "timeframe": "4h",
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T00:00:00",  # Full year for comprehensive testing
            "optimization_config": {
                "max_evals": 25,  # Reasonable number for each strategy
                "timeout_minutes": 30  # 30 minutes per strategy
            }
        },
        "strategy_specific_configs": {},  # No strategy-specific overrides
        "parallel_jobs": 1  # Sequential execution (fixed serialization issues)
    }
    
    headers = {
        "X-API-Key": "dev_key_123",
        "Content-Type": "application/json"
    }
    
    print("📋 Optimization Configuration:")
    print(f"   • Asset: {optimization_request['common_config']['asset']}")
    print(f"   • Timeframe: {optimization_request['common_config']['timeframe']}")
    print(f"   • Date Range: {optimization_request['common_config']['start_date'][:10]} to {optimization_request['common_config']['end_date'][:10]}")
    print(f"   • Max Evaluations per Strategy: {optimization_request['common_config']['optimization_config']['max_evals']}")
    print(f"   • Timeout per Strategy: {optimization_request['common_config']['optimization_config']['timeout_minutes']} minutes")
    print(f"   • Parallel Jobs: {optimization_request['parallel_jobs']}")
    print(f"   • Total Estimated Time: ~{len(all_strategies) * optimization_request['common_config']['optimization_config']['timeout_minutes'] / 60:.1f} hours")
    print()
    
    try:
        # Check if server is running
        health_response = requests.get("http://127.0.0.1:8001/", timeout=10)
        print(f"✅ Server is running (status: {health_response.status_code})")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        print("💡 Please start the server with:")
        print("   uvicorn src.api.main:app --host 127.0.0.1 --port 8001 --reload")
        return False
    
    print("\n🚀 Starting comprehensive batch optimization...")
    print("⚠️  This is a LONG-RUNNING process. Expected duration: ~13+ hours")
    print("⚠️  The server will process strategies sequentially to avoid serialization issues")
    print()
    
    # Confirm before starting
    confirm = input("Continue with comprehensive optimization? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Optimization cancelled.")
        return False
    
    start_time = time.time()
    
    try:
        print("📤 Sending batch optimization request...")
        response = requests.post(
            "http://127.0.0.1:8001/api/v1/optimize/batch",
            json=optimization_request,
            headers=headers,
            timeout=600  # 10 minutes for initial request submission
        )
        
        print(f"📊 Request submitted: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n🎉 BATCH OPTIMIZATION COMPLETED!")
            print("=" * 50)
            
            # Parse results
            submitted_jobs = result.get('submitted_jobs', [])
            failed_submissions = result.get('failed_submissions', [])
            
            print(f"📈 Total Jobs: {len(submitted_jobs) + len(failed_submissions)}")
            print(f"✅ Successful: {len(submitted_jobs)}")
            print(f"❌ Failed: {len(failed_submissions)}")
            
            if submitted_jobs:
                print(f"\n🏆 SUCCESS RATE: {len(submitted_jobs)/(len(submitted_jobs) + len(failed_submissions))*100:.1f}%")
                
                print("\n📊 SUCCESSFUL OPTIMIZATIONS:")
                print("-" * 50)
                
                # Sort by performance score
                successful_jobs = [job for job in submitted_jobs if job.get('status') == 'completed']
                successful_jobs.sort(key=lambda x: x.get('result', {}).get('best_score', 0), reverse=True)
                
                for i, job in enumerate(successful_jobs, 1):
                    result_data = job.get('result', {})
                    score = result_data.get('best_score', 0)
                    runtime = job.get('execution_time_seconds', 0)
                    
                    print(f"{i:2d}. {job['strategy_name']:<25} | Score: {score:.4f} | Runtime: {runtime:6.1f}s")
                
                # Show top performers
                if successful_jobs:
                    print(f"\n🥇 TOP 5 PERFORMERS:")
                    print("-" * 30)
                    for i, job in enumerate(successful_jobs[:5], 1):
                        result_data = job.get('result', {})
                        score = result_data.get('best_score', 0)
                        params = result_data.get('best_params', {})
                        print(f"{i}. {job['strategy_name']}: {score:.4f}")
                        
                        # Show key parameters (first few)
                        key_params = list(params.items())[:3]
                        if key_params:
                            param_str = ", ".join([f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in key_params])
                            print(f"   Params: {param_str}")
            
            if failed_submissions:
                print(f"\n❌ FAILED OPTIMIZATIONS:")
                print("-" * 30)
                for job in failed_submissions:
                    error_msg = job.get('error_message', 'Unknown error')[:100]
                    print(f"   • {job['strategy_name']}: {error_msg}")
            
            elapsed_time = time.time() - start_time
            print(f"\n⏱️  Total execution time: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
            print(f"📊 Average time per strategy: {elapsed_time/len(all_strategies):.1f} seconds")
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_optimization_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"💾 Detailed results saved to: {filename}")
            
            return True
            
        else:
            print(f"❌ Request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {error_detail}")
            except:
                print(f"Response text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - optimization may still be running in background")
        print("Check server logs for progress")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    optimize_all_strategies() 