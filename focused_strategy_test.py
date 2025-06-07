#!/usr/bin/env python3
"""
🔍 Focused Strategy Performance Test
Test the 3 available API strategies with detailed analysis
"""

import asyncio
import json
import httpx
import pandas as pd
from datetime import datetime
import time

# Available strategies from API
API_STRATEGIES = [
    "MovingAverageCrossover",
    "RSIMeanReversion", 
    "MACDMomentum"
]

async def test_api_strategies():
    """
    Test the 3 available API strategies with detailed analysis
    """
    
    print("🔍 FOCUSED STRATEGY PERFORMANCE TEST")
    print("="*50)
    print(f"Testing {len(API_STRATEGIES)} available API strategies")
    print("Using 30 trials per strategy for faster results")
    print("="*50)
    
    # Test configuration
    base_config = {
        "symbol": "BTCUSDT",
        "timeframe": "4h", 
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "optimization_config": {
            "trials": 30,  # Fewer trials for faster testing
            "timeout": 300,  # 5 minute timeout
            "optimization_metric": "sharpe_ratio"
        }
    }
    
    results = []
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        
        for i, strategy_name in enumerate(API_STRATEGIES, 1):
            print(f"\n🔍 [{i}/{len(API_STRATEGIES)}] Testing {strategy_name}...")
            
            try:
                # Submit optimization job
                start_time = time.time()
                
                job_response = await client.post(
                    "http://localhost:8000/api/v1/optimize/single",
                    headers={"X-API-Key": "dev_key_123"},
                    json={**base_config, "strategy_name": strategy_name}
                )
                
                print(f"   📤 Job submission status: {job_response.status_code}")
                
                if job_response.status_code != 200:
                    response_text = job_response.text
                    print(f"   ❌ Failed to submit: {response_text[:200]}...")
                    continue
                
                job_data = job_response.json()
                job_id = job_data["job_id"]
                print(f"   📝 Job ID: {job_id}")
                
                # Wait for completion
                max_wait = 400  # 6.5 minutes total
                start_wait = time.time()
                
                while (time.time() - start_wait) < max_wait:
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                    # Check status
                    status_response = await client.get(
                        f"http://localhost:8000/api/v1/optimize/status/{job_id}",
                        headers={"X-API-Key": "dev_key_123"}
                    )
                    
                    if status_response.status_code != 200:
                        print(f"   ❌ Status check failed: {status_response.status_code}")
                        break
                        
                    status_data = status_response.json()
                    status = status_data["status"]
                    
                    if status == "completed":
                        # Get detailed results
                        results_response = await client.get(
                            f"http://localhost:8000/api/v1/optimize/results/{job_id}",
                            headers={"X-API-Key": "dev_key_123"}
                        )
                        
                        if results_response.status_code == 200:
                            result_data = results_response.json()
                            
                            # Debug: Print raw result structure
                            print(f"   🔍 Raw result keys: {list(result_data.keys())}")
                            
                            metrics = result_data.get("performance_metrics", {})
                            print(f"   🔍 Metrics keys: {list(metrics.keys())}")
                            
                            # Extract and validate metrics
                            total_return = metrics.get("total_return", 0)
                            sharpe_ratio = metrics.get("sharpe_ratio", 0)
                            max_drawdown = metrics.get("max_drawdown", 0)
                            win_rate = metrics.get("win_rate", 0)
                            profit_factor = metrics.get("profit_factor", 0)
                            total_trades = metrics.get("total_trades", 0)
                            
                            # Validate metrics make sense
                            if total_trades == 0:
                                print(f"   ⚠️  WARNING: 0 trades generated - check strategy logic")
                            
                            if total_return > 10:  # More than 1000%
                                print(f"   ⚠️  WARNING: Extremely high return {total_return*100:.1f}% - potential error")
                            
                            optimization_time = time.time() - start_time
                            
                            result_summary = {
                                "strategy": strategy_name,
                                "total_return_pct": total_return * 100,  # Convert to percentage
                                "sharpe_ratio": sharpe_ratio,
                                "max_drawdown_pct": max_drawdown * 100,
                                "win_rate_pct": win_rate * 100,
                                "profit_factor": profit_factor,
                                "total_trades": total_trades,
                                "optimization_time": optimization_time,
                                "best_parameters": result_data.get("best_parameters", {}),
                                "raw_metrics": metrics  # Include raw for debugging
                            }
                            
                            results.append(result_summary)
                            
                            print(f"   ✅ {strategy_name} COMPLETED:")
                            print(f"      💰 Total Return: {total_return*100:.1f}%")
                            print(f"      ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
                            print(f"      📉 Max Drawdown: {max_drawdown*100:.1f}%")
                            print(f"      🎯 Win Rate: {win_rate*100:.1f}%")
                            print(f"      🔢 Total Trades: {total_trades}")
                            print(f"      ⏱️  Optimization Time: {optimization_time:.0f}s")
                            
                            # Show best parameters
                            if result_data.get("best_parameters"):
                                print(f"      🎛️  Best Parameters:")
                                for param, value in result_data["best_parameters"].items():
                                    print(f"         {param}: {value}")
                            
                        else:
                            print(f"   ❌ Failed to get results: {results_response.status_code}")
                        break
                        
                    elif status == "failed":
                        error_msg = status_data.get("error", "Unknown error")
                        print(f"   ❌ Optimization failed: {error_msg}")
                        break
                    
                    elif status in ["pending", "running"]:
                        elapsed = time.time() - start_wait
                        progress = status_data.get("progress", {})
                        current_trial = progress.get("current_trial", "?")
                        total_trials = progress.get("total_trials", "?")
                        print(f"   ⏳ {status} - Trial {current_trial}/{total_trials} ({elapsed:.0f}s)")
                        continue
                    
                else:
                    print(f"   ⏰ Timeout after {max_wait}s")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                continue
    
    # Analyze results
    if results:
        print("\n" + "="*60)
        print("📊 STRATEGY COMPARISON RESULTS")
        print("="*60)
        
        df = pd.DataFrame(results)
        
        # Sort by total return
        df_sorted = df.sort_values('total_return_pct', ascending=False)
        
        print(f"\n🏆 RANKING BY TOTAL RETURN:")
        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            print(f"   {emoji} {idx}. {row['strategy']:20} | {row['total_return_pct']:7.1f}% | Sharpe: {row['sharpe_ratio']:5.2f} | DD: {row['max_drawdown_pct']:6.1f}% | Trades: {row['total_trades']:3.0f}")
        
        # Sort by Sharpe ratio  
        df_sharpe = df.sort_values('sharpe_ratio', ascending=False)
        print(f"\n⚡ RANKING BY SHARPE RATIO:")
        for idx, (_, row) in enumerate(df_sharpe.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            print(f"   {emoji} {idx}. {row['strategy']:20} | {row['total_return_pct']:7.1f}% | Sharpe: {row['sharpe_ratio']:5.2f} | DD: {row['max_drawdown_pct']:6.1f}% | Trades: {row['total_trades']:3.0f}")
        
        # Champion analysis
        champion = df_sorted.iloc[0]
        print(f"\n🏆 OVERALL CHAMPION: {champion['strategy']}")
        print(f"   💰 Total Return: {champion['total_return_pct']:.1f}%")
        print(f"   ⚡ Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
        print(f"   📉 Max Drawdown: {champion['max_drawdown_pct']:.1f}%")
        print(f"   🎯 Win Rate: {champion['win_rate_pct']:.1f}%")
        print(f"   💹 Profit Factor: {champion['profit_factor']:.2f}")
        print(f"   🔢 Total Trades: {champion['total_trades']:.0f}")
        
        # Compare with MovingAverageCrossover baseline if available
        ma_results = df[df['strategy'] == 'MovingAverageCrossover']
        if not ma_results.empty:
            ma_return = ma_results['total_return_pct'].iloc[0]
            print(f"\n🎯 BASELINE COMPARISON:")
            print(f"   MovingAverageCrossover baseline: {ma_return:.1f}%")
            
            better_strategies = df[df['total_return_pct'] > ma_return]
            if len(better_strategies) > 0:
                best_improvement = better_strategies['total_return_pct'].max() - ma_return
                print(f"   Strategies beating baseline: {len(better_strategies)}/{len(df)}")
                print(f"   Best improvement: +{best_improvement:.1f}%")
            else:
                print(f"   MovingAverageCrossover was the best performer!")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'focused_strategy_test_{timestamp}.csv', index=False)
        print(f"\n💾 Results saved to: focused_strategy_test_{timestamp}.csv")
        
        return df
    
    else:
        print("\n❌ No successful optimizations completed")
        return None

if __name__ == "__main__":
    print("🚀 Starting Focused Strategy Test...")
    results = asyncio.run(test_api_strategies())
    print("\n�� Test complete!") 