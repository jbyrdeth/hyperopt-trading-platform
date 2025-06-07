#!/usr/bin/env python3
"""
🏆 HyperOpt Strategy Tournament
Comprehensive performance comparison across all 24 available strategies
"""

import asyncio
import json
import httpx
import pandas as pd
from datetime import datetime
import time
import sys

# All 24 available strategies from the factory
STRATEGIES = [
    "MovingAverageCrossover", "MACD", "RSI", "BollingerBands", "Momentum",
    "ROC", "Stochastic", "WilliamsR", "UltimateOscillator", "VWAP",
    "OBV", "AD", "CMF", "ATR", "BollingerSqueeze", "KeltnerChannel",
    "HistoricalVolatility", "SupportResistance", "PivotPoints", 
    "FibonacciRetracement", "DoubleTopBottom", "MTFTrendAnalysis", 
    "MTFRSI", "MTFMACD"
]

async def run_strategy_tournament():
    """
    Run optimization for all available strategies and compare performance
    """
    
    print("🚀 HYPEROPT STRATEGY TOURNAMENT")
    print("="*60)
    print(f"Testing {len(STRATEGIES)} strategies on BTC 2023 data")
    print("Optimization: 50 trials per strategy for robust comparison")
    print("="*60)
    
    # Optimization configuration - same for all strategies for fair comparison
    base_config = {
        "symbol": "BTCUSDT",
        "timeframe": "4h", 
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "optimization_config": {
            "trials": 50,  # More trials for better comparison
            "timeout": 600,  # 10 minute timeout per strategy
            "optimization_metric": "sharpe_ratio"
        }
    }
    
    results = []
    failed_strategies = []
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        # Test each strategy
        for i, strategy_name in enumerate(STRATEGIES, 1):
            print(f"\n🔍 [{i:2d}/{len(STRATEGIES)}] Testing {strategy_name}...")
            
            try:
                # Submit optimization job
                start_time = time.time()
                
                job_response = await client.post(
                    "http://localhost:8000/api/v1/optimize/single",
                    headers={"X-API-Key": "dev_key_123"},
                    json={**base_config, "strategy_name": strategy_name}
                )
                
                if job_response.status_code != 200:
                    print(f"❌ {strategy_name}: Failed to submit job - {job_response.status_code}")
                    failed_strategies.append(strategy_name)
                    continue
                
                job_data = job_response.json()
                job_id = job_data["job_id"]
                print(f"   📝 Job ID: {job_id}")
                
                # Wait for completion with timeout
                max_wait_time = 720  # 12 minutes total timeout
                start_wait = time.time()
                
                while (time.time() - start_wait) < max_wait_time:
                    await asyncio.sleep(10)  # Check every 10 seconds
                    
                    # Check status
                    status_response = await client.get(
                        f"http://localhost:8000/api/v1/optimize/status/{job_id}",
                        headers={"X-API-Key": "dev_key_123"}
                    )
                    
                    if status_response.status_code != 200:
                        print(f"❌ {strategy_name}: Status check failed")
                        break
                        
                    status_data = status_response.json()
                    status = status_data["status"]
                    
                    if status == "completed":
                        # Get results
                        results_response = await client.get(
                            f"http://localhost:8000/api/v1/optimize/results/{job_id}",
                            headers={"X-API-Key": "dev_key_123"}
                        )
                        
                        if results_response.status_code == 200:
                            result_data = results_response.json()
                            metrics = result_data.get("performance_metrics", {})
                            
                            # Extract key performance metrics
                            total_return = metrics.get("total_return", 0) * 100  # Convert to percentage
                            sharpe_ratio = metrics.get("sharpe_ratio", 0)
                            max_drawdown = metrics.get("max_drawdown", 0) * 100
                            win_rate = metrics.get("win_rate", 0) * 100
                            profit_factor = metrics.get("profit_factor", 0)
                            total_trades = metrics.get("total_trades", 0)
                            optimization_time = time.time() - start_time
                            
                            results.append({
                                "strategy": strategy_name,
                                "total_return": total_return,
                                "sharpe_ratio": sharpe_ratio,
                                "max_drawdown": max_drawdown,
                                "win_rate": win_rate,
                                "profit_factor": profit_factor,
                                "total_trades": total_trades,
                                "optimization_time": optimization_time,
                                "best_parameters": result_data.get("best_parameters", {}),
                                "job_id": job_id
                            })
                            
                            print(f"   ✅ {strategy_name}: {total_return:.1f}% return, {sharpe_ratio:.2f} Sharpe, {max_drawdown:.1f}% DD")
                            print(f"      🎯 {total_trades} trades, {win_rate:.1f}% win rate, {optimization_time:.0f}s optimization")
                            
                        else:
                            print(f"❌ {strategy_name}: Failed to get results")
                            failed_strategies.append(strategy_name)
                        break
                        
                    elif status == "failed":
                        print(f"❌ {strategy_name}: Optimization failed")
                        failed_strategies.append(strategy_name)
                        break
                    
                    elif status in ["pending", "running"]:
                        elapsed = time.time() - start_wait
                        print(f"   ⏳ {strategy_name}: {status} ({elapsed:.0f}s elapsed)")
                        continue
                    
                else:
                    print(f"⏰ {strategy_name}: Timeout after {max_wait_time}s")
                    failed_strategies.append(strategy_name)
                    
            except Exception as e:
                print(f"❌ {strategy_name}: Error - {str(e)}")
                failed_strategies.append(strategy_name)
                continue
    
    # Analyze and display results
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("🏆 STRATEGY TOURNAMENT RESULTS")
        print("="*80)
        
        print(f"\n📊 TOURNAMENT SUMMARY:")
        print(f"   ✅ Successful: {len(results)} strategies")
        print(f"   ❌ Failed: {len(failed_strategies)} strategies")
        if failed_strategies:
            print(f"   Failed strategies: {', '.join(failed_strategies)}")
        
        # Sort by different metrics and show top 5 for each
        print(f"\n🥇 TOP 5 STRATEGIES BY TOTAL RETURN:")
        top_returns = df.nlargest(5, 'total_return')
        for idx, (_, row) in enumerate(top_returns.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "🏅"
            print(f"   {emoji} {idx}. {row['strategy']:25} | {row['total_return']:6.1f}% | Sharpe: {row['sharpe_ratio']:5.2f} | DD: {row['max_drawdown']:5.1f}% | Trades: {row['total_trades']:3.0f}")
        
        print(f"\n⚡ TOP 5 STRATEGIES BY SHARPE RATIO:")
        top_sharpe = df.nlargest(5, 'sharpe_ratio')
        for idx, (_, row) in enumerate(top_sharpe.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "🏅"
            print(f"   {emoji} {idx}. {row['strategy']:25} | {row['total_return']:6.1f}% | Sharpe: {row['sharpe_ratio']:5.2f} | DD: {row['max_drawdown']:5.1f}% | Trades: {row['total_trades']:3.0f}")
        
        print(f"\n🛡️  TOP 5 STRATEGIES BY RISK-ADJUSTED PERFORMANCE:")
        df['risk_adjusted_score'] = df['total_return'] / (df['max_drawdown'] + 1)  # Add 1 to avoid division by zero
        top_risk_adj = df.nlargest(5, 'risk_adjusted_score')
        for idx, (_, row) in enumerate(top_risk_adj.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "🏅"
            print(f"   {emoji} {idx}. {row['strategy']:25} | {row['total_return']:6.1f}% | Sharpe: {row['sharpe_ratio']:5.2f} | DD: {row['max_drawdown']:5.1f}% | R/A: {row['risk_adjusted_score']:5.2f}")
        
        print(f"\n💰 TOP 5 STRATEGIES BY PROFIT FACTOR:")
        top_profit = df.nlargest(5, 'profit_factor')
        for idx, (_, row) in enumerate(top_profit.iterrows(), 1):
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "🏅"
            print(f"   {emoji} {idx}. {row['strategy']:25} | {row['total_return']:6.1f}% | PF: {row['profit_factor']:5.2f} | WR: {row['win_rate']:5.1f}% | Trades: {row['total_trades']:3.0f}")
        
        # Overall champion analysis
        print(f"\n🏆 OVERALL CHAMPION ANALYSIS:")
        
        # Best overall performer (highest return)
        champion = df.loc[df['total_return'].idxmax()]
        print(f"\n   🥇 HIGHEST TOTAL RETURN CHAMPION:")
        print(f"      Strategy: {champion['strategy']}")
        print(f"      Total Return: {champion['total_return']:.1f}%")
        print(f"      Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
        print(f"      Max Drawdown: {champion['max_drawdown']:.1f}%")
        print(f"      Win Rate: {champion['win_rate']:.1f}%")
        print(f"      Profit Factor: {champion['profit_factor']:.2f}")
        print(f"      Total Trades: {champion['total_trades']:.0f}")
        
        # Best Sharpe ratio
        sharpe_champion = df.loc[df['sharpe_ratio'].idxmax()]
        print(f"\n   ⚡ BEST SHARPE RATIO CHAMPION:")
        print(f"      Strategy: {sharpe_champion['strategy']}")
        print(f"      Total Return: {sharpe_champion['total_return']:.1f}%")
        print(f"      Sharpe Ratio: {sharpe_champion['sharpe_ratio']:.2f}")
        print(f"      Max Drawdown: {sharpe_champion['max_drawdown']:.1f}%")
        
        # Most consistent (lowest drawdown with decent return)
        consistent_strategies = df[df['total_return'] > 20]  # At least 20% return
        if not consistent_strategies.empty:
            most_consistent = consistent_strategies.loc[consistent_strategies['max_drawdown'].idxmin()]
            print(f"\n   🛡️  MOST CONSISTENT CHAMPION (>20% return, lowest DD):")
            print(f"      Strategy: {most_consistent['strategy']}")
            print(f"      Total Return: {most_consistent['total_return']:.1f}%")
            print(f"      Sharpe Ratio: {most_consistent['sharpe_ratio']:.2f}")
            print(f"      Max Drawdown: {most_consistent['max_drawdown']:.1f}%")
        
        # Performance statistics
        print(f"\n📈 TOURNAMENT STATISTICS:")
        print(f"   Average Return: {df['total_return'].mean():.1f}%")
        print(f"   Median Return: {df['total_return'].median():.1f}%")
        print(f"   Best Return: {df['total_return'].max():.1f}%")
        print(f"   Worst Return: {df['total_return'].min():.1f}%")
        print(f"   Average Sharpe: {df['sharpe_ratio'].mean():.2f}")
        print(f"   Best Sharpe: {df['sharpe_ratio'].max():.2f}")
        print(f"   Average Drawdown: {df['max_drawdown'].mean():.1f}%")
        print(f"   Worst Drawdown: {df['max_drawdown'].max():.1f}%")
        
        # Compare with MovingAverageCrossover baseline
        ma_result = df[df['strategy'] == 'MovingAverageCrossover']
        if not ma_result.empty:
            ma_return = ma_result['total_return'].iloc[0]
            better_strategies = df[df['total_return'] > ma_return]
            print(f"\n🎯 BASELINE COMPARISON (vs MovingAverageCrossover {ma_return:.1f}%):")
            print(f"   Strategies beating baseline: {len(better_strategies)}/{len(df)}")
            if len(better_strategies) > 0:
                improvement = better_strategies['total_return'].max() - ma_return
                print(f"   Best improvement: +{improvement:.1f}% over baseline")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f'strategy_tournament_results_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 DETAILED RESULTS SAVED: {csv_filename}")
        
        # Save summary JSON
        summary = {
            "tournament_date": datetime.now().isoformat(),
            "total_strategies_tested": len(results),
            "failed_strategies": len(failed_strategies),
            "champion": {
                "strategy": champion['strategy'],
                "total_return": float(champion['total_return']),
                "sharpe_ratio": float(champion['sharpe_ratio']),
                "max_drawdown": float(champion['max_drawdown'])
            },
            "statistics": {
                "average_return": float(df['total_return'].mean()),
                "median_return": float(df['total_return'].median()),
                "best_return": float(df['total_return'].max()),
                "worst_return": float(df['total_return'].min()),
                "average_sharpe": float(df['sharpe_ratio'].mean()),
                "best_sharpe": float(df['sharpe_ratio'].max())
            }
        }
        
        json_filename = f'strategy_tournament_summary_{timestamp}.json'
        with open(json_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 SUMMARY SAVED: {json_filename}")
        
        return df
    
    else:
        print("❌ No successful optimizations completed")
        return None

# Run the tournament
if __name__ == "__main__":
    print("🚀 Starting HyperOpt Strategy Tournament...")
    print("This will take approximately 30-60 minutes to complete.")
    print("Each strategy gets 50 optimization trials for robust comparison.")
    
    start_time = time.time()
    results_df = asyncio.run(run_strategy_tournament())
    total_time = time.time() - start_time
    
    print(f"\n⏱️  TOURNAMENT COMPLETED in {total_time/60:.1f} minutes")
    
    if results_df is not None:
        print(f"\n🎯 KEY FINDINGS:")
        champion = results_df.loc[results_df['total_return'].idxmax()]
        ma_baseline = results_df[results_df['strategy'] == 'MovingAverageCrossover']
        
        if not ma_baseline.empty:
            ma_return = ma_baseline['total_return'].iloc[0]
            improvement = champion['total_return'] - ma_return
            print(f"   • Champion strategy: {champion['strategy']}")
            print(f"   • Champion return: {champion['total_return']:.1f}%")
            print(f"   • Baseline (MovingAverageCrossover): {ma_return:.1f}%")
            if improvement > 0:
                print(f"   • Improvement over baseline: +{improvement:.1f}%")
                print(f"   • The 45.2% baseline was {'NOT' if improvement > 0 else ''} the best performer!")
            else:
                print(f"   • The MovingAverageCrossover baseline was actually the champion!")
        else:
            print(f"   • Champion strategy: {champion['strategy']} with {champion['total_return']:.1f}% return")
    
    print("\n🏆 Tournament complete! Check the saved files for detailed results.") 