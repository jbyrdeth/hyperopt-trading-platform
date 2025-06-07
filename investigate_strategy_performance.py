#!/usr/bin/env python3
"""
🚨 CRITICAL PERFORMANCE INVESTIGATION

Comprehensive diagnostic investigation to understand why trading strategies
are performing extremely poorly (2.22% returns, 3 trades/year).

This script tests multiple scenarios to isolate root causes:
- Different market periods (bull/bear/volatile)
- Different assets (BTC, ETH, SOL)  
- Different timeframes (1h, 4h, 1d)
- Different parameter ranges
- Data quality validation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import asyncio
import httpx
import yfinance as yf
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

class PerformanceDiagnostics:
    """Comprehensive strategy performance diagnostics."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine(initial_capital=100000)
        self.results = {}
        
    def create_synthetic_data(self, scenario: str, days: int = 365) -> pd.DataFrame:
        """Create synthetic market data for different scenarios."""
        
        dates = pd.date_range(start='2023-01-01', periods=days*6, freq='4h')  # 4h intervals
        np.random.seed(42)
        
        if scenario == "strong_bull":
            # Strong uptrend with volatility - should generate many signals
            trend = np.linspace(100, 200, len(dates))  # 100% growth
            volatility = np.random.normal(0, 5, len(dates))
            prices = trend + volatility
            
        elif scenario == "strong_bear":
            # Strong downtrend - should generate short signals
            trend = np.linspace(100, 50, len(dates))  # -50% decline
            volatility = np.random.normal(0, 3, len(dates))
            prices = trend + volatility
            
        elif scenario == "high_volatility":
            # Sideways but very volatile - should generate many mean reversion signals
            base = 100
            volatility = np.random.normal(0, 10, len(dates))
            prices = base + volatility
            
        elif scenario == "trending_volatile":
            # Moderate uptrend with high volatility - ideal for most strategies
            trend = np.linspace(100, 150, len(dates))  # 50% growth
            volatility = np.random.normal(0, 8, len(dates))
            prices = trend + volatility
            
        else:  # "flat_low_vol"
            # Flat market with low volatility - should generate few signals
            base = 100
            volatility = np.random.normal(0, 1, len(dates))
            prices = base + volatility
        
        # Ensure prices are positive
        prices = np.maximum(prices, 10)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
            'close': prices,
            'volume': np.random.uniform(10000, 100000, len(dates))
        }, index=dates)
        
        # Ensure OHLC relationships are correct
        data['high'] = np.maximum.reduce([data['open'], data['high'], data['low'], data['close']])
        data['low'] = np.minimum.reduce([data['open'], data['high'], data['low'], data['close']])
        
        return data
    
    def get_aggressive_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get more aggressive parameters that should generate more signals."""
        
        aggressive_params = {
            'MovingAverageCrossover': {
                'fast_period': 5,    # Shorter periods = more signals
                'slow_period': 15,
                'ma_type': 'SMA',
                'signal_threshold': 0.005,  # Lower threshold = more signals
                'position_size_pct': 0.8,   # Larger positions
                'stop_loss_pct': 0.03,      # Tighter stops
                'take_profit_pct': 0.06
            },
            'RSIMeanReversion': {
                'period': 10,        # Shorter period = more sensitive
                'overbought': 65,    # Less extreme levels = more signals
                'oversold': 35,
                'exit_strategy': 'opposite',
                'position_size_pct': 0.8,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'MACD': {
                'fast_period': 8,    # Faster MACD = more signals
                'slow_period': 21,
                'signal_period': 6,
                'signal_threshold': 0.005,
                'position_size_pct': 0.8,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'BollingerBands': {
                'period': 15,        # Shorter period = more signals
                'std_dev': 1.5,      # Narrower bands = more breakouts
                'strategy_type': 'breakout',
                'position_size_pct': 0.8,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'WilliamsR': {
                'period': 10,        # Shorter period = more signals
                'overbought': -15,   # Less extreme levels
                'oversold': -85,
                'position_size_pct': 0.8,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            }
        }
        
        return aggressive_params.get(strategy_name, {
            'period': 10,
            'threshold': 0.005,
            'position_size_pct': 0.8,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        })
    
    def test_strategy_scenarios(self, strategy_name: str) -> Dict[str, Any]:
        """Test a strategy across multiple market scenarios."""
        
        print(f"\n🧪 TESTING {strategy_name}")
        print("-" * 50)
        
        scenarios = {
            "Strong Bull": "strong_bull",
            "Strong Bear": "strong_bear", 
            "High Volatility": "high_volatility",
            "Trending Volatile": "trending_volatile",
            "Flat Low Vol": "flat_low_vol"
        }
        
        strategy_results = {}
        
        for scenario_name, scenario_type in scenarios.items():
            print(f"  📊 {scenario_name}...")
            
            try:
                # Create test data
                data = self.create_synthetic_data(scenario_type, days=180)  # 6 months
                
                # Get aggressive parameters
                parameters = self.get_aggressive_parameters(strategy_name)
                
                # Create strategy instance
                strategy_class = self.factory.registry.get_strategy_class(strategy_name)
                strategy = strategy_class(**parameters)
                
                # Reset and run backtest
                self.engine.reset()
                result = self.engine.backtest_strategy(strategy, data, f"{scenario_type}/USDT")
                
                # Calculate metrics
                period_years = len(data) / (365 * 6)  # 6 intervals per day
                annual_return = ((1 + result.total_return) ** (1 / period_years) - 1) if period_years > 0 else result.total_return
                
                scenario_result = {
                    'total_return_pct': result.total_return * 100,
                    'annual_return_pct': annual_return * 100,
                    'total_trades': result.total_trades,
                    'win_rate_pct': result.win_rate * 100,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown_pct': result.max_drawdown * 100,
                    'profit_factor': result.profit_factor,
                    'market_return_pct': ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100,
                    'trades_per_month': result.total_trades / 6  # 6 month period
                }
                
                strategy_results[scenario_name] = scenario_result
                
                print(f"    Return: {scenario_result['annual_return_pct']:6.1f}% | "
                      f"Trades: {scenario_result['total_trades']:3d} | "
                      f"Market: {scenario_result['market_return_pct']:5.1f}%")
                
                # Flag concerning results
                if scenario_result['total_trades'] == 0:
                    print(f"    ⚠️ NO TRADES GENERATED")
                elif scenario_result['annual_return_pct'] < 5 and scenario_result['market_return_pct'] > 20:
                    print(f"    ⚠️ SEVERELY UNDERPERFORMING MARKET")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                strategy_results[scenario_name] = {'error': str(e)}
        
        return strategy_results
    
    def analyze_parameter_sensitivity(self, strategy_name: str) -> Dict[str, Any]:
        """Test how parameter changes affect signal generation."""
        
        print(f"\n🔍 PARAMETER SENSITIVITY: {strategy_name}")
        print("-" * 50)
        
        # Use trending volatile data (should be good for most strategies)
        data = self.create_synthetic_data("trending_volatile", days=90)
        
        if strategy_name == "MovingAverageCrossover":
            param_tests = [
                {'fast_period': 5, 'slow_period': 10, 'signal_threshold': 0.001},
                {'fast_period': 5, 'slow_period': 15, 'signal_threshold': 0.005},
                {'fast_period': 10, 'slow_period': 20, 'signal_threshold': 0.01},
                {'fast_period': 20, 'slow_period': 50, 'signal_threshold': 0.02}
            ]
        elif strategy_name == "RSIMeanReversion":
            param_tests = [
                {'period': 7, 'overbought': 60, 'oversold': 40},
                {'period': 14, 'overbought': 70, 'oversold': 30},
                {'period': 21, 'overbought': 80, 'oversold': 20}
            ]
        else:
            return {'message': 'Parameter sensitivity not implemented for this strategy'}
        
        sensitivity_results = {}
        
        for i, params in enumerate(param_tests):
            try:
                # Add common parameters
                full_params = {
                    **params,
                    'position_size_pct': 0.5,
                    'stop_loss_pct': 0.05,
                    'take_profit_pct': 0.10
                }
                
                strategy_class = self.factory.registry.get_strategy_class(strategy_name)
                strategy = strategy_class(**full_params)
                
                self.engine.reset()
                result = self.engine.backtest_strategy(strategy, data, "PARAM_TEST/USDT")
                
                sensitivity_results[f"Config_{i+1}"] = {
                    'parameters': params,
                    'trades': result.total_trades,
                    'return_pct': result.total_return * 100,
                    'win_rate_pct': result.win_rate * 100
                }
                
                print(f"  Config {i+1}: {params} → {result.total_trades} trades, {result.total_return*100:5.1f}%")
                
            except Exception as e:
                print(f"  Config {i+1}: Error - {e}")
        
        return sensitivity_results
    
    def check_data_quality_impact(self) -> Dict[str, Any]:
        """Test how data quality affects strategy performance."""
        
        print(f"\n📊 DATA QUALITY IMPACT ANALYSIS")
        print("-" * 50)
        
        # Create data with different quality issues
        base_data = self.create_synthetic_data("trending_volatile", days=90)
        
        data_scenarios = {
            "Clean Data": base_data.copy(),
            "Missing Data": self._introduce_gaps(base_data.copy()),
            "Noisy Data": self._add_noise(base_data.copy()),
            "Low Volume": self._reduce_volume(base_data.copy())
        }
        
        test_strategy = "MovingAverageCrossover"
        params = self.get_aggressive_parameters(test_strategy)
        
        quality_results = {}
        
        for scenario_name, data in data_scenarios.items():
            try:
                strategy_class = self.factory.registry.get_strategy_class(test_strategy)
                strategy = strategy_class(**params)
                
                self.engine.reset()
                result = self.engine.backtest_strategy(strategy, data, f"{scenario_name}/USDT")
                
                quality_results[scenario_name] = {
                    'trades': result.total_trades,
                    'return_pct': result.total_return * 100,
                    'data_points': len(data),
                    'data_completeness': (len(data) / len(base_data)) * 100
                }
                
                print(f"  {scenario_name:15}: {result.total_trades:3d} trades, {result.total_return*100:6.1f}% return")
                
            except Exception as e:
                print(f"  {scenario_name:15}: Error - {e}")
        
        return quality_results
    
    def _introduce_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Introduce random gaps in data."""
        # Remove 10% of data points randomly
        drop_indices = np.random.choice(data.index, size=int(len(data) * 0.1), replace=False)
        return data.drop(drop_indices)
    
    def _add_noise(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add noise to price data."""
        noise = np.random.normal(0, data['close'].std() * 0.1, len(data))
        data['close'] += noise
        data['open'] += noise * 0.5
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        return data
    
    def _reduce_volume(self, data: pd.DataFrame) -> pd.DataFrame:
        """Reduce volume to very low levels."""
        data['volume'] = data['volume'] * 0.01  # 1% of original volume
        return data
    
    async def test_live_api_performance(self) -> Dict[str, Any]:
        """Test strategies using the live optimization API."""
        
        print(f"\n🌐 LIVE API PERFORMANCE TEST")
        print("-" * 50)
        
        api_results = {}
        test_strategies = ['MovingAverageCrossover', 'RSIMeanReversion', 'WilliamsR']
        
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                for strategy_name in test_strategies:
                    print(f"  🔄 Testing {strategy_name} via API...")
                    
                    try:
                        # More aggressive optimization settings
                        optimization_config = {
                            "trials": 30,
                            "timeout": 180,
                            "optimization_metric": "total_return"
                        }
                        
                        # Start optimization job
                        job_response = await client.post(
                            "http://localhost:8000/api/v1/optimize/single",
                            headers={"X-API-Key": "dev_key_123"},
                            json={
                                "strategy_name": strategy_name,
                                "symbol": "BTCUSDT", 
                                "timeframe": "4h",
                                "start_date": "2023-01-01",
                                "end_date": "2023-06-30",  # 6 months
                                "optimization_config": optimization_config
                            }
                        )
                        
                        if job_response.status_code == 200:
                            job_data = job_response.json()
                            job_id = job_data.get("job_id")
                            
                            # Wait for completion
                            for attempt in range(20):  # 3 minute timeout
                                await asyncio.sleep(10)
                                
                                status_response = await client.get(
                                    f"http://localhost:8000/api/v1/optimize/status/{job_id}",
                                    headers={"X-API-Key": "dev_key_123"}
                                )
                                
                                if status_response.status_code == 200:
                                    status_data = status_response.json()
                                    
                                    if status_data["status"] == "completed":
                                        results_response = await client.get(
                                            f"http://localhost:8000/api/v1/optimize/results/{job_id}",
                                            headers={"X-API-Key": "dev_key_123"}
                                        )
                                        
                                        if results_response.status_code == 200:
                                            result_data = results_response.json()
                                            metrics = result_data.get("performance_metrics", {})
                                            
                                            api_results[strategy_name] = {
                                                'annual_return_pct': metrics.get('annual_return', 0) * 100,
                                                'total_trades': metrics.get('total_trades', 0),
                                                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                                                'win_rate_pct': metrics.get('win_rate', 0) * 100,
                                                'max_drawdown_pct': metrics.get('max_drawdown', 0) * 100,
                                                'trials_completed': result_data.get('trials_completed', 0),
                                                'best_params': result_data.get('best_parameters', {})
                                            }
                                            
                                            print(f"    ✅ Return: {api_results[strategy_name]['annual_return_pct']:6.1f}% | "
                                                  f"Trades: {api_results[strategy_name]['total_trades']:3d} | "
                                                  f"Trials: {api_results[strategy_name]['trials_completed']}")
                                        break
                                        
                                    elif status_data["status"] == "failed":
                                        print(f"    ❌ Failed: {status_data.get('error', 'Unknown')}")
                                        break
                        else:
                            print(f"    ❌ API request failed: {job_response.status_code}")
                            
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                    
                    await asyncio.sleep(2)  # Rate limiting
                    
        except Exception as e:
            print(f"❌ API testing failed: {e}")
        
        return api_results
    
    def generate_diagnostic_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive diagnostic report."""
        
        report = []
        report.append("🚨 STRATEGY PERFORMANCE DIAGNOSTIC REPORT")
        report.append("=" * 60)
        
        # Executive Summary
        report.append("\n📋 EXECUTIVE SUMMARY")
        report.append("-" * 30)
        
        total_tests = 0
        high_performance_tests = 0
        zero_trade_tests = 0
        
        for section_name, section_data in all_results.items():
            if isinstance(section_data, dict):
                for test_name, test_data in section_data.items():
                    if isinstance(test_data, dict) and 'total_trades' in test_data:
                        total_tests += 1
                        if test_data.get('annual_return_pct', 0) > 15:
                            high_performance_tests += 1
                        if test_data.get('total_trades', 1) == 0:
                            zero_trade_tests += 1
        
        if total_tests > 0:
            report.append(f"Total Tests Conducted: {total_tests}")
            report.append(f"High Performance (>15% annual): {high_performance_tests} ({high_performance_tests/total_tests*100:.1f}%)")
            report.append(f"Zero Trade Tests: {zero_trade_tests} ({zero_trade_tests/total_tests*100:.1f}%)")
        
        # Detailed findings
        for section_name, section_data in all_results.items():
            report.append(f"\n📊 {section_name.upper()}")
            report.append("-" * 30)
            
            if isinstance(section_data, dict):
                for test_name, test_data in section_data.items():
                    if isinstance(test_data, dict):
                        if 'error' in test_data:
                            report.append(f"  ❌ {test_name}: {test_data['error']}")
                        else:
                            return_pct = test_data.get('annual_return_pct', test_data.get('return_pct', 0))
                            trades = test_data.get('total_trades', test_data.get('trades', 0))
                            report.append(f"  📈 {test_name}: {return_pct:6.1f}% return, {trades:3d} trades")
        
        # Recommendations
        report.append(f"\n💡 DIAGNOSTIC RECOMMENDATIONS")
        report.append("-" * 30)
        
        if zero_trade_tests > total_tests * 0.5:
            report.append("🚨 CRITICAL: >50% of tests generated zero trades")
            report.append("   → Check signal generation logic")
            report.append("   → Verify parameter thresholds are not too restrictive")
            report.append("   → Review strategy entry/exit conditions")
        
        if high_performance_tests < total_tests * 0.2:
            report.append("⚠️ WARNING: <20% of tests achieved >15% annual returns")
            report.append("   → Parameter optimization may be insufficient")
            report.append("   → Consider expanding parameter search spaces")
            report.append("   → Test with longer optimization periods")
        
        report.append("\n🎯 NEXT STEPS:")
        report.append("1. Focus on fixing zero-trade scenarios first")
        report.append("2. Expand parameter ranges for optimization")
        report.append("3. Test with different market periods and conditions")
        report.append("4. Validate data quality and completeness")
        report.append("5. Consider strategy-specific optimizations")
        
        return "\n".join(report)

async def main():
    """Run comprehensive strategy performance investigation."""
    
    print("🚨 CRITICAL STRATEGY PERFORMANCE INVESTIGATION")
    print("=" * 60)
    print("Mission: Diagnose why strategies are severely underperforming")
    print("Target: Identify root causes of 2.22% returns and 3 trades/year")
    print()
    
    diagnostics = PerformanceDiagnostics()
    all_results = {}
    
    # Test 1: Multi-scenario testing
    print("🧪 PHASE 1: MULTI-SCENARIO TESTING")
    print("=" * 60)
    
    test_strategies = ['MovingAverageCrossover', 'RSIMeanReversion', 'WilliamsR', 'MACD']
    scenario_results = {}
    
    for strategy in test_strategies:
        scenario_results[strategy] = diagnostics.test_strategy_scenarios(strategy)
        time.sleep(1)  # Brief pause
    
    all_results['scenario_testing'] = scenario_results
    
    # Test 2: Parameter sensitivity
    print(f"\n🔍 PHASE 2: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    sensitivity_results = {}
    for strategy in ['MovingAverageCrossover', 'RSIMeanReversion']:
        sensitivity_results[strategy] = diagnostics.analyze_parameter_sensitivity(strategy)
    
    all_results['parameter_sensitivity'] = sensitivity_results
    
    # Test 3: Data quality impact
    print(f"\n📊 PHASE 3: DATA QUALITY IMPACT")
    print("=" * 60)
    
    quality_results = diagnostics.check_data_quality_impact()
    all_results['data_quality'] = quality_results
    
    # Test 4: Live API testing (if API is running)
    print(f"\n🌐 PHASE 4: LIVE API TESTING")
    print("=" * 60)
    
    try:
        api_results = await diagnostics.test_live_api_performance()
        all_results['api_testing'] = api_results
    except Exception as e:
        print(f"⚠️ API testing skipped: {e}")
    
    # Generate comprehensive report
    print(f"\n📋 GENERATING DIAGNOSTIC REPORT")
    print("=" * 60)
    
    report = diagnostics.generate_diagnostic_report(all_results)
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"strategy_performance_diagnostics_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📁 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Market context for comparison
    print(f"\n📈 MARKET CONTEXT (for comparison)")
    print("-" * 30)
    
    try:
        # Get BTC performance in 2023 for comparison
        btc_data = yf.download("BTC-USD", start="2023-01-01", end="2023-12-31", progress=False)
        if not btc_data.empty:
            btc_return = ((btc_data['Close'][-1] / btc_data['Close'][0]) - 1) * 100
            print(f"BTC Buy & Hold 2023: {btc_return:.1f}%")
            
            if btc_return > 50:
                print("🚨 CRITICAL: BTC had strong performance in 2023")
                print("   → Strategies severely underperforming market")
            elif btc_return < 0:
                print("📉 INFO: BTC declined in 2023")
                print("   → Bear market may explain some poor performance")
        
    except Exception as e:
        print(f"Could not fetch BTC data: {e}")
    
    print(f"\n🎯 INVESTIGATION COMPLETE")
    print("=" * 60)
    print("Review the detailed results to identify root causes.")
    print("Focus on scenarios that generated the most trades and highest returns.")
    print("Use findings to fix parameter ranges and optimization settings.")
    
    return all_results

if __name__ == "__main__":
    results = asyncio.run(main()) 