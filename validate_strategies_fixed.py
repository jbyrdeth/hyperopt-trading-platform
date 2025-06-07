#!/usr/bin/env python3
"""
24-Strategy Validation Script - Fixed Version

Tests all 24 strategies with the fixed backtesting engine using proper imports
and parameter handling.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

def create_test_data():
    """Create simple test data for validation."""
    dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='4h')
    np.random.seed(42)
    
    # Create trending data with some volatility
    base_prices = np.linspace(100, 130, len(dates))  # 30% uptrend over 2 months
    noise = np.random.normal(0, 3, len(dates))
    prices = base_prices + noise
    
    # Ensure prices are positive
    prices = np.maximum(prices, 50)
    
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.003,
        'low': prices * 0.997,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    return data

def get_strategy_defaults(strategy_name: str) -> Dict[str, Any]:
    """Get reasonable default parameters for a strategy."""
    
    # Define default parameters for each strategy
    defaults = {
        'MovingAverageCrossover': {
            'fast_period': 10,
            'slow_period': 20,
            'ma_type': 'SMA',
            'signal_threshold': 0.01,
            'position_size_pct': 0.5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'MACD': {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9,
            'signal_threshold': 0.01,
            'position_size_pct': 0.5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'RSI': {
            'period': 14,
            'overbought': 70,
            'oversold': 30,
            'exit_strategy': 'opposite',
            'position_size_pct': 0.5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'BollingerBands': {
            'period': 20,
            'std_dev': 2.0,
            'strategy_type': 'breakout',
            'position_size_pct': 0.5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        },
        'Momentum': {
            'period': 14,
            'threshold': 0.02,
            'position_size_pct': 0.5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        }
    }
    
    # Return defaults for known strategies, or generic defaults
    return defaults.get(strategy_name, {
        'period': 14,
        'threshold': 0.01,
        'position_size_pct': 0.5,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.10
    })

def test_single_strategy(strategy_name: str, data: pd.DataFrame, factory: StrategyFactory, engine: BacktestingEngine) -> Dict[str, Any]:
    """Test a single strategy."""
    
    result = {
        'name': strategy_name,
        'success': False,
        'error': None,
        'metrics': {}
    }
    
    try:
        # Get default parameters
        parameters = get_strategy_defaults(strategy_name)
        
        # Create strategy using the factory's create method
        strategy_class = factory.registry.get_strategy_class(strategy_name)
        strategy = strategy_class(**parameters)
        
        # Reset engine
        engine.reset()
        
        # Run backtest
        backtest_result = engine.backtest_strategy(strategy, data, "TEST/USDT")
        
        # Store metrics
        result['metrics'] = {
            'total_return': backtest_result.total_return,
            'annual_return': backtest_result.annual_return,
            'sharpe_ratio': backtest_result.sharpe_ratio,
            'max_drawdown': backtest_result.max_drawdown,
            'total_trades': backtest_result.total_trades,
            'win_rate': backtest_result.win_rate,
            'profit_factor': backtest_result.profit_factor,
            'volatility': backtest_result.volatility
        }
        
        result['success'] = True
        
        print(f"   ✅ {strategy_name}: {backtest_result.total_return*100:.2f}% return, {backtest_result.total_trades} trades")
        
        # Check for the old bug
        if backtest_result.total_trades == 0 and abs(backtest_result.total_return) > 0.05:
            print(f"   ⚠️ WARNING: Possible bug - 0 trades but {backtest_result.total_return*100:.2f}% return")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"   ❌ {strategy_name}: {e}")
    
    return result

def main():
    """Main validation function."""
    
    print("🚀 24-STRATEGY VALIDATION - FIXED VERSION")
    print("=" * 50)
    print("Mission: Validate all 24 strategies with fixed backtesting engine")
    print()
    
    # Initialize components
    print("🔧 Initializing components...")
    factory = StrategyFactory()
    engine = BacktestingEngine(initial_capital=100000)
    
    # Get all strategies
    strategies = factory.get_all_strategies()
    print(f"📋 Found {len(strategies)} strategies to test")
    
    # Create test data
    print("📊 Creating test data...")
    data = create_test_data()
    print(f"   Data points: {len(data)}")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"   Total price change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # Test all strategies
    print(f"\n🧪 Testing all {len(strategies)} strategies...")
    results = {}
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, strategy_name in enumerate(strategies, 1):
        print(f"\n[{i:2d}/{len(strategies)}] Testing {strategy_name}...")
        
        result = test_single_strategy(strategy_name, data, factory, engine)
        results[strategy_name] = result
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    end_time = time.time()
    
    # Generate summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Total Strategies: {len(strategies)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(successful/len(strategies)*100):.1f}%")
    print(f"Time Taken: {end_time - start_time:.1f} seconds")
    
    # Show top performers
    successful_strategies = [
        (name, data) for name, data in results.items() 
        if data['success'] and data['metrics']['total_trades'] > 0
    ]
    
    if successful_strategies:
        # Sort by total return
        successful_strategies.sort(
            key=lambda x: x[1]['metrics']['total_return'], 
            reverse=True
        )
        
        print(f"\n🏆 TOP 10 PERFORMERS:")
        for i, (name, data) in enumerate(successful_strategies[:10], 1):
            metrics = data['metrics']
            print(f"   {i:2d}. {name}: {metrics['total_return']*100:6.2f}% return, "
                  f"{metrics['total_trades']:3d} trades, {metrics['sharpe_ratio']:5.2f} Sharpe")
    
    # Show failed strategies
    failed_strategies = [name for name, data in results.items() if not data['success']]
    if failed_strategies:
        print(f"\n❌ FAILED STRATEGIES ({len(failed_strategies)}):")
        for name in failed_strategies:
            error = results[name]['error']
            print(f"   - {name}: {error}")
    
    # Check for bug patterns
    zero_trade_strategies = [
        name for name, data in results.items() 
        if data['success'] and data['metrics']['total_trades'] == 0 and abs(data['metrics']['total_return']) > 0.05
    ]
    
    if zero_trade_strategies:
        print(f"\n⚠️ POTENTIAL BUG DETECTED:")
        print(f"   Strategies with 0 trades but >5% return: {zero_trade_strategies}")
    else:
        print(f"\n✅ NO BUG PATTERNS DETECTED")
        print(f"   All strategies with significant returns also have trades recorded")
    
    # Save results
    output_file = "validation_results_24_strategies_fixed.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📁 Results saved to {output_file}")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")
    
    print(f"\n🎯 VALIDATION COMPLETE!")
    if successful > 0:
        print(f"   ✅ {successful} strategies validated successfully")
        print(f"   🎯 Ready for enhanced tournament analysis")
    else:
        print(f"   ❌ No strategies validated - need to investigate issues")
    
    return results

if __name__ == "__main__":
    results = main() 