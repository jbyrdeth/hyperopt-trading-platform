#!/usr/bin/env python3
"""
Fix DatetimeIndex Issue - Critical Data Format Fix

The backtesting engine expects DatetimeIndex for time-based calculations
but our test data has RangeIndex. This script fixes the data format.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

def create_proper_test_data() -> pd.DataFrame:
    """Create test data with proper DatetimeIndex for backtesting engine."""
    
    # Create proper datetime index
    start_date = datetime(2023, 1, 1)
    periods = 360  # 60 days * 6 (4-hour intervals per day)
    dates = pd.date_range(start=start_date, periods=periods, freq='4H')
    
    np.random.seed(42)
    
    # Create realistic crypto-like price data
    base_price = 30000  # Starting price like BTC
    trend_component = np.linspace(0, 0.15, len(dates))  # 15% uptrend
    volatility = 0.03
    
    returns = np.random.normal(0, volatility, len(dates))
    returns += trend_component * volatility * 0.5  # Add trend
    
    # Add some volatility clustering
    for i in range(1, len(returns)):
        returns[i] += returns[i-1] * 0.1
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate realistic OHLCV data
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices))))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices))))
    volumes = np.random.lognormal(15, 0.5, len(prices))  # Realistic volumes
    
    # Create DataFrame with proper DatetimeIndex
    data = pd.DataFrame({
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    }, index=dates)  # THIS IS THE KEY FIX - DatetimeIndex as index
    
    # Ensure OHLC logic is correct
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
    
    return data

def test_data_format():
    """Test different data formats to identify what works."""
    
    print("🔧 TESTING DATA FORMAT FIXES")
    print("=" * 60)
    
    # Test 1: RangeIndex (current failing approach)
    print("\n🧪 Test 1: RangeIndex (current failing approach)")
    dates = pd.date_range(start='2023-01-01', periods=100, freq='4H')
    data1 = pd.DataFrame({
        'open': np.random.random(100) * 1000 + 30000,
        'high': np.random.random(100) * 1000 + 30500,
        'low': np.random.random(100) * 1000 + 29500,
        'close': np.random.random(100) * 1000 + 30000,
        'volume': np.random.random(100) * 1000000,
        'timestamp': dates  # Timestamp as column
    })
    print(f"   Index type: {type(data1.index)}")
    print(f"   Index name: {data1.index.name}")
    print(f"   Has timestamp column: {'timestamp' in data1.columns}")
    
    # Test 2: DatetimeIndex (proper approach)
    print("\n🧪 Test 2: DatetimeIndex (proper approach)")
    data2 = pd.DataFrame({
        'open': np.random.random(100) * 1000 + 30000,
        'high': np.random.random(100) * 1000 + 30500,
        'low': np.random.random(100) * 1000 + 29500,
        'close': np.random.random(100) * 1000 + 30000,
        'volume': np.random.random(100) * 1000000
    }, index=dates)  # DatetimeIndex as index
    print(f"   Index type: {type(data2.index)}")
    print(f"   Index name: {data2.index.name}")
    print(f"   Has timestamp column: {'timestamp' in data2.columns}")
    
    return data1, data2

def test_strategy_with_proper_data():
    """Test a strategy with properly formatted data."""
    
    print(f"\n🧪 TESTING STRATEGY WITH PROPER DATA FORMAT")
    print("=" * 60)
    
    # Create proper test data
    data = create_proper_test_data()
    
    print(f"📊 Data format verification:")
    print(f"   Shape: {data.shape}")
    print(f"   Index type: {type(data.index)}")
    print(f"   Index name: {data.index.name}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print()
    
    # Test strategy creation and backtesting
    factory = StrategyFactory()
    engine = BacktestingEngine()
    
    # Test MovingAverageCrossover with proper data
    print("🧪 Testing MovingAverageCrossover with proper DatetimeIndex...")
    
    try:
        # Create strategy using direct instantiation
        strategy_class = factory.registry.get_strategy_class('MovingAverageCrossover')
        
        params = {
            'fast_period': 5,
            'slow_period': 15,
            'ma_type': 'EMA',
            'signal_threshold': 0.002,
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.05
        }
        
        strategy = strategy_class(**params)
        print(f"  ✅ Strategy created: {strategy.name}")
        
        # Test backtesting
        print("  🧪 Running backtest...")
        result = engine.backtest_strategy(strategy, data)
        
        print(f"  ✅ SUCCESS: Backtest completed!")
        print(f"     📊 Trades: {result.total_trades}")
        print(f"     📈 Return: {result.total_return:.2f}%")
        print(f"     💰 Final value: ${result.final_portfolio_value:.2f}")
        
        return True, result
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False, None

def test_multiple_strategies():
    """Test multiple strategies with the fixed data format."""
    
    print(f"\n🧪 TESTING MULTIPLE STRATEGIES WITH FIXED DATA")
    print("=" * 60)
    
    # Create proper test data
    data = create_proper_test_data()
    
    factory = StrategyFactory()
    engine = BacktestingEngine()
    
    # Test strategies with corrected parameters
    test_strategies = {
        'MovingAverageCrossover': {
            'fast_period': 5,
            'slow_period': 15,
            'ma_type': 'EMA',
            'signal_threshold': 0.002,
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.05
        },
        'MACD': {
            'fast_period': 6,
            'slow_period': 18,
            'signal_period': 6,
            'histogram_threshold': 0.005,
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.05
        },
        'RSI': {
            'rsi_period': 8,
            'overbought': 65,
            'oversold': 35,
            'exit_signal': 'opposite',
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
    }
    
    successful_strategies = []
    
    for strategy_name, params in test_strategies.items():
        print(f"\n📋 Testing {strategy_name}...")
        
        try:
            # Create strategy
            strategy_class = factory.registry.get_strategy_class(strategy_name)
            strategy = strategy_class(**params)
            
            # Run backtest
            result = engine.backtest_strategy(strategy, data)
            
            print(f"  ✅ SUCCESS: {strategy.name}")
            print(f"     📊 Trades: {result.total_trades}, Return: {result.total_return:.2f}%")
            successful_strategies.append(strategy_name)
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print(f"\n📊 RESULTS SUMMARY")
    print("-" * 30)
    print(f"Total tested: {len(test_strategies)}")
    print(f"Successful: {len(successful_strategies)}")
    print(f"Success rate: {len(successful_strategies)/len(test_strategies)*100:.1f}%")
    
    if successful_strategies:
        print(f"\n✅ WORKING STRATEGIES:")
        for strategy in successful_strategies:
            print(f"   • {strategy}")
    
    return successful_strategies

def main():
    """Main function to test and fix DatetimeIndex issue."""
    
    print("🔧 DATETIME INDEX ISSUE DIAGNOSIS & FIX")
    print("=" * 60)
    print("Mission: Fix data format for backtesting engine compatibility")
    print("Goal: Enable strategies to run with proper DatetimeIndex")
    print()
    
    # Test data format differences
    data1, data2 = test_data_format()
    
    # Test single strategy with proper data
    success, result = test_strategy_with_proper_data()
    
    if success:
        print(f"\n🎉 SUCCESS: DatetimeIndex fix works!")
        
        # Test multiple strategies
        successful_strategies = test_multiple_strategies()
        
        print(f"\n🎯 DATETIME INDEX FIX COMPLETE")
        print("=" * 60)
        print("Key finding: Use DatetimeIndex as DataFrame index (not column)")
        print("Next step: Update enhanced validation with proper data format")
        
        return {
            'fix_successful': True,
            'successful_strategies': successful_strategies,
            'key_fix': 'use_datetime_index_not_column'
        }
    else:
        print(f"\n❌ DatetimeIndex fix needs further investigation")
        return {'fix_successful': False}

if __name__ == "__main__":
    results = main() 