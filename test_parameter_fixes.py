#!/usr/bin/env python3
"""
Test Parameter Constraint Fixes

Verify that the expanded parameter ranges now allow more aggressive
trading configurations that should generate significantly more signals.
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

def create_test_data():
    """Create test data with clear trend patterns."""
    dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='4h')
    np.random.seed(42)
    
    # Create strong trending data with volatility
    base_prices = np.linspace(100, 140, len(dates))  # 40% uptrend
    noise = np.random.normal(0, 3, len(dates))
    prices = base_prices + noise
    
    # Add some volume correlation
    volumes = 1000 + np.random.normal(0, 200, len(dates))
    volumes = np.maximum(volumes, 100)
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.01, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.02, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.02, len(dates)))),
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return data

def test_aggressive_parameters():
    """Test strategies with aggressive parameters that should generate more signals."""
    
    print("🔧 TESTING PARAMETER CONSTRAINT FIXES")
    print("="*60)
    
    data = create_test_data()
    factory = StrategyFactory()
    engine = BacktestingEngine(initial_capital=10000)
    
    # Test 1: MovingAverageCrossover with aggressive short periods
    print("\n🧪 TEST 1: MovingAverageCrossover - Aggressive Settings")
    test_params = [
        # These should now work (previously blocked)
        {"fast_period": 3, "slow_period": 8, "signal_threshold": 0.0005},   # Very short periods
        {"fast_period": 2, "slow_period": 5, "signal_threshold": 0.001},    # Super short
        {"fast_period": 5, "slow_period": 15, "signal_threshold": 0.0008},  # Medium aggressive
    ]
    
    for i, params in enumerate(test_params, 1):
        try:
            strategy = factory.create_strategy("MovingAverageCrossover", **params)
            result = engine.backtest_strategy(strategy, data)
            
            print(f"  ✅ Config {i}: fast={params['fast_period']}, slow={params['slow_period']}")
            print(f"     📊 Trades: {result.total_trades}, Return: {result.total_return:.2f}%")
            
            if result.total_trades > 0:
                print(f"     🎯 SUCCESS: Generated {result.total_trades} trades!")
            else:
                print(f"     ⚠️  Still no trades - may need further investigation")
                
        except Exception as e:
            print(f"  ❌ Config {i} FAILED: {e}")
    
    # Test 2: MACD with aggressive settings
    print(f"\n🧪 TEST 2: MACD - Aggressive Settings")
    macd_params = [
        # These should now work (previously blocked)
        {"fast_period": 3, "slow_period": 12, "signal_period": 3},    # Very fast
        {"fast_period": 5, "slow_period": 15, "signal_period": 4},    # Fast
        {"fast_period": 8, "slow_period": 21, "signal_period": 5},    # Moderately aggressive
    ]
    
    for i, params in enumerate(macd_params, 1):
        try:
            strategy = factory.create_strategy("MACD", **params)
            result = engine.backtest_strategy(strategy, data)
            
            print(f"  ✅ Config {i}: fast={params['fast_period']}, slow={params['slow_period']}, signal={params['signal_period']}")
            print(f"     📊 Trades: {result.total_trades}, Return: {result.total_return:.2f}%")
            
            if result.total_trades > 0:
                print(f"     🎯 SUCCESS: Generated {result.total_trades} trades!")
            else:
                print(f"     ⚠️  Still no trades - may need further investigation")
                
        except Exception as e:
            print(f"  ❌ Config {i} FAILED: {e}")
    
    print(f"\n🏆 CONSTRAINT FIX VALIDATION COMPLETE")
    print(f"💡 Next step: Run full strategy optimization with these expanded ranges")

if __name__ == "__main__":
    test_aggressive_parameters() 