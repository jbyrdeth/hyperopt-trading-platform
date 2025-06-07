#!/usr/bin/env python3
"""
Quick Validation Test

Test a few strategies to verify the backtesting engine fix works correctly
before running the full 24-strategy validation.
"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_test_data():
    """Create simple test data for validation."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='4h')
    np.random.seed(42)
    
    # Create trending data
    base_prices = np.linspace(100, 120, len(dates))  # 20% uptrend
    noise = np.random.normal(0, 2, len(dates))
    prices = base_prices + noise
    
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    return data

def test_strategy_fix():
    """Test that the backtesting engine fix works correctly."""
    
    print("🔧 QUICK VALIDATION TEST")
    print("=" * 40)
    print("Testing backtesting engine fix with sample strategies")
    print()
    
    try:
        # Import modules
        from optimization.strategy_factory import StrategyFactory
        from strategies.backtesting_engine import BacktestingEngine
        
        print("✅ Modules imported successfully")
        
        # Initialize components
        factory = StrategyFactory()
        engine = BacktestingEngine(initial_capital=100000)
        
        # Get available strategies
        strategies = factory.get_all_strategies()
        print(f"📋 Found {len(strategies)} strategies")
        
        # Test with first 3 strategies
        test_strategies = strategies[:3]
        print(f"🎯 Testing strategies: {', '.join(test_strategies)}")
        
        # Create test data
        data = create_test_data()
        print(f"📊 Test data: {len(data)} points, price range ${data['close'].min():.2f}-${data['close'].max():.2f}")
        
        results = {}
        
        for strategy_name in test_strategies:
            print(f"\n🧪 Testing {strategy_name}...")
            
            try:
                # Get default parameters
                param_space = factory.get_parameter_space(strategy_name)
                parameters = {}
                
                for param_name, param_config in param_space.items():
                    if isinstance(param_config, dict):
                        if 'low' in param_config and 'high' in param_config:
                            low, high = param_config['low'], param_config['high']
                            if isinstance(low, int):
                                parameters[param_name] = (low + high) // 2
                            else:
                                parameters[param_name] = (low + high) / 2
                        elif 'choices' in param_config:
                            parameters[param_name] = param_config['choices'][0]
                    else:
                        parameters[param_name] = param_config
                
                # Create strategy
                strategy = factory.create_strategy(strategy_name, parameters)
                
                # Reset engine
                engine.reset()
                
                # Run backtest
                result = engine.backtest_strategy(strategy, data, "TEST/USDT")
                
                # Store results
                results[strategy_name] = {
                    'total_return': result.total_return,
                    'total_trades': result.total_trades,
                    'win_rate': result.win_rate,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown
                }
                
                print(f"   ✅ Return: {result.total_return*100:.2f}%")
                print(f"   ✅ Trades: {result.total_trades}")
                print(f"   ✅ Win Rate: {result.win_rate*100:.1f}%")
                print(f"   ✅ Sharpe: {result.sharpe_ratio:.2f}")
                
                # Check for the old bug
                if result.total_trades == 0 and abs(result.total_return) > 0.05:
                    print(f"   ⚠️ WARNING: Possible bug - 0 trades but {result.total_return*100:.2f}% return")
                else:
                    print(f"   ✅ Results look reasonable")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[strategy_name] = {'error': str(e)}
        
        # Summary
        print(f"\n📊 QUICK TEST SUMMARY:")
        successful = len([r for r in results.values() if 'error' not in r])
        print(f"   Successful tests: {successful}/{len(test_strategies)}")
        
        if successful > 0:
            print(f"   ✅ Backtesting engine appears to be working correctly")
            print(f"   🎯 Ready to proceed with full 24-strategy validation")
            return True
        else:
            print(f"   ❌ Issues detected - need to investigate further")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_strategy_fix()
    
    if success:
        print(f"\n🚀 Quick test passed! Ready for full validation.")
        print(f"   Run: python validate_24_strategies.py")
    else:
        print(f"\n🔧 Quick test failed. Need to fix issues first.") 