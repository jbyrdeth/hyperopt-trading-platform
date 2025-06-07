#!/usr/bin/env python3
"""
Test script for the Strategy Framework and Backtesting Engine

This script tests the base strategy class, backtesting engine, and simple MA strategy
to ensure everything works correctly together.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.data.data_fetcher import DataFetcher
from src.strategies.base_strategy import BaseStrategy, Signal, Position
from src.strategies.backtesting_engine import BacktestingEngine, CostModel
from src.strategies.simple_ma_strategy import SimpleMAStrategy


def create_sample_data(days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Generate price data with some trend and volatility
    base_price = 50000  # Starting price
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns with slight upward bias
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close + np.random.normal(0, close * 0.005)
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df


def test_base_strategy():
    """Test the base strategy class."""
    print("\n" + "="*60)
    print("TESTING BASE STRATEGY CLASS")
    print("="*60)
    
    try:
        # Test that we can't instantiate abstract base class
        try:
            strategy = BaseStrategy("test")
            print("❌ ERROR: Should not be able to instantiate abstract BaseStrategy")
            return False
        except TypeError:
            print("✅ Correctly prevented instantiation of abstract BaseStrategy")
        
        # Test SimpleMAStrategy instantiation
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
        print(f"✅ Created SimpleMAStrategy: {strategy}")
        
        # Test parameter validation
        invalid_strategy = SimpleMAStrategy(fast_period=10, slow_period=5)  # Invalid: fast >= slow
        if not invalid_strategy.validate_parameters():
            print("✅ Parameter validation correctly rejected invalid parameters")
        else:
            print("❌ Parameter validation failed to reject invalid parameters")
            return False
        
        # Test with sample data
        data = create_sample_data(50)
        strategy.initialize(data)
        
        if strategy.is_initialized:
            print("✅ Strategy initialization successful")
        else:
            print("❌ Strategy initialization failed")
            return False
        
        # Test signal generation
        test_time = data.index[30]  # Use a timestamp with enough history
        test_data = data.loc[test_time]
        signal = strategy.generate_signal(test_time, test_data)
        
        print(f"✅ Generated signal: {signal.action} (strength: {signal.strength:.2f})")
        
        # Test position sizing
        position_size = strategy.calculate_position_size(
            signal, test_data['close'], 100000, 0.02
        )
        print(f"✅ Calculated position size: {position_size:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Base strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtesting_engine():
    """Test the backtesting engine."""
    print("\n" + "="*60)
    print("TESTING BACKTESTING ENGINE")
    print("="*60)
    
    try:
        # Create sample data
        data = create_sample_data(100)
        print(f"✅ Created sample data: {len(data)} days")
        
        # Create strategy
        strategy = SimpleMAStrategy(fast_period=5, slow_period=20)
        print(f"✅ Created strategy: {strategy}")
        
        # Create backtesting engine
        engine = BacktestingEngine(initial_capital=100000)
        print(f"✅ Created backtesting engine with $100,000 initial capital")
        
        # Run backtest
        print("🔄 Running backtest...")
        results = engine.backtest_strategy(strategy, data, "TEST/USDT")
        
        # Display results
        print(f"✅ Backtest completed successfully!")
        print(f"   Total Return: {results.total_return:.2%}")
        print(f"   Annual Return: {results.annual_return:.2%}")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {results.max_drawdown:.2%}")
        print(f"   Win Rate: {results.win_rate:.2%}")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        print(f"   Total Commission: ${results.total_commission:.2f}")
        print(f"   Total Slippage: ${results.total_slippage:.2f}")
        
        # Validate results
        if results.total_trades > 0:
            print("✅ Strategy generated trades")
        else:
            print("⚠️  Strategy generated no trades (may be normal for short test period)")
        
        if len(results.equity_curve) == len(data):
            print("✅ Equity curve has correct length")
        else:
            print(f"❌ Equity curve length mismatch: {len(results.equity_curve)} vs {len(data)}")
            return False
        
        # Test cost model
        cost_model = CostModel(commission_rate=0.002, slippage_base=0.001)
        engine_with_costs = BacktestingEngine(initial_capital=100000, cost_model=cost_model)
        results_with_costs = engine_with_costs.backtest_strategy(strategy, data, "TEST/USDT")
        
        if results_with_costs.total_costs > results.total_costs:
            print("✅ Higher cost model correctly increased total costs")
        else:
            print("⚠️  Cost model may not be working as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_real_data():
    """Test the framework with real market data."""
    print("\n" + "="*60)
    print("TESTING WITH REAL MARKET DATA")
    print("="*60)
    
    try:
        # Setup logging
        logger = setup_logging(level="INFO", console_output=True)
        
        # Load configuration and fetch real data
        config_manager = ConfigManager("config")
        exchange_config = config_manager.get_config("exchanges")
        data_fetcher = DataFetcher(exchange_config)
        
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        print("🔄 Fetching real market data...")
        result = await data_fetcher.fetch_data(
            symbol="BTC/USDT",
            timeframe="1d",
            start_date=start_date,
            end_date=end_date
        )
        
        if not result.success:
            print(f"⚠️  Could not fetch real data: {result.error_message}")
            print("   This is expected if no API keys are configured")
            return True  # Not a failure, just no real data available
        
        print(f"✅ Fetched {len(result.data)} days of real BTC data")
        
        # Test strategy with real data
        strategy = SimpleMAStrategy(fast_period=5, slow_period=15)
        engine = BacktestingEngine(initial_capital=100000)
        
        print("🔄 Running backtest on real data...")
        backtest_results = engine.backtest_strategy(strategy, result.data, "BTC/USDT")
        
        print(f"✅ Real data backtest completed!")
        print(f"   Period: {backtest_results.start_date.date()} to {backtest_results.end_date.date()}")
        print(f"   Total Return: {backtest_results.total_return:.2%}")
        print(f"   Annual Return: {backtest_results.annual_return:.2%}")
        print(f"   Sharpe Ratio: {backtest_results.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {backtest_results.max_drawdown:.2%}")
        print(f"   Total Trades: {backtest_results.total_trades}")
        print(f"   Data Quality: {result.quality_metrics.quality_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    try:
        # Test with very small dataset
        small_data = create_sample_data(5)
        strategy = SimpleMAStrategy(fast_period=3, slow_period=10)  # Slow period > data length
        engine = BacktestingEngine(initial_capital=1000)
        
        results = engine.backtest_strategy(strategy, small_data, "TEST/USDT")
        print(f"✅ Handled small dataset: {results.total_trades} trades")
        
        # Test with zero volatility data
        flat_data = create_sample_data(20)
        flat_data['close'] = 50000  # Flat price
        flat_data['open'] = 50000
        flat_data['high'] = 50000
        flat_data['low'] = 50000
        
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
        results = engine.backtest_strategy(strategy, flat_data, "TEST/USDT")
        print(f"✅ Handled flat price data: {results.total_trades} trades")
        
        # Test with very low capital
        low_capital_engine = BacktestingEngine(initial_capital=100)
        data = create_sample_data(50)
        data['close'] = data['close'] * 1000  # Make prices very high relative to capital
        
        results = low_capital_engine.backtest_strategy(strategy, data, "TEST/USDT")
        print(f"✅ Handled low capital scenario: {results.total_trades} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("="*80)
    print("TRADING STRATEGY FRAMEWORK TEST SUITE")
    print("="*80)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Base Strategy Class", test_base_strategy),
        ("Backtesting Engine", test_backtesting_engine),
        ("Real Market Data", test_with_real_data),
        ("Edge Cases", test_edge_cases)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} tests...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                print(f"✅ {test_name} tests PASSED")
            else:
                print(f"❌ {test_name} tests FAILED")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ {test_name} tests FAILED with exception: {e}")
            all_tests_passed = False
    
    # Final results
    print("\n" + "="*80)
    if all_tests_passed:
        print("🎉 ALL TESTS PASSED! Strategy framework is working correctly.")
        print("\nNext steps:")
        print("1. Implement more sophisticated strategies")
        print("2. Add technical indicators library")
        print("3. Implement hyperparameter optimization")
        print("4. Add portfolio management features")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
        sys.exit(1)
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main()) 