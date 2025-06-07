#!/usr/bin/env python3
"""
Multi-Timeframe Strategies Test Suite

This script tests all multi-timeframe strategies to ensure they work correctly
with initialization, signal generation, strategy factory integration, and backtesting.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(periods=300):
    """Create realistic test data with proper datetime index."""
    np.random.seed(42)
    
    # Create datetime index
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=periods, freq='1h')
    
    # Generate realistic price data with trend and volatility
    base_price = 50000
    trend = np.linspace(0, 0.5, periods)  # Upward trend
    noise = np.random.normal(0, 0.02, periods)  # 2% volatility
    returns = trend + noise
    
    # Calculate prices
    prices = [base_price]
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, periods))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, periods))
    data['volume'] = np.random.uniform(1000, 10000, periods)
    
    return data

def test_strategy_initialization(strategy_class, strategy_name):
    """Test strategy initialization."""
    print(f"\n--- Testing {strategy_name} Initialization ---")
    
    try:
        # Test with default parameters
        strategy = strategy_class()
        print(f"✓ {strategy_name} created with default parameters")
        
        # Test parameter validation
        if hasattr(strategy, 'validate_parameters'):
            is_valid = strategy.validate_parameters()
            if is_valid:
                print("✓ Parameter validation: Passed")
            else:
                print("✗ Parameter validation: Failed")
                return False
        
        # Test with custom parameters
        custom_params = {
            'primary_timeframe': '4H',
            'secondary_timeframe': '1D',
            'position_size_pct': 0.8
        }
        strategy_custom = strategy_class(**custom_params)
        print(f"✓ {strategy_name} created with custom parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ {strategy_name} initialization failed: {e}")
        return False

def test_strategy_with_data(strategy_class, strategy_name, data):
    """Test strategy with actual data."""
    print(f"\n--- Testing {strategy_name} with Data ---")
    
    try:
        strategy = strategy_class()
        
        # Initialize with data
        strategy.initialize(data)
        
        if strategy.is_initialized:
            print(f"✓ {strategy_name} initialized with {len(data)} data points")
        else:
            print(f"✗ {strategy_name} failed to initialize")
            return False
        
        # Test signal generation
        signals_generated = 0
        test_periods = min(50, len(data) - strategy.required_periods)
        
        for i in range(strategy.required_periods, strategy.required_periods + test_periods):
            current_data = data.iloc[i]
            signal = strategy.generate_signal(data.index[i], current_data)
            
            if signal.action != 'hold':
                signals_generated += 1
        
        print(f"✓ Generated {signals_generated} signals out of {test_periods} test periods")
        
        # Test position sizing
        if signals_generated > 0:
            test_signal = None
            for i in range(strategy.required_periods, len(data)):
                current_data = data.iloc[i]
                signal = strategy.generate_signal(data.index[i], current_data)
                if signal.action != 'hold':
                    test_signal = signal
                    break
            
            if test_signal:
                position_size = strategy.calculate_position_size(
                    test_signal, 
                    test_signal.price, 
                    100000, 
                    0.02
                )
                print(f"✓ Position sizing works: {position_size:.4f}")
        
        # Test strategy info
        if hasattr(strategy, 'get_current_indicators'):
            indicators = strategy.get_current_indicators()
            print(f"✓ Strategy info: {len(indicators)} fields")
        
        return True
        
    except Exception as e:
        print(f"✗ {strategy_name} data test failed: {e}")
        return False

def test_strategy_factory_integration():
    """Test integration with strategy factory."""
    print(f"\n--- Testing Strategy Factory Integration ---")
    
    try:
        from src.optimization.strategy_factory import StrategyFactory
        
        factory = StrategyFactory()
        mtf_strategies = factory.get_strategies_by_category('multi_timeframe')
        
        for strategy_name in mtf_strategies:
            try:
                # Test parameter space retrieval
                param_space = factory.get_parameter_space(strategy_name)
                print(f"✓ {strategy_name} registered in factory")
                print(f"✓ {strategy_name} parameter space: {len(param_space)} parameters")
                
                # Test strategy creation via factory
                strategy = factory.create_strategy(strategy_name)
                print(f"✓ {strategy_name} created via factory")
                
            except Exception as e:
                print(f"✗ {strategy_name} factory integration failed: {e}")
                return False
        
        print(f"✓ Multi-timeframe category has {len(mtf_strategies)} strategies")
        return True
        
    except Exception as e:
        print(f"✗ Strategy factory integration failed: {e}")
        return False

def test_backtesting_integration():
    """Test integration with backtesting engine."""
    print(f"\n--- Testing Backtesting Integration ---")
    
    try:
        from src.strategies.backtesting_engine import BacktestingEngine, CostModel
        from src.strategies.mtf_trend_analysis_strategy import MTFTrendAnalysisStrategy
        
        # Create test data
        data = create_test_data(periods=200)
        
        # Create strategy
        strategy = MTFTrendAnalysisStrategy()
        
        # Create cost model
        cost_model = CostModel(
            commission_rate=0.001,
            slippage_base=0.0005
        )
        
        # Create backtesting engine
        engine = BacktestingEngine(
            initial_capital=100000,
            cost_model=cost_model
        )
        
        # Run backtest
        results = engine.backtest_strategy(strategy, data)
        
        print("✓ Backtest completed")
        print(f"  - Total Return: {results.total_return:.2f}%")
        print(f"  - Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"  - Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"  - Total Trades: {results.total_trades}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtesting integration failed: {e}")
        return False

def main():
    """Run all multi-timeframe strategy tests."""
    print("=" * 60)
    print("MULTI-TIMEFRAME STRATEGIES TEST SUITE")
    print("=" * 60)
    
    # Import strategies
    try:
        from src.strategies.mtf_trend_analysis_strategy import MTFTrendAnalysisStrategy
        from src.strategies.mtf_rsi_strategy import MTFRSIStrategy
        from src.strategies.mtf_macd_strategy import MTFMACDStrategy
        print("✓ Successfully imported 3 multi-timeframe strategies")
    except ImportError as e:
        print(f"✗ Failed to import strategies: {e}")
        return False
    
    # Create test data
    print(f"\n--- Creating Test Data ---")
    data = create_test_data(periods=300)
    print(f"✓ Created test data with {len(data)} periods")
    print(f"  - Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  - Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
    
    # Test strategies
    strategies = [
        (MTFTrendAnalysisStrategy, "MTFTrendAnalysis"),
        (MTFRSIStrategy, "MTFRSI"),
        (MTFMACDStrategy, "MTFMACD")
    ]
    
    results = {}
    
    for strategy_class, strategy_name in strategies:
        # Test initialization
        init_success = test_strategy_initialization(strategy_class, strategy_name)
        
        # Test with data
        data_success = test_strategy_with_data(strategy_class, strategy_name, data)
        
        results[strategy_name] = {
            'initialization': init_success,
            'data_processing': data_success
        }
    
    # Test factory integration
    factory_success = test_strategy_factory_integration()
    
    # Test backtesting integration
    backtest_success = test_backtesting_integration()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for strategy_name, result in results.items():
        init_status = "✓ PASS" if result['initialization'] else "✗ FAIL"
        data_status = "✓ PASS" if result['data_processing'] else "✗ FAIL"
        print(f"{strategy_name:<20} Init: {init_status}   Data: {data_status}")
    
    factory_status = "✓ PASS" if factory_success else "✗ FAIL"
    backtest_status = "✓ PASS" if backtest_success else "✗ FAIL"
    
    print(f"\nOverall Results:")
    print(f"  - Strategies tested: {len(strategies)}")
    print(f"  - Factory integration: {factory_status}")
    print(f"  - Backtesting integration: {backtest_status}")
    
    # Check if all tests passed
    all_passed = (
        all(result['initialization'] and result['data_processing'] for result in results.values()) and
        factory_success and
        backtest_success
    )
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 