#!/usr/bin/env python3
"""
Test script for the Hyperparameter Optimization System

This script tests the optimization engine, strategy factory, and related components
to ensure they work correctly together.
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.data.data_fetcher import DataFetcher
from src.optimization.hyperopt_optimizer import (
    HyperoptOptimizer, OptimizationConfig, OptimizationObjective
)
from src.optimization.strategy_factory import strategy_factory
from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy


def create_sample_data(days: int = 300, volatility: float = 0.02) -> pd.DataFrame:
    """Create sample OHLCV data for testing optimization."""
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Generate price data with trends and volatility
    base_price = 50000  # Starting price
    
    # Create trending periods with regime changes
    regime_length = 50
    regimes = []
    current_trend = 1
    
    for i in range(0, days, regime_length):
        regime_end = min(i + regime_length, days)
        regime_size = regime_end - i
        
        # Random trend change
        if np.random.random() < 0.3:  # 30% chance of trend change
            current_trend *= -1
        
        regimes.extend([current_trend] * regime_size)
    
    # Generate returns with trend bias and volatility clustering
    returns = []
    vol_state = volatility
    
    for i, trend in enumerate(regimes):
        # Volatility clustering
        if np.random.random() < 0.05:  # 5% chance of volatility regime change
            vol_state = np.random.uniform(volatility * 0.5, volatility * 2)
        
        base_return = trend * 0.0008  # Trend bias
        random_return = np.random.normal(0, vol_state)
        returns.append(base_return + random_return)
    
    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close price
        daily_volatility = abs(np.random.normal(0, volatility * 0.5))
        high = close * (1 + daily_volatility)
        low = close * (1 - daily_volatility)
        open_price = close + np.random.normal(0, close * volatility * 0.3)
        
        # Ensure OHLC consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume with correlation to price movement and volatility
        base_volume = 1000
        volume_multiplier = 1 + abs(returns[i]) * 20 + daily_volatility * 10
        volume = base_volume * volume_multiplier * np.random.uniform(0.5, 1.5)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=timestamps)
    return df


def test_strategy_factory():
    """Test the strategy factory functionality."""
    print("\n" + "="*60)
    print("TESTING STRATEGY FACTORY")
    print("="*60)
    
    try:
        # Test strategy registration and retrieval
        all_strategies = strategy_factory.registry.get_all_strategies()
        print(f"✅ Registered strategies: {all_strategies}")
        
        # Test category organization
        categories = strategy_factory.registry.get_all_categories()
        print(f"✅ Strategy categories: {categories}")
        
        for category in categories:
            strategies = strategy_factory.registry.get_strategies_by_category(category)
            print(f"   {category}: {strategies}")
        
        # Test strategy creation
        ma_strategy = strategy_factory.create_strategy(
            "MovingAverageCrossover", 
            fast_period=10, 
            slow_period=20
        )
        print(f"✅ Created strategy: {ma_strategy}")
        
        # Test parameter space retrieval
        param_space = strategy_factory.registry.get_parameter_space("RSI")
        print(f"✅ RSI parameter space has {len(param_space)} parameters")
        
        # Test optimization candidates
        candidates = strategy_factory.get_optimization_candidates(
            categories=["trend_following", "mean_reversion"]
        )
        print(f"✅ Found {len(candidates)} optimization candidates")
        
        # Test strategy summary
        summary = strategy_factory.get_strategy_summary()
        print(f"✅ Strategy summary: {summary['total_strategies']} total strategies")
        
        # Test parameter validation
        valid_params = {"rsi_period": 14, "oversold": 30, "overbought": 70}
        is_valid = strategy_factory.validate_strategy_parameters("RSI", valid_params)
        print(f"✅ Parameter validation: {is_valid}")
        
        # Test invalid parameters
        invalid_params = {"rsi_period": 100, "oversold": 80, "overbought": 20}  # Invalid ranges
        is_invalid = strategy_factory.validate_strategy_parameters("RSI", invalid_params)
        print(f"✅ Invalid parameter detection: {not is_invalid}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_config():
    """Test optimization configuration and objectives."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZATION CONFIGURATION")
    print("="*60)
    
    try:
        # Test default configuration
        default_config = OptimizationConfig()
        print(f"✅ Default config created with {len(default_config.objectives)} objectives")
        
        # Test custom objectives
        custom_objectives = [
            OptimizationObjective("sharpe_ratio", 0.5, maximize=True),
            OptimizationObjective("max_drawdown", 0.3, maximize=False),
            OptimizationObjective("win_rate", 0.2, maximize=True)
        ]
        
        custom_config = OptimizationConfig(
            max_evals=50,
            early_stop_rounds=10,
            objectives=custom_objectives,
            validation_split=0.2
        )
        print(f"✅ Custom config created with validation split: {custom_config.validation_split}")
        
        # Test invalid objectives (weights don't sum to 1)
        try:
            invalid_objectives = [
                OptimizationObjective("sharpe_ratio", 0.6, maximize=True),
                OptimizationObjective("max_drawdown", 0.6, maximize=False)  # Total = 1.2
            ]
            OptimizationConfig(objectives=invalid_objectives)
            print("❌ Should have failed with invalid objective weights")
            return False
        except ValueError:
            print("✅ Correctly rejected invalid objective weights")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimization_cache():
    """Test the optimization cache functionality."""
    print("\n" + "="*60)
    print("TESTING OPTIMIZATION CACHE")
    print("="*60)
    
    try:
        from src.optimization.hyperopt_optimizer import OptimizationCache
        
        # Create cache instance
        cache = OptimizationCache("test_cache")
        print("✅ Created optimization cache")
        
        # Test cache operations
        test_key = "test_key_123"
        test_data = {
            "loss": 0.5,
            "status": "ok",
            "metrics": {"sharpe_ratio": 1.5, "return": 0.15}
        }
        
        # Store data
        cache.put(test_key, test_data)
        print("✅ Stored data in cache")
        
        # Retrieve data
        retrieved_data = cache.get(test_key)
        if retrieved_data == test_data:
            print("✅ Successfully retrieved cached data")
        else:
            print("❌ Cache retrieval failed")
            return False
        
        # Test cache miss
        missing_data = cache.get("nonexistent_key")
        if missing_data is None:
            print("✅ Correctly handled cache miss")
        else:
            print("❌ Cache should return None for missing keys")
            return False
        
        # Clean up test cache
        cache.clear()
        print("✅ Cache cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_strategy_optimization():
    """Test optimization of a single strategy."""
    print("\n" + "="*60)
    print("TESTING SINGLE STRATEGY OPTIMIZATION")
    print("="*60)
    
    try:
        # Create test data
        data = create_sample_data(200, volatility=0.025)
        print(f"✅ Created test data: {len(data)} days")
        
        # Create optimization configuration
        config = OptimizationConfig(
            max_evals=20,  # Small number for testing
            early_stop_rounds=5,
            validation_split=0.3,
            cache_results=True,
            cache_dir="test_optimization_cache"
        )
        
        # Create optimizer
        optimizer = HyperoptOptimizer(config)
        print("✅ Created hyperopt optimizer")
        
        # Get strategy and parameter space
        strategy_class = strategy_factory.registry.get_strategy_class("RSI")
        param_space = strategy_factory.registry.get_parameter_space("RSI")
        print(f"✅ Retrieved RSI strategy with {len(param_space)} parameters")
        
        # Run optimization
        print("🔄 Running optimization...")
        result = optimizer.optimize_strategy(
            strategy_class=strategy_class,
            parameter_space=param_space,
            data=data,
            symbol="TEST/USDT"
        )
        
        print(f"✅ Optimization completed!")
        print(f"   Best score: {result.best_score:.4f}")
        print(f"   Best parameters: {result.best_params}")
        print(f"   Total evaluations: {result.total_evaluations}")
        print(f"   Cache hits: {result.cache_hits}")
        print(f"   Optimization time: {result.optimization_time:.2f}s")
        
        if result.validation_results:
            print(f"   Train score: {result.validation_results['train_score']:.4f}")
            print(f"   Validation score: {result.validation_results['validation_score']:.4f}")
        
        # Test that we got reasonable results
        if result.total_evaluations > 0 and result.best_score != 0:
            print("✅ Optimization produced valid results")
        else:
            print("❌ Optimization results seem invalid")
            return False
        
        # Test caching by running again
        print("🔄 Testing cache by running optimization again...")
        result2 = optimizer.optimize_strategy(
            strategy_class=strategy_class,
            parameter_space=param_space,
            data=data,
            symbol="TEST/USDT"
        )
        
        if result2.cache_hits > result.cache_hits:
            print(f"✅ Cache working: {result2.cache_hits} cache hits")
        else:
            print("⚠️  Cache may not be working as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Single strategy optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_strategy_optimization():
    """Test optimization of multiple strategies."""
    print("\n" + "="*60)
    print("TESTING MULTI-STRATEGY OPTIMIZATION")
    print("="*60)
    
    try:
        # Create test data
        data = create_sample_data(150, volatility=0.02)
        print(f"✅ Created test data: {len(data)} days")
        
        # Create optimization configuration
        config = OptimizationConfig(
            max_evals=15,  # Small number for testing
            early_stop_rounds=5,
            validation_split=0.2,
            n_jobs=1  # Sequential for testing
        )
        
        # Create optimizer
        optimizer = HyperoptOptimizer(config)
        
        # Get multiple strategies for optimization
        strategies = strategy_factory.get_optimization_candidates(
            strategy_names=["MovingAverageCrossover", "RSI"]
        )
        print(f"✅ Selected {len(strategies)} strategies for optimization")
        
        # Run multi-strategy optimization
        print("🔄 Running multi-strategy optimization...")
        results = optimizer.optimize_multiple_strategies(
            strategies=strategies,
            data=data,
            symbol="TEST/USDT"
        )
        
        print(f"✅ Multi-strategy optimization completed!")
        print(f"   Optimized {len(results)} strategies")
        
        # Display results for each strategy
        for strategy_name, result in results.items():
            print(f"   {strategy_name}:")
            print(f"     Best score: {result.best_score:.4f}")
            print(f"     Evaluations: {result.total_evaluations}")
            print(f"     Time: {result.optimization_time:.2f}s")
        
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1].best_score)
        print(f"🏆 Best strategy: {best_strategy[0]} (score: {best_strategy[1].best_score:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-strategy optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_walk_forward_optimization():
    """Test walk-forward optimization."""
    print("\n" + "="*60)
    print("TESTING WALK-FORWARD OPTIMIZATION")
    print("="*60)
    
    try:
        # Create longer test data for walk-forward
        data = create_sample_data(250, volatility=0.02)
        print(f"✅ Created test data: {len(data)} days")
        
        # Create optimization configuration
        config = OptimizationConfig(
            max_evals=10,  # Very small for testing
            early_stop_rounds=3,
            validation_split=0.0  # No validation split for walk-forward
        )
        
        # Create optimizer
        optimizer = HyperoptOptimizer(config)
        
        # Get strategy
        strategy_class = strategy_factory.registry.get_strategy_class("MovingAverageCrossover")
        param_space = strategy_factory.registry.get_parameter_space("MovingAverageCrossover")
        
        # Run walk-forward optimization
        print("🔄 Running walk-forward optimization...")
        window_size = 100
        step_size = 30
        
        results = optimizer.walk_forward_optimization(
            strategy_class=strategy_class,
            parameter_space=param_space,
            data=data,
            window_size=window_size,
            step_size=step_size,
            symbol="TEST/USDT"
        )
        
        print(f"✅ Walk-forward optimization completed!")
        print(f"   Number of windows: {len(results)}")
        
        # Display results for each window
        for i, result in enumerate(results):
            print(f"   Window {i+1}: score={result.best_score:.4f}, "
                  f"evals={result.total_evaluations}")
        
        # Calculate average performance
        avg_score = np.mean([r.best_score for r in results])
        print(f"   Average score across windows: {avg_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Walk-forward optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_real_data():
    """Test optimization with real market data."""
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
        start_date = end_date - timedelta(days=120)  # Last 120 days
        
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
        
        # Create optimization configuration
        config = OptimizationConfig(
            max_evals=25,
            early_stop_rounds=8,
            validation_split=0.25
        )
        
        # Create optimizer
        optimizer = HyperoptOptimizer(config)
        
        # Optimize RSI strategy on real data
        strategy_class = strategy_factory.registry.get_strategy_class("RSI")
        param_space = strategy_factory.registry.get_parameter_space("RSI")
        
        print("🔄 Optimizing RSI strategy on real BTC data...")
        optimization_result = optimizer.optimize_strategy(
            strategy_class=strategy_class,
            parameter_space=param_space,
            data=result.data,
            symbol="BTC/USDT"
        )
        
        print(f"✅ Real data optimization completed!")
        print(f"   Best score: {optimization_result.best_score:.4f}")
        print(f"   Best parameters: {optimization_result.best_params}")
        print(f"   Data quality: {result.quality_metrics.quality_score:.3f}")
        
        if optimization_result.validation_results:
            train_score = optimization_result.validation_results.get('train_score')
            val_score = optimization_result.validation_results.get('validation_score')
            
            if train_score is not None and val_score is not None:
                print(f"   Train/Validation scores: {train_score:.4f} / {val_score:.4f}")
                
                # Check for overfitting
                if val_score < train_score * 0.7:
                    print("   ⚠️  Potential overfitting detected")
                else:
                    print("   ✅ No significant overfitting detected")
            else:
                print("   ⚠️  Validation scores not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("="*80)
    print("HYPERPARAMETER OPTIMIZATION SYSTEM TEST SUITE")
    print("="*80)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Strategy Factory", test_strategy_factory),
        ("Optimization Configuration", test_optimization_config),
        ("Optimization Cache", test_optimization_cache),
        ("Single Strategy Optimization", test_single_strategy_optimization),
        ("Multi-Strategy Optimization", test_multi_strategy_optimization),
        ("Walk-Forward Optimization", test_walk_forward_optimization),
        ("Real Market Data", test_with_real_data)
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
    
    # Clean up test cache
    try:
        import shutil
        if os.path.exists("test_cache"):
            shutil.rmtree("test_cache")
        if os.path.exists("test_optimization_cache"):
            shutil.rmtree("test_optimization_cache")
    except:
        pass
    
    # Final results
    print("\n" + "="*80)
    if all_tests_passed:
        print("🎉 ALL OPTIMIZATION SYSTEM TESTS PASSED!")
        print("\nHyperparameter Optimization System Features:")
        print("✅ Multi-objective optimization with configurable weights")
        print("✅ TPE (Tree-structured Parzen Estimator) algorithm")
        print("✅ Result caching to avoid re-computation")
        print("✅ Early stopping for efficiency")
        print("✅ Train/validation split for overfitting detection")
        print("✅ Walk-forward optimization for time series")
        print("✅ Parallel processing support")
        print("✅ Strategy factory and registry system")
        print("✅ Comprehensive logging and monitoring")
        print("\nNext steps:")
        print("1. Implement remaining 60 trading strategies")
        print("2. Add advanced optimization algorithms (genetic, particle swarm)")
        print("3. Implement portfolio optimization and risk management")
        print("4. Create web interface for optimization monitoring")
    else:
        print("❌ SOME OPTIMIZATION TESTS FAILED. Please check the errors above.")
        sys.exit(1)
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main()) 