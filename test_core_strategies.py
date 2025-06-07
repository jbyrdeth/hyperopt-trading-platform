#!/usr/bin/env python3
"""
Test script for Core Trading Strategies

This script tests all the core trading strategies we've implemented:
- Moving Average Crossover
- MACD Strategy
- RSI Strategy
- Bollinger Bands Strategy
- Momentum Strategy
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

# Import all our strategies
from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.bollinger_bands_strategy import BollingerBandsStrategy
from src.strategies.momentum_strategy import MomentumStrategy


def create_sample_data(days: int = 200, volatility: float = 0.02) -> pd.DataFrame:
    """Create sample OHLCV data for testing with realistic price movements."""
    np.random.seed(42)  # For reproducible results
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Generate price data with trends and volatility
    base_price = 50000  # Starting price
    
    # Create trending periods
    trend_changes = np.random.choice([0, 1], size=days, p=[0.95, 0.05])  # 5% chance of trend change
    current_trend = 1  # Start with uptrend
    trends = []
    
    for change in trend_changes:
        if change:
            current_trend *= -1
        trends.append(current_trend)
    
    # Generate returns with trend bias
    returns = []
    for i, trend in enumerate(trends):
        base_return = trend * 0.001  # Trend bias
        random_return = np.random.normal(0, volatility)
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
        
        # Generate volume with some correlation to price movement
        base_volume = 1000
        volume_multiplier = 1 + abs(returns[i]) * 10  # Higher volume on big moves
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


def test_strategy(strategy_class, strategy_name: str, data: pd.DataFrame, **kwargs):
    """Test a single strategy with the given data."""
    print(f"\n{'='*60}")
    print(f"TESTING {strategy_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Create strategy instance
        strategy = strategy_class(**kwargs)
        print(f"✅ Created {strategy}")
        
        # Test parameter validation
        if not strategy.validate_parameters():
            print(f"❌ Parameter validation failed for {strategy}")
            return False
        print(f"✅ Parameter validation passed")
        
        # Initialize strategy
        strategy.initialize(data)
        if not strategy.is_initialized:
            print(f"❌ Strategy initialization failed")
            return False
        print(f"✅ Strategy initialized successfully")
        
        # Test signal generation on a few data points
        test_signals = []
        for i in range(max(50, len(data) // 4), min(len(data), max(50, len(data) // 4) + 10)):
            timestamp = data.index[i]
            current_data = data.iloc[i]
            signal = strategy.generate_signal(timestamp, current_data)
            test_signals.append(signal)
            
            if signal.action != "hold":
                print(f"   📊 Signal at {timestamp.date()}: {signal.action} "
                      f"(strength: {signal.strength:.3f}, confidence: {signal.confidence:.3f})")
        
        signal_count = sum(1 for s in test_signals if s.action != "hold")
        print(f"✅ Generated {signal_count} signals out of {len(test_signals)} test points")
        
        # Test backtesting
        engine = BacktestingEngine(initial_capital=100000)
        results = engine.backtest_strategy(strategy, data, "TEST/USDT")
        
        print(f"✅ Backtest completed successfully!")
        print(f"   📈 Total Return: {results.total_return:.2%}")
        print(f"   📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   📉 Max Drawdown: {results.max_drawdown:.2%}")
        print(f"   🔄 Total Trades: {results.total_trades}")
        print(f"   💰 Win Rate: {results.win_rate:.2%}")
        
        # Test indicator access
        indicators = strategy.get_current_indicators()
        if indicators:
            print(f"✅ Current indicators: {list(indicators.keys())}")
        
        # Test parameter space
        param_space = strategy.get_parameter_space()
        if param_space:
            print(f"✅ Parameter space defined with {len(param_space)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ {strategy_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_combinations():
    """Test multiple strategies running on the same data."""
    print(f"\n{'='*60}")
    print("TESTING STRATEGY COMBINATIONS")
    print(f"{'='*60}")
    
    try:
        # Create test data
        data = create_sample_data(150, volatility=0.025)
        
        # Create multiple strategies
        strategies = [
            MovingAverageCrossoverStrategy(fast_period=10, slow_period=30, ma_type="EMA"),
            MACDStrategy(fast_period=12, slow_period=26, signal_period=9),
            RSIStrategy(rsi_period=14, oversold=30, overbought=70),
            BollingerBandsStrategy(period=20, std_dev=2.0, entry_method="both"),
            MomentumStrategy(period=14, threshold=0.03)
        ]
        
        # Test all strategies on the same data
        engine = BacktestingEngine(initial_capital=100000)
        results = []
        
        for strategy in strategies:
            strategy.initialize(data)
            if strategy.is_initialized:
                result = engine.backtest_strategy(strategy, data, "TEST/USDT")
                results.append((strategy.name, result))
                print(f"   {strategy.name}: Return={result.total_return:.2%}, "
                      f"Trades={result.total_trades}, Sharpe={result.sharpe_ratio:.2f}")
        
        print(f"✅ Successfully tested {len(results)} strategies on the same dataset")
        
        # Find best performing strategy
        if results:
            best_strategy = max(results, key=lambda x: x[1].sharpe_ratio)
            print(f"🏆 Best performing strategy: {best_strategy[0]} "
                  f"(Sharpe: {best_strategy[1].sharpe_ratio:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy combination test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_real_data():
    """Test strategies with real market data."""
    print(f"\n{'='*60}")
    print("TESTING WITH REAL MARKET DATA")
    print(f"{'='*60}")
    
    try:
        # Setup logging
        logger = setup_logging(level="INFO", console_output=True)
        
        # Load configuration and fetch real data
        config_manager = ConfigManager("config")
        exchange_config = config_manager.get_config("exchanges")
        data_fetcher = DataFetcher(exchange_config)
        
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)  # Last 100 days
        
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
        
        # Test a few strategies with real data
        strategies_to_test = [
            ("Moving Average", MovingAverageCrossoverStrategy, {"fast_period": 10, "slow_period": 21}),
            ("RSI", RSIStrategy, {"rsi_period": 14, "oversold": 30, "overbought": 70}),
            ("MACD", MACDStrategy, {"fast_period": 12, "slow_period": 26, "signal_period": 9})
        ]
        
        engine = BacktestingEngine(initial_capital=100000)
        
        for name, strategy_class, params in strategies_to_test:
            strategy = strategy_class(**params)
            strategy.initialize(result.data)
            
            if strategy.is_initialized:
                backtest_results = engine.backtest_strategy(strategy, result.data, "BTC/USDT")
                print(f"   {name}: Return={backtest_results.total_return:.2%}, "
                      f"Trades={backtest_results.total_trades}, "
                      f"Sharpe={backtest_results.sharpe_ratio:.2f}")
        
        print(f"✅ Real data testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test strategies with edge case scenarios."""
    print(f"\n{'='*60}")
    print("TESTING EDGE CASES")
    print(f"{'='*60}")
    
    try:
        # Test with very volatile data
        volatile_data = create_sample_data(100, volatility=0.08)
        strategy = RSIStrategy(rsi_period=14)
        strategy.initialize(volatile_data)
        
        engine = BacktestingEngine(initial_capital=100000)
        results = engine.backtest_strategy(strategy, volatile_data, "VOLATILE/USDT")
        print(f"✅ Handled high volatility data: {results.total_trades} trades")
        
        # Test with trending data
        trending_data = create_sample_data(100, volatility=0.01)
        # Add strong trend
        trending_data['close'] = trending_data['close'] * np.linspace(1, 1.5, len(trending_data))
        trending_data['high'] = trending_data['high'] * np.linspace(1, 1.5, len(trending_data))
        trending_data['low'] = trending_data['low'] * np.linspace(1, 1.5, len(trending_data))
        trending_data['open'] = trending_data['open'] * np.linspace(1, 1.5, len(trending_data))
        
        strategy = MovingAverageCrossoverStrategy(fast_period=5, slow_period=20)
        strategy.initialize(trending_data)
        results = engine.backtest_strategy(strategy, trending_data, "TREND/USDT")
        print(f"✅ Handled trending data: {results.total_trades} trades")
        
        # Test with minimal data
        minimal_data = create_sample_data(50, volatility=0.02)
        strategy = BollingerBandsStrategy(period=20)
        strategy.initialize(minimal_data)
        results = engine.backtest_strategy(strategy, minimal_data, "MINIMAL/USDT")
        print(f"✅ Handled minimal data: {results.total_trades} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("="*80)
    print("CORE TRADING STRATEGIES TEST SUITE")
    print("="*80)
    
    # Create test data
    test_data = create_sample_data(200, volatility=0.02)
    print(f"📊 Created test dataset with {len(test_data)} days of data")
    
    all_tests_passed = True
    
    # Test individual strategies
    strategy_tests = [
        (MovingAverageCrossoverStrategy, "Moving Average Crossover", {
            "fast_period": 10, "slow_period": 30, "ma_type": "EMA"
        }),
        (MACDStrategy, "MACD Strategy", {
            "fast_period": 12, "slow_period": 26, "signal_period": 9
        }),
        (RSIStrategy, "RSI Strategy", {
            "rsi_period": 14, "oversold": 30, "overbought": 70, "exit_signal": "opposite"
        }),
        (BollingerBandsStrategy, "Bollinger Bands Strategy", {
            "period": 20, "std_dev": 2.0, "entry_method": "both"
        }),
        (MomentumStrategy, "Momentum Strategy", {
            "period": 14, "threshold": 0.03
        })
    ]
    
    for strategy_class, name, params in strategy_tests:
        success = test_strategy(strategy_class, name, test_data, **params)
        if not success:
            all_tests_passed = False
    
    # Test strategy combinations
    print(f"\n🧪 Running strategy combination tests...")
    success = test_strategy_combinations()
    if not success:
        all_tests_passed = False
    
    # Test with real data
    print(f"\n🧪 Running real data tests...")
    success = await test_with_real_data()
    if not success:
        all_tests_passed = False
    
    # Test edge cases
    print(f"\n🧪 Running edge case tests...")
    success = test_edge_cases()
    if not success:
        all_tests_passed = False
    
    # Final results
    print("\n" + "="*80)
    if all_tests_passed:
        print("🎉 ALL CORE STRATEGY TESTS PASSED!")
        print("\nImplemented strategies:")
        print("✅ Moving Average Crossover (SMA, EMA, WMA, TEMA)")
        print("✅ MACD Strategy (with histogram analysis)")
        print("✅ RSI Strategy (with multiple exit strategies)")
        print("✅ Bollinger Bands (breakout and mean reversion)")
        print("✅ Momentum Strategy (ROC-based)")
        print("\nNext steps:")
        print("1. Implement remaining 60 strategies")
        print("2. Create strategy factory and registry")
        print("3. Implement hyperparameter optimization engine")
        print("4. Add portfolio management and risk controls")
    else:
        print("❌ SOME STRATEGY TESTS FAILED. Please check the errors above.")
        sys.exit(1)
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main()) 