#!/usr/bin/env python3
"""
Strategy Diagnostic Tool
Check why champion strategies are showing low returns and minimal trading
"""

import sys
import os

# Fix path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

import pandas as pd
import numpy as np

# Import with try/catch to handle path issues
try:
    from src.optimization.strategy_factory import StrategyFactory
    from src.strategies.backtesting_engine import BacktestingEngine
except ImportError:
    try:
        from optimization.strategy_factory import StrategyFactory
        from strategies.backtesting_engine import BacktestingEngine
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure you're running from the project root directory")
        sys.exit(1)

def run_strategy_diagnostic():
    """Run comprehensive diagnostic on champion strategies."""
    print("🔍 STRATEGY DIAGNOSTIC ANALYSIS")
    print("=" * 50)
    
    factory = StrategyFactory()
    engine = BacktestingEngine(initial_capital=100000)
    
    # Test our champion strategies
    champion_strategies = [
        'MovingAverageCrossover',
        'AD',  # Accumulation/Distribution
        'HistoricalVolatility',
        'MTFMACD',
        'VWAP'
    ]
    
    # Create test data with clear trending behavior
    print("📊 Generating test data...")
    dates = pd.date_range('2023-01-01', periods=2000, freq='H')
    base_price = 45000
    
    # Create strong uptrend scenario
    trend = 0.0008  # Strong trend
    volatility = 0.025  # Good volatility for trading
    
    returns = np.random.normal(trend, volatility, len(dates))
    # Add some momentum for more realistic behavior
    for i in range(1, len(returns)):
        returns[i] += 0.15 * returns[i-1]  # Add momentum
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create realistic OHLCV data
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, volatility/3)))
        low = price * (1 - abs(np.random.normal(0, volatility/3)))
        volume = np.random.uniform(2000, 8000)
        
        data.append({
            'open': price,
            'high': max(price, high),
            'low': min(price, low), 
            'close': price,
            'volume': volume
        })
    
    test_data = pd.DataFrame(data, index=dates)
    
    market_return = (test_data['close'].iloc[-1] / test_data['close'].iloc[0]) - 1
    print(f"Market data: {len(test_data)} points")
    print(f"Price range: ${test_data['close'].min():.0f} - ${test_data['close'].max():.0f}")
    print(f"Market return: {market_return:.1%}")
    print(f"Market volatility: {test_data['close'].pct_change().std() * np.sqrt(365*24):.1%}")
    
    print(f"\n🧪 Testing {len(champion_strategies)} champion strategies...")
    
    results = []
    
    for strategy_name in champion_strategies:
        print(f"\n🔄 Testing {strategy_name}...")
        
        try:
            # Get strategy parameters
            params = factory.get_default_parameters(strategy_name)
            print(f"   Parameters: {params}")
            
            # Create strategy instance
            strategy_class = factory.registry.get_strategy_class(strategy_name)
            strategy = strategy_class(**params)
            
            # Test signal generation
            signals_count = 0
            buy_signals = 0
            sell_signals = 0
            
            # Check signals in chunks to see activity
            for i in range(100, len(test_data), 200):
                chunk_data = test_data.iloc[:i]
                try:
                    current_time = chunk_data.index[i-1]  # Get the timestamp
                    current_data = chunk_data.iloc[i-1]   # Get the data row
                    signal = strategy.generate_signal(current_time, current_data)
                    if signal and hasattr(signal, 'action'):
                        if signal.action == 'buy':
                            buy_signals += 1
                        elif signal.action == 'sell':
                            sell_signals += 1
                        signals_count += 1
                except Exception as e:
                    print(f"   ⚠️ Signal error at point {i}: {e}")
                    break
            
            print(f"   Signal activity: {signals_count} total, {buy_signals} buy, {sell_signals} sell")
            
            # Run full backtest
            result = engine.backtest_strategy(strategy, test_data)
            
            print(f"   📈 Results:")
            print(f"      Total trades: {result.total_trades}")
            print(f"      Total return: {result.total_return:.1%}")
            print(f"      Annual return: {result.annual_return:.1%}")
            print(f"      Sharpe ratio: {result.sharpe_ratio:.2f}")
            print(f"      Win rate: {result.win_rate:.1%}")
            
            # Check if there's a position sizing issue
            if hasattr(result, 'trades') and len(result.trades) > 0:
                avg_trade_size = np.mean([abs(trade.size * trade.entry_price) for trade in result.trades])
                print(f"      Avg trade size: ${avg_trade_size:.2f}")
                print(f"      Position sizing: {avg_trade_size / 100000:.1%} of capital")
            
            results.append({
                'strategy': strategy_name,
                'trades': result.total_trades,
                'return': result.total_return,
                'annual_return': result.annual_return,
                'signals': signals_count,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            results.append({
                'strategy': strategy_name,
                'error': str(e),
                'status': 'failed'
            })
    
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['status'] == 'success']
    failed_tests = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Successful tests: {len(successful_tests)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\n🏆 PERFORMANCE SUMMARY:")
        for result in successful_tests:
            print(f"   {result['strategy']}: {result['trades']} trades, {result['return']:.1%} return")
        
        # Check for common issues
        low_activity_strategies = [r for r in successful_tests if r['trades'] < 5]
        low_return_strategies = [r for r in successful_tests if abs(r['return']) < 0.1]
        
        if low_activity_strategies:
            print(f"\n⚠️ LOW ACTIVITY DETECTED:")
            for result in low_activity_strategies:
                print(f"   {result['strategy']}: Only {result['trades']} trades")
        
        if low_return_strategies:
            print(f"\n⚠️ LOW RETURNS DETECTED:")
            for result in low_return_strategies:
                print(f"   {result['strategy']}: Only {result['return']:.1%} return")
    
    if failed_tests:
        print(f"\n❌ FAILED STRATEGIES:")
        for result in failed_tests:
            print(f"   {result['strategy']}: {result['error']}")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if len([r for r in successful_tests if r['trades'] < 5]) > 0:
        print("   📊 Consider more aggressive position sizing")
        print("   🎚️ Reduce signal generation thresholds")
        print("   📈 Increase market data volatility for testing")
    
    if len([r for r in successful_tests if abs(r['return']) < 0.1]) > 0:
        print("   💰 Check position sizing parameters")
        print("   ⚖️ Verify risk management isn't too conservative")
        print("   🔧 Review strategy parameter optimization")

if __name__ == "__main__":
    run_strategy_diagnostic() 