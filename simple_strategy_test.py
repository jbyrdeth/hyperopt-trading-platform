#!/usr/bin/env python3
"""
Simple Strategy Test

Simple test to identify fundamental issues with strategy performance.
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

def create_simple_uptrend_data(days: int = 10) -> pd.DataFrame:
    """Create simple uptrending data that should be profitable."""
    np.random.seed(42)
    
    # Simple linear uptrend
    start_price = 50000
    end_price = 60000  # 20% gain over period
    n_points = days * 24
    
    prices = np.linspace(start_price, end_price, n_points)
    
    # Add minimal noise
    noise = np.random.normal(0, 100, n_points)  # Small $100 noise
    prices = prices + noise
    
    # Generate OHLCV
    opens = np.roll(prices, 1)
    opens[0] = start_price
    
    highs = prices * 1.002  # 0.2% above close
    lows = prices * 0.998   # 0.2% below close
    volume = np.full(n_points, 1000000)  # Constant volume
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=days)
    datetime_index = pd.date_range(start=start_date, periods=n_points, freq='H')
    
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volume
    }, index=datetime_index)
    
    return data

def test_basic_strategy():
    """Test basic strategy with minimal parameters."""
    print("🔍 BASIC STRATEGY TEST")
    print("=" * 40)
    
    # Create simple data
    data = create_simple_uptrend_data(days=10)
    market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
    print(f"📊 Market return: {market_return:.1f}%")
    print(f"📅 Data points: {len(data)}")
    
    # Test MovingAverageCrossover with minimal settings
    factory = StrategyFactory()
    engine = BacktestingEngine(initial_capital=100000)
    
    # Super simple parameters
    params = {
        'fast_period': 2,       # Ultra fast
        'slow_period': 5,       # Ultra slow
        'ma_type': 'SMA',       # Simple MA
        'signal_threshold': 0.0005,  # Minimum valid threshold
        'position_size_pct': 0.50,  # 50% position size
        'stop_loss_pct': 0.10,     # 10% stop loss (wide)
        'take_profit_pct': 0.15    # 15% take profit (wide)
    }
    
    print(f"\n🎯 Testing with params: {params}")
    
    try:
        strategy = factory.create_strategy("MovingAverageCrossover", **params)
        result = engine.backtest_strategy(strategy, data)
        
        print(f"\n📊 RESULTS:")
        print(f"   Total trades: {result.total_trades}")
        print(f"   Total return: {result.total_return*100:.1f}%")
        print(f"   Annual return: {result.annual_return*100:.1f}%")
        print(f"   Win rate: {result.win_rate*100:.1f}%")
        print(f"   Final capital: ${engine.current_capital:,.2f}")
        
        # Check individual trades
        if hasattr(result, 'trades') and result.trades:
            print(f"\n💰 INDIVIDUAL TRADES:")
            for i, trade in enumerate(result.trades):
                print(f"   Trade {i+1}: Entry ${trade.entry_price:.0f} → Exit ${trade.exit_price:.0f} = {trade.return_pct*100:.1f}%")
        
        # Manual calculation check
        print(f"\n🔍 MANUAL VERIFICATION:")
        initial_capital = engine.initial_capital
        final_capital = engine.current_capital
        manual_return = (final_capital / initial_capital - 1) * 100
        print(f"   Initial capital: ${initial_capital:,.2f}")
        print(f"   Final capital: ${final_capital:,.2f}")
        print(f"   Manual return calc: {manual_return:.1f}%")
        
        # Check if MA signals are working
        print(f"\n📈 MOVING AVERAGE ANALYSIS:")
        data_copy = data.copy()
        
        # Calculate MAs manually
        data_copy['MA_fast'] = data_copy['close'].rolling(window=params['fast_period']).mean()
        data_copy['MA_slow'] = data_copy['close'].rolling(window=params['slow_period']).mean()
        data_copy['signal'] = (data_copy['MA_fast'] > data_copy['MA_slow']).astype(int)
        data_copy['signal_change'] = data_copy['signal'].diff()
        
        buy_signals = len(data_copy[data_copy['signal_change'] == 1])
        sell_signals = len(data_copy[data_copy['signal_change'] == -1])
        
        print(f"   Buy signals detected: {buy_signals}")
        print(f"   Sell signals detected: {sell_signals}")
        print(f"   Fast MA end: ${data_copy['MA_fast'].iloc[-1]:.0f}")
        print(f"   Slow MA end: ${data_copy['MA_slow'].iloc[-1]:.0f}")
        
        if buy_signals == 0:
            print("   ⚠️ NO BUY SIGNALS - MA crossover not happening")
        
        return result
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_buy_and_hold():
    """Test simple buy and hold strategy."""
    print(f"\n🏪 BUY AND HOLD TEST")
    print("=" * 30)
    
    data = create_simple_uptrend_data(days=10)
    
    start_price = data['close'].iloc[0]
    end_price = data['close'].iloc[-1]
    
    # Simulate buying at start, selling at end
    initial_capital = 100000
    shares = initial_capital / start_price
    final_value = shares * end_price
    buy_hold_return = (final_value / initial_capital - 1) * 100
    
    print(f"   Start price: ${start_price:.0f}")
    print(f"   End price: ${end_price:.0f}")
    print(f"   Shares bought: {shares:.4f}")
    print(f"   Final value: ${final_value:.0f}")
    print(f"   Buy & Hold return: {buy_hold_return:.1f}%")
    
    return buy_hold_return

def main():
    """Run simple strategy tests."""
    print("🧪 SIMPLE STRATEGY DIAGNOSTIC")
    print("=" * 50)
    print("🎯 Testing basic strategy logic")
    
    # Test buy and hold baseline
    buy_hold_return = test_buy_and_hold()
    
    # Test basic strategy
    strategy_result = test_basic_strategy()
    
    print(f"\n📋 COMPARISON:")
    if strategy_result:
        strategy_return = strategy_result.total_return * 100
        print(f"   Buy & Hold: {buy_hold_return:.1f}%")
        print(f"   Strategy: {strategy_return:.1f}%")
        
        if strategy_return < buy_hold_return:
            print(f"   ⚠️ Strategy underperforming buy & hold by {buy_hold_return - strategy_return:.1f}%")
        else:
            print(f"   ✅ Strategy outperforming buy & hold by {strategy_return - buy_hold_return:.1f}%")
    
    print(f"\n🎉 SIMPLE TEST COMPLETE!")

if __name__ == "__main__":
    main() 