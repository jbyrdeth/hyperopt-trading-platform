#!/usr/bin/env python3
"""
Simple Backtest Debug Script

Investigate the "0 trades but high returns" bug in the backtesting engine
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Import our modules
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from strategies.backtesting_engine import BacktestingEngine
from utils.logger import get_logger

def create_test_data():
    """Create simple test data with obvious signals."""
    
    # Create 30 days of hourly data
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="1h")
    
    # Create a simple trend: price goes from 100 to 200 over 30 days
    base_prices = np.linspace(100, 200, len(dates))
    
    # Add some volatility
    np.random.seed(42)
    noise = np.random.normal(0, 2, len(dates))
    prices = base_prices + noise
    
    # Ensure prices are always positive and create OHLCV data
    data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    }, index=dates)
    
    return data

def debug_strategy_step_by_step():
    """Debug the strategy execution step by step."""
    
    print("🔍 DEBUGGING BACKTESTING ENGINE - STEP BY STEP")
    print("=" * 60)
    
    # 1. Create test data
    print("\n📊 Creating test data...")
    data = create_test_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Show price progression
    print(f"   Starting price: ${data['close'].iloc[0]:.2f}")
    print(f"   Ending price: ${data['close'].iloc[-1]:.2f}")
    print(f"   Total price change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    # 2. Create strategy with simple parameters
    print("\n🎯 Creating strategy...")
    strategy = MovingAverageCrossoverStrategy(
        fast_period=5,   # Very short period
        slow_period=10,  # Short period for quick signals
        ma_type="SMA",
        signal_threshold=0.01,  # 1% threshold
        position_size_pct=1.0,  # Full position
        stop_loss_pct=0.10,     # 10% stop loss
        take_profit_pct=0.20    # 20% take profit
    )
    
    print(f"   Strategy parameters: {strategy.parameters}")
    
    # 3. Initialize strategy
    print("\n🔧 Initializing strategy...")
    strategy.initialize(data)
    
    # Check moving averages
    if hasattr(strategy, 'fast_ma') and strategy.fast_ma is not None:
        print(f"   Fast MA calculated: {len(strategy.fast_ma)} points")
        print(f"   Slow MA calculated: {len(strategy.slow_ma)} points")
        
        # Show some MA values
        valid_start = max(strategy.fast_period, strategy.slow_period)
        print(f"   Fast MA [start]: {strategy.fast_ma.iloc[valid_start:valid_start+3].values}")
        print(f"   Slow MA [start]: {strategy.slow_ma.iloc[valid_start:valid_start+3].values}")
        print(f"   Fast MA [end]: {strategy.fast_ma.iloc[-3:].values}")
        print(f"   Slow MA [end]: {strategy.slow_ma.iloc[-3:].values}")
    
    # 4. Test signal generation manually
    print("\n🎮 Testing signal generation...")
    signals_generated = 0
    buy_signals = 0
    sell_signals = 0
    
    # Test signals for every 24th hour (daily)
    test_indices = range(max(strategy.fast_period, strategy.slow_period), len(data), 24)
    
    for i in test_indices[:10]:  # Test first 10 days
        timestamp = data.index[i]
        current_data = data.iloc[i]
        
        # Generate signal
        signal = strategy.generate_signal(timestamp, current_data)
        
        if signal.action != "hold":
            signals_generated += 1
            if signal.action == "buy":
                buy_signals += 1
            elif signal.action == "sell":
                sell_signals += 1
                
            print(f"   Signal {signals_generated}: {signal.action} @ ${signal.price:.2f}")
            print(f"      Time: {timestamp}")
            print(f"      Strength: {signal.strength:.3f}")
            print(f"      Confidence: {signal.confidence:.3f}")
            print(f"      Reason: {signal.metadata.get('reason', 'N/A')}")
            
            # Show MA values at this time
            if hasattr(strategy, 'fast_ma'):
                fast_val = strategy.fast_ma.iloc[i]
                slow_val = strategy.slow_ma.iloc[i]
                print(f"      Fast MA: {fast_val:.2f}, Slow MA: {slow_val:.2f}")
                print(f"      MA Diff: {((fast_val - slow_val) / slow_val * 100):.3f}%")
    
    print(f"\n📈 Signal Summary:")
    print(f"   Total signals: {signals_generated}")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    
    # 5. Run full backtest
    print("\n🏃 Running full backtest...")
    engine = BacktestingEngine(initial_capital=100000)
    
    try:
        print(f"   Engine initial capital: ${engine.initial_capital:,.2f}")
        
        # Run backtest
        results = engine.backtest_strategy(strategy, data, "TESTBTC")
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Total Return: {results.total_return*100:.2f}%")
        print(f"   Annual Return: {results.annual_return*100:.2f}%") 
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {results.max_drawdown*100:.2f}%")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Win Rate: {results.win_rate*100:.2f}%")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        
        # Check equity curve
        print(f"\n💰 EQUITY ANALYSIS:")
        print(f"   Starting equity: ${results.equity_curve.iloc[0]:,.2f}")
        print(f"   Ending equity: ${results.equity_curve.iloc[-1]:,.2f}")
        print(f"   Equity curve length: {len(results.equity_curve)}")
        
        # Check for suspicious patterns
        if results.total_trades == 0 and results.total_return > 0.1:
            print(f"\n⚠️ BUG DETECTED:")
            print(f"   0 trades but {results.total_return*100:.2f}% return!")
            print(f"   This confirms the bug we're investigating.")
            
            # Look at equity curve changes
            equity_changes = results.equity_curve.pct_change().dropna()
            large_changes = equity_changes[abs(equity_changes) > 0.01]
            
            if len(large_changes) > 0:
                print(f"   Large equity changes: {len(large_changes)}")
                print(f"   Max change: {large_changes.max()*100:.2f}%")
                print(f"   These changes happened without trades!")
            
        # Show first few trades if any
        if results.trades and len(results.trades) > 0:
            print(f"\n💼 FIRST FEW TRADES:")
            for i, trade in enumerate(results.trades[:3]):
                print(f"   Trade {i+1}: {trade.side} {trade.size:.4f} @ ${trade.entry_price:.2f}")
                print(f"      Entry: {trade.entry_time}")
                print(f"      Exit: {trade.exit_time} @ ${trade.exit_price:.2f}")
                print(f"      P&L: ${trade.net_pnl:.2f} ({trade.return_pct*100:.2f}%)")
        else:
            print(f"\n❌ NO TRADES RECORDED!")
            print(f"   Despite generating {signals_generated} signals, no trades were executed.")
            print(f"   This suggests the bug is in the trade execution logic.")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Starting backtest debug session")
    
    results = debug_strategy_step_by_step()
    
    if results:
        print(f"\n🎯 DEBUG COMPLETE!")
        print(f"   Check the output above for the root cause of the 0 trades bug.")
    else:
        print(f"\n❌ DEBUG FAILED!")
        print(f"   Could not complete the backtest analysis.") 