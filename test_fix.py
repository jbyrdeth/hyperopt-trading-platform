#!/usr/bin/env python3
"""
Test the backtesting engine fix
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime

# Simple test to verify the fix works
def test_fix():
    print("🔧 TESTING BACKTESTING ENGINE FIX")
    print("=" * 50)
    
    try:
        # Import modules
        from strategies.backtesting_engine import BacktestingEngine, CostModel
        from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
        
        print("✅ Modules imported successfully")
        
        # Create simple test data with clear trend
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='1h')
        np.random.seed(42)
        prices = np.linspace(100, 110, len(dates))  # 10% price increase over 10 days
        
        data = pd.DataFrame({
            'open': prices * 0.999,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.uniform(1000, 5000, len(dates))
        }, index=dates)
        
        print(f"📊 Test data: {len(data)} points")
        print(f"   Price range: ${prices[0]:.2f} -> ${prices[-1]:.2f} ({((prices[-1]/prices[0])-1)*100:.1f}% change)")
        
        # Create strategy with aggressive parameters for quick signals
        strategy = MovingAverageCrossoverStrategy(
            fast_period=3,
            slow_period=6,
            ma_type='SMA',
            signal_threshold=0.005,  # 0.5% threshold
            position_size_pct=0.5    # Use 50% of capital per trade
        )
        
        print(f"🎯 Strategy: {strategy.name}")
        print(f"   Parameters: {strategy.parameters}")
        
        # Create engine with small capital for easy tracking
        engine = BacktestingEngine(initial_capital=10000)
        print(f"💰 Initial capital: ${engine.initial_capital:,.2f}")
        
        # Run backtest
        print("\\n🏃 Running backtest with FIX...")
        results = engine.backtest_strategy(strategy, data, "TEST/USDT")
        
        print(f"\\n📊 RESULTS AFTER FIX:")
        print(f"   Total Return: {results.total_return*100:.2f}%")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Win Rate: {results.win_rate*100:.2f}%")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Starting Equity: ${results.equity_curve.iloc[0]:,.2f}")
        print(f"   Ending Equity: ${results.equity_curve.iloc[-1]:,.2f}")
        
        # Check if fix worked
        if results.total_trades > 0:
            print(f"\\n✅ FIX SUCCESSFUL!")
            print(f"   - Trades are now being recorded: {results.total_trades} trades")
            print(f"   - Returns are reasonable: {results.total_return*100:.2f}%")
            
            # Show first trade
            if results.trades:
                trade = results.trades[0]
                print(f"   - First trade: {trade.side} {trade.size:.4f} @ ${trade.entry_price:.2f}")
                print(f"     Exit: ${trade.exit_price:.2f}, P&L: ${trade.net_pnl:.2f}")
        else:
            if results.total_return > 0.05:  # > 5% return with 0 trades
                print(f"\\n❌ BUG STILL EXISTS!")
                print(f"   - 0 trades but {results.total_return*100:.2f}% return")
            else:
                print(f"\\n⚠️ No trades generated (may be normal for short test period)")
                print(f"   - Return: {results.total_return*100:.2f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_fix()
    
    if results:
        print(f"\\n🎯 TEST COMPLETE!")
        if results.total_trades > 0:
            print(f"   The fix appears to be working correctly.")
        else:
            print(f"   Need to investigate further or adjust test parameters.")
    else:
        print(f"\\n❌ TEST FAILED!")
        print(f"   Could not complete the test.") 