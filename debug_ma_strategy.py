#!/usr/bin/env python3
"""
Debug MovingAverageCrossover Strategy

Investigate why the strategy is generating 0 trades but showing high returns
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
from strategies.backtesting_engine import BacktestingEngine
from utils.logger import get_logger

def get_sample_data():
    """Get sample BTCUSDT data for testing."""
    try:
        # Try to get real data
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(start="2023-01-01", end="2023-03-31", interval="4h")
        data.columns = [col.lower() for col in data.columns]
        return data
    except:
        # Generate synthetic data if yfinance fails
        print("⚠️ Using synthetic data for testing")
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="4h")
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [40000]  # Starting BTC price
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        return data

def debug_strategy():
    """Debug the MovingAverageCrossover strategy step by step."""
    
    print("🔍 DEBUGGING MOVINGAVERAGECROSSOVER STRATEGY")
    print("=" * 60)
    
    # 1. Get sample data
    print("\n📊 Getting sample data...")
    data = get_sample_data()
    print(f"   Data shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 2. Initialize strategy
    print("\n🎯 Initializing strategy...")
    strategy = MovingAverageCrossoverStrategy(
        fast_period=10,
        slow_period=20,
        ma_type="SMA",
        signal_threshold=0.005,  # 0.5% threshold
        position_size_pct=0.95,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    print(f"   Strategy: {strategy}")
    print(f"   Parameters: {strategy.parameters}")
    
    # 3. Initialize strategy with data
    print("\n🔧 Initializing strategy with data...")
    strategy.initialize(data)
    print(f"   Strategy initialized: {strategy.is_initialized}")
    
    if hasattr(strategy, 'fast_ma') and strategy.fast_ma is not None:
        print(f"   Fast MA samples: {strategy.fast_ma.tail(3).values}")
        print(f"   Slow MA samples: {strategy.slow_ma.tail(3).values}")
    
    # 4. Test signal generation manually
    print("\n🎮 Testing signal generation...")
    signal_count = 0
    buy_signals = 0
    sell_signals = 0
    
    # Test signals for 20 random timestamps
    test_dates = np.random.choice(data.index[25:], size=min(20, len(data)-25), replace=False)
    
    for i, test_date in enumerate(sorted(test_dates)):
        current_data = data.loc[test_date]
        signal = strategy.generate_signal(test_date, current_data)
        
        if signal.action != "hold":
            signal_count += 1
            if signal.action == "buy":
                buy_signals += 1
            elif signal.action == "sell":
                sell_signals += 1
            
            print(f"   Signal {signal_count}: {signal.action} @ {signal.price:.2f} "
                  f"(strength: {signal.strength:.3f}, confidence: {signal.confidence:.3f})")
            print(f"      Metadata: {signal.metadata}")
    
    print(f"\n📈 Signal Summary:")
    print(f"   Total signals generated: {signal_count}")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    
    # 5. Run backtest
    print("\n🏃 Running backtest...")
    engine = BacktestingEngine(initial_capital=100000)
    
    try:
        results = engine.backtest_strategy(strategy, data, "BTCUSDT")
        
        print(f"\n📊 BACKTEST RESULTS:")
        print(f"   Total Return: {results.total_return*100:.2f}%")
        print(f"   Annual Return: {results.annual_return*100:.2f}%")
        print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {results.max_drawdown*100:.2f}%")
        print(f"   Total Trades: {results.total_trades}")
        print(f"   Win Rate: {results.win_rate*100:.2f}%")
        print(f"   Profit Factor: {results.profit_factor:.2f}")
        
        # 6. Analyze individual trades
        if results.trades:
            print(f"\n💰 INDIVIDUAL TRADES:")
            for i, trade in enumerate(results.trades[:5]):  # Show first 5 trades
                print(f"   Trade {i+1}: {trade.side} {trade.size:.4f} @ {trade.entry_price:.2f}")
                print(f"      Entry: {trade.entry_time}")
                print(f"      Exit: {trade.exit_time} @ {trade.exit_price:.2f}")
                print(f"      P&L: {trade.net_pnl:.2f} ({trade.return_pct*100:.2f}%)")
                print(f"      Duration: {trade.duration}")
        else:
            print(f"\n⚠️ NO TRADES EXECUTED!")
            print(f"   This explains the 0 trades issue!")
            
            # Debug why no trades
            print(f"\n🔍 DEBUGGING NO TRADES:")
            
            # Check if strategy position is being updated
            print(f"   Strategy trades executed: {strategy.trades_executed}")
            print(f"   Strategy signals generated: {strategy.signals_generated}")
            print(f"   Current position: {strategy.current_position}")
            
            # Test signal generation again with verbose output
            print(f"\n🔬 DETAILED SIGNAL ANALYSIS:")
            for i, test_date in enumerate(sorted(test_dates)[:5]):
                current_data = data.loc[test_date]
                signal = strategy.generate_signal(test_date, current_data)
                
                # Get MA values at this time
                try:
                    current_idx = data.index.get_loc(test_date)
                    if hasattr(strategy, 'fast_ma') and strategy.fast_ma is not None:
                        fast_ma = strategy.fast_ma.iloc[current_idx]
                        slow_ma = strategy.slow_ma.iloc[current_idx]
                        prev_fast = strategy.fast_ma.iloc[current_idx-1] if current_idx > 0 else fast_ma
                        prev_slow = strategy.slow_ma.iloc[current_idx-1] if current_idx > 0 else slow_ma
                        
                        print(f"   {test_date}:")
                        print(f"      Price: {current_data['close']:.2f}")
                        print(f"      Fast MA: {fast_ma:.2f} (prev: {prev_fast:.2f})")
                        print(f"      Slow MA: {slow_ma:.2f} (prev: {prev_slow:.2f})")
                        print(f"      Diff%: {((fast_ma - slow_ma) / slow_ma * 100):.3f}%")
                        print(f"      Signal: {signal.action} (strength: {signal.strength:.3f})")
                        print(f"      Reason: {signal.metadata.get('reason', 'N/A')}")
                        
                except Exception as e:
                    print(f"      Error analyzing {test_date}: {e}")
        
        # 7. Equity curve analysis
        print(f"\n📈 EQUITY CURVE ANALYSIS:")
        print(f"   Starting equity: {results.equity_curve.iloc[0]:.2f}")
        print(f"   Ending equity: {results.equity_curve.iloc[-1]:.2f}")
        print(f"   Equity curve length: {len(results.equity_curve)}")
        
        # Check for unusual equity movements
        equity_changes = results.equity_curve.pct_change().dropna()
        large_moves = equity_changes[abs(equity_changes) > 0.1]
        if len(large_moves) > 0:
            print(f"   Large equity moves (>10%): {len(large_moves)}")
            print(f"   Largest move: {large_moves.max()*100:.2f}%")
            print(f"   This might explain the high return with 0 trades!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results if 'results' in locals() else None

if __name__ == "__main__":
    results = debug_strategy()
    print(f"\n🎯 Debug complete! Check output above for insights.") 