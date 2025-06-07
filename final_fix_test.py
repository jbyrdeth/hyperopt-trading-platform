#!/usr/bin/env python3
"""
Final Fix Test

Step-by-step analysis of the equity curve and capital tracking.
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

def create_simple_data(days: int = 5) -> pd.DataFrame:
    """Create very simple test data."""
    np.random.seed(42)
    
    # Create step function: 50000 -> 55000 -> 60000
    prices = [50000] * 48 + [55000] * 48 + [60000] * 24  # 5 days of hourly data
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices)
    })
    
    # Add datetime index
    start_date = datetime.now() - timedelta(days=days)
    data.index = pd.date_range(start=start_date, periods=len(prices), freq='H')
    
    return data

def debug_backtesting_step_by_step():
    """Debug the backtesting process step by step."""
    print("🔍 STEP-BY-STEP BACKTESTING DEBUG")
    print("=" * 50)
    
    # Create simple data
    data = create_simple_data(days=5)
    print(f"📊 Data overview:")
    print(f"   Start price: ${data['close'].iloc[0]:,.0f}")
    print(f"   Mid price: ${data['close'].iloc[len(data)//2]:,.0f}")
    print(f"   End price: ${data['close'].iloc[-1]:,.0f}")
    print(f"   Market return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1)*100:.1f}%")
    
    # Create strategy with minimal parameters
    factory = StrategyFactory()
    
    # Use a custom BacktestingEngine that logs everything
    class DebugBacktestingEngine(BacktestingEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.portfolio_debug = []
            
        def backtest_strategy(self, strategy, data, symbol="BTC/USDT"):
            """Override to add debugging."""
            print(f"\n🎯 Starting backtest with initial capital: ${self.initial_capital:,.2f}")
            
            # Reset engine state
            self.reset()
            
            # Initialize tracking
            portfolio_values = []
            
            # Add volatility column if not present
            if 'volatility' not in data.columns:
                data = data.copy()
                data['volatility'] = data['close'].pct_change().rolling(20).std().fillna(0.02)
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                current_price = row['close']
                volatility = row['volatility']
                
                # Update strategy with current data
                strategy.update(data.iloc[:i+1])
                
                # Check for signals
                signal = strategy.generate_signal(data.iloc[:i+1])
                
                if signal and not signal.is_neutral:
                    print(f"\n📡 Signal at {i}: {signal.direction} @ ${current_price:.0f}")
                    # Execute signal
                    self._execute_signal(strategy, signal, current_price, timestamp, volatility, symbol)
                    print(f"   Capital after signal: ${self.current_capital:.2f}")
                
                # Calculate portfolio value
                if not strategy.current_position.is_flat:
                    position = strategy.current_position
                    unrealized_pnl = (current_price - position.entry_price) * position.size
                    total_value = self.current_capital + unrealized_pnl
                    
                    if i % 24 == 0:  # Print daily
                        print(f"   Day {i//24}: Position ${position.size:.2f} @ ${position.entry_price:.0f}, Current ${current_price:.0f}, Unrealized P&L: ${unrealized_pnl:.2f}, Portfolio: ${total_value:.2f}")
                else:
                    total_value = self.current_capital
                    if i % 24 == 0:  # Print daily
                        print(f"   Day {i//24}: No position, Portfolio: ${total_value:.2f}")
                
                portfolio_values.append(total_value)
                self.timestamps.append(timestamp)
            
            # Close any remaining position
            if not strategy.current_position.is_flat:
                final_price = data.iloc[-1]['close']
                final_volatility = data.iloc[-1]['volatility']
                print(f"\n🔚 Closing final position @ ${final_price:.0f}")
                self._execute_exit(strategy, final_price, data.index[-1], final_volatility, symbol, "end_of_data")
                print(f"   Final capital: ${self.current_capital:.2f}")
            
            # Calculate results
            results = self._calculate_results(strategy, data, portfolio_values, symbol)
            
            print(f"\n📋 FINAL RESULTS:")
            print(f"   Portfolio values: Start ${portfolio_values[0]:.0f}, End ${portfolio_values[-1]:.0f}")
            print(f"   Calculated total return: {results.total_return*100:.1f}%")
            print(f"   Manual total return: {(self.current_capital / self.initial_capital - 1)*100:.1f}%")
            print(f"   Total trades: {results.total_trades}")
            
            return results
    
    engine = DebugBacktestingEngine(initial_capital=100000)
    
    # Simple parameters - use valid ranges
    params = {
        'fast_period': 5,      # Minimum valid value
        'slow_period': 10,     # Valid value > fast_period
        'ma_type': 'SMA',
        'signal_threshold': 0.0005,
        'position_size_pct': 0.50,
        'stop_loss_pct': None,
        'take_profit_pct': None
    }
    
    strategy = factory.create_strategy("MovingAverageCrossover", **params)
    result = engine.backtest_strategy(strategy, data)
    
    print(f"\n🎉 Debug complete!")
    return result

def main():
    """Run final fix test."""
    print("🔧 FINAL FIX TEST")
    print("=" * 30)
    
    result = debug_backtesting_step_by_step()

if __name__ == "__main__":
    main() 