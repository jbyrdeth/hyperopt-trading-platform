#!/usr/bin/env python3
"""
Debug Capital Tracking

This script will trace capital and position values step by step
to understand why we're getting astronomical returns.
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

def create_optimal_test_data(days: int = 60) -> pd.DataFrame:
    """Create optimal test data designed for profitable strategies (same as breakthrough script)."""
    np.random.seed(42)
    
    # Create hourly data points
    n_points = days * 24
    
    # Create a mixed market with multiple profitable phases
    
    # Phase 1: Strong bull run (30% of data)
    bull_points = int(n_points * 0.3)
    bull_trend = np.cumsum(np.random.normal(0.0008, 0.01, bull_points))
    
    # Phase 2: Volatile sideways (25% of data) 
    sideways_points = int(n_points * 0.25)
    sideways = np.cumsum(np.random.normal(0.0001, 0.02, sideways_points))
    
    # Phase 3: Moderate decline (20% of data)
    bear_points = int(n_points * 0.2)
    bear_trend = np.cumsum(np.random.normal(-0.0003, 0.015, bear_points))
    
    # Phase 4: Recovery rally (remaining 25%)
    recovery_points = n_points - bull_points - sideways_points - bear_points
    recovery_trend = np.cumsum(np.random.normal(0.0006, 0.018, recovery_points))
    
    # Combine all phases
    price_changes = np.concatenate([bull_trend, sideways, bear_trend, recovery_trend])
    
    # Generate realistic price series
    base_price = 50000
    prices = base_price * np.exp(price_changes)  # EXPONENTIAL - this might cause issues!
    
    # Generate OHLCV
    opens = np.roll(prices, 1)
    opens[0] = base_price
    
    highs = prices * (1 + np.abs(np.random.normal(0, 0.008, len(prices))))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.008, len(prices))))
    volume = np.random.lognormal(15, 0.8, len(prices))
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=days)
    datetime_index = pd.date_range(start=start_date, periods=len(prices), freq='H')
    
    data = pd.DataFrame({
        'open': opens,
        'high': np.maximum.reduce([opens, prices, highs]),
        'low': np.minimum.reduce([opens, prices, lows]),
        'close': prices,
        'volume': volume
    }, index=datetime_index)
    
    return data

class CapitalTracker:
    """Track capital and positions during backtesting."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine(initial_capital=100000)
        
    def debug_single_trade(self):
        """Debug a single trade execution."""
        print("🔍 DEBUGGING SINGLE TRADE EXECUTION")
        print("=" * 50)
        
        # Create simple test data
        data = create_optimal_test_data(days=60)  # More data for signals
        print(f"📊 Test data: {len(data)} points")
        print(f"💰 Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        # Use the EXACT breakthrough parameters causing issues
        params = {
            'fast_period': 3,      # Ultra-fast from breakthrough
            'slow_period': 8,      # Short from breakthrough  
            'ma_type': 'EMA',      # EMA from breakthrough
            'signal_threshold': 0.0005,  # MINIMUM valid threshold 
            'position_size_pct': 0.1,    # REASONABLE position size (10% instead of 100%)
            'stop_loss_pct': 0.05,       # From breakthrough
            'take_profit_pct': 0.08      # From breakthrough
        }
        
        print(f"\n🎯 Using breakthrough parameters:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        
        strategy = self.factory.create_strategy("MovingAverageCrossover", **params)
        
        print(f"\n💰 Initial capital: ${self.engine.initial_capital:,.2f}")
        
        # Manual debug: Check position size calculation
        print(f"\n🔍 MANUAL POSITION SIZE DEBUG:")
        print(f"   Available capital for position sizing: ${self.engine.current_capital * 0.998:,.2f}")
        print(f"   Position size pct: {params['position_size_pct']}")
        
        # Test position size calculation manually
        test_price = 50000
        test_available = self.engine.current_capital * 0.998
        expected_position_size = test_available * params['position_size_pct'] / test_price
        print(f"   Expected position size at $50k: {expected_position_size:.8f}")
        
        # Enable detailed logging for debugging
        original_logger_level = self.engine.logger.level
        self.engine.logger.setLevel(10)  # DEBUG level
        
        # Also enable strategy logger
        strategy_logger = strategy.logger
        original_strategy_level = strategy_logger.level
        strategy_logger.setLevel(10)  # DEBUG level
        
        # Run backtest with detailed tracking
        self.engine.reset()
        result = self.engine.backtest_strategy(strategy, data)
        
        # Restore logger level
        self.engine.logger.setLevel(original_logger_level)
        strategy_logger.setLevel(original_strategy_level)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Total trades: {result.total_trades}")
        print(f"   Total return: {result.total_return:.8f}")
        print(f"   Annual return: {result.annual_return:.8f}")
        print(f"   Final capital: ${self.engine.current_capital:,.2f}")
        
        # Show equity curve details
        print(f"\n📈 EQUITY CURVE ANALYSIS:")
        equity_curve = result.equity_curve
        print(f"   Start value: ${equity_curve.iloc[0]:,.2f}")
        print(f"   End value: ${equity_curve.iloc[-1]:,.2f}")
        print(f"   Min value: ${equity_curve.min():,.2f}")
        print(f"   Max value: ${equity_curve.max():,.2f}")
        
        # Debug portfolio value calculation
        print(f"\n🔍 PORTFOLIO VALUE DEBUG:")
        print(f"   Initial capital: ${self.engine.initial_capital:,.2f}")
        print(f"   Final current_capital: ${self.engine.current_capital:,.2f}")
        print(f"   Total transaction costs: ${self.engine.total_transaction_costs:.2f}")
        print(f"   Invested capital: ${self.engine.invested_capital:.2f}")
        
        # Check if there's an open position
        if hasattr(strategy, 'current_position') and not strategy.current_position.is_flat:
            pos = strategy.current_position
            current_price = data['close'].iloc[-1]
            unrealized_pnl = (current_price - pos.entry_price) * pos.size
            print(f"\n🔍 OPEN POSITION DEBUG:")
            print(f"   Position size: {pos.size:.8f}")
            print(f"   Entry price: ${pos.entry_price:.2f}")
            print(f"   Current price: ${current_price:.2f}")
            print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
            expected_portfolio = self.engine.initial_capital - self.engine.total_transaction_costs + unrealized_pnl
            print(f"   Expected portfolio value: ${expected_portfolio:.2f}")
        else:
            print(f"\n🔍 NO OPEN POSITION")
            expected_portfolio = self.engine.initial_capital - self.engine.total_transaction_costs
            print(f"   Expected portfolio value: ${expected_portfolio:.2f}")
            
        print(f"   Actual final equity: ${equity_curve.iloc[-1]:,.2f}")
        
        # Check for extreme values in equity curve
        extreme_changes = equity_curve.pct_change().abs()
        large_changes = extreme_changes[extreme_changes > 0.5]  # >50% changes
        if len(large_changes) > 0:
            print(f"\n⚠️ LARGE EQUITY JUMPS DETECTED:")
            print(f"   Count: {len(large_changes)}")
            print(f"   Max jump: {extreme_changes.max():.1%}")
            print(f"   Min jump: {extreme_changes.min():.1%}")
            
            # Show first few large changes
            for i, (timestamp, change) in enumerate(large_changes.head(3).items()):
                idx = equity_curve.index.get_loc(timestamp)
                if idx > 0:
                    prev_val = equity_curve.iloc[idx-1]
                    curr_val = equity_curve.iloc[idx]
                    print(f"   Jump {i+1}: ${prev_val:,.2f} → ${curr_val:,.2f} ({change:.1%})")
        
        # Show individual trades if any
        if result.trades:
            print(f"\n💰 INDIVIDUAL TRADES:")
            for i, trade in enumerate(result.trades[:3]):  # Show first 3
                print(f"   Trade {i+1}:")
                print(f"     Entry: {trade.entry_price:.2f} @ {trade.entry_time}")
                print(f"     Exit:  {trade.exit_price:.2f} @ {trade.exit_time}")
                print(f"     Size:  {trade.size:.8f}")
                print(f"     P&L:   {trade.net_pnl:.2f}")
                print(f"     Return: {trade.return_pct*100:.3f}%")
                
        # Check for extreme values
        if abs(result.total_return) > 10:
            print(f"\n⚠️ EXTREME RETURN DETECTED!")
            print(f"   Return: {result.total_return:.6f} ({result.total_return*100:.2f}%)")
            print(f"   This suggests position size calculation issues")
            
            # Check position sizes in trades
            if result.trades:
                position_sizes = [t.size for t in result.trades]
                print(f"   Position sizes: {position_sizes[:5]}...")  # First 5
        
        return result

def main():
    """Run capital tracking debug."""
    tracker = CapitalTracker()
    
    print("🔍 CAPITAL TRACKING DEBUG")
    print("=" * 40)
    print("🎯 Goal: Understand why returns are astronomical")
    
    result = tracker.debug_single_trade()
    
    print(f"\n🎯 DEBUG COMPLETE")
    print("=" * 30)

if __name__ == "__main__":
    main() 