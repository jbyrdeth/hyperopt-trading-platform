#!/usr/bin/env python3
"""
Debug Strategy Performance - Deep Investigation

This script investigates why even aggressively expanded parameter ranges
are not producing profitable trades. We'll examine:

1. Signal generation frequency
2. Trade execution logic
3. Market data characteristics
4. Parameter constraint effectiveness
5. Backtesting engine behavior

Goal: Identify and fix the root cause preventing profitable signal generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

class StrategyPerformanceDebugger:
    """Deep strategy performance investigation."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine()
        
    def create_simple_trending_data(self, days: int = 30) -> pd.DataFrame:
        """Create simple trending data that should be profitable for basic strategies."""
        np.random.seed(42)
        
        n_points = days * 24  # Hourly data
        base_price = 50000
        
        # Create a clear uptrend with some noise
        trend = np.linspace(0, 0.3, n_points)  # 30% total gain
        noise = np.random.normal(0, 0.01, n_points)  # 1% noise
        
        # Combine trend and noise
        price_changes = trend + noise
        prices = base_price * (1 + price_changes)
        
        # Generate OHLCV
        opens = np.roll(prices, 1)
        opens[0] = base_price
        
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices))))
        volume = np.random.lognormal(15, 0.5, len(prices))
        
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
        
    def test_ultra_simple_strategy(self, data: pd.DataFrame) -> dict:
        """Test the simplest possible profitable strategy."""
        print("🧪 Testing ultra-simple MovingAverageCrossover...")
        
        # Ultra-simple parameters that should work on trending data
        params = {
            'fast_period': 2,      # Shortest possible
            'slow_period': 5,      # Very short
            'ma_type': 'SMA',      # Simple
            'signal_threshold': 0.0001,  # Minimal threshold
            'position_size_pct': 1.0,    # Max position
            'stop_loss_pct': 0.1,        # Wide stop loss
            'take_profit_pct': 0.2       # Wide take profit
        }
        
        try:
            strategy = self.factory.create_strategy("MovingAverageCrossover", **params)
            result = self.engine.backtest(strategy, data)
            
            print(f"   📊 Trades: {result['total_trades']}")
            print(f"   📈 Return: {result['total_return']:.4f}")
            print(f"   📉 Max DD: {result.get('max_drawdown', 'N/A')}")
            print(f"   🎯 Win Rate: {result.get('win_rate', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return {'error': str(e)}
            
    def analyze_signal_generation(self, data: pd.DataFrame) -> dict:
        """Analyze signal generation in detail."""
        print("\n🔍 SIGNAL GENERATION ANALYSIS")
        print("=" * 50)
        
        # Test simple MA crossover manually
        fast_ma = data['close'].rolling(window=2).mean()
        slow_ma = data['close'].rolling(window=5).mean()
        
        # Calculate crossover signals
        cross_above = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_below = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        buy_signals = cross_above.sum()
        sell_signals = cross_below.sum()
        
        print(f"📊 Manual Signal Analysis:")
        print(f"   📈 Buy signals: {buy_signals}")
        print(f"   📉 Sell signals: {sell_signals}")
        print(f"   📅 Data points: {len(data)}")
        print(f"   🔢 Signal frequency: {(buy_signals + sell_signals) / len(data) * 100:.2f}%")
        
        # Show first few signals
        buy_dates = data.index[cross_above][:5]
        sell_dates = data.index[cross_below][:5]
        
        print(f"\n📅 First 5 buy signals:")
        for i, date in enumerate(buy_dates):
            price = data.loc[date, 'close']
            print(f"   {i+1}. {date.strftime('%Y-%m-%d %H:%M')} at ${price:,.2f}")
            
        print(f"\n📅 First 5 sell signals:")
        for i, date in enumerate(sell_dates):
            price = data.loc[date, 'close']
            print(f"   {i+1}. {date.strftime('%Y-%m-%d %H:%M')} at ${price:,.2f}")
            
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': buy_signals + sell_signals,
            'signal_frequency_pct': (buy_signals + sell_signals) / len(data) * 100
        }
        
    def test_parameter_sensitivity(self, data: pd.DataFrame) -> dict:
        """Test how sensitive strategies are to different parameters."""
        print("\n🎛️ PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 50)
        
        results = []
        
        # Test various fast/slow period combinations
        fast_periods = [2, 3, 5, 8]
        slow_periods = [10, 15, 20, 30]
        thresholds = [0.0, 0.001, 0.005, 0.01]
        
        print("Testing parameter combinations:")
        
        for fast in fast_periods:
            for slow in slow_periods:
                for threshold in thresholds:
                    if fast >= slow:
                        continue
                        
                    params = {
                        'fast_period': fast,
                        'slow_period': slow,
                        'ma_type': 'SMA',
                        'signal_threshold': threshold,
                        'position_size_pct': 1.0,
                        'stop_loss_pct': 0.05,
                        'take_profit_pct': 0.1
                    }
                    
                    try:
                        strategy = self.factory.create_strategy("MovingAverageCrossover", **params)
                        result = self.engine.backtest(strategy, data)
                        
                        if result['total_trades'] > 0:
                            annual_return = result['total_return'] * (365 / (len(data) / 24))
                            results.append({
                                'fast': fast,
                                'slow': slow,
                                'threshold': threshold,
                                'trades': result['total_trades'],
                                'return': result['total_return'],
                                'annual_return': annual_return,
                                'profitable': result['total_return'] > 0
                            })
                            
                            print(f"   {fast:2d}/{slow:2d}, thr={threshold:5.3f}: "
                                  f"{result['total_trades']:2d} trades, "
                                  f"{result['total_return']:7.3f} return, "
                                  f"{annual_return:6.1f}% annual")
                    except Exception as e:
                        continue
                        
        if results:
            profitable = [r for r in results if r['profitable']]
            print(f"\n📊 Summary:")
            print(f"   🧪 Total combinations tested: {len(results)}")
            print(f"   ✅ Profitable combinations: {len(profitable)}")
            print(f"   📈 Success rate: {len(profitable)/len(results)*100:.1f}%")
            
            if profitable:
                best = max(profitable, key=lambda x: x['annual_return'])
                print(f"   🏆 Best: {best['fast']}/{best['slow']}, "
                      f"{best['annual_return']:.1f}% annual, {best['trades']} trades")
        else:
            print("   ❌ No working combinations found")
            
        return {'results': results, 'profitable_count': len(profitable) if results else 0}
        
    def investigate_backtesting_engine(self, data: pd.DataFrame) -> dict:
        """Investigate if there are issues with the backtesting engine itself."""
        print("\n🔧 BACKTESTING ENGINE INVESTIGATION")
        print("=" * 50)
        
        # Create a simple strategy manually
        params = {
            'fast_period': 3,
            'slow_period': 10,
            'ma_type': 'SMA',
            'signal_threshold': 0.001,
            'position_size_pct': 1.0,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1
        }
        
        try:
            strategy = self.factory.create_strategy("MovingAverageCrossover", **params)
            
            # Enable debug logging
            self.engine.debug = True
            result = self.engine.backtest(strategy, data)
            
            print(f"📊 Engine Debug Results:")
            print(f"   📈 Total return: {result['total_return']}")
            print(f"   🔄 Total trades: {result['total_trades']}")
            print(f"   💰 Final capital: {result.get('final_capital', 'N/A')}")
            print(f"   📊 Equity curve length: {len(result.get('equity_curve', []))}")
            
            # Check if equity curve exists and makes sense
            if 'equity_curve' in result:
                equity = result['equity_curve']
                print(f"   📈 Equity start: {equity.iloc[0]:.4f}")
                print(f"   📈 Equity end: {equity.iloc[-1]:.4f}")
                print(f"   📊 Equity range: {equity.min():.4f} to {equity.max():.4f}")
                
            return result
            
        except Exception as e:
            print(f"   ❌ Engine error: {e}")
            return {'error': str(e)}
            
    def run_comprehensive_debug(self) -> dict:
        """Run comprehensive debugging analysis."""
        print("🐛 COMPREHENSIVE STRATEGY PERFORMANCE DEBUG")
        print("=" * 60)
        print("🎯 Goal: Identify why strategies aren't generating profitable trades")
        
        # Create simple trending data that should be profitable
        print("\n📊 Creating simple trending data (30% gain over 30 days)...")
        data = self.create_simple_trending_data(days=30)
        
        market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        print(f"   📈 Market return: {market_return:.2f}%")
        print(f"   📅 Period: {len(data)} hourly data points")
        print(f"   💰 Price range: ${data['close'].min():,.0f} to ${data['close'].max():,.0f}")
        
        results = {}
        
        # 1. Test ultra-simple strategy
        results['simple_strategy'] = self.test_ultra_simple_strategy(data)
        
        # 2. Analyze signal generation
        results['signal_analysis'] = self.analyze_signal_generation(data)
        
        # 3. Test parameter sensitivity  
        results['parameter_sensitivity'] = self.test_parameter_sensitivity(data)
        
        # 4. Investigate backtesting engine
        results['engine_investigation'] = self.investigate_backtesting_engine(data)
        
        # Generate summary
        results['summary'] = {
            'market_return_pct': market_return,
            'data_points': len(data),
            'investigation_complete': True
        }
        
        return results
        
def main():
    """Run comprehensive strategy debugging."""
    debugger = StrategyPerformanceDebugger()
    
    print("🐛 STRATEGY PERFORMANCE DEBUGGING")
    print("=" * 50)
    print("🔍 Deep investigation into strategy profitability issues")
    
    results = debugger.run_comprehensive_debug()
    
    # Save results
    filename = f"strategy_debug_results_{int(datetime.now().timestamp())}.json"
    with open(filename, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return obj
            
        json.dump(results, f, indent=2, default=convert_types)
    
    print(f"\n📁 Debug results saved to: {filename}")
    print("\n🎯 DEBUG INVESTIGATION COMPLETE")
    print("=" * 50)
    
if __name__ == "__main__":
    main() 