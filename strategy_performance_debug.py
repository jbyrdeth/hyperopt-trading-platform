#!/usr/bin/env python3
"""
Strategy Performance Debug

Deep analysis to understand why strategies are performing poorly
and identify the root causes for negative returns.
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

class StrategyPerformanceDebugger:
    """Debug strategy performance issues."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine(initial_capital=100000)
        
    def create_profitable_test_data(self, days: int = 30) -> pd.DataFrame:
        """Create test data specifically designed to be profitable for most strategies."""
        np.random.seed(42)
        n_points = days * 24
        
        # Create a strong trending market with periodic pullbacks
        # This should be profitable for trend-following strategies
        
        # Base trend: 2% daily growth (should be very profitable)
        daily_trend = 0.02 / 24  # Convert to hourly
        
        # Add some noise and pullbacks
        trends = []
        for i in range(n_points):
            if i % 200 == 0:  # Pullback every ~8 days
                trend = -daily_trend * 2  # 2x pullback
            elif i % 50 == 0:  # Small correction every ~2 days
                trend = -daily_trend * 0.5
            else:
                trend = daily_trend
            trends.append(trend)
        
        # Add volatility
        volatility = 0.01
        price_changes = []
        for i, trend in enumerate(trends):
            change = trend + np.random.normal(0, volatility)
            price_changes.append(change)
        
        # Convert to cumulative prices
        cumulative_changes = np.cumsum(price_changes)
        base_price = 50000
        prices = base_price * np.exp(cumulative_changes)
        
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
        
    def debug_strategy_signals(self, strategy_name: str, data: pd.DataFrame) -> dict:
        """Debug why a strategy might not be generating signals."""
        print(f"\n🔍 DEBUGGING {strategy_name} SIGNALS")
        print("=" * 50)
        
        # Test with very aggressive parameters to force signals
        aggressive_params = {
            'position_size_pct': 0.20,  # 20% position size
            'stop_loss_pct': 0.10,      # Wide stops
            'take_profit_pct': 0.15     # Wide targets
        }
        
        # Strategy-specific aggressive parameters
        if strategy_name == "MovingAverageCrossover":
            aggressive_params.update({
                'fast_period': 3,
                'slow_period': 8,
                'ma_type': 'EMA',
                'signal_threshold': 0.001  # Very sensitive
            })
        elif strategy_name == "MACD":
            aggressive_params.update({
                'fast_period': 8,
                'slow_period': 17,
                'signal_period': 6
            })
        elif strategy_name == "RSI":
            aggressive_params.update({
                'period': 10,
                'overbought': 65,  # More sensitive
                'oversold': 35
            })
        elif strategy_name == "BollingerBands":
            aggressive_params.update({
                'period': 15,
                'std_dev': 1.5  # Tighter bands = more signals
            })
        elif strategy_name == "Momentum":
            aggressive_params.update({
                'period': 5,
                'threshold': 0.005  # More sensitive
            })
        
        try:
            strategy = self.factory.create_strategy(strategy_name, **aggressive_params)
            result = self.engine.backtest_strategy(strategy, data)
            
            market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            
            analysis = {
                'strategy': strategy_name,
                'market_return': market_return,
                'total_trades': result.total_trades,
                'total_return': result.total_return * 100,
                'annual_return': result.annual_return * 100,
                'win_rate': result.win_rate * 100,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown * 100,
                'params_used': aggressive_params,
                'success': True
            }
            
            print(f"📊 Market Return: {market_return:.1f}%")
            print(f"🔢 Total Trades: {result.total_trades}")
            print(f"💰 Strategy Return: {result.total_return*100:.1f}%")
            print(f"📈 Annual Return: {result.annual_return*100:.1f}%")
            print(f"🎯 Win Rate: {result.win_rate*100:.1f}%")
            
            if result.total_trades == 0:
                print("⚠️ NO TRADES GENERATED - Strategy parameters too restrictive")
            elif result.total_return < 0:
                print("⚠️ NEGATIVE RETURNS - Strategy logic or parameters need tuning")
            else:
                print("✅ Strategy generating profitable trades")
                
            return analysis
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return {
                'strategy': strategy_name,
                'success': False,
                'error': str(e),
                'total_trades': 0,
                'total_return': -100
            }
            
    def test_parameter_sensitivity(self, strategy_name: str, data: pd.DataFrame) -> list:
        """Test how parameter changes affect strategy performance."""
        print(f"\n🎛️ PARAMETER SENSITIVITY TEST: {strategy_name}")
        print("=" * 50)
        
        results = []
        
        if strategy_name == "MovingAverageCrossover":
            # Test different MA periods
            test_configs = [
                {'fast_period': 3, 'slow_period': 8, 'position_size_pct': 0.25},
                {'fast_period': 5, 'slow_period': 15, 'position_size_pct': 0.25},
                {'fast_period': 8, 'slow_period': 21, 'position_size_pct': 0.25},
                {'fast_period': 3, 'slow_period': 12, 'position_size_pct': 0.30},
                {'fast_period': 4, 'slow_period': 10, 'position_size_pct': 0.30}
            ]
            
            for config in test_configs:
                config.update({
                    'ma_type': 'EMA',
                    'signal_threshold': 0.001,
                    'stop_loss_pct': 0.08,
                    'take_profit_pct': 0.12
                })
                
                try:
                    strategy = self.factory.create_strategy(strategy_name, **config)
                    result = self.engine.backtest_strategy(strategy, data)
                    
                    results.append({
                        'config': f"Fast:{config['fast_period']}, Slow:{config['slow_period']}, Size:{config['position_size_pct']}",
                        'trades': result.total_trades,
                        'return': result.total_return * 100,
                        'annual': result.annual_return * 100,
                        'win_rate': result.win_rate * 100,
                        'sharpe': result.sharpe_ratio
                    })
                    
                    print(f"   {config['fast_period']}/{config['slow_period']} @ {config['position_size_pct']*100:.0f}%: {result.total_trades} trades, {result.annual_return*100:.1f}% annual")
                    
                except Exception as e:
                    print(f"   {config}: ERROR - {str(e)[:50]}")
                    
        return results
        
    def analyze_transaction_costs(self, strategy_name: str, data: pd.DataFrame) -> dict:
        """Analyze if transaction costs are eating into profits."""
        print(f"\n💸 TRANSACTION COST ANALYSIS: {strategy_name}")
        print("=" * 50)
        
        # Test with different commission levels
        original_commission = self.engine.commission
        
        results = {}
        
        for commission in [0.0, 0.001, 0.002, 0.005]:  # 0%, 0.1%, 0.2%, 0.5%
            self.engine.commission = commission
            
            # Use aggressive parameters
            params = {
                'fast_period': 3,
                'slow_period': 8,
                'ma_type': 'EMA',
                'signal_threshold': 0.001,
                'position_size_pct': 0.25,
                'stop_loss_pct': 0.08,
                'take_profit_pct': 0.12
            }
            
            try:
                strategy = self.factory.create_strategy(strategy_name, **params)
                result = self.engine.backtest_strategy(strategy, data)
                
                results[commission] = {
                    'trades': result.total_trades,
                    'annual_return': result.annual_return * 100,
                    'total_costs': self.engine.total_transaction_costs
                }
                
                print(f"   Commission {commission*100:.1f}%: {result.annual_return*100:.1f}% annual, ${self.engine.total_transaction_costs:.2f} costs")
                
            except Exception as e:
                print(f"   Commission {commission*100:.1f}%: ERROR - {str(e)[:50]}")
                
        # Restore original commission
        self.engine.commission = original_commission
        
        return results
        
    def run_comprehensive_debug(self):
        """Run comprehensive debugging analysis."""
        print("🚨 STRATEGY PERFORMANCE DEBUGGING")
        print("=" * 60)
        print("🎯 Goal: Understand why strategies have negative returns")
        
        # Create profitable test data
        print("\n📊 Creating profitable test data...")
        data = self.create_profitable_test_data(days=30)
        market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        print(f"   Market return: {market_return:.1f}%")
        print(f"   Start price: ${data['close'].iloc[0]:,.0f}")
        print(f"   End price: ${data['close'].iloc[-1]:,.0f}")
        
        # Test key strategies
        key_strategies = [
            "MovingAverageCrossover",
            "MACD", 
            "RSI",
            "BollingerBands",
            "Momentum"
        ]
        
        debug_results = []
        
        for strategy_name in key_strategies:
            # Debug signals
            analysis = self.debug_strategy_signals(strategy_name, data)
            debug_results.append(analysis)
            
            # Test parameter sensitivity for MovingAverageCrossover
            if strategy_name == "MovingAverageCrossover" and analysis.get('success'):
                param_results = self.test_parameter_sensitivity(strategy_name, data)
                
                # Analyze transaction costs
                cost_analysis = self.analyze_transaction_costs(strategy_name, data)
        
        # Summary
        print(f"\n📋 DEBUG SUMMARY")
        print("=" * 30)
        
        successful_strategies = [r for r in debug_results if r.get('success', False)]
        profitable_strategies = [r for r in successful_strategies if r.get('annual_return', -100) > 0]
        
        print(f"✅ Successful strategies: {len(successful_strategies)}/{len(key_strategies)}")
        print(f"💰 Profitable strategies: {len(profitable_strategies)}/{len(successful_strategies)}")
        
        if profitable_strategies:
            print(f"\n🏆 BEST PERFORMERS:")
            sorted_profitable = sorted(profitable_strategies, key=lambda x: x.get('annual_return', -100), reverse=True)
            for strategy in sorted_profitable[:3]:
                print(f"   {strategy['strategy']}: {strategy['annual_return']:.1f}% annual ({strategy['total_trades']} trades)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        no_trade_strategies = [r for r in successful_strategies if r.get('total_trades', 0) == 0]
        if no_trade_strategies:
            print(f"   🔧 Fix signal generation for: {[s['strategy'] for s in no_trade_strategies]}")
            
        negative_strategies = [r for r in successful_strategies if r.get('annual_return', -100) < 0 and r.get('total_trades', 0) > 0]
        if negative_strategies:
            print(f"   📈 Optimize parameters for: {[s['strategy'] for s in negative_strategies]}")
            
        if len(profitable_strategies) > 0:
            print(f"   ✅ Use profitable configs in production")
        else:
            print(f"   ⚠️ CRITICAL: No profitable strategies found - major fixes needed")
            
        return debug_results

def main():
    """Run strategy performance debugging."""
    debugger = StrategyPerformanceDebugger()
    
    print("🔍 STRATEGY PERFORMANCE DEBUGGING")
    print("=" * 50)
    print("🎯 Identifying why strategies show negative returns")
    
    results = debugger.run_comprehensive_debug()
    
    print(f"\n🎉 DEBUGGING COMPLETE!")
    print(f"   📊 Check results above for specific fixes needed")

if __name__ == "__main__":
    main() 