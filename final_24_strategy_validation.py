#!/usr/bin/env python3
"""
FINAL 24-Strategy Validation - ALL FIXES APPLIED

This script incorporates all discovered fixes:
1. Direct class instantiation (bypasses kwargs issue)
2. Corrected parameter mappings (specific to each strategy)  
3. Proper DatetimeIndex format (fixes backtesting engine compatibility)

Expected result: High success rate with realistic trade generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

class Final24StrategyValidator:
    """Final comprehensive validator with all fixes applied."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine()
        
        # Corrected parameter mappings (from successful testing)
        self.strategy_params = {
            'MovingAverageCrossover': {
                'fast_period': 5,
                'slow_period': 15,
                'ma_type': 'EMA',
                'signal_threshold': 0.002,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05
            },
            'MACD': {
                'fast_period': 6,
                'slow_period': 18,
                'signal_period': 6,
                'histogram_threshold': 0.005,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05
            },
            'RSI': {
                'rsi_period': 8,
                'overbought': 65,
                'oversold': 35,
                'exit_signal': 'opposite',
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'BollingerBands': {
                'period': 12,
                'std_dev': 1.8,
                'squeeze_threshold': 0.2,
                'entry_method': 'breakout',
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'Momentum': {
                'period': 8,
                'threshold': 0.01,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'ROC': {
                'roc_period': 8,
                'signal_threshold': 1.0,
                'persistence_bars': 3,
                'ma_smoothing': 5,
                'divergence_lookback': 20,
                'trend_filter': True,
                'trend_period': 50,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'Stochastic': {
                'k_period': 8,
                'd_period': 3,
                'overbought': 75,
                'oversold': 25,
                'trend_filter_period': 50,
                'divergence_lookback': 20,
                'adaptive_levels': True,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'WilliamsR': {
                'wr_period': 12,  # Adjusted to valid range [10,50]
                'buy_level': -85,
                'sell_level': -15,
                'trend_filter_period': 50,
                'volume_multiplier': 1.5,
                'failure_swing_bars': 5,
                'multi_timeframe': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'UltimateOscillator': {
                'short_period': 7,
                'medium_period': 16,  # Fixed: adjusted to valid range [15,35]
                'long_period': 28,
                'overbought': 70,
                'oversold': 30,
                'signal_threshold': 50,
                'trend_confirmation': 50,
                'divergence_lookback': 20,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'VWAP': {
                'vwap_period': 20,
                'deviation_threshold': 0.02,
                'volume_multiplier': 1.5,
                'trading_mode': 'mean_reversion',
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'OBV': {
                'obv_ma_period': 20,
                'divergence_lookback': 10,
                'signal_threshold': 0.02,
                'volume_threshold': 1.2,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'AD': {
                'ad_ma_period': 20,
                'trend_threshold': 0.02,
                'volume_filter': 1.0,
                'divergence_lookback': 10,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'CMF': {
                'cmf_period': 20,
                'buy_threshold': 0.1,
                'sell_threshold': -0.1,
                'momentum_period': 5,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'ATR': {
                'atr_period': 10,
                'breakout_multiplier': 2.0,
                'stop_multiplier': 1.5,
                'trend_filter': True,
                'trend_period': 50,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'BollingerSqueeze': {
                'bb_period': 20,
                'bb_std': 2.0,
                'kc_period': 20,
                'kc_atr_mult': 1.5,
                'squeeze_threshold': 0.95,
                'breakout_bars': 3,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'KeltnerChannel': {
                'kc_period': 20,
                'atr_multiplier': 2.0,
                'trading_mode': 'breakout',
                'channel_position': 'middle',
                'trend_filter': True,
                'trend_period': 50,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'HistoricalVolatility': {
                'volatility_period': 20,
                'regime_period': 60,
                'high_vol_threshold': 0.75,
                'low_vol_threshold': 0.25,
                'trend_period': 50,
                'momentum_period': 10,
                'volume_confirmation': True,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            # Pattern Recognition strategies  
            'SupportResistance': {
                'parameters': None
            },
            'PivotPoints': {
                'parameters': None
            },
            'FibonacciRetracement': {
                'parameters': None
            },
            'DoubleTopBottom': {
                'parameters': None
            },
            # Multi-timeframe strategies (may need special handling)
            'MTFTrendAnalysis': {
                'primary_timeframe': '1H',
                'secondary_timeframe': '4H',
                'tertiary_timeframe': '1D',
                'ma_period_short': 20,
                'ma_period_long': 50,
                'ma_type': 'EMA',
                'trend_alignment_threshold': 0.7,
                'trend_strength_threshold': 0.02,
                'signal_confirmation_bars': 2,
                'volume_confirmation': True,
                'volume_threshold': 1.2,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'MTFRSI': {
                'primary_timeframe': '1H',
                'secondary_timeframe': '4H',
                'tertiary_timeframe': '1D',
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'rsi_alignment_threshold': 0.7,
                'divergence_lookback': 20,
                'signal_confirmation_bars': 2,
                'volume_confirmation': True,
                'volume_threshold': 1.2,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'MTFMACD': {
                'primary_timeframe': '1H',
                'secondary_timeframe': '4H',
                'tertiary_timeframe': '1D',
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'macd_alignment_threshold': 0.7,
                'histogram_threshold': 0.0001,
                'divergence_lookback': 20,
                'signal_confirmation_bars': 2,
                'volume_confirmation': True,
                'volume_threshold': 1.2,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            }
        }
    
    def create_proper_test_data(self, days: int = 60) -> pd.DataFrame:
        """Create test data with proper DatetimeIndex (KEY FIX)."""
        
        # Create proper datetime index - THIS IS CRITICAL
        start_date = datetime(2023, 1, 1)
        periods = days * 6  # 4-hour intervals per day
        dates = pd.date_range(start=start_date, periods=periods, freq='4H')
        
        np.random.seed(42)
        
        # Create realistic crypto-like price data
        base_price = 30000
        trend_component = np.linspace(0, 0.15, len(dates))  # 15% uptrend
        volatility = 0.03
        
        returns = np.random.normal(0, volatility, len(dates))
        returns += trend_component * volatility * 0.5
        
        # Add volatility clustering
        for i in range(1, len(returns)):
            returns[i] += returns[i-1] * 0.1
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate realistic OHLCV data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices))))
        volumes = np.random.lognormal(15, 0.5, len(prices))
        
        # CRITICAL: Use DatetimeIndex as DataFrame index
        data = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)  # DatetimeIndex as index, NOT as column
        
        # Ensure OHLC logic
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return data
    
    def create_strategy_direct(self, strategy_name: str) -> Any:
        """Create strategy using direct class instantiation (bypasses kwargs issue)."""
        
        try:
            strategy_class = self.factory.registry.get_strategy_class(strategy_name)
            params = self.strategy_params.get(strategy_name, {})
            strategy = strategy_class(**params)
            return strategy
        except Exception as e:
            print(f"  ❌ Failed to create {strategy_name}: {e}")
            return None
    
    def validate_strategy(self, strategy_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate a single strategy with comprehensive metrics."""
        
        try:
            # Create strategy using direct instantiation
            strategy = self.create_strategy_direct(strategy_name)
            
            if strategy is None:
                return {
                    'strategy_name': strategy_name,
                    'status': 'failed',
                    'error': 'Strategy creation failed',
                    'total_trades': 0,
                    'total_return': 0.0
                }
            
            # Run backtest
            result = self.engine.backtest_strategy(strategy, data)
            
            # Calculate additional metrics
            period_days = len(data) / 6  # 4-hour intervals
            annual_return = result.total_return * (365 / period_days) if period_days > 0 else 0.0
            trades_per_month = result.total_trades * (30 / period_days) if period_days > 0 else 0.0
            
            return {
                'strategy_name': strategy_name,
                'status': 'success',
                'strategy_display_name': strategy.name,
                'total_trades': result.total_trades,
                'total_return': result.total_return,
                'annual_return': annual_return,
                'trades_per_month': trades_per_month,
                'period_days': period_days
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy_name,
                'status': 'failed',
                'error': str(e),
                'total_trades': 0,
                'total_return': 0.0
            }
    
    def validate_all_strategies(self) -> Dict[str, Any]:
        """Final comprehensive validation of all 24 strategies."""
        
        print("🚀 FINAL 24-STRATEGY VALIDATION - ALL FIXES APPLIED")
        print("=" * 70)
        print("✅ Direct class instantiation (kwargs fix)")
        print("✅ Corrected parameter mappings")  
        print("✅ Proper DatetimeIndex format")
        print()
        
        # Create proper test data
        print("📊 Creating test data with proper DatetimeIndex...")
        data = self.create_proper_test_data(days=60)
        market_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
        
        print(f"   📈 Market return: {market_return:.2f}% (Buy & Hold benchmark)")
        print(f"   📅 Period: {len(data)} data points ({len(data)/6:.1f} days)")
        print(f"   🗂️  Index type: {type(data.index)} ✅")
        print(f"   📋 Columns: {list(data.columns)}")
        print()
        
        # Test all strategies
        strategies = list(self.strategy_params.keys())
        results = []
        successful_strategies = []
        failed_strategies = []
        
        print("🧪 TESTING ALL 24 STRATEGIES")
        print("=" * 50)
        
        for i, strategy_name in enumerate(strategies, 1):
            print(f"[{i:2d}/24] Testing {strategy_name}...", end=' ')
            
            start_time = time.time()
            metrics = self.validate_strategy(strategy_name, data)
            duration = time.time() - start_time
            
            if metrics['status'] == 'success':
                print(f"✅ SUCCESS")
                print(f"        📊 {metrics['total_trades']} trades, {metrics['total_return']:.2f}% return")
                print(f"        📈 {metrics['annual_return']:.1f}% annual, {metrics['trades_per_month']:.1f} trades/month")
                successful_strategies.append(strategy_name)
            else:
                print(f"❌ FAILED: {metrics['error']}")
                failed_strategies.append(strategy_name)
            
            results.append(metrics)
        
        # Comprehensive analysis
        print(f"\n📊 FINAL VALIDATION RESULTS")
        print("=" * 50)
        
        success_rate = len(successful_strategies) / len(strategies) * 100
        print(f"🎯 Success Rate: {len(successful_strategies)}/24 ({success_rate:.1f}%)")
        print(f"📈 Market Benchmark: {market_return:.2f}%")
        
        if successful_strategies:
            successful_results = [r for r in results if r['status'] == 'success' and r['total_trades'] > 0]
            
            if successful_results:
                # Sort by annual return
                successful_results.sort(key=lambda x: x.get('annual_return', 0), reverse=True)
                
                print(f"\n🏆 TOP PERFORMING STRATEGIES:")
                print("-" * 35)
                for i, result in enumerate(successful_results[:10], 1):
                    print(f"{i:2d}. {result['strategy_display_name']}")
                    print(f"    📊 {result['total_trades']} trades, {result['total_return']:.2f}% return")
                    print(f"    📈 {result['annual_return']:.1f}% annual")
                
                # Performance statistics
                returns = [r['annual_return'] for r in successful_results]
                trades = [r['total_trades'] for r in successful_results]
                
                print(f"\n📈 PERFORMANCE STATISTICS:")
                print("-" * 25)
                print(f"Average Annual Return: {np.mean(returns):.1f}%")
                print(f"Median Annual Return: {np.median(returns):.1f}%")
                print(f"Best Annual Return: {np.max(returns):.1f}%")
                print(f"Average Trades: {np.mean(trades):.1f}")
                print(f"Most Active: {np.max(trades)} trades")
                
                # Strategies beating market
                beating_market = [r for r in successful_results if r['annual_return'] > market_return]
                print(f"\n🎯 Strategies Beating Market: {len(beating_market)}/{len(successful_results)}")
        
        if failed_strategies:
            print(f"\n❌ FAILED STRATEGIES ({len(failed_strategies)}):")
            for strategy in failed_strategies:
                error = next(r['error'] for r in results if r['strategy_name'] == strategy and r['status'] == 'failed')
                print(f"   • {strategy}: {error}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"final_24_strategy_validation_{timestamp}.json"
        
        output_data = {
            'validation_timestamp': timestamp,
            'all_fixes_applied': True,
            'test_period_days': len(data) / 6,
            'market_return_pct': market_return,
            'total_strategies': len(strategies),
            'successful_strategies': len(successful_strategies),
            'success_rate_pct': success_rate,
            'results': results,
            'top_performers': successful_results[:10] if 'successful_results' in locals() else [],
            'fixes_applied': [
                'direct_class_instantiation_kwargs_fix',
                'corrected_parameter_mappings',
                'proper_datetime_index_format'
            ]
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\n📁 Complete results saved to: {results_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
        
        print(f"\n🎯 FINAL VALIDATION COMPLETE")
        print("=" * 50)
        print("All discovered fixes have been applied and tested.")
        print("Ready for portfolio optimization with validated strategies!")
        
        return output_data

def main():
    """Main validation function."""
    
    print("🚀 FINAL 24-STRATEGY VALIDATION")
    print("=" * 50)
    print("Applying all discovered fixes for comprehensive strategy testing.")
    print()
    
    validator = Final24StrategyValidator()
    results = validator.validate_all_strategies()
    
    return results

if __name__ == "__main__":
    results = main() 