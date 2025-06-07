#!/usr/bin/env python3
"""
Enhanced Strategy Validation - FIXED VERSION

Using the successful direct class instantiation approach and corrected
parameter mappings to validate all 24 strategies with proper signal generation.

This script resolves the kwargs issue and parameter compatibility problems.
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

class EnhancedStrategyValidator:
    """Enhanced strategy validator using fixed parameter mappings."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine()
        
        # Load corrected parameter mappings (based on our successful fix)
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
                'wr_period': 12,  # Fixed: adjusted to valid range
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
                'medium_period': 14,
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
            # Pattern Recognition strategies (these work with None parameters)
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
            # Multi-timeframe strategies
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
    
    def create_test_data(self, days: int = 60) -> pd.DataFrame:
        """Create comprehensive test data with various market patterns."""
        
        dates = pd.date_range(start='2023-01-01', periods=days*6, freq='4h')  # 4-hour intervals
        np.random.seed(42)
        
        # Create realistic crypto-like price data
        base_price = 30000  # Starting price like BTC
        trend_component = np.linspace(0, 0.15, len(dates))  # 15% uptrend
        volatility = 0.03
        
        returns = np.random.normal(0, volatility, len(dates))
        returns += trend_component * volatility * 0.5  # Add trend
        
        # Add some volatility clustering
        for i in range(1, len(returns)):
            returns[i] += returns[i-1] * 0.1
        
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate realistic OHLCV data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices))))
        volumes = np.random.lognormal(15, 0.5, len(prices))  # Realistic volumes
        
        data = pd.DataFrame({
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes,
            'timestamp': dates
        })
        
        # Ensure OHLC logic is correct
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return data
    
    def create_strategy_direct(self, strategy_name: str) -> Any:
        """Create strategy using direct class instantiation (avoids kwargs issue)."""
        
        try:
            # Get strategy class
            strategy_class = self.factory.registry.get_strategy_class(strategy_name)
            
            # Get parameters for this strategy
            params = self.strategy_params.get(strategy_name, {})
            
            # Create strategy directly
            strategy = strategy_class(**params)
            
            return strategy
            
        except Exception as e:
            print(f"  ❌ Failed to create {strategy_name}: {e}")
            return None
    
    def validate_strategy(self, strategy_name: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate a single strategy with enhanced metrics."""
        
        try:
            # Create strategy using direct instantiation
            strategy = self.create_strategy_direct(strategy_name)
            
            if strategy is None:
                return {
                    'strategy_name': strategy_name,
                    'status': 'failed',
                    'error': 'Strategy creation failed',
                    'trades': 0,
                    'return': 0.0
                }
            
            # Run backtest
            result = self.engine.backtest_strategy(strategy, data)
            
            # Enhanced metrics
            metrics = {
                'strategy_name': strategy_name,
                'status': 'success',
                'strategy_display_name': strategy.name,
                'total_trades': result.total_trades,
                'total_return': result.total_return,
                'win_rate': result.win_rate if hasattr(result, 'win_rate') else 0.0,
                'max_drawdown': result.max_drawdown if hasattr(result, 'max_drawdown') else 0.0,
                'sharpe_ratio': result.sharpe_ratio if hasattr(result, 'sharpe_ratio') else 0.0,
                'avg_trade_return': result.avg_trade_return if hasattr(result, 'avg_trade_return') else 0.0,
                'data_points': len(data),
                'period_days': len(data) / 6,  # 4-hour intervals, 6 per day
                'annual_return': result.total_return * (365 / (len(data) / 6)) if len(data) > 0 else 0.0,
                'trades_per_month': result.total_trades * (30 / (len(data) / 6)) if len(data) > 0 else 0.0
            }
            
            return metrics
            
        except Exception as e:
            return {
                'strategy_name': strategy_name,
                'status': 'failed',
                'error': str(e),
                'trades': 0,
                'return': 0.0
            }
    
    def validate_all_strategies(self) -> Dict[str, Any]:
        """Validate all 24 strategies with comprehensive analysis."""
        
        print("🚀 ENHANCED STRATEGY VALIDATION - FIXED VERSION")
        print("=" * 60)
        print("Mission: Validate all 24 strategies with corrected parameters")
        print("Approach: Direct class instantiation (bypasses kwargs issue)")
        print()
        
        # Create test data
        print("📊 Creating comprehensive test data...")
        data = self.create_test_data(days=60)  # 2 months of 4-hour data
        market_return = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
        print(f"   📈 Market return: {market_return:.2f}% (Buy & Hold benchmark)")
        print(f"   📅 Period: {len(data)} data points ({len(data)/6:.1f} days)")
        print()
        
        # Test all strategies
        strategies = list(self.strategy_params.keys())
        results = []
        successful_strategies = []
        failed_strategies = []
        
        for i, strategy_name in enumerate(strategies, 1):
            print(f"🧪 [{i:2d}/24] Testing {strategy_name}...")
            
            start_time = time.time()
            metrics = self.validate_strategy(strategy_name, data)
            duration = time.time() - start_time
            
            if metrics['status'] == 'success':
                print(f"  ✅ SUCCESS: {metrics['strategy_display_name']}")
                print(f"     📊 Trades: {metrics['total_trades']}, Return: {metrics['total_return']:.2f}%")
                print(f"     📈 Annual: {metrics['annual_return']:.1f}%, Trades/month: {metrics['trades_per_month']:.1f}")
                print(f"     ⏱️  Duration: {duration:.2f}s")
                successful_strategies.append(strategy_name)
            else:
                print(f"  ❌ FAILED: {metrics['error']}")
                failed_strategies.append(strategy_name)
            
            results.append(metrics)
            print()
        
        # Analysis and summary
        print("📊 COMPREHENSIVE VALIDATION RESULTS")
        print("=" * 60)
        
        success_rate = len(successful_strategies) / len(strategies) * 100
        print(f"Success Rate: {len(successful_strategies)}/24 ({success_rate:.1f}%)")
        print(f"Market Benchmark: {market_return:.2f}%")
        print()
        
        if successful_strategies:
            successful_results = [r for r in results if r['status'] == 'success' and r['total_trades'] > 0]
            
            if successful_results:
                # Sort by annual return
                successful_results.sort(key=lambda x: x.get('annual_return', 0), reverse=True)
                
                print("🏆 TOP PERFORMING STRATEGIES:")
                print("-" * 40)
                for i, result in enumerate(successful_results[:10], 1):
                    print(f"{i:2d}. {result['strategy_display_name']}")
                    print(f"    📊 {result['total_trades']} trades, {result['total_return']:.2f}% return")
                    print(f"    📈 {result['annual_return']:.1f}% annual, {result['trades_per_month']:.1f} trades/month")
                
                print()
                print("📈 PERFORMANCE STATISTICS:")
                print("-" * 30)
                returns = [r['annual_return'] for r in successful_results]
                trades = [r['total_trades'] for r in successful_results]
                
                print(f"Average Annual Return: {np.mean(returns):.1f}%")
                print(f"Median Annual Return: {np.median(returns):.1f}%")
                print(f"Best Annual Return: {np.max(returns):.1f}%")
                print(f"Average Trades: {np.mean(trades):.1f}")
                print(f"Median Trades: {np.median(trades):.1f}")
                print(f"Most Active: {np.max(trades)} trades")
                
                # Identify strategies beating market
                beating_market = [r for r in successful_results if r['annual_return'] > market_return]
                print(f"\n🎯 Strategies Beating Market: {len(beating_market)}/{len(successful_results)}")
                
        if failed_strategies:
            print(f"\n❌ FAILED STRATEGIES ({len(failed_strategies)}):")
            for strategy in failed_strategies:
                error = next(r['error'] for r in results if r['strategy_name'] == strategy and r['status'] == 'failed')
                print(f"   • {strategy}: {error}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"enhanced_validation_results_{timestamp}.json"
        
        output_data = {
            'validation_timestamp': timestamp,
            'test_period_days': len(data) / 6,
            'market_return_pct': market_return,
            'total_strategies': len(strategies),
            'successful_strategies': len(successful_strategies),
            'success_rate_pct': success_rate,
            'results': results,
            'top_performers': successful_results[:10] if 'successful_results' in locals() else []
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\n📁 Detailed results saved to: {results_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
        
        return output_data

def main():
    """Main validation function."""
    
    print("🚀 ENHANCED STRATEGY VALIDATION - FIXED VERSION")
    print("=" * 60)
    print("Using corrected parameter mappings and direct class instantiation")
    print("to resolve kwargs issue and validate all 24 trading strategies.")
    print()
    
    validator = EnhancedStrategyValidator()
    results = validator.validate_all_strategies()
    
    print(f"\n🎯 VALIDATION COMPLETE")
    print("=" * 60)
    print("All strategies tested with fixed parameter compatibility")
    print("Next step: Portfolio optimization with validated strategies")
    
    return results

if __name__ == "__main__":
    results = main() 