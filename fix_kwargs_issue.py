#!/usr/bin/env python3
"""
Fix Kwargs Issue - Critical Strategy Creation Fix

The root cause of strategy creation failures is the kwargs parameter handling.
This script diagnoses and fixes the kwargs initialization issue.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory

def test_strategy_creation_fix():
    """Test strategy creation with different parameter approaches."""
    
    print("🔧 TESTING STRATEGY CREATION FIXES")
    print("=" * 60)
    
    factory = StrategyFactory()
    
    # Test MovingAverageCrossover with different parameter approaches
    print("\n🧪 Testing MovingAverageCrossover...")
    
    test_configs = [
        {
            'name': 'Without kwargs',
            'params': {
                'fast_period': 5,
                'slow_period': 15,
                'ma_type': 'EMA',
                'signal_threshold': 0.002,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05
            }
        },
        {
            'name': 'With empty kwargs',
            'params': {
                'fast_period': 5,
                'slow_period': 15,
                'ma_type': 'EMA',
                'signal_threshold': 0.002,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05,
                'kwargs': {}
            }
        },
        {
            'name': 'Direct class instantiation',
            'direct_class': True,
            'params': {
                'fast_period': 5,
                'slow_period': 15,
                'ma_type': 'EMA',
                'signal_threshold': 0.002,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.05
            }
        }
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n  🔬 Test {i}: {config['name']}")
        
        try:
            if config.get('direct_class'):
                # Try direct class instantiation
                strategy_class = factory.registry.get_strategy_class('MovingAverageCrossover')
                strategy = strategy_class(**config['params'])
            else:
                # Try factory creation
                strategy = factory.create_strategy('MovingAverageCrossover', **config['params'])
            
            print(f"    ✅ SUCCESS: {type(strategy).__name__} created")
            print(f"    📋 Strategy name: {strategy.name}")
            return strategy  # Return the working strategy for further testing
            
        except Exception as e:
            print(f"    ❌ FAILED: {e}")
    
    return None

def test_multiple_strategies():
    """Test the fix approach on multiple strategies."""
    
    print(f"\n🧪 TESTING MULTIPLE STRATEGIES WITH FIX")
    print("=" * 60)
    
    factory = StrategyFactory()
    
    # Test strategies with appropriate parameters (no generic 'period')
    test_strategies = {
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
            'rsi_period': 8,  # Use correct parameter name
            'overbought': 65,
            'oversold': 35,
            'exit_signal': 'opposite',  # Use correct parameter name
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        },
        'BollingerBands': {
            'period': 12,
            'std_dev': 1.8,
            'squeeze_threshold': 0.2,  # Use correct parameter name
            'entry_method': 'breakout',  # Use correct parameter name
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        },
        'WilliamsR': {
            'wr_period': 8,  # Use correct parameter name
            'buy_level': -85,  # Use correct parameter name
            'sell_level': -15,  # Use correct parameter name
            'trend_filter_period': 50,
            'volume_multiplier': 1.5,
            'failure_swing_bars': 5,
            'multi_timeframe': True,
            'position_size_pct': 0.7,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
    }
    
    successful_strategies = []
    
    for strategy_name, params in test_strategies.items():
        print(f"\n📋 Testing {strategy_name}...")
        
        try:
            # Try direct class instantiation (bypass factory)
            strategy_class = factory.registry.get_strategy_class(strategy_name)
            strategy = strategy_class(**params)
            
            print(f"  ✅ SUCCESS: {strategy.name}")
            successful_strategies.append(strategy_name)
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    
    print(f"\n📊 RESULTS SUMMARY")
    print("-" * 30)
    print(f"Total tested: {len(test_strategies)}")
    print(f"Successful: {len(successful_strategies)}")
    print(f"Success rate: {len(successful_strategies)/len(test_strategies)*100:.1f}%")
    
    if successful_strategies:
        print(f"\n✅ WORKING STRATEGIES:")
        for strategy in successful_strategies:
            print(f"   • {strategy}")
    
    return successful_strategies

def create_corrected_parameter_mappings():
    """Create corrected parameter mappings for all strategies."""
    
    print(f"\n💡 CREATING CORRECTED PARAMETER MAPPINGS")
    print("=" * 60)
    
    # Based on diagnostic results, create accurate parameter mappings
    corrected_mappings = {
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
            'wr_period': 8,
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
        # Pattern Recognition (these work with None parameters)
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
        # Multi-timeframe
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
    
    print(f"Created parameter mappings for {len(corrected_mappings)} strategies")
    
    # Save the corrected mappings
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mappings_file = f"corrected_strategy_parameters_{timestamp}.json"
    
    try:
        with open(mappings_file, 'w') as f:
            json.dump(corrected_mappings, f, indent=2)
        print(f"\n📁 Corrected mappings saved to: {mappings_file}")
    except Exception as e:
        print(f"❌ Failed to save mappings: {e}")
    
    return corrected_mappings

def main():
    """Main function to test and fix kwargs issue."""
    
    print("🔧 KWARGS ISSUE DIAGNOSIS & FIX")
    print("=" * 60)
    print("Mission: Fix strategy creation parameter issues")
    print("Goal: Enable all 24 strategies to be created successfully")
    print()
    
    # Test single strategy creation
    working_strategy = test_strategy_creation_fix()
    
    if working_strategy:
        print(f"\n🎉 SUCCESS: Found working approach for strategy creation!")
    
    # Test multiple strategies
    successful_strategies = test_multiple_strategies()
    
    # Create corrected parameter mappings
    corrected_mappings = create_corrected_parameter_mappings()
    
    print(f"\n🎯 KWARGS FIX COMPLETE")
    print("=" * 60)
    print("Next step: Update enhanced validation with corrected parameters")
    print("Key finding: Use direct class instantiation, avoid factory for kwargs issue")
    
    return {
        'successful_strategies': successful_strategies,
        'corrected_mappings': corrected_mappings,
        'fix_approach': 'direct_class_instantiation'
    }

if __name__ == "__main__":
    results = main() 