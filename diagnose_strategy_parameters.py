#!/usr/bin/env python3
"""
Strategy Parameter Diagnostics

Diagnose the exact parameter requirements for each strategy to fix
the parameter compatibility issues blocking validation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import inspect
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory

class StrategyParameterDiagnostic:
    """Diagnose strategy parameter requirements."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        
    def get_strategy_signature(self, strategy_name: str) -> Dict[str, Any]:
        """Get the parameter signature for a strategy."""
        
        try:
            strategy_class = self.factory.registry.get_strategy_class(strategy_name)
            init_signature = inspect.signature(strategy_class.__init__)
            
            parameters = {}
            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue
                
                param_info = {
                    'name': param_name,
                    'annotation': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    'default': param.default if param.default != inspect.Parameter.empty else 'REQUIRED',
                    'kind': str(param.kind)
                }
                parameters[param_name] = param_info
            
            return {
                'strategy_name': strategy_name,
                'class_name': strategy_class.__name__,
                'parameters': parameters,
                'total_params': len(parameters),
                'required_params': len([p for p in parameters.values() if p['default'] == 'REQUIRED']),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy_name,
                'status': 'failed',
                'error': str(e)
            }
    
    def get_compatible_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get compatible parameters for a strategy based on its signature."""
        
        signature = self.get_strategy_signature(strategy_name)
        
        if signature['status'] == 'failed':
            return {'error': signature['error']}
        
        params = signature['parameters']
        compatible_params = {}
        
        # Map common parameter patterns
        param_mappings = {
            # Trend Following
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
            
            # Mean Reversion
            'RSI': {
                'period': 8,
                'overbought': 65,
                'oversold': 35,
                'exit_strategy': 'opposite',
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'WilliamsR': {
                'period': 8,
                'overbought': -15,
                'oversold': -85,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'Stochastic': {
                'k_period': 8,
                'd_period': 3,
                'overbought': 75,
                'oversold': 25,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            
            # Volatility
            'BollingerBands': {
                'period': 12,
                'std_dev': 1.8,
                'strategy_type': 'breakout',
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'ATR': {
                'period': 10,
                'atr_multiplier': 2.0,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'KeltnerChannel': {
                'period': 10,
                'atr_period': 10,
                'multiplier': 2.0,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'BollingerSqueeze': {
                'bb_period': 20,
                'bb_std': 2.0,
                'kc_period': 20,
                'kc_multiplier': 1.5,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'HistoricalVolatility': {
                'period': 20,
                'threshold': 0.02,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            
            # Momentum
            'Momentum': {
                'period': 8,
                'threshold': 0.01,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'ROC': {
                'period': 8,
                'threshold': 1.0,
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
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            
            # Volume
            'VWAP': {
                'period': 20,
                'deviation_threshold': 0.5,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'OBV': {
                'ma_period': 20,
                'signal_threshold': 0.01,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'AD': {
                'ma_period': 20,
                'signal_threshold': 0.01,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'CMF': {
                'period': 20,
                'threshold': 0.1,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            
            # Pattern Recognition - These need specific parameters
            'SupportResistance': {
                'window': 20,
                'min_touches': 2,
                'tolerance': 0.01,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'PivotPoints': {
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'FibonacciRetracement': {
                'window': 50,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'DoubleTopBottom': {
                'window': 20,
                'min_distance': 10,
                'tolerance': 0.02,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            
            # Multi-timeframe
            'MTFTrendAnalysis': {
                'short_window': 10,
                'long_window': 20,
                'timeframes': ['1h', '4h'],
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'MTFRSI': {
                'rsi_period': 14,
                'timeframes': ['1h', '4h'],
                'overbought': 70,
                'oversold': 30,
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            },
            'MTFMACD': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9,
                'timeframes': ['1h', '4h'],
                'position_size_pct': 0.7,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06
            }
        }
        
        # Get strategy-specific parameters
        strategy_params = param_mappings.get(strategy_name, {})
        
        # Filter parameters that exist in the strategy signature
        for param_name, param_value in strategy_params.items():
            if param_name in params:
                compatible_params[param_name] = param_value
        
        # Add any missing required parameters with defaults
        for param_name, param_info in params.items():
            if param_info['default'] == 'REQUIRED' and param_name not in compatible_params:
                # Provide sensible defaults for required parameters
                if 'period' in param_name.lower():
                    compatible_params[param_name] = 14
                elif 'threshold' in param_name.lower():
                    compatible_params[param_name] = 0.01
                elif 'window' in param_name.lower():
                    compatible_params[param_name] = 20
                elif param_name in ['overbought', 'oversold']:
                    compatible_params[param_name] = 70 if param_name == 'overbought' else 30
                else:
                    # Try to infer from annotation
                    annotation = param_info.get('annotation', '')
                    if 'float' in annotation.lower():
                        compatible_params[param_name] = 0.5
                    elif 'int' in annotation.lower():
                        compatible_params[param_name] = 10
                    elif 'str' in annotation.lower():
                        compatible_params[param_name] = 'default'
                    elif 'bool' in annotation.lower():
                        compatible_params[param_name] = True
                    else:
                        compatible_params[param_name] = None
        
        return compatible_params
    
    def test_strategy_creation(self, strategy_name: str) -> Dict[str, Any]:
        """Test creating a strategy with compatible parameters."""
        
        try:
            compatible_params = self.get_compatible_parameters(strategy_name)
            
            if 'error' in compatible_params:
                return {
                    'strategy_name': strategy_name,
                    'status': 'failed',
                    'error': compatible_params['error']
                }
            
            # Test strategy creation
            strategy = self.factory.create_strategy(strategy_name, **compatible_params)
            
            return {
                'strategy_name': strategy_name,
                'status': 'success',
                'parameters_used': compatible_params,
                'strategy_created': True
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy_name,
                'status': 'failed',
                'error': str(e),
                'parameters_used': compatible_params if 'compatible_params' in locals() else {}
            }
    
    def diagnose_all_strategies(self) -> Dict[str, Any]:
        """Diagnose all strategies and create corrected parameter mappings."""
        
        print("🔍 STRATEGY PARAMETER DIAGNOSTICS")
        print("=" * 60)
        
        strategies = self.factory.get_all_strategies()
        diagnostics = {}
        successful_strategies = []
        failed_strategies = []
        
        for strategy_name in strategies:
            print(f"\n📋 {strategy_name}")
            print("-" * 40)
            
            # Get signature
            signature = self.get_strategy_signature(strategy_name)
            if signature['status'] == 'success':
                params = signature['parameters']
                print(f"  Parameters ({signature['total_params']}):")
                for param_name, param_info in params.items():
                    default_str = f" = {param_info['default']}" if param_info['default'] != 'REQUIRED' else " (REQUIRED)"
                    print(f"    {param_name}: {param_info['annotation']}{default_str}")
            else:
                print(f"  ❌ Error getting signature: {signature['error']}")
                continue
            
            # Test creation
            creation_test = self.test_strategy_creation(strategy_name)
            if creation_test['status'] == 'success':
                print(f"  ✅ Strategy creation: SUCCESS")
                successful_strategies.append(strategy_name)
            else:
                print(f"  ❌ Strategy creation: {creation_test['error']}")
                failed_strategies.append(strategy_name)
            
            diagnostics[strategy_name] = {
                'signature': signature,
                'creation_test': creation_test
            }
        
        print(f"\n📊 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        print(f"Total strategies: {len(strategies)}")
        print(f"Successfully creatable: {len(successful_strategies)} ({len(successful_strategies)/len(strategies)*100:.1f}%)")
        print(f"Failed creation: {len(failed_strategies)}")
        
        if successful_strategies:
            print(f"\n✅ WORKING STRATEGIES:")
            for strategy in successful_strategies:
                print(f"   • {strategy}")
        
        if failed_strategies:
            print(f"\n❌ FAILED STRATEGIES:")
            for strategy in failed_strategies:
                print(f"   • {strategy}")
        
        return {
            'diagnostics': diagnostics,
            'successful_strategies': successful_strategies,
            'failed_strategies': failed_strategies,
            'success_rate': len(successful_strategies) / len(strategies)
        }

def main():
    """Main diagnostic function."""
    
    print("🔍 STRATEGY PARAMETER COMPATIBILITY DIAGNOSTICS")
    print("=" * 60)
    print("Mission: Fix parameter compatibility issues")
    print("Goal: Enable all strategies to run successfully")
    print()
    
    diagnostic = StrategyParameterDiagnostic()
    results = diagnostic.diagnose_all_strategies()
    
    # Generate corrected parameter mapping
    print(f"\n💡 GENERATING CORRECTED PARAMETER MAPPINGS")
    print("=" * 60)
    
    corrected_mappings = {}
    for strategy_name in results['successful_strategies']:
        creation_test = results['diagnostics'][strategy_name]['creation_test']
        if creation_test['status'] == 'success':
            corrected_mappings[strategy_name] = creation_test['parameters_used']
    
    print(f"Generated mappings for {len(corrected_mappings)} strategies")
    
    # Save results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"strategy_parameter_diagnostics_{timestamp}.json"
    
    output_data = {
        'diagnostics': results,
        'corrected_parameter_mappings': corrected_mappings,
        'timestamp': timestamp
    }
    
    try:
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n📁 Results saved to: {results_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    print(f"\n🎯 DIAGNOSTICS COMPLETE")
    print("=" * 60)
    print("Next step: Update enhanced validation with corrected parameters")
    
    return output_data

if __name__ == "__main__":
    results = main() 