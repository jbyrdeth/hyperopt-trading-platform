#!/usr/bin/env python3
"""
Quick test of imports using main.py pattern
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

print('✅ Imports working correctly')
factory = StrategyFactory()
strategies = factory.get_all_strategies()
print(f'📋 Found {len(strategies)} strategies')
print(f'First 5: {strategies[:5]}')

# Test creating a strategy
if strategies:
    strategy_name = strategies[0]
    print(f'\n🧪 Testing strategy creation: {strategy_name}')
    
    param_space = factory.get_parameter_space(strategy_name)
    print(f'Parameter space: {param_space}')
    
    # Get default parameters
    parameters = {}
    for param_name, param_config in param_space.items():
        if isinstance(param_config, dict):
            if 'low' in param_config and 'high' in param_config:
                low, high = param_config['low'], param_config['high']
                if isinstance(low, int):
                    parameters[param_name] = (low + high) // 2
                else:
                    parameters[param_name] = (low + high) / 2
            elif 'choices' in param_config:
                parameters[param_name] = param_config['choices'][0]
        else:
            parameters[param_name] = param_config
    
    print(f'Default parameters: {parameters}')
    
    try:
        strategy = factory.create_strategy(strategy_name, parameters)
        print(f'✅ Strategy created successfully: {strategy.name}')
    except Exception as e:
        print(f'❌ Strategy creation failed: {e}') 