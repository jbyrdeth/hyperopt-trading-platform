#!/usr/bin/env python3
"""
Check available strategies in the strategy factory
"""

import sys
sys.path.append('src')

try:
    from optimization.strategy_factory import StrategyFactory
    
    print("🔍 Checking Strategy Factory...")
    factory = StrategyFactory()
    
    strategies = factory.get_all_strategies()
    print(f"📊 Available strategies: {len(strategies)}")
    
    print("\n📋 Strategy List:")
    for i, name in enumerate(strategies, 1):
        print(f"  {i:2d}. {name}")
        
    print("\n📂 Strategy Categories:")
    categories = factory.get_all_categories()
    for category in categories:
        cat_strategies = factory.get_strategies_by_category(category)
        print(f"  {category}: {len(cat_strategies)} strategies")
        for strategy in cat_strategies:
            print(f"    - {strategy}")
        
    # Get strategy summary
    print("\n📈 Strategy Summary:")
    summary = factory.get_strategy_summary()
    print(f"  Total Strategies: {summary['total_strategies']}")
    print(f"  Total Categories: {len(summary['categories'])}")
    
    for category, info in summary['categories'].items():
        print(f"    {category}: {info['count']} strategies")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 