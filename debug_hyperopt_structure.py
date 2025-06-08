#!/usr/bin/env python3
"""
Debug script to examine hyperopt object structure.
"""

import sys
import os
import pickle

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hyperopt import hp

def examine_hyperopt_object(name, obj):
    """Examine the structure of a hyperopt object."""
    print(f"\n🔍 Examining {name}:")
    print(f"  Type: {type(obj)}")
    print(f"  Dir: {[attr for attr in dir(obj) if not attr.startswith('_')]}")
    
    if hasattr(obj, 'name'):
        print(f"  Name: {obj.name}")
    if hasattr(obj, 'pos_args'):
        print(f"  Pos args: {obj.pos_args}")
    if hasattr(obj, 'kwargs'):
        print(f"  Kwargs: {obj.kwargs}")
    
    # Try to pickle it
    try:
        pickled = pickle.dumps(obj)
        print(f"  ✅ Pickleable: Yes ({len(pickled)} bytes)")
    except Exception as e:
        print(f"  ❌ Pickleable: No - {e}")

def main():
    print("🔍 Examining hyperopt object structures...")
    
    # Create various hyperopt objects
    choice_obj = hp.choice('test_choice', [1, 2, 3])
    uniform_obj = hp.uniform('test_uniform', 0, 1)
    randint_obj = hp.randint('test_randint', 10)
    
    examine_hyperopt_object("hp.choice", choice_obj)
    examine_hyperopt_object("hp.uniform", uniform_obj)
    examine_hyperopt_object("hp.randint", randint_obj)
    
    # Test a complete parameter space
    param_space = {
        'choice_param': choice_obj,
        'uniform_param': uniform_obj,
        'randint_param': randint_obj
    }
    
    print(f"\n🧪 Testing complete parameter space:")
    try:
        pickled = pickle.dumps(param_space)
        print(f"  ✅ Complete space pickleable: Yes ({len(pickled)} bytes)")
    except Exception as e:
        print(f"  ❌ Complete space pickleable: No - {e}")

if __name__ == "__main__":
    main() 