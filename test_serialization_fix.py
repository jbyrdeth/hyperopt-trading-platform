#!/usr/bin/env python3
"""
Test script to verify hyperopt parameter space serialization fix.

This script tests that we can properly serialize and deserialize hyperopt
parameter spaces for use in multiprocessing scenarios.
"""

import sys
import os
import pickle
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from hyperopt import hp
    from src.optimization.hyperopt_optimizer import ParameterSpaceSerializer
    from src.optimization.strategy_factory import StrategyFactory
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_parameter_space_serialization():
    """Test that parameter spaces can be serialized and deserialized."""
    print("\n🧪 Testing parameter space serialization...")
    
    # Create a sample parameter space with hyperopt objects
    original_space = {
        'rsi_period': hp.choice('rsi_period', list(range(5, 51))),
        'oversold': hp.uniform('oversold', 10, 40),
        'overbought': hp.uniform('overbought', 60, 90),
        'exit_signal': hp.choice('exit_signal', ["opposite", "middle", "trailing"]),
        'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
        'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
        'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
    }
    
    serializer = ParameterSpaceSerializer()
    
    # Test 1: Check if original space is not serializable
    print("  📋 Testing original space serializability...")
    is_original_serializable = serializer.is_serializable(original_space)
    print(f"    Original space serializable: {is_original_serializable}")
    
    if is_original_serializable:
        print("    ⚠️  Warning: Original space is already serializable (unexpected)")
    else:
        print("    ✅ Original space is not serializable (expected)")
    
    # Test 2: Serialize the parameter space
    print("  🔄 Serializing parameter space...")
    try:
        serialized_space = serializer.serialize_parameter_space(original_space)
        print("    ✅ Successfully serialized parameter space")
        print(f"    📊 Serialized keys: {list(serialized_space.keys())}")
        
        # Check if serialized space is actually serializable
        is_serialized_serializable = serializer.is_serializable(serialized_space)
        print(f"    Serialized space serializable: {is_serialized_serializable}")
        
        if is_serialized_serializable:
            print("    ✅ Serialized space is now serializable")
        else:
            print("    ❌ Serialized space is still not serializable")
            return False
            
    except Exception as e:
        print(f"    ❌ Failed to serialize: {e}")
        return False
    
    # Test 3: Deserialize the parameter space
    print("  🔄 Deserializing parameter space...")
    try:
        deserialized_space = serializer.deserialize_parameter_space(serialized_space)
        print("    ✅ Successfully deserialized parameter space")
        print(f"    📊 Deserialized keys: {list(deserialized_space.keys())}")
        
        # Check if deserialized space has the same structure
        if set(original_space.keys()) == set(deserialized_space.keys()):
            print("    ✅ Deserialized space has same keys as original")
        else:
            print("    ❌ Deserialized space has different keys")
            return False
            
    except Exception as e:
        print(f"    ❌ Failed to deserialize: {e}")
        return False
    
    # Test 4: Test pickle serialization of serialized space
    print("  🥒 Testing pickle serialization...")
    try:
        pickled_data = pickle.dumps(serialized_space)
        unpickled_data = pickle.loads(pickled_data)
        print("    ✅ Successfully pickled and unpickled serialized space")
        
        if serialized_space == unpickled_data:
            print("    ✅ Pickled data matches original")
        else:
            print("    ❌ Pickled data doesn't match original")
            return False
            
    except Exception as e:
        print(f"    ❌ Failed to pickle: {e}")
        return False
    
    return True


def test_strategy_factory_integration():
    """Test integration with strategy factory parameter spaces."""
    print("\n🏭 Testing strategy factory integration...")
    
    try:
        factory = StrategyFactory()
        serializer = ParameterSpaceSerializer()
        
        # Test with RSI strategy
        print("  📊 Testing RSI strategy parameter space...")
        rsi_space = factory.get_parameter_space("RSI")
        print(f"    Original RSI space keys: {list(rsi_space.keys())}")
        
        # Check serializability
        is_serializable = serializer.is_serializable(rsi_space)
        print(f"    RSI space serializable: {is_serializable}")
        
        if not is_serializable:
            # Serialize and deserialize
            serialized = serializer.serialize_parameter_space(rsi_space)
            deserialized = serializer.deserialize_parameter_space(serialized)
            
            print(f"    Serialized RSI space keys: {list(serialized.keys())}")
            print(f"    Deserialized RSI space keys: {list(deserialized.keys())}")
            
            if set(rsi_space.keys()) == set(deserialized.keys()):
                print("    ✅ RSI strategy serialization successful")
            else:
                print("    ❌ RSI strategy serialization failed")
                return False
        
        # Test with MovingAverageCrossover strategy
        print("  📊 Testing MovingAverageCrossover strategy parameter space...")
        ma_space = factory.get_parameter_space("MovingAverageCrossover")
        print(f"    Original MA space keys: {list(ma_space.keys())}")
        
        # Check serializability
        is_serializable = serializer.is_serializable(ma_space)
        print(f"    MA space serializable: {is_serializable}")
        
        if not is_serializable:
            # Serialize and deserialize
            serialized = serializer.serialize_parameter_space(ma_space)
            deserialized = serializer.deserialize_parameter_space(serialized)
            
            print(f"    Serialized MA space keys: {list(serialized.keys())}")
            print(f"    Deserialized MA space keys: {list(deserialized.keys())}")
            
            if set(ma_space.keys()) == set(deserialized.keys()):
                print("    ✅ MovingAverageCrossover strategy serialization successful")
            else:
                print("    ❌ MovingAverageCrossover strategy serialization failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Strategy factory integration test failed: {e}")
        return False


def main():
    """Run all serialization tests."""
    print("🚀 Starting hyperopt parameter space serialization tests...")
    
    # Run tests
    test1_passed = test_parameter_space_serialization()
    test2_passed = test_strategy_factory_integration()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"  Parameter space serialization: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Strategy factory integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Serialization fix is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 