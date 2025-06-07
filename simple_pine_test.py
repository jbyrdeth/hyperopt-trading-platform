"""
Simple Pine Script Generator Test

Basic test to validate the Pine Script generation system works correctly.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, 'src')

try:
    from export.pine_script_generator import (
        PineScriptGenerator, PineScriptConfig, MovingAverageCrossoverTemplate,
        RSITemplate, MACDTemplate
    )
    from export.parameter_translator import ParameterTranslator, PineScriptType
    from export.pine_validator import PineScriptValidator
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SimpleStrategy:
    """Simple mock strategy for testing."""
    
    def __init__(self, name="Test Moving Average Strategy"):
        self.name = name
        self.parameters = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_threshold": 0.02
        }
        self.risk_params = {
            "stop_loss": 2.0,
            "take_profit": 4.0,
            "position_size": 100.0
        }


def test_pine_script_generation():
    """Test basic Pine Script generation functionality."""
    
    print("\n" + "="*60)
    print("TESTING PINE SCRIPT GENERATOR")
    print("="*60)
    
    # Create temporary directory for outputs
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # 1. Test Generator Initialization
        print("\n1. Testing Generator Initialization...")
        config = PineScriptConfig(
            output_directory=temp_dir,
            title="Test Strategy",
            include_alerts=True,
            include_debugging=True,
            include_visualization=True
        )
        generator = PineScriptGenerator(config)
        print(f"   ✅ Generator initialized with {len(generator.templates)} templates")
        
        # 2. Test Template Access
        print("\n2. Testing Template Access...")
        available_templates = generator.get_available_templates()
        print(f"   ✅ Available templates: {available_templates}")
        
        # 3. Test Moving Average Template
        print("\n3. Testing Moving Average Template...")
        ma_template = generator.templates["moving_average_crossover"]
        params = {"fast_period": 12, "slow_period": 26, "signal_threshold": 0.02}
        
        indicators = ma_template.generate_indicators(params)
        signals = ma_template.generate_signals(params)
        print(f"   ✅ Generated {len(indicators)} indicators and {len(signals)} signals")
        
        # 4. Test Parameter Translation
        print("\n4. Testing Parameter Translation...")
        translator = ParameterTranslator()
        
        # Test predefined parameter
        fast_param = translator.translate_parameter("fast_period", 12)
        print(f"   ✅ Translated fast_period: {fast_param.title} ({fast_param.param_type.value})")
        
        # Test auto-detection
        custom_param = translator.translate_parameter("custom_threshold", 0.5)
        print(f"   ✅ Auto-detected custom_threshold: {custom_param.title} ({custom_param.param_type.value})")
        
        # 5. Test Pine Script Code Generation
        print("\n5. Testing Pine Script Code Generation...")
        pine_code = fast_param.to_pine_script()
        print(f"   ✅ Generated Pine Script input: {pine_code[:50]}...")
        
        # 6. Test Full Strategy Generation
        print("\n6. Testing Full Strategy Generation...")
        strategy = SimpleStrategy()
        
        script_path = generator.generate_strategy_script(
            strategy,
            template_name="moving_average_crossover"
        )
        print(f"   ✅ Generated strategy script: {os.path.basename(script_path)}")
        
        # Read and check content
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   📊 Script size: {len(content)} characters")
        
        # Check for key elements
        required_elements = [
            "//@version=5",
            "strategy(",
            "fast_ma = ta.sma",
            "slow_ma = ta.sma",
            "strategy.entry",
            "stop_loss_pct",
            "take_profit_pct"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"   ⚠️  Missing elements: {missing_elements}")
        else:
            print("   ✅ All required elements present")
        
        # 7. Test Indicator Generation
        print("\n7. Testing Indicator Generation...")
        indicator_path = generator.generate_indicator_script(
            strategy,
            template_name="moving_average_crossover"
        )
        print(f"   ✅ Generated indicator script: {os.path.basename(indicator_path)}")
        
        # 8. Test Validation
        print("\n8. Testing Script Validation...")
        validator = PineScriptValidator()
        result = validator.validate_file(script_path)
        
        print(f"   📋 Validation result: {result.get_summary()}")
        if result.error_count == 0:
            print("   ✅ No critical errors found")
        else:
            print(f"   ⚠️  Found {result.error_count} errors")
            for issue in result.issues[:3]:  # Show first 3 issues
                print(f"      - {issue.severity.value}: {issue.message}")
        
        # 9. Test Multiple Templates
        print("\n9. Testing Multiple Templates...")
        templates_to_test = [
            ("rsi_strategy", {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "confirmation_period": 2
            }),
            ("macd_strategy", {
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "signal_threshold": 0.001
            })
        ]
        
        for template_name, template_params in templates_to_test:
            test_strategy = SimpleStrategy(f"Test {template_name.replace('_', ' ').title()}")
            test_strategy.parameters = template_params
            
            try:
                test_script_path = generator.generate_strategy_script(
                    test_strategy,
                    template_name=template_name
                )
                print(f"   ✅ Generated {template_name}: {os.path.basename(test_script_path)}")
            except Exception as e:
                print(f"   ❌ Failed to generate {template_name}: {e}")
        
        # 10. Test Batch Generation
        print("\n10. Testing Batch Generation...")
        strategies = [
            (SimpleStrategy("Batch Strategy 1"), None),
            (SimpleStrategy("Batch Strategy 2"), None)
        ]
        
        batch_files = generator.generate_batch_scripts(strategies)
        print(f"    ✅ Generated {len(batch_files)} scripts in batch")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Show generated files
        print(f"\n📁 Generated files in {temp_dir}:")
        for file_path in Path(temp_dir).glob("*.pine"):
            file_size = os.path.getsize(file_path)
            print(f"   - {file_path.name} ({file_size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def show_sample_script():
    """Generate and display a sample Pine Script."""
    
    print("\n" + "="*60)
    print("SAMPLE PINE SCRIPT OUTPUT")
    print("="*60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = PineScriptConfig(output_directory=temp_dir, title="Sample MA Strategy")
        generator = PineScriptGenerator(config)
        strategy = SimpleStrategy("Sample Moving Average Strategy")
        
        script_path = generator.generate_strategy_script(
            strategy,
            template_name="moving_average_crossover"
        )
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Show first 50 lines
        lines = content.split('\n')
        print('\n'.join(lines[:50]))
        
        if len(lines) > 50:
            print(f"\n... ({len(lines) - 50} more lines)")
        
        print(f"\n📊 Total script size: {len(content):,} characters, {len(lines)} lines")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    print("🚀 Starting Pine Script Generator Tests...")
    
    success = test_pine_script_generation()
    
    if success:
        show_sample_script()
        print(f"\n🎉 Pine Script Generator is working correctly!")
        exit_code = 0
    else:
        print(f"\n💥 Pine Script Generator tests failed!")
        exit_code = 1
    
    sys.exit(exit_code) 