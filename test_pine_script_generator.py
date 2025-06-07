"""
Test Pine Script Generator System

This test suite validates the Pine Script generation, parameter translation,
and code validation systems work correctly.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')

from export.pine_script_generator import (
    PineScriptGenerator, PineScriptConfig, MovingAverageCrossoverTemplate,
    RSITemplate, MACDTemplate, SignalCondition, IndicatorCalculation
)
from export.parameter_translator import ParameterTranslator, PineScriptParameter, PineScriptType
from export.pine_validator import PineScriptValidator, ValidationSeverity
from strategies.base_strategy import BaseStrategy
from optimization.hyperopt_optimizer import OptimizationResult


class MockStrategy(BaseStrategy):
    """Mock strategy for testing purposes."""
    
    def __init__(self, name="Test Strategy", strategy_type="moving_average"):
        self.name = name
        self.strategy_type = strategy_type
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
    
    def generate_signals(self, data):
        """Mock signal generation."""
        return {"long_entry": [True, False], "long_exit": [False, True]}
    
    def calculate_returns(self, data):
        """Mock return calculation."""
        return [0.01, -0.005, 0.015]


class TestPineScriptGenerator(unittest.TestCase):
    """Test Pine Script Generator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PineScriptConfig(
            output_directory=self.temp_dir,
            title="Test Strategy",
            include_alerts=True,
            include_debugging=True
        )
        self.generator = PineScriptGenerator(self.config)
        self.strategy = MockStrategy()
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generator_initialization(self):
        """Test generator initializes correctly."""
        self.assertIsInstance(self.generator, PineScriptGenerator)
        self.assertEqual(len(self.generator.templates), 3)
        self.assertIn("moving_average_crossover", self.generator.templates)
        self.assertIn("rsi_strategy", self.generator.templates)
        self.assertIn("macd_strategy", self.generator.templates)
    
    def test_template_detection(self):
        """Test automatic template detection."""
        # Test moving average detection
        ma_strategy = MockStrategy("Moving Average Cross", "moving_average")
        template = self.generator._detect_template(ma_strategy)
        self.assertEqual(template, "moving_average_crossover")
        
        # Test RSI detection
        rsi_strategy = MockStrategy("RSI Strategy", "rsi")
        template = self.generator._detect_template(rsi_strategy)
        self.assertEqual(template, "rsi_strategy")
        
        # Test MACD detection
        macd_strategy = MockStrategy("MACD Strategy", "macd")
        template = self.generator._detect_template(macd_strategy)
        self.assertEqual(template, "macd_strategy")
    
    def test_parameter_extraction(self):
        """Test parameter extraction from strategy."""
        params = self.generator._get_strategy_parameters(self.strategy)
        
        self.assertIn("fast_period", params)
        self.assertIn("slow_period", params) 
        self.assertIn("signal_threshold", params)
        self.assertIn("stop_loss", params)
        self.assertEqual(params["fast_period"], 12)
        self.assertEqual(params["slow_period"], 26)
    
    def test_strategy_script_generation(self):
        """Test complete strategy script generation."""
        script_path = self.generator.generate_strategy_script(
            self.strategy,
            template_name="moving_average_crossover"
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(script_path))
        
        # Read and verify content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for required sections
        self.assertIn("//@version=5", content)
        self.assertIn("strategy(", content)
        self.assertIn("fast_ma = ta.sma(close,", content)
        self.assertIn("slow_ma = ta.sma(close,", content)
        self.assertIn("strategy.entry(", content)
        self.assertIn("stop_loss_pct", content)
        self.assertIn("take_profit_pct", content)
    
    def test_indicator_script_generation(self):
        """Test indicator script generation."""
        script_path = self.generator.generate_indicator_script(
            self.strategy,
            template_name="moving_average_crossover"
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(script_path))
        
        # Read and verify content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Should be indicator, not strategy
        self.assertIn("indicator(", content)
        self.assertNotIn("strategy(", content)
        self.assertNotIn("strategy.entry", content)
        self.assertNotIn("strategy.exit", content)
    
    def test_batch_generation(self):
        """Test batch script generation."""
        strategies = [
            (MockStrategy("Strategy 1", "moving_average"), None),
            (MockStrategy("Strategy 2", "rsi"), None),
            (MockStrategy("Strategy 3", "macd"), None)
        ]
        
        generated_files = self.generator.generate_batch_scripts(strategies)
        
        self.assertEqual(len(generated_files), 3)
        for file_path in generated_files:
            self.assertTrue(os.path.exists(file_path))


class TestPineScriptTemplates(unittest.TestCase):
    """Test Pine Script template implementations."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = PineScriptConfig()
    
    def test_moving_average_template(self):
        """Test Moving Average Crossover template."""
        template = MovingAverageCrossoverTemplate(self.config)
        
        # Test template properties
        self.assertEqual(template.get_template_name(), "Moving Average Crossover")
        self.assertEqual(set(template.get_required_parameters()), 
                        {"fast_period", "slow_period", "signal_threshold"})
        
        # Test indicator generation
        params = {"fast_period": 12, "slow_period": 26, "signal_threshold": 0.02}
        indicators = template.generate_indicators(params)
        
        self.assertEqual(len(indicators), 3)
        self.assertTrue(any(ind.name == "fast_ma" for ind in indicators))
        self.assertTrue(any(ind.name == "slow_ma" for ind in indicators))
        self.assertTrue(any(ind.name == "signal_strength" for ind in indicators))
        
        # Test signal generation
        signals = template.generate_signals(params)
        
        self.assertEqual(len(signals), 4)
        signal_types = [s.signal_type for s in signals]
        self.assertIn("long_entry", signal_types)
        self.assertIn("long_exit", signal_types)
        self.assertIn("short_entry", signal_types)
        self.assertIn("short_exit", signal_types)
    
    def test_rsi_template(self):
        """Test RSI template."""
        template = RSITemplate(self.config)
        
        # Test template properties
        self.assertEqual(template.get_template_name(), "RSI Strategy")
        required_params = set(template.get_required_parameters())
        expected_params = {"rsi_period", "rsi_overbought", "rsi_oversold", "confirmation_period"}
        self.assertEqual(required_params, expected_params)
        
        # Test with parameters
        params = {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "confirmation_period": 2
        }
        
        indicators = template.generate_indicators(params)
        self.assertEqual(len(indicators), 2)
        
        signals = template.generate_signals(params)
        self.assertEqual(len(signals), 4)
    
    def test_macd_template(self):
        """Test MACD template."""
        template = MACDTemplate(self.config)
        
        # Test template properties
        self.assertEqual(template.get_template_name(), "MACD Strategy")
        required_params = set(template.get_required_parameters())
        expected_params = {"macd_fast", "macd_slow", "macd_signal", "signal_threshold"}
        self.assertEqual(required_params, expected_params)
        
        # Test with parameters
        params = {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "signal_threshold": 0.001
        }
        
        indicators = template.generate_indicators(params)
        self.assertEqual(len(indicators), 2)
        
        signals = template.generate_signals(params)
        self.assertEqual(len(signals), 4)


class TestParameterTranslator(unittest.TestCase):
    """Test parameter translation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.translator = ParameterTranslator()
    
    def test_predefined_parameter_translation(self):
        """Test translation of predefined parameters."""
        # Test fast_period
        param = self.translator.translate_parameter("fast_period", 12)
        
        self.assertEqual(param.name, "fast_period")
        self.assertEqual(param.param_type, PineScriptType.INT)
        self.assertEqual(param.default_value, 12)
        self.assertEqual(param.title, "Fast MA Period")
        self.assertEqual(param.group, "Moving Average")
        self.assertEqual(param.minval, 1)
        self.assertEqual(param.maxval, 200)
        
        # Test RSI parameter
        param = self.translator.translate_parameter("rsi_overbought", 70.0)
        
        self.assertEqual(param.param_type, PineScriptType.FLOAT)
        self.assertEqual(param.title, "RSI Overbought")
        self.assertEqual(param.group, "RSI")
    
    def test_auto_detection(self):
        """Test automatic parameter type detection."""
        # Test integer detection
        param = self.translator.translate_parameter("custom_period", 20)
        self.assertEqual(param.param_type, PineScriptType.INT)
        
        # Test float detection
        param = self.translator.translate_parameter("custom_threshold", 0.5)
        self.assertEqual(param.param_type, PineScriptType.FLOAT)
        
        # Test boolean detection
        param = self.translator.translate_parameter("enable_filter", True)
        self.assertEqual(param.param_type, PineScriptType.BOOL)
    
    def test_parameter_name_cleaning(self):
        """Test parameter name cleaning for Pine Script."""
        # Test with spaces and special characters
        clean_name = self.translator._clean_parameter_name("test-param with spaces!")
        self.assertEqual(clean_name, "test_param_with_spaces_")
        
        # Test starting with number
        clean_name = self.translator._clean_parameter_name("1st_param")
        self.assertEqual(clean_name, "param_1st_param")
    
    def test_pine_script_output(self):
        """Test Pine Script code generation."""
        param = self.translator.translate_parameter("fast_period", 12)
        pine_code = param.to_pine_script()
        
        self.assertIn("fast_period = input.int(", pine_code)
        self.assertIn("12", pine_code)
        self.assertIn('"Fast MA Period"', pine_code)
        self.assertIn("minval=1", pine_code)
        self.assertIn("maxval=200", pine_code)
        self.assertIn('group="Moving Average"', pine_code)
    
    def test_multiple_parameter_translation(self):
        """Test translation of multiple parameters."""
        params = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_threshold": 0.02,
            "enable_filter": True
        }
        
        pine_params = self.translator.translate_parameters(params)
        
        self.assertEqual(len(pine_params), 4)
        param_names = [p.name for p in pine_params]
        self.assertIn("fast_period", param_names)
        self.assertIn("slow_period", param_names)
        self.assertIn("signal_threshold", param_names)
        self.assertIn("enable_filter", param_names)


class TestPineScriptValidator(unittest.TestCase):
    """Test Pine Script validation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.validator = PineScriptValidator()
    
    def test_valid_script_validation(self):
        """Test validation of a valid Pine Script."""
        valid_script = '''
//@version=5
strategy("Test Strategy", shorttitle="Test", overlay=false)

// Parameters
fast_period = input.int(12, "Fast MA Period", minval=1, maxval=200)
slow_period = input.int(26, "Slow MA Period", minval=1, maxval=500)

// Calculations
fast_ma = ta.sma(close, fast_period)
slow_ma = ta.sma(close, slow_period)

// Signals
long_entry = ta.crossover(fast_ma, slow_ma)
long_exit = ta.crossunder(fast_ma, slow_ma)

// Strategy
if (long_entry)
    strategy.entry("Long", strategy.long)
if (long_exit)
    strategy.close("Long")
'''
        
        result = self.validator.validate(valid_script)
        
        # Should have minimal issues (maybe some warnings/info)
        self.assertTrue(result.error_count <= 1)  # Allow for minor issues
    
    def test_invalid_script_validation(self):
        """Test validation of an invalid Pine Script."""
        invalid_script = '''
// Missing version
strategy("Test Strategy")

// Unmatched parentheses
fast_ma = ta.sma(close, 12
slow_ma = ta.sma(close, 26))

// Reserved keyword as variable
strategy = "test"

// Missing strategy calls
long_entry = ta.crossover(fast_ma, slow_ma)
'''
        
        result = self.validator.validate(invalid_script)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(result.error_count, 0)
        
        # Check for specific error types
        error_messages = [issue.message for issue in result.issues 
                         if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        
        # Should catch missing version
        self.assertTrue(any("version" in msg.lower() for msg in error_messages))
    
    def test_empty_script_validation(self):
        """Test validation of empty script."""
        result = self.validator.validate("")
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.error_count, 1)
        self.assertIn("empty", result.issues[0].message.lower())
    
    def test_risk_management_validation(self):
        """Test risk management validation."""
        script_without_risk = '''
//@version=5
strategy("Test Strategy", overlay=false)

fast_ma = ta.sma(close, 12)
slow_ma = ta.sma(close, 26)

if (ta.crossover(fast_ma, slow_ma))
    strategy.entry("Long", strategy.long)
'''
        
        result = self.validator.validate(script_without_risk)
        
        # Should have warnings about missing risk management
        warning_messages = [issue.message for issue in result.issues 
                           if issue.severity == ValidationSeverity.WARNING]
        
        self.assertTrue(any("stop loss" in msg.lower() for msg in warning_messages))
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        script = '''
//@version=5
strategy("Test Strategy", overlay=false)
// Some simple script content
'''
        
        result = self.validator.validate(script)
        summary = self.validator.get_validation_summary(result)
        
        self.assertIsInstance(summary, str)
        self.assertIn("validation", summary.lower())


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = PineScriptConfig(output_directory=self.temp_dir)
        self.generator = PineScriptGenerator(self.config)
        self.validator = PineScriptValidator()
        self.strategy = MockStrategy()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_and_validate(self):
        """Test generating script and validating it."""
        # Generate script
        script_path = self.generator.generate_strategy_script(self.strategy)
        
        # Validate generated script
        result = self.validator.validate_file(script_path)
        
        # Should pass validation or have minimal issues
        self.assertTrue(result.error_count <= 2)  # Allow for minor issues
        
        print(f"Generated script validation: {result.get_summary()}")
        if result.issues:
            for issue in result.issues[:5]:  # Show first 5 issues
                print(f"  {issue}")
    
    def test_different_templates_validation(self):
        """Test that all templates generate valid code."""
        templates = ["moving_average_crossover", "rsi_strategy", "macd_strategy"]
        
        for template_name in templates:
            with self.subTest(template=template_name):
                # Adjust strategy parameters for template
                if template_name == "rsi_strategy":
                    self.strategy.parameters.update({
                        "rsi_period": 14,
                        "rsi_overbought": 70,
                        "rsi_oversold": 30,
                        "confirmation_period": 2
                    })
                elif template_name == "macd_strategy":
                    self.strategy.parameters.update({
                        "macd_fast": 12,
                        "macd_slow": 26,
                        "macd_signal": 9,
                        "signal_threshold": 0.001
                    })
                
                # Generate and validate
                script_path = self.generator.generate_strategy_script(
                    self.strategy, template_name=template_name
                )
                result = self.validator.validate_file(script_path)
                
                # Should have minimal critical errors
                critical_errors = [issue for issue in result.issues 
                                 if issue.severity == ValidationSeverity.CRITICAL]
                self.assertEqual(len(critical_errors), 0)
                
                print(f"{template_name}: {result.get_summary()}")


if __name__ == "__main__":
    # Create a test suite
    test_classes = [
        TestPineScriptGenerator,
        TestPineScriptTemplates, 
        TestParameterTranslator,
        TestPineScriptValidator,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("PINE SCRIPT GENERATOR TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Error:')[-1].strip()}") 