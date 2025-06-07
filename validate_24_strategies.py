#!/usr/bin/env python3
"""
Comprehensive 24-Strategy Validation Script

Tests all 24 strategies with the fixed backtesting engine to establish
accurate baseline performance metrics for the "Perfect & Deploy" mission.
"""

import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Import our modules with proper error handling
try:
    from optimization.strategy_factory import StrategyFactory
    from strategies.backtesting_engine import BacktestingEngine
    from data.data_fetcher import DataFetcher
    from utils.logger import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

class Strategy24Validator:
    """Comprehensive validator for all 24 trading strategies."""
    
    def __init__(self):
        self.logger = get_logger("strategy_validator")
        self.strategy_factory = StrategyFactory()
        self.data_fetcher = DataFetcher()
        self.engine = BacktestingEngine(initial_capital=100000)
        
        # Test configuration
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.test_period_days = 90  # 3 months of data
        self.results = {}
        
        self.logger.info("Initialized 24-Strategy Validator")
    
    def fetch_test_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch test data for validation."""
        self.logger.info(f"Fetching test data for {len(self.test_symbols)} symbols")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.test_period_days)
        
        data = {}
        for symbol in self.test_symbols:
            try:
                self.logger.info(f"Fetching data for {symbol}")
                df = self.data_fetcher.fetch_data(
                    symbol=symbol,
                    timeframe='4h',
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and len(df) > 100:  # Ensure sufficient data
                    data[symbol] = df
                    self.logger.info(f"✅ {symbol}: {len(df)} data points")
                else:
                    self.logger.warning(f"❌ {symbol}: Insufficient data")
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Failed to fetch data - {e}")
        
        return data
    
    def get_default_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get reasonable default parameters for a strategy."""
        try:
            param_space = self.strategy_factory.get_parameter_space(strategy_name)
            defaults = {}
            
            for param_name, param_config in param_space.items():
                if isinstance(param_config, dict):
                    if 'low' in param_config and 'high' in param_config:
                        # Use middle value for range parameters
                        low = param_config['low']
                        high = param_config['high']
                        if isinstance(low, int) and isinstance(high, int):
                            defaults[param_name] = (low + high) // 2
                        else:
                            defaults[param_name] = (low + high) / 2
                    elif 'choices' in param_config:
                        # Use first choice for choice parameters
                        defaults[param_name] = param_config['choices'][0]
                    else:
                        defaults[param_name] = param_config
                else:
                    defaults[param_name] = param_config
            
            return defaults
            
        except Exception as e:
            self.logger.error(f"Error getting defaults for {strategy_name}: {e}")
            return {}
    
    def validate_single_strategy(
        self, 
        strategy_name: str, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Validate a single strategy across all test symbols."""
        
        self.logger.info(f"🎯 Validating strategy: {strategy_name}")
        
        strategy_results = {
            'name': strategy_name,
            'category': 'unknown',
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'symbol_results': {},
            'aggregate_metrics': {},
            'errors': []
        }
        
        try:
            # Get strategy info
            strategy_info = self.strategy_factory.registry.get_strategy_info(strategy_name)
            strategy_results['category'] = strategy_info.get('category', 'unknown')
            
            # Get default parameters
            parameters = self.get_default_parameters(strategy_name)
            if not parameters:
                strategy_results['errors'].append("Could not determine default parameters")
                return strategy_results
            
            # Test on each symbol
            all_returns = []
            all_sharpe_ratios = []
            all_max_drawdowns = []
            all_trade_counts = []
            all_win_rates = []
            
            for symbol, df in data.items():
                strategy_results['total_tests'] += 1
                
                try:
                    # Create strategy instance
                    strategy = self.strategy_factory.create_strategy(strategy_name, parameters)
                    
                    # Reset engine for each test
                    self.engine.reset()
                    
                    # Run backtest
                    results = self.engine.backtest_strategy(strategy, df, symbol)
                    
                    # Store results
                    symbol_result = {
                        'symbol': symbol,
                        'total_return': results.total_return,
                        'annual_return': results.annual_return,
                        'sharpe_ratio': results.sharpe_ratio,
                        'max_drawdown': results.max_drawdown,
                        'total_trades': results.total_trades,
                        'win_rate': results.win_rate,
                        'profit_factor': results.profit_factor,
                        'volatility': results.volatility,
                        'calmar_ratio': results.calmar_ratio,
                        'data_points': len(df),
                        'start_date': df.index[0].isoformat(),
                        'end_date': df.index[-1].isoformat()
                    }
                    
                    strategy_results['symbol_results'][symbol] = symbol_result
                    strategy_results['successful_tests'] += 1
                    
                    # Collect for aggregation
                    all_returns.append(results.total_return)
                    all_sharpe_ratios.append(results.sharpe_ratio)
                    all_max_drawdowns.append(results.max_drawdown)
                    all_trade_counts.append(results.total_trades)
                    all_win_rates.append(results.win_rate)
                    
                    self.logger.info(
                        f"  ✅ {symbol}: {results.total_return*100:.2f}% return, "
                        f"{results.total_trades} trades, {results.sharpe_ratio:.2f} Sharpe"
                    )
                    
                except Exception as e:
                    strategy_results['failed_tests'] += 1
                    error_msg = f"{symbol}: {str(e)}"
                    strategy_results['errors'].append(error_msg)
                    self.logger.error(f"  ❌ {error_msg}")
            
            # Calculate aggregate metrics
            if all_returns:
                strategy_results['aggregate_metrics'] = {
                    'avg_return': np.mean(all_returns),
                    'std_return': np.std(all_returns),
                    'avg_sharpe_ratio': np.mean([s for s in all_sharpe_ratios if not np.isnan(s) and not np.isinf(s)]),
                    'avg_max_drawdown': np.mean(all_max_drawdowns),
                    'avg_trades': np.mean(all_trade_counts),
                    'avg_win_rate': np.mean(all_win_rates),
                    'consistency_score': len([r for r in all_returns if r > 0]) / len(all_returns),
                    'parameters_used': parameters
                }
                
                self.logger.info(
                    f"  📊 Aggregate: {strategy_results['aggregate_metrics']['avg_return']*100:.2f}% avg return, "
                    f"{strategy_results['aggregate_metrics']['consistency_score']*100:.1f}% consistency"
                )
            
        except Exception as e:
            strategy_results['errors'].append(f"Strategy creation failed: {str(e)}")
            self.logger.error(f"❌ Strategy {strategy_name} failed: {e}")
        
        return strategy_results
    
    def validate_all_strategies(self) -> Dict[str, Any]:
        """Validate all 24 strategies."""
        
        self.logger.info("🚀 Starting comprehensive 24-strategy validation")
        
        # Fetch test data
        data = self.fetch_test_data()
        if not data:
            self.logger.error("❌ No test data available - cannot proceed")
            return {}
        
        self.logger.info(f"✅ Test data ready for {len(data)} symbols")
        
        # Get all strategies
        all_strategies = self.strategy_factory.get_all_strategies()
        self.logger.info(f"📋 Found {len(all_strategies)} strategies to validate")
        
        # Validate each strategy
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'symbols': self.test_symbols,
                'period_days': self.test_period_days,
                'initial_capital': self.engine.initial_capital,
                'data_points_per_symbol': {symbol: len(df) for symbol, df in data.items()}
            },
            'strategies': {},
            'summary': {
                'total_strategies': len(all_strategies),
                'successful_strategies': 0,
                'failed_strategies': 0,
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0
            }
        }
        
        for i, strategy_name in enumerate(all_strategies, 1):
            self.logger.info(f"\n📈 [{i}/{len(all_strategies)}] Validating {strategy_name}")
            
            strategy_result = self.validate_single_strategy(strategy_name, data)
            validation_results['strategies'][strategy_name] = strategy_result
            
            # Update summary
            validation_results['summary']['total_tests'] += strategy_result['total_tests']
            validation_results['summary']['successful_tests'] += strategy_result['successful_tests']
            validation_results['summary']['failed_tests'] += strategy_result['failed_tests']
            
            if strategy_result['successful_tests'] > 0:
                validation_results['summary']['successful_strategies'] += 1
            else:
                validation_results['summary']['failed_strategies'] += 1
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
        
        return validation_results
    
    def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a summary report of validation results."""
        
        if not results or 'strategies' not in results:
            return "❌ No validation results available"
        
        report = []
        report.append("🏆 24-STRATEGY VALIDATION SUMMARY")
        report.append("=" * 50)
        
        # Overall summary
        summary = results['summary']
        report.append(f"\n📊 OVERALL RESULTS:")
        report.append(f"   Total Strategies: {summary['total_strategies']}")
        report.append(f"   Successful: {summary['successful_strategies']}")
        report.append(f"   Failed: {summary['failed_strategies']}")
        report.append(f"   Success Rate: {(summary['successful_strategies']/summary['total_strategies']*100):.1f}%")
        report.append(f"   Total Tests: {summary['total_tests']}")
        report.append(f"   Test Success Rate: {(summary['successful_tests']/summary['total_tests']*100):.1f}%")
        
        # Category breakdown
        categories = {}
        for strategy_name, strategy_data in results['strategies'].items():
            category = strategy_data.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'count': 0, 'successful': 0}
            categories[category]['count'] += 1
            if strategy_data['successful_tests'] > 0:
                categories[category]['successful'] += 1
        
        report.append(f"\n📂 BY CATEGORY:")
        for category, stats in categories.items():
            success_rate = (stats['successful'] / stats['count'] * 100) if stats['count'] > 0 else 0
            report.append(f"   {category}: {stats['successful']}/{stats['count']} ({success_rate:.1f}%)")
        
        # Top performers
        successful_strategies = [
            (name, data) for name, data in results['strategies'].items()
            if data['successful_tests'] > 0 and 'aggregate_metrics' in data
        ]
        
        if successful_strategies:
            # Sort by average return
            successful_strategies.sort(
                key=lambda x: x[1]['aggregate_metrics'].get('avg_return', 0), 
                reverse=True
            )
            
            report.append(f"\n🥇 TOP 10 PERFORMERS (by avg return):")
            for i, (name, data) in enumerate(successful_strategies[:10], 1):
                metrics = data['aggregate_metrics']
                report.append(
                    f"   {i:2d}. {name}: "
                    f"{metrics['avg_return']*100:6.2f}% return, "
                    f"{metrics.get('avg_sharpe_ratio', 0):5.2f} Sharpe, "
                    f"{metrics['consistency_score']*100:4.1f}% consistency"
                )
        
        # Failed strategies
        failed_strategies = [
            name for name, data in results['strategies'].items()
            if data['successful_tests'] == 0
        ]
        
        if failed_strategies:
            report.append(f"\n❌ FAILED STRATEGIES ({len(failed_strategies)}):")
            for name in failed_strategies:
                errors = results['strategies'][name].get('errors', [])
                report.append(f"   - {name}: {'; '.join(errors[:2])}")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = "validation_results_24_strategies.json"):
        """Save validation results to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"✅ Results saved to {filename}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save results: {e}")

def main():
    """Main validation function."""
    
    print("🚀 STARTING 24-STRATEGY VALIDATION")
    print("=" * 50)
    print("Mission: Perfect & Deploy - Validate existing strategies")
    print("Scope: All 24 strategies across 3 major crypto pairs")
    print("Goal: Establish accurate baseline performance metrics")
    print()
    
    # Initialize validator
    validator = Strategy24Validator()
    
    # Run validation
    start_time = time.time()
    results = validator.validate_all_strategies()
    end_time = time.time()
    
    if results:
        # Generate and display summary
        summary = validator.generate_summary_report(results)
        print(summary)
        
        # Save results
        validator.save_results(results)
        
        print(f"\n⏱️ Validation completed in {end_time - start_time:.1f} seconds")
        print(f"📁 Detailed results saved to validation_results_24_strategies.json")
        print("\n🎯 Next Steps:")
        print("   1. Review top performing strategies")
        print("   2. Investigate any failed strategies")
        print("   3. Proceed to multi-strategy portfolio optimization")
        
        return results
    else:
        print("❌ Validation failed - no results generated")
        return None

if __name__ == "__main__":
    results = main() 