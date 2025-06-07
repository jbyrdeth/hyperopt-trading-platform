#!/usr/bin/env python3
"""
Trading Strategy Optimization System
Main application entry point

This is a professional-grade trading strategy optimization system that finds
the best trading strategies for cryptocurrencies using hyperparameter optimization.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.logger import setup_logging
from src.utils.config_manager import ConfigManager
from src.data.data_fetcher import DataFetcher
from src.optimization.optimizer import StrategyOptimizer
from src.validation.cross_validator import CrossValidator
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.export.pine_script_generator import PineScriptGenerator


class TradingOptimizer:
    """
    Main trading strategy optimization system.
    
    This class orchestrates the entire optimization process from data fetching
    to strategy validation and Pine Script generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading optimizer.
        
        Args:
            config_path: Optional path to configuration directory
        """
        self.config_path = config_path or "config"
        self.config_manager = ConfigManager(self.config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_fetcher: Optional[DataFetcher] = None
        self.optimizer: Optional[StrategyOptimizer] = None
        self.validator: Optional[CrossValidator] = None
        self.analyzer: Optional[PerformanceAnalyzer] = None
        self.pine_generator: Optional[PineScriptGenerator] = None
        
        self.logger.info("Trading Strategy Optimization System initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_config = self.config_manager.get_config('optimization').get('logging', {})
        return setup_logging(
            level=log_config.get('level', 'INFO'),
            log_file=log_config.get('optimization_log', 'logs/optimization.log')
        )
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize data fetcher
            exchange_config = self.config_manager.get_config('exchanges')
            self.data_fetcher = DataFetcher(exchange_config)
            
            # Initialize optimizer
            opt_config = self.config_manager.get_config('optimization')
            strategy_config = self.config_manager.get_config('strategies')
            self.optimizer = StrategyOptimizer(opt_config, strategy_config)
            
            # Initialize validator
            self.validator = CrossValidator(opt_config['validation'])
            
            # Initialize analyzer
            self.analyzer = PerformanceAnalyzer()
            
            # Initialize Pine Script generator
            self.pine_generator = PineScriptGenerator()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_optimization(
        self,
        assets: list = None,
        timeframes: list = None,
        strategies: list = None,
        tournament_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete optimization process.
        
        Args:
            assets: List of assets to optimize (default: BTC, ETH, SOL)
            timeframes: List of timeframes (default: 4h)
            strategies: List of strategies to test (default: all)
            tournament_mode: Whether to use tournament optimization
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            self.logger.info("Starting optimization process...")
            
            # Set defaults
            assets = assets or ['BTC', 'ETH', 'SOL']
            timeframes = timeframes or ['4h']
            
            # Fetch data
            self.logger.info(f"Fetching data for assets: {assets}, timeframes: {timeframes}")
            data = self.data_fetcher.fetch_multiple_assets(assets, timeframes)
            
            # Run optimization
            if tournament_mode:
                self.logger.info("Running tournament optimization...")
                results = self.optimizer.run_tournament(data, strategies)
            else:
                self.logger.info("Running standard optimization...")
                results = self.optimizer.optimize_strategies(data, strategies)
            
            # Validate results
            self.logger.info("Validating optimization results...")
            validation_results = self.validator.validate_strategies(results, data)
            
            # Analyze performance
            self.logger.info("Analyzing performance...")
            analysis = self.analyzer.analyze_results(validation_results)
            
            # Combine results
            final_results = {
                'optimization': results,
                'validation': validation_results,
                'analysis': analysis,
                'metadata': {
                    'assets': assets,
                    'timeframes': timeframes,
                    'timestamp': datetime.now().isoformat(),
                    'total_strategies_tested': len(results),
                    'tournament_mode': tournament_mode
                }
            }
            
            self.logger.info("Optimization process completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Optimization process failed: {e}")
            raise
    
    def generate_pine_scripts(self, results: Dict[str, Any], top_n: int = 5) -> Dict[str, str]:
        """
        Generate Pine Script code for top strategies.
        
        Args:
            results: Optimization results
            top_n: Number of top strategies to generate scripts for
            
        Returns:
            Dictionary mapping strategy names to Pine Script code
        """
        try:
            self.logger.info(f"Generating Pine Scripts for top {top_n} strategies...")
            
            # Get top strategies
            top_strategies = self.analyzer.get_top_strategies(results, top_n)
            
            # Generate Pine Scripts
            pine_scripts = {}
            for strategy_name, strategy_data in top_strategies.items():
                script = self.pine_generator.generate_script(strategy_name, strategy_data)
                pine_scripts[strategy_name] = script
                
                # Save to file
                script_path = f"pine_scripts/{strategy_name}.pine"
                os.makedirs(os.path.dirname(script_path), exist_ok=True)
                with open(script_path, 'w') as f:
                    f.write(script)
                
                self.logger.info(f"Generated Pine Script for {strategy_name}")
            
            return pine_scripts
            
        except Exception as e:
            self.logger.error(f"Pine Script generation failed: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results"):
        """
        Save optimization results to files.
        
        Args:
            results: Results to save
            output_dir: Output directory
        """
        try:
            self.logger.info(f"Saving results to {output_dir}...")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Save main results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as JSON
            import json
            with open(f"{output_dir}/optimization_results_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            summary = self.analyzer.generate_summary_report(results)
            with open(f"{output_dir}/summary_report_{timestamp}.txt", 'w') as f:
                f.write(summary)
            
            # Generate visualizations
            self.analyzer.generate_visualizations(results, output_dir)
            
            self.logger.info("Results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Trading Strategy Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --assets BTC ETH --timeframes 4h 1d
  python main.py --tournament --top-strategies 10
  python main.py --config custom_config/ --assets SOL
        """
    )
    
    parser.add_argument(
        '--assets',
        nargs='+',
        default=['BTC', 'ETH', 'SOL'],
        help='Assets to optimize (default: BTC ETH SOL)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['4h'],
        help='Timeframes to use (default: 4h)'
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        help='Specific strategies to test (default: all)'
    )
    
    parser.add_argument(
        '--tournament',
        action='store_true',
        help='Use tournament optimization mode'
    )
    
    parser.add_argument(
        '--top-strategies',
        type=int,
        default=5,
        help='Number of top strategies to generate Pine Scripts for (default: 5)'
    )
    
    parser.add_argument(
        '--config',
        default='config',
        help='Configuration directory path (default: config)'
    )
    
    parser.add_argument(
        '--output',
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform a dry run without actual optimization'
    )
    
    return parser


def main():
    """Main application entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Initialize optimizer
        optimizer = TradingOptimizer(config_path=args.config)
        
        # Set log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        # Initialize components
        optimizer.initialize_components()
        
        if args.dry_run:
            optimizer.logger.info("Dry run mode - skipping actual optimization")
            optimizer.logger.info(f"Would optimize assets: {args.assets}")
            optimizer.logger.info(f"Would use timeframes: {args.timeframes}")
            optimizer.logger.info(f"Tournament mode: {args.tournament}")
            return
        
        # Run optimization
        results = optimizer.run_optimization(
            assets=args.assets,
            timeframes=args.timeframes,
            strategies=args.strategies,
            tournament_mode=args.tournament
        )
        
        # Generate Pine Scripts for top strategies
        pine_scripts = optimizer.generate_pine_scripts(results, args.top_strategies)
        
        # Save results
        optimizer.save_results(results, args.output)
        
        # Print summary
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Assets optimized: {', '.join(args.assets)}")
        print(f"Timeframes used: {', '.join(args.timeframes)}")
        print(f"Total strategies tested: {results['metadata']['total_strategies_tested']}")
        print(f"Pine Scripts generated: {len(pine_scripts)}")
        print(f"Results saved to: {args.output}")
        print("="*80)
        
        # Show top strategies
        top_strategies = optimizer.analyzer.get_top_strategies(results, 5)
        print("\nTOP 5 STRATEGIES:")
        for i, (name, data) in enumerate(top_strategies.items(), 1):
            metrics = data.get('metrics', {})
            print(f"{i}. {name}")
            print(f"   Annual Return: {metrics.get('annual_return', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 