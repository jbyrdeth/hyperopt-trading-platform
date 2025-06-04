"""
Cross-Asset Validation System

This module provides functionality to validate trading strategies across multiple
cryptocurrencies to test generalizability and robustness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from ..utils.logger import get_logger
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, CostModel, BacktestResults
    from .data_splitter import DataSplitter
except ImportError:
    from src.utils.logger import get_logger
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, CostModel, BacktestResults
    from src.validation.data_splitter import DataSplitter


@dataclass
class AssetPerformance:
    """Performance metrics for a single asset."""
    asset: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    
    # Additional metrics
    profit_factor: float
    expectancy: float
    avg_trade_duration: float
    
    # Raw results for detailed analysis
    backtest_results: BacktestResults
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy analysis."""
        return {
            'asset': self.asset,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'volatility': self.volatility,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'avg_trade_duration': self.avg_trade_duration
        }


@dataclass
class CrossAssetResults:
    """Results from cross-asset validation."""
    strategy_name: str
    asset_performances: List[AssetPerformance]
    correlation_matrix: pd.DataFrame
    performance_summary: Dict[str, Any]
    generalization_score: float
    consistency_score: float
    robustness_metrics: Dict[str, float]
    
    # Transfer learning results
    transfer_results: Optional[Dict[str, Any]] = None
    
    # Asset groupings based on correlation
    asset_groups: Optional[Dict[str, List[str]]] = None
    
    def get_performance_dataframe(self) -> pd.DataFrame:
        """Get performance metrics as a DataFrame."""
        return pd.DataFrame([perf.to_dict() for perf in self.asset_performances])
    
    def get_best_assets(self, metric: str = 'sharpe_ratio', top_n: int = 3) -> List[str]:
        """Get top performing assets by specified metric."""
        df = self.get_performance_dataframe()
        return df.nlargest(top_n, metric)['asset'].tolist()
    
    def get_worst_assets(self, metric: str = 'sharpe_ratio', bottom_n: int = 3) -> List[str]:
        """Get worst performing assets by specified metric."""
        df = self.get_performance_dataframe()
        return df.nsmallest(bottom_n, metric)['asset'].tolist()


class CrossAssetValidator:
    """
    Cross-asset validation system for trading strategies.
    
    Tests strategy performance across multiple cryptocurrencies to evaluate
    generalizability, robustness, and consistency.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        cost_model: Optional[CostModel] = None,
        correlation_threshold: float = 0.7,
        min_correlation_periods: int = 252,  # 1 year of daily data
        parallel_execution: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize the CrossAssetValidator.
        
        Args:
            initial_capital: Starting capital for each backtest
            cost_model: Cost model for transaction costs
            correlation_threshold: Threshold for grouping correlated assets
            min_correlation_periods: Minimum periods for correlation calculation
            parallel_execution: Whether to run backtests in parallel
            max_workers: Maximum number of parallel workers
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.correlation_threshold = correlation_threshold
        self.min_correlation_periods = min_correlation_periods
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
        
        self.logger = get_logger("cross_asset_validator")
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(
            initial_capital=initial_capital,
            cost_model=self.cost_model
        )
        
        # Data splitter for train/test splits
        self.data_splitter = DataSplitter()
    
    def validate_strategy(
        self,
        strategy: BaseStrategy,
        asset_data: Dict[str, pd.DataFrame],
        test_mode: str = 'full_data'
    ) -> CrossAssetResults:
        """
        Validate a strategy across multiple assets.
        
        Args:
            strategy: Strategy to validate
            asset_data: Dictionary mapping asset names to OHLCV DataFrames
            test_mode: 'full_data', 'out_of_sample', or 'transfer_learning'
            
        Returns:
            CrossAssetResults object with comprehensive validation results
        """
        self.logger.info(f"Starting cross-asset validation for {strategy.name} on {len(asset_data)} assets")
        
        # Validate input data
        validated_data = self._validate_asset_data(asset_data)
        
        if len(validated_data) < 2:
            raise ValueError("Need at least 2 assets for cross-asset validation")
        
        # Calculate asset correlations
        correlation_matrix = self._calculate_asset_correlations(validated_data)
        
        # Run backtests based on test mode
        if test_mode == 'full_data':
            asset_performances = self._run_full_data_backtests(strategy, validated_data)
        elif test_mode == 'out_of_sample':
            asset_performances = self._run_out_of_sample_backtests(strategy, validated_data)
        elif test_mode == 'transfer_learning':
            asset_performances, transfer_results = self._run_transfer_learning_backtests(strategy, validated_data)
        else:
            raise ValueError(f"Unknown test mode: {test_mode}")
        
        # Calculate performance summary and scores
        performance_summary = self._calculate_performance_summary(asset_performances)
        generalization_score = self._calculate_generalization_score(asset_performances)
        consistency_score = self._calculate_consistency_score(asset_performances)
        robustness_metrics = self._calculate_robustness_metrics(asset_performances, correlation_matrix)
        
        # Group assets by correlation
        asset_groups = self._group_assets_by_correlation(correlation_matrix)
        
        results = CrossAssetResults(
            strategy_name=strategy.name,
            asset_performances=asset_performances,
            correlation_matrix=correlation_matrix,
            performance_summary=performance_summary,
            generalization_score=generalization_score,
            consistency_score=consistency_score,
            robustness_metrics=robustness_metrics,
            asset_groups=asset_groups
        )
        
        if test_mode == 'transfer_learning':
            results.transfer_results = transfer_results
        
        self.logger.info(f"Cross-asset validation completed. Generalization score: {generalization_score:.3f}")
        
        return results
    
    def _validate_asset_data(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Validate and clean asset data."""
        validated_data = {}
        
        for asset, data in asset_data.items():
            try:
                # Check required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_cols):
                    self.logger.warning(f"Asset {asset} missing required columns, skipping")
                    continue
                
                # Check for sufficient data
                if len(data) < 100:
                    self.logger.warning(f"Asset {asset} has insufficient data ({len(data)} periods), skipping")
                    continue
                
                # Check for datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    self.logger.warning(f"Asset {asset} does not have datetime index, skipping")
                    continue
                
                # Remove rows with missing critical data
                clean_data = data.dropna(subset=['close', 'volume'])
                
                if len(clean_data) < len(data) * 0.9:  # More than 10% missing
                    self.logger.warning(f"Asset {asset} has too much missing data, skipping")
                    continue
                
                validated_data[asset] = clean_data.sort_index()
                self.logger.debug(f"Validated asset {asset}: {len(clean_data)} periods")
                
            except Exception as e:
                self.logger.error(f"Error validating asset {asset}: {e}")
                continue
        
        return validated_data
    
    def _calculate_asset_correlations(self, asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between asset returns."""
        # Align data to common time periods
        common_start = max(data.index[0] for data in asset_data.values())
        common_end = min(data.index[-1] for data in asset_data.values())
        
        returns_data = {}
        
        for asset, data in asset_data.items():
            # Filter to common period
            aligned_data = data.loc[common_start:common_end]
            
            if len(aligned_data) >= self.min_correlation_periods:
                # Calculate returns
                returns = aligned_data['close'].pct_change().dropna()
                returns_data[asset] = returns
        
        if len(returns_data) < 2:
            self.logger.warning("Insufficient data for correlation calculation")
            return pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        self.logger.info(f"Calculated correlations for {len(returns_data)} assets")
        
        return correlation_matrix
    
    def _run_full_data_backtests(
        self,
        strategy: BaseStrategy,
        asset_data: Dict[str, pd.DataFrame]
    ) -> List[AssetPerformance]:
        """Run backtests on full datasets for each asset."""
        
        def backtest_asset(asset_name, data):
            try:
                # Create a fresh strategy instance for each asset
                strategy_copy = strategy.__class__(**strategy.parameters, **strategy.risk_params)
                
                # Run backtest
                results = self.backtesting_engine.backtest_strategy(strategy_copy, data, asset_name)
                
                return AssetPerformance(
                    asset=asset_name,
                    total_return=results.total_return,
                    annual_return=results.annual_return,
                    sharpe_ratio=results.sharpe_ratio,
                    max_drawdown=results.max_drawdown,
                    win_rate=results.win_rate,
                    total_trades=results.total_trades,
                    volatility=results.volatility,
                    calmar_ratio=results.calmar_ratio,
                    sortino_ratio=results.sortino_ratio,
                    profit_factor=results.profit_factor,
                    expectancy=results.expectancy,
                    avg_trade_duration=results.avg_trade_duration,
                    backtest_results=results
                )
                
            except Exception as e:
                self.logger.error(f"Backtest failed for asset {asset_name}: {e}")
                return None
        
        performances = []
        
        if self.parallel_execution and len(asset_data) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_asset = {
                    executor.submit(backtest_asset, asset, data): asset 
                    for asset, data in asset_data.items()
                }
                
                for future in as_completed(future_to_asset):
                    result = future.result()
                    if result is not None:
                        performances.append(result)
        else:
            # Sequential execution
            for asset, data in asset_data.items():
                result = backtest_asset(asset, data)
                if result is not None:
                    performances.append(result)
        
        return performances
    
    def _run_out_of_sample_backtests(
        self,
        strategy: BaseStrategy,
        asset_data: Dict[str, pd.DataFrame]
    ) -> List[AssetPerformance]:
        """Run out-of-sample backtests using train/test splits."""
        performances = []
        
        for asset, data in asset_data.items():
            try:
                # Split data
                split = self.data_splitter.chronological_split(data)
                
                # Train strategy on training data (if needed for parameter optimization)
                # For now, we'll just test on the test set with given parameters
                
                # Create strategy instance
                strategy_copy = strategy.__class__(**strategy.parameters, **strategy.risk_params)
                
                # Run backtest on test data only
                results = self.backtesting_engine.backtest_strategy(strategy_copy, split.test, asset)
                
                performance = AssetPerformance(
                    asset=asset,
                    total_return=results.total_return,
                    annual_return=results.annual_return,
                    sharpe_ratio=results.sharpe_ratio,
                    max_drawdown=results.max_drawdown,
                    win_rate=results.win_rate,
                    total_trades=results.total_trades,
                    volatility=results.volatility,
                    calmar_ratio=results.calmar_ratio,
                    sortino_ratio=results.sortino_ratio,
                    profit_factor=results.profit_factor,
                    expectancy=results.expectancy,
                    avg_trade_duration=results.avg_trade_duration,
                    backtest_results=results
                )
                
                performances.append(performance)
                
            except Exception as e:
                self.logger.error(f"Out-of-sample backtest failed for asset {asset}: {e}")
                continue
        
        return performances
    
    def _run_transfer_learning_backtests(
        self,
        strategy: BaseStrategy,
        asset_data: Dict[str, pd.DataFrame]
    ) -> Tuple[List[AssetPerformance], Dict[str, Any]]:
        """Run transfer learning backtests (train on one asset, test on others)."""
        performances = []
        transfer_results = {}
        
        assets = list(asset_data.keys())
        
        for train_asset in assets:
            train_data = asset_data[train_asset]
            
            # Split training asset data
            try:
                train_split = self.data_splitter.chronological_split(train_data)
                
                # For now, we'll use the given strategy parameters
                # In a full implementation, you might optimize parameters on train_split.train
                
                test_results = {}
                
                for test_asset in assets:
                    if test_asset == train_asset:
                        continue
                    
                    try:
                        test_data = asset_data[test_asset]
                        test_split = self.data_splitter.chronological_split(test_data)
                        
                        # Create strategy instance
                        strategy_copy = strategy.__class__(**strategy.parameters, **strategy.risk_params)
                        
                        # Test on test asset
                        results = self.backtesting_engine.backtest_strategy(
                            strategy_copy, test_split.test, test_asset
                        )
                        
                        test_results[test_asset] = {
                            'total_return': results.total_return,
                            'sharpe_ratio': results.sharpe_ratio,
                            'max_drawdown': results.max_drawdown,
                            'win_rate': results.win_rate
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Transfer learning test failed for {train_asset} -> {test_asset}: {e}")
                        continue
                
                transfer_results[train_asset] = test_results
                
            except Exception as e:
                self.logger.error(f"Transfer learning training failed for asset {train_asset}: {e}")
                continue
        
        # Also run regular backtests for comparison
        performances = self._run_out_of_sample_backtests(strategy, asset_data)
        
        return performances, transfer_results
    
    def _calculate_performance_summary(self, performances: List[AssetPerformance]) -> Dict[str, Any]:
        """Calculate summary statistics across all assets."""
        if not performances:
            return {}
        
        metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown', 
                  'win_rate', 'volatility', 'calmar_ratio', 'sortino_ratio', 
                  'profit_factor', 'expectancy']
        
        summary = {}
        
        for metric in metrics:
            values = [getattr(perf, metric) for perf in performances if not np.isnan(getattr(perf, metric))]
            
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_median'] = np.median(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
        
        # Additional summary metrics
        summary['total_assets'] = len(performances)
        summary['profitable_assets'] = sum(1 for p in performances if p.total_return > 0)
        summary['profitable_ratio'] = summary['profitable_assets'] / summary['total_assets']
        
        return summary
    
    def _calculate_generalization_score(self, performances: List[AssetPerformance]) -> float:
        """Calculate a generalization score based on consistency across assets."""
        if len(performances) < 2:
            return 0.0
        
        # Use Sharpe ratio as primary metric for generalization
        sharpe_ratios = [p.sharpe_ratio for p in performances if not np.isnan(p.sharpe_ratio)]
        
        if not sharpe_ratios:
            return 0.0
        
        # Calculate coefficient of variation (lower is better for consistency)
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        if mean_sharpe <= 0:
            return 0.0
        
        cv = std_sharpe / mean_sharpe
        
        # Convert to score (0-1, higher is better)
        # Good generalization: low CV and positive mean Sharpe
        consistency_score = max(0, 1 - cv)
        performance_score = max(0, min(1, mean_sharpe / 2))  # Normalize assuming good Sharpe is ~2
        
        generalization_score = (consistency_score + performance_score) / 2
        
        return generalization_score
    
    def _calculate_consistency_score(self, performances: List[AssetPerformance]) -> float:
        """Calculate consistency score based on multiple metrics."""
        if len(performances) < 2:
            return 0.0
        
        metrics = ['sharpe_ratio', 'calmar_ratio', 'win_rate']
        consistency_scores = []
        
        for metric in metrics:
            values = [getattr(p, metric) for p in performances if not np.isnan(getattr(p, metric))]
            
            if len(values) >= 2:
                # Calculate coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val > 0:
                    cv = std_val / mean_val
                    consistency_scores.append(max(0, 1 - cv))
        
        return np.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_robustness_metrics(
        self,
        performances: List[AssetPerformance],
        correlation_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate robustness metrics."""
        metrics = {}
        
        if not performances:
            return metrics
        
        # Performance stability
        sharpe_ratios = [p.sharpe_ratio for p in performances if not np.isnan(p.sharpe_ratio)]
        if sharpe_ratios:
            metrics['sharpe_stability'] = 1 - (np.std(sharpe_ratios) / max(abs(np.mean(sharpe_ratios)), 0.1))
        
        # Drawdown consistency
        drawdowns = [abs(p.max_drawdown) for p in performances]
        if drawdowns:
            metrics['drawdown_consistency'] = 1 - (np.std(drawdowns) / max(np.mean(drawdowns), 0.01))
        
        # Win rate stability
        win_rates = [p.win_rate for p in performances]
        if win_rates:
            metrics['win_rate_stability'] = 1 - (np.std(win_rates) / max(np.mean(win_rates), 0.1))
        
        # Correlation-adjusted performance
        if not correlation_matrix.empty and len(performances) > 1:
            # Calculate performance diversity score
            asset_names = [p.asset for p in performances]
            available_assets = [asset for asset in asset_names if asset in correlation_matrix.index]
            
            if len(available_assets) > 1:
                # Get correlations between available assets
                subset_corr = correlation_matrix.loc[available_assets, available_assets]
                avg_correlation = subset_corr.values[np.triu_indices_from(subset_corr.values, k=1)].mean()
                
                # Lower correlation means more diverse test
                metrics['asset_diversity'] = max(0, 1 - avg_correlation)
        
        return metrics
    
    def _group_assets_by_correlation(self, correlation_matrix: pd.DataFrame) -> Dict[str, List[str]]:
        """Group assets by correlation similarity."""
        if correlation_matrix.empty:
            return {}
        
        # Simple clustering based on correlation threshold
        assets = correlation_matrix.index.tolist()
        groups = {}
        assigned = set()
        group_id = 0
        
        for asset in assets:
            if asset in assigned:
                continue
            
            # Find highly correlated assets
            correlated_assets = [asset]
            correlations = correlation_matrix.loc[asset]
            
            for other_asset in assets:
                if (other_asset != asset and 
                    other_asset not in assigned and 
                    abs(correlations[other_asset]) >= self.correlation_threshold):
                    correlated_assets.append(other_asset)
            
            # Create group
            group_name = f"group_{group_id}"
            groups[group_name] = correlated_assets
            assigned.update(correlated_assets)
            group_id += 1
        
        return groups
    
    def create_performance_report(
        self,
        results: CrossAssetResults,
        save_path: Optional[str] = None
    ) -> str:
        """Create a comprehensive performance report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"CROSS-ASSET VALIDATION REPORT")
        report_lines.append(f"Strategy: {results.strategy_name}")
        report_lines.append(f"Assets Tested: {len(results.asset_performances)}")
        report_lines.append("=" * 80)
        
        # Overall Scores
        report_lines.append("\nOVERALL SCORES:")
        report_lines.append(f"Generalization Score: {results.generalization_score:.3f}")
        report_lines.append(f"Consistency Score: {results.consistency_score:.3f}")
        
        # Performance Summary
        summary = results.performance_summary
        report_lines.append("\nPERFORMANCE SUMMARY:")
        report_lines.append(f"Mean Annual Return: {summary.get('annual_return_mean', 0):.2%}")
        report_lines.append(f"Mean Sharpe Ratio: {summary.get('sharpe_ratio_mean', 0):.3f}")
        report_lines.append(f"Mean Max Drawdown: {summary.get('max_drawdown_mean', 0):.2%}")
        report_lines.append(f"Profitable Assets: {summary.get('profitable_assets', 0)}/{summary.get('total_assets', 0)}")
        
        # Individual Asset Performance
        report_lines.append("\nINDIVIDUAL ASSET PERFORMANCE:")
        df = results.get_performance_dataframe()
        
        for _, row in df.iterrows():
            report_lines.append(f"\n{row['asset']}:")
            report_lines.append(f"  Annual Return: {row['annual_return']:.2%}")
            report_lines.append(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            report_lines.append(f"  Max Drawdown: {row['max_drawdown']:.2%}")
            report_lines.append(f"  Win Rate: {row['win_rate']:.2%}")
            report_lines.append(f"  Total Trades: {row['total_trades']}")
        
        # Best and Worst Performers
        best_assets = results.get_best_assets('sharpe_ratio', 3)
        worst_assets = results.get_worst_assets('sharpe_ratio', 3)
        
        report_lines.append(f"\nBEST PERFORMERS (Sharpe Ratio): {', '.join(best_assets)}")
        report_lines.append(f"WORST PERFORMERS (Sharpe Ratio): {', '.join(worst_assets)}")
        
        # Robustness Metrics
        if results.robustness_metrics:
            report_lines.append("\nROBUSTNESS METRICS:")
            for metric, value in results.robustness_metrics.items():
                report_lines.append(f"{metric}: {value:.3f}")
        
        # Asset Groups
        if results.asset_groups:
            report_lines.append("\nASSET CORRELATION GROUPS:")
            for group_name, assets in results.asset_groups.items():
                report_lines.append(f"{group_name}: {', '.join(assets)}")
        
        # Transfer Learning Results
        if results.transfer_results:
            report_lines.append("\nTRANSFER LEARNING RESULTS:")
            for train_asset, test_results in results.transfer_results.items():
                report_lines.append(f"\nTrained on {train_asset}:")
                for test_asset, metrics in test_results.items():
                    report_lines.append(f"  {test_asset}: Sharpe {metrics['sharpe_ratio']:.3f}, Return {metrics['total_return']:.2%}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")
        
        return report
    
    def plot_performance_comparison(
        self,
        results: CrossAssetResults,
        save_path: Optional[str] = None
    ):
        """Create performance comparison plots."""
        df = results.get_performance_dataframe()
        
        if df.empty:
            self.logger.warning("No data to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Cross-Asset Performance: {results.strategy_name}', fontsize=16)
        
        # Sharpe Ratio comparison
        axes[0, 0].bar(df['asset'], df['sharpe_ratio'])
        axes[0, 0].set_title('Sharpe Ratio by Asset')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Annual Return comparison
        axes[0, 1].bar(df['asset'], df['annual_return'] * 100)
        axes[0, 1].set_title('Annual Return by Asset')
        axes[0, 1].set_ylabel('Annual Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Max Drawdown comparison
        axes[1, 0].bar(df['asset'], df['max_drawdown'] * 100)
        axes[1, 0].set_title('Max Drawdown by Asset')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Win Rate comparison
        axes[1, 1].bar(df['asset'], df['win_rate'] * 100)
        axes[1, 1].set_title('Win Rate by Asset')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Performance plots saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(
        self,
        results: CrossAssetResults,
        save_path: Optional[str] = None
    ):
        """Plot asset correlation heatmap."""
        if results.correlation_matrix.empty:
            self.logger.warning("No correlation data to plot")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            results.correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title(f'Asset Return Correlations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show() 