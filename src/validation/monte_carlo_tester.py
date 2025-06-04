"""
Monte Carlo Simulation for Trading Strategy Testing

This module provides functionality to test trading strategies across randomly
selected time periods to assess consistency and robustness.
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
import random

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
class MonteCarloRun:
    """Results from a single Monte Carlo simulation run."""
    run_id: int
    start_date: datetime
    end_date: datetime
    period_length: int
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    profit_factor: float
    expectancy: float
    avg_trade_duration: float
    
    # Raw results for detailed analysis
    backtest_results: BacktestResults
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy analysis."""
        return {
            'run_id': self.run_id,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'period_length': self.period_length,
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
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    strategy_name: str
    asset_name: str
    n_simulations: int
    period_length: int
    simulation_runs: List[MonteCarloRun]
    
    # Statistical summary
    performance_stats: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Dict[str, float]]
    
    # Distribution analysis
    distribution_tests: Dict[str, Dict[str, Any]]
    
    # Bootstrap results
    bootstrap_results: Optional[Dict[str, Any]] = None
    
    def get_performance_dataframe(self) -> pd.DataFrame:
        """Get performance metrics as a DataFrame."""
        return pd.DataFrame([run.to_dict() for run in self.simulation_runs])
    
    def get_metric_distribution(self, metric: str) -> np.ndarray:
        """Get distribution of a specific metric."""
        df = self.get_performance_dataframe()
        return df[metric].values
    
    def get_percentile(self, metric: str, percentile: float) -> float:
        """Get percentile value for a metric."""
        values = self.get_metric_distribution(metric)
        return np.percentile(values, percentile)
    
    def get_probability_positive(self, metric: str = 'total_return') -> float:
        """Get probability of positive performance."""
        values = self.get_metric_distribution(metric)
        return np.mean(values > 0)


class MonteCarloTester:
    """
    Monte Carlo simulation framework for trading strategy testing.
    
    Tests strategies across randomly selected time periods to assess
    consistency, robustness, and statistical significance.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        cost_model: Optional[CostModel] = None,
        random_seed: Optional[int] = None,
        parallel_execution: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize the MonteCarloTester.
        
        Args:
            initial_capital: Starting capital for each simulation
            cost_model: Cost model for transaction costs
            random_seed: Random seed for reproducible results
            parallel_execution: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.random_seed = random_seed
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers
        
        self.logger = get_logger("monte_carlo_tester")
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(
            initial_capital=initial_capital,
            cost_model=self.cost_model
        )
    
    def run_simulation(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        asset_name: str = "Asset",
        n_simulations: int = 100,
        period_length: Optional[int] = None,
        min_period_length: int = 252,  # 1 year of daily data
        max_period_length: Optional[int] = None,
        overlap_allowed: bool = True,
        confidence_levels: List[float] = [0.90, 0.95, 0.99]
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation on a strategy.
        
        Args:
            strategy: Strategy to test
            data: Historical OHLCV data
            asset_name: Name of the asset being tested
            n_simulations: Number of simulation runs
            period_length: Fixed period length (if None, will vary randomly)
            min_period_length: Minimum period length for random periods
            max_period_length: Maximum period length for random periods
            overlap_allowed: Whether periods can overlap
            confidence_levels: Confidence levels for intervals
            
        Returns:
            MonteCarloResults object with comprehensive simulation results
        """
        self.logger.info(f"Starting Monte Carlo simulation for {strategy.name} on {asset_name}")
        self.logger.info(f"Simulations: {n_simulations}, Period length: {period_length or 'variable'}")
        
        # Validate input data
        if len(data) < min_period_length:
            raise ValueError(f"Insufficient data: {len(data)} < {min_period_length}")
        
        # Set default max period length
        if max_period_length is None:
            max_period_length = min(len(data) // 2, 1000)  # Half of data or 1000 periods
        
        # Generate random periods
        periods = self._generate_random_periods(
            data=data,
            n_simulations=n_simulations,
            period_length=period_length,
            min_period_length=min_period_length,
            max_period_length=max_period_length,
            overlap_allowed=overlap_allowed
        )
        
        # Run simulations
        simulation_runs = self._run_simulations(strategy, data, periods, asset_name)
        
        # Calculate statistical summary
        performance_stats = self._calculate_performance_statistics(simulation_runs)
        confidence_intervals = self._calculate_confidence_intervals(simulation_runs, confidence_levels)
        distribution_tests = self._perform_distribution_tests(simulation_runs)
        
        # Perform bootstrap analysis
        bootstrap_results = self._perform_bootstrap_analysis(simulation_runs)
        
        results = MonteCarloResults(
            strategy_name=strategy.name,
            asset_name=asset_name,
            n_simulations=len(simulation_runs),
            period_length=period_length or -1,  # -1 indicates variable length
            simulation_runs=simulation_runs,
            performance_stats=performance_stats,
            confidence_intervals=confidence_intervals,
            distribution_tests=distribution_tests,
            bootstrap_results=bootstrap_results
        )
        
        self.logger.info(f"Monte Carlo simulation completed. {len(simulation_runs)} successful runs")
        
        return results
    
    def _generate_random_periods(
        self,
        data: pd.DataFrame,
        n_simulations: int,
        period_length: Optional[int],
        min_period_length: int,
        max_period_length: int,
        overlap_allowed: bool
    ) -> List[Tuple[int, int, int]]:
        """Generate random time periods for simulation."""
        periods = []
        used_ranges = []
        
        for i in range(n_simulations):
            max_attempts = 1000  # Prevent infinite loops
            attempts = 0
            
            while attempts < max_attempts:
                # Determine period length
                if period_length is not None:
                    length = period_length
                else:
                    length = np.random.randint(min_period_length, max_period_length + 1)
                
                # Ensure we have enough data
                if length >= len(data):
                    length = len(data) - 1
                
                # Random start position
                max_start = len(data) - length
                if max_start <= 0:
                    break
                
                start_idx = np.random.randint(0, max_start)
                end_idx = start_idx + length
                
                # Check for overlap if not allowed
                if not overlap_allowed:
                    overlap = any(
                        not (end_idx <= used_start or start_idx >= used_end)
                        for used_start, used_end, _ in used_ranges
                    )
                    if overlap:
                        attempts += 1
                        continue
                
                periods.append((start_idx, end_idx, length))
                used_ranges.append((start_idx, end_idx, length))
                break
                
            else:
                self.logger.warning(f"Could not generate non-overlapping period for simulation {i}")
        
        self.logger.info(f"Generated {len(periods)} random periods")
        return periods
    
    def _run_simulations(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        periods: List[Tuple[int, int, int]],
        asset_name: str
    ) -> List[MonteCarloRun]:
        """Run simulations for all periods."""
        
        def run_single_simulation(run_data):
            run_id, (start_idx, end_idx, length) = run_data
            
            try:
                # Extract period data
                period_data = data.iloc[start_idx:end_idx].copy()
                
                if len(period_data) < strategy.required_periods:
                    return None
                
                # Create fresh strategy instance
                strategy_copy = strategy.__class__(**strategy.parameters, **strategy.risk_params)
                
                # Run backtest
                results = self.backtesting_engine.backtest_strategy(
                    strategy_copy, period_data, f"{asset_name}_MC_{run_id}"
                )
                
                return MonteCarloRun(
                    run_id=run_id,
                    start_date=period_data.index[0],
                    end_date=period_data.index[-1],
                    period_length=length,
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
                self.logger.error(f"Simulation {run_id} failed: {e}")
                return None
        
        simulation_runs = []
        
        if self.parallel_execution and len(periods) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_run = {
                    executor.submit(run_single_simulation, (i, period)): i 
                    for i, period in enumerate(periods)
                }
                
                for future in as_completed(future_to_run):
                    result = future.result()
                    if result is not None:
                        simulation_runs.append(result)
        else:
            # Sequential execution
            for i, period in enumerate(periods):
                result = run_single_simulation((i, period))
                if result is not None:
                    simulation_runs.append(result)
        
        return simulation_runs
    
    def _calculate_performance_statistics(
        self,
        simulation_runs: List[MonteCarloRun]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistical summary of performance metrics."""
        if not simulation_runs:
            return {}
        
        metrics = [
            'total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'volatility', 'calmar_ratio', 'sortino_ratio',
            'profit_factor', 'expectancy', 'total_trades', 'avg_trade_duration'
        ]
        
        stats_summary = {}
        
        for metric in metrics:
            values = [getattr(run, metric) for run in simulation_runs if not np.isnan(getattr(run, metric))]
            
            if values:
                stats_summary[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'iqr': np.percentile(values, 75) - np.percentile(values, 25)
                }
        
        return stats_summary
    
    def _calculate_confidence_intervals(
        self,
        simulation_runs: List[MonteCarloRun],
        confidence_levels: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for key metrics."""
        if not simulation_runs:
            return {}
        
        key_metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']
        intervals = {}
        
        for metric in key_metrics:
            values = [getattr(run, metric) for run in simulation_runs if not np.isnan(getattr(run, metric))]
            
            if values:
                intervals[metric] = {}
                for confidence in confidence_levels:
                    alpha = 1 - confidence
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    
                    intervals[metric][f'ci_{int(confidence*100)}'] = {
                        'lower': np.percentile(values, lower_percentile),
                        'upper': np.percentile(values, upper_percentile)
                    }
        
        return intervals
    
    def _perform_distribution_tests(
        self,
        simulation_runs: List[MonteCarloRun]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical tests on metric distributions."""
        if not simulation_runs:
            return {}
        
        key_metrics = ['total_return', 'annual_return', 'sharpe_ratio']
        test_results = {}
        
        for metric in key_metrics:
            values = [getattr(run, metric) for run in simulation_runs if not np.isnan(getattr(run, metric))]
            
            if len(values) >= 8:  # Minimum for meaningful tests
                test_results[metric] = {}
                
                # Normality test (Shapiro-Wilk)
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(values)
                    test_results[metric]['normality'] = {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                except Exception as e:
                    self.logger.warning(f"Normality test failed for {metric}: {e}")
                
                # One-sample t-test (test if mean is significantly different from 0)
                try:
                    t_stat, t_p = stats.ttest_1samp(values, 0)
                    test_results[metric]['t_test'] = {
                        'statistic': t_stat,
                        'p_value': t_p,
                        'significantly_positive': t_p < 0.05 and t_stat > 0,
                        'significantly_negative': t_p < 0.05 and t_stat < 0
                    }
                except Exception as e:
                    self.logger.warning(f"T-test failed for {metric}: {e}")
                
                # Jarque-Bera test for normality
                try:
                    jb_stat, jb_p = stats.jarque_bera(values)
                    test_results[metric]['jarque_bera'] = {
                        'statistic': jb_stat,
                        'p_value': jb_p,
                        'is_normal': jb_p > 0.05
                    }
                except Exception as e:
                    self.logger.warning(f"Jarque-Bera test failed for {metric}: {e}")
        
        return test_results
    
    def _perform_bootstrap_analysis(
        self,
        simulation_runs: List[MonteCarloRun],
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """Perform bootstrap resampling analysis."""
        if not simulation_runs or len(simulation_runs) < 10:
            return {}
        
        key_metrics = ['total_return', 'annual_return', 'sharpe_ratio']
        bootstrap_results = {}
        
        for metric in key_metrics:
            values = np.array([getattr(run, metric) for run in simulation_runs if not np.isnan(getattr(run, metric))])
            
            if len(values) >= 10:
                # Bootstrap resampling
                bootstrap_means = []
                bootstrap_stds = []
                
                for _ in range(n_bootstrap):
                    # Resample with replacement
                    bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                    bootstrap_stds.append(np.std(bootstrap_sample))
                
                bootstrap_results[metric] = {
                    'bootstrap_mean_distribution': bootstrap_means,
                    'bootstrap_std_distribution': bootstrap_stds,
                    'mean_ci_95': {
                        'lower': np.percentile(bootstrap_means, 2.5),
                        'upper': np.percentile(bootstrap_means, 97.5)
                    },
                    'std_ci_95': {
                        'lower': np.percentile(bootstrap_stds, 2.5),
                        'upper': np.percentile(bootstrap_stds, 97.5)
                    }
                }
        
        return bootstrap_results
    
    def create_simulation_report(
        self,
        results: MonteCarloResults,
        save_path: Optional[str] = None
    ) -> str:
        """Create a comprehensive simulation report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"MONTE CARLO SIMULATION REPORT")
        report_lines.append(f"Strategy: {results.strategy_name}")
        report_lines.append(f"Asset: {results.asset_name}")
        report_lines.append(f"Simulations: {results.n_simulations}")
        report_lines.append(f"Period Length: {results.period_length if results.period_length > 0 else 'Variable'}")
        report_lines.append("=" * 80)
        
        # Performance Statistics
        report_lines.append("\nPERFORMANCE STATISTICS:")
        for metric, stats in results.performance_stats.items():
            if metric in ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']:
                report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
                report_lines.append(f"  Mean: {stats['mean']:.4f}")
                report_lines.append(f"  Median: {stats['median']:.4f}")
                report_lines.append(f"  Std Dev: {stats['std']:.4f}")
                report_lines.append(f"  Min: {stats['min']:.4f}")
                report_lines.append(f"  Max: {stats['max']:.4f}")
                report_lines.append(f"  Skewness: {stats['skewness']:.4f}")
                report_lines.append(f"  Kurtosis: {stats['kurtosis']:.4f}")
        
        # Confidence Intervals
        report_lines.append("\nCONFIDENCE INTERVALS:")
        for metric, intervals in results.confidence_intervals.items():
            report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
            for ci_level, bounds in intervals.items():
                level = ci_level.replace('ci_', '')
                report_lines.append(f"  {level}% CI: [{bounds['lower']:.4f}, {bounds['upper']:.4f}]")
        
        # Probability Analysis
        report_lines.append("\nPROBABILITY ANALYSIS:")
        prob_positive_return = results.get_probability_positive('total_return')
        prob_positive_sharpe = results.get_probability_positive('sharpe_ratio')
        
        report_lines.append(f"Probability of Positive Return: {prob_positive_return:.2%}")
        report_lines.append(f"Probability of Positive Sharpe: {prob_positive_sharpe:.2%}")
        
        # Distribution Tests
        if results.distribution_tests:
            report_lines.append("\nDISTRIBUTION TESTS:")
            for metric, tests in results.distribution_tests.items():
                report_lines.append(f"\n{metric.replace('_', ' ').title()}:")
                
                if 'normality' in tests:
                    norm_test = tests['normality']
                    report_lines.append(f"  Normality (Shapiro-Wilk): p={norm_test['p_value']:.4f} ({'Normal' if norm_test['is_normal'] else 'Non-normal'})")
                
                if 't_test' in tests:
                    t_test = tests['t_test']
                    significance = "Significantly positive" if t_test['significantly_positive'] else \
                                 "Significantly negative" if t_test['significantly_negative'] else \
                                 "Not significant"
                    report_lines.append(f"  T-test vs 0: p={t_test['p_value']:.4f} ({significance})")
        
        # Bootstrap Results Summary
        if results.bootstrap_results:
            report_lines.append("\nBOOTSTRAP ANALYSIS:")
            for metric, bootstrap in results.bootstrap_results.items():
                if 'mean_ci_95' in bootstrap:
                    ci = bootstrap['mean_ci_95']
                    report_lines.append(f"{metric.replace('_', ' ').title()} Mean 95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")
        
        return report
    
    def plot_distribution_analysis(
        self,
        results: MonteCarloResults,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ):
        """Create distribution plots for Monte Carlo results."""
        if not results.simulation_runs:
            self.logger.warning("No simulation data to plot")
            return
        
        if metrics is None:
            metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']
        
        # Filter metrics that exist in the data
        available_metrics = [m for m in metrics if hasattr(results.simulation_runs[0], m)]
        
        if not available_metrics:
            self.logger.warning("No valid metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Monte Carlo Distribution Analysis: {results.strategy_name}', fontsize=16)
        
        for i, metric in enumerate(available_metrics):
            values = results.get_metric_distribution(metric)
            
            # Histogram with KDE
            axes[i].hist(values, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add KDE if we have enough data points
            if len(values) > 10:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 100)
                    axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                except:
                    pass
            
            # Add vertical lines for mean and median
            mean_val = np.mean(values)
            median_val = np.median(values)
            axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[i].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(metric.replace("_", " ").title())
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(available_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Distribution plots saved to {save_path}")
        
        plt.show()
    
    def plot_performance_timeline(
        self,
        results: MonteCarloResults,
        save_path: Optional[str] = None
    ):
        """Plot performance metrics over time periods."""
        if not results.simulation_runs:
            self.logger.warning("No simulation data to plot")
            return
        
        df = results.get_performance_dataframe()
        
        # Sort by start date
        df = df.sort_values('start_date')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Monte Carlo Performance Timeline: {results.strategy_name}', fontsize=16)
        
        # Total Return over time
        axes[0, 0].scatter(df['start_date'], df['total_return'], alpha=0.6, color='blue')
        axes[0, 0].set_title('Total Return by Period Start Date')
        axes[0, 0].set_ylabel('Total Return')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sharpe Ratio over time
        axes[0, 1].scatter(df['start_date'], df['sharpe_ratio'], alpha=0.6, color='green')
        axes[0, 1].set_title('Sharpe Ratio by Period Start Date')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max Drawdown over time
        axes[1, 0].scatter(df['start_date'], df['max_drawdown'], alpha=0.6, color='red')
        axes[1, 0].set_title('Max Drawdown by Period Start Date')
        axes[1, 0].set_ylabel('Max Drawdown')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win Rate over time
        axes[1, 1].scatter(df['start_date'], df['win_rate'], alpha=0.6, color='orange')
        axes[1, 1].set_title('Win Rate by Period Start Date')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Timeline plots saved to {save_path}")
        
        plt.show() 