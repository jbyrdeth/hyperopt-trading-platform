"""
Anti-Overfitting Data Splitting Framework

This module provides enhanced data splitting capabilities specifically designed
to prevent overfitting in trading strategy development and optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import warnings
from scipy import stats
import logging

try:
    from ..validation.data_splitter import DataSplitter, DataSplit
    from ..strategies.backtesting_engine import BacktestResults
    from ..utils.logger import get_logger
except ImportError:
    from src.validation.data_splitter import DataSplitter, DataSplit
    from src.strategies.backtesting_engine import BacktestResults
    from src.utils.logger import get_logger


@dataclass
class OverfittingMetrics:
    """Metrics for detecting overfitting in strategy performance."""
    
    # Performance consistency metrics
    train_test_performance_ratio: float
    train_test_sharpe_ratio: float
    train_test_drawdown_ratio: float
    train_test_win_rate_ratio: float
    
    # Statistical significance
    performance_p_value: float
    returns_correlation: float
    
    # Stability metrics
    rolling_performance_stability: float
    parameter_sensitivity: float
    
    # Overfitting indicators
    overfitting_score: float  # 0-100, higher = more overfitted
    overfitting_risk: str  # "Low", "Medium", "High"
    
    # Detailed analysis
    warning_flags: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AntiOverfittingSplit:
    """Enhanced data split with anti-overfitting specific features."""
    
    # Standard splits
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    
    # Anti-overfitting specific splits
    optimization_set: pd.DataFrame  # For hyperparameter optimization
    final_validation_set: pd.DataFrame  # Final unseen validation
    
    # Metadata
    split_info: Dict[str, Any]
    overfitting_metrics: Optional[OverfittingMetrics] = None
    
    def __post_init__(self):
        """Validate the enhanced split after initialization."""
        total_original = self.split_info.get('original_length', 0)
        total_split = (len(self.train) + len(self.validation) + len(self.test) + 
                      len(self.optimization_set) + len(self.final_validation_set))
        
        if abs(total_split - total_original) > 1:  # Allow for small rounding differences
            warnings.warn(f"Data length mismatch: original={total_original}, split={total_split}")
    
    @property
    def optimization_ratio(self) -> float:
        """Get optimization set ratio."""
        total = (len(self.train) + len(self.validation) + len(self.test) + 
                len(self.optimization_set) + len(self.final_validation_set))
        return len(self.optimization_set) / total if total > 0 else 0.0
    
    @property
    def final_validation_ratio(self) -> float:
        """Get final validation set ratio."""
        total = (len(self.train) + len(self.validation) + len(self.test) + 
                len(self.optimization_set) + len(self.final_validation_set))
        return len(self.final_validation_set) / total if total > 0 else 0.0


class AntiOverfittingDataSplitter:
    """
    Enhanced data splitter with anti-overfitting measures.
    
    Provides specialized data splitting strategies designed to prevent overfitting
    during strategy development and hyperparameter optimization.
    """
    
    def __init__(
        self,
        train_ratio: float = 0.5,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        optimization_ratio: float = 0.1,
        final_validation_ratio: float = 0.1,
        min_periods_per_set: int = 252,  # Minimum 1 year of daily data
        gap_days: int = 30,  # Gap between sets to prevent data leakage
        max_optimization_iterations: int = 100,
        require_minimum_trades: bool = True,
        min_trades_per_year: int = 12
    ):
        """
        Initialize the AntiOverfittingDataSplitter.
        
        Args:
            train_ratio: Proportion for initial training
            validation_ratio: Proportion for validation during development
            test_ratio: Proportion for testing
            optimization_ratio: Proportion for hyperparameter optimization
            final_validation_ratio: Proportion for final unseen validation
            min_periods_per_set: Minimum periods required in each set
            gap_days: Days to skip between sets to prevent data leakage
            max_optimization_iterations: Maximum allowed optimization iterations
            require_minimum_trades: Whether to enforce minimum trade requirements
            min_trades_per_year: Minimum trades per year requirement
        """
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.optimization_ratio = optimization_ratio
        self.final_validation_ratio = final_validation_ratio
        self.min_periods_per_set = min_periods_per_set
        self.gap_days = gap_days
        self.max_optimization_iterations = max_optimization_iterations
        self.require_minimum_trades = require_minimum_trades
        self.min_trades_per_year = min_trades_per_year
        
        # Validate ratios
        total_ratio = (train_ratio + validation_ratio + test_ratio + 
                      optimization_ratio + final_validation_ratio)
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Initialize base splitter
        self.base_splitter = DataSplitter(
            train_ratio=0.7,  # Will be adjusted
            validation_ratio=0.15,
            test_ratio=0.15,
            min_periods_per_set=min_periods_per_set,
            ensure_temporal_order=True
        )
        
        self.logger = get_logger("anti_overfitting_splitter")
        self.optimization_count = 0
        
    def create_anti_overfitting_split(
        self,
        data: pd.DataFrame,
        strategy_name: str = "Unknown"
    ) -> AntiOverfittingSplit:
        """
        Create an anti-overfitting data split with multiple validation sets.
        
        Args:
            data: Historical market data
            strategy_name: Name of strategy for logging
            
        Returns:
            AntiOverfittingSplit with multiple validation sets
        """
        if len(data) < self.min_periods_per_set * 5:
            raise ValueError(f"Insufficient data: need at least {self.min_periods_per_set * 5} periods")
        
        self.logger.info(f"Creating anti-overfitting split for {strategy_name}")
        
        # Sort data chronologically
        data_sorted = data.sort_index()
        total_periods = len(data_sorted)
        
        # Calculate split points with gaps
        gap_periods = self._calculate_gap_periods(data_sorted)
        
        # Calculate actual split sizes accounting for gaps
        effective_periods = total_periods - (gap_periods * 4)  # 4 gaps between 5 sets
        
        train_size = int(effective_periods * self.train_ratio)
        val_size = int(effective_periods * self.validation_ratio)
        test_size = int(effective_periods * self.test_ratio)
        opt_size = int(effective_periods * self.optimization_ratio)
        final_val_size = effective_periods - train_size - val_size - test_size - opt_size
        
        # Create splits with gaps
        current_idx = 0
        
        # Training set
        train_end = current_idx + train_size
        train_data = data_sorted.iloc[current_idx:train_end].copy()
        current_idx = train_end + gap_periods
        
        # Validation set
        val_end = current_idx + val_size
        validation_data = data_sorted.iloc[current_idx:val_end].copy()
        current_idx = val_end + gap_periods
        
        # Test set
        test_end = current_idx + test_size
        test_data = data_sorted.iloc[current_idx:test_end].copy()
        current_idx = test_end + gap_periods
        
        # Optimization set
        opt_end = current_idx + opt_size
        optimization_data = data_sorted.iloc[current_idx:opt_end].copy()
        current_idx = opt_end + gap_periods
        
        # Final validation set
        final_validation_data = data_sorted.iloc[current_idx:current_idx + final_val_size].copy()
        
        # Validate minimum periods
        sets = [
            ("train", train_data),
            ("validation", validation_data),
            ("test", test_data),
            ("optimization", optimization_data),
            ("final_validation", final_validation_data)
        ]
        
        for name, dataset in sets:
            if len(dataset) < self.min_periods_per_set:
                raise ValueError(f"{name} set too small: {len(dataset)} < {self.min_periods_per_set}")
        
        split_info = {
            'method': 'anti_overfitting',
            'strategy_name': strategy_name,
            'original_length': len(data),
            'gap_days': self.gap_days,
            'gap_periods': gap_periods,
            'ratios': {
                'train': self.train_ratio,
                'validation': self.validation_ratio,
                'test': self.test_ratio,
                'optimization': self.optimization_ratio,
                'final_validation': self.final_validation_ratio
            },
            'created_at': datetime.now()
        }
        
        self.logger.info(f"Created split: train={len(train_data)}, val={len(validation_data)}, "
                        f"test={len(test_data)}, opt={len(optimization_data)}, "
                        f"final_val={len(final_validation_data)}")
        
        return AntiOverfittingSplit(
            train=train_data,
            validation=validation_data,
            test=test_data,
            optimization_set=optimization_data,
            final_validation_set=final_validation_data,
            split_info=split_info
        )
    
    def create_walk_forward_splits(
        self,
        data: pd.DataFrame,
        window_months: int = 6,
        step_months: int = 1,
        min_splits: int = 10
    ) -> List[AntiOverfittingSplit]:
        """
        Create walk-forward splits for robust validation.
        
        Args:
            data: Historical market data
            window_months: Size of each training window in months
            step_months: Step size between windows in months
            min_splits: Minimum number of splits to create
            
        Returns:
            List of AntiOverfittingSplit objects
        """
        self.logger.info(f"Creating walk-forward splits: window={window_months}m, step={step_months}m")
        
        data_sorted = data.sort_index()
        splits = []
        
        # Calculate window size in periods
        window_periods = self._months_to_periods(data_sorted, window_months)
        step_periods = self._months_to_periods(data_sorted, step_months)
        
        # Minimum data needed for each split
        min_periods_needed = window_periods + self.min_periods_per_set  # Training + test
        
        start_idx = 0
        while start_idx + min_periods_needed <= len(data_sorted):
            # Training window
            train_end = start_idx + window_periods
            train_data = data_sorted.iloc[start_idx:train_end].copy()
            
            # Test window (next period after training)
            test_start = train_end
            test_end = min(test_start + self.min_periods_per_set, len(data_sorted))
            test_data = data_sorted.iloc[test_start:test_end].copy()
            
            # Skip if test set is too small
            if len(test_data) < self.min_periods_per_set:
                break
            
            # Create smaller validation sets within training data
            train_split_point = int(len(train_data) * 0.8)
            train_subset = train_data.iloc[:train_split_point].copy()
            val_subset = train_data.iloc[train_split_point:].copy()
            
            # Create empty sets for consistency (not used in walk-forward)
            empty_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=data.columns)
            
            split_info = {
                'method': 'walk_forward',
                'split_number': len(splits) + 1,
                'window_months': window_months,
                'step_months': step_months,
                'original_length': len(data),
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1]
            }
            
            split = AntiOverfittingSplit(
                train=train_subset,
                validation=val_subset,
                test=test_data,
                optimization_set=empty_df,
                final_validation_set=empty_df,
                split_info=split_info
            )
            
            splits.append(split)
            start_idx += step_periods
        
        if len(splits) < min_splits:
            self.logger.warning(f"Only created {len(splits)} splits, requested {min_splits}")
        
        self.logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def _calculate_gap_periods(self, data: pd.DataFrame) -> int:
        """Calculate gap periods based on data frequency."""
        if len(data) < 2:
            return self.gap_days
        
        # Estimate data frequency
        time_diff = data.index[1] - data.index[0]
        periods_per_day = timedelta(days=1) / time_diff
        
        return max(1, int(self.gap_days * periods_per_day))
    
    def _months_to_periods(self, data: pd.DataFrame, months: int) -> int:
        """Convert months to periods based on data frequency."""
        if len(data) < 2:
            return months * 30  # Fallback
        
        # Estimate periods per month
        time_diff = data.index[1] - data.index[0]
        periods_per_day = timedelta(days=1) / time_diff
        periods_per_month = periods_per_day * 30.44  # Average days per month
        
        return max(self.min_periods_per_set, int(months * periods_per_month))


class OverfittingDetector:
    """
    Detector for identifying overfitting in trading strategies.
    
    Analyzes performance differences between training and testing sets
    to identify potential overfitting issues.
    """
    
    def __init__(
        self,
        performance_threshold: float = 0.3,  # 30% performance degradation threshold
        p_value_threshold: float = 0.05,
        stability_threshold: float = 0.5
    ):
        """
        Initialize the OverfittingDetector.
        
        Args:
            performance_threshold: Maximum acceptable performance degradation
            p_value_threshold: Statistical significance threshold
            stability_threshold: Minimum stability score required
        """
        self.performance_threshold = performance_threshold
        self.p_value_threshold = p_value_threshold
        self.stability_threshold = stability_threshold
        self.logger = get_logger("overfitting_detector")
    
    def detect_overfitting(
        self,
        train_results: BacktestResults,
        test_results: BacktestResults,
        validation_results: Optional[BacktestResults] = None
    ) -> OverfittingMetrics:
        """
        Detect overfitting by comparing training and testing performance.
        
        Args:
            train_results: Results from training set
            test_results: Results from test set
            validation_results: Optional validation set results
            
        Returns:
            OverfittingMetrics with detailed analysis
        """
        self.logger.info("Analyzing overfitting indicators")
        
        # Calculate performance ratios
        train_test_performance_ratio = self._safe_ratio(
            test_results.total_return, train_results.total_return
        )
        train_test_sharpe_ratio = self._safe_ratio(
            test_results.sharpe_ratio, train_results.sharpe_ratio
        )
        train_test_drawdown_ratio = self._safe_ratio(
            abs(test_results.max_drawdown), abs(train_results.max_drawdown)
        )
        train_test_win_rate_ratio = self._safe_ratio(
            test_results.win_rate, train_results.win_rate
        )
        
        # Statistical significance testing
        train_returns = train_results.equity_curve.pct_change().dropna()
        test_returns = test_results.equity_curve.pct_change().dropna()
        
        # T-test for mean difference
        try:
            _, p_value = stats.ttest_ind(train_returns, test_returns)
        except:
            p_value = 1.0
        
        # Correlation between train and test returns
        try:
            # Align returns by date if possible
            common_dates = train_returns.index.intersection(test_returns.index)
            if len(common_dates) > 10:
                correlation = train_returns[common_dates].corr(test_returns[common_dates])
            else:
                correlation = 0.0
        except:
            correlation = 0.0
        
        # Rolling performance stability
        stability = self._calculate_stability(train_returns, test_returns)
        
        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(
            train_test_performance_ratio,
            train_test_sharpe_ratio,
            p_value,
            stability
        )
        
        # Determine risk level
        if overfitting_score >= 70:
            risk_level = "High"
        elif overfitting_score >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Generate warnings and recommendations
        warnings, recommendations = self._generate_warnings_and_recommendations(
            train_test_performance_ratio,
            train_test_sharpe_ratio,
            p_value,
            stability,
            overfitting_score
        )
        
        return OverfittingMetrics(
            train_test_performance_ratio=train_test_performance_ratio,
            train_test_sharpe_ratio=train_test_sharpe_ratio,
            train_test_drawdown_ratio=train_test_drawdown_ratio,
            train_test_win_rate_ratio=train_test_win_rate_ratio,
            performance_p_value=p_value,
            returns_correlation=correlation,
            rolling_performance_stability=stability,
            parameter_sensitivity=0.0,  # Would need parameter sweep data
            overfitting_score=overfitting_score,
            overfitting_risk=risk_level,
            warning_flags=warnings,
            recommendations=recommendations
        )
    
    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        """Calculate ratio safely, handling edge cases."""
        if abs(denominator) < 1e-8:
            return 0.0 if abs(numerator) < 1e-8 else float('inf')
        return numerator / denominator
    
    def _calculate_stability(self, train_returns: pd.Series, test_returns: pd.Series) -> float:
        """Calculate performance stability between train and test sets."""
        try:
            # Calculate rolling metrics for both sets
            window = min(30, len(train_returns) // 4, len(test_returns) // 4)
            if window < 5:
                return 0.5  # Default moderate stability
            
            train_rolling_sharpe = train_returns.rolling(window).mean() / train_returns.rolling(window).std()
            test_rolling_sharpe = test_returns.rolling(window).mean() / test_returns.rolling(window).std()
            
            # Calculate stability as inverse of coefficient of variation
            train_cv = train_rolling_sharpe.std() / abs(train_rolling_sharpe.mean()) if train_rolling_sharpe.mean() != 0 else 1
            test_cv = test_rolling_sharpe.std() / abs(test_rolling_sharpe.mean()) if test_rolling_sharpe.mean() != 0 else 1
            
            # Average stability (lower CV = higher stability)
            avg_cv = (train_cv + test_cv) / 2
            stability = max(0, min(1, 1 - avg_cv))
            
            return stability
        except:
            return 0.5  # Default moderate stability
    
    def _calculate_overfitting_score(
        self,
        performance_ratio: float,
        sharpe_ratio: float,
        p_value: float,
        stability: float
    ) -> float:
        """Calculate overall overfitting score (0-100)."""
        
        score = 0
        
        # Performance degradation component (0-40 points)
        if performance_ratio < 0.5:  # >50% degradation
            score += 40
        elif performance_ratio < 0.7:  # 30-50% degradation
            score += 30
        elif performance_ratio < 0.9:  # 10-30% degradation
            score += 20
        elif performance_ratio < 1.0:  # Some degradation
            score += 10
        
        # Sharpe ratio degradation (0-30 points)
        if sharpe_ratio < 0.3:
            score += 30
        elif sharpe_ratio < 0.5:
            score += 20
        elif sharpe_ratio < 0.8:
            score += 10
        
        # Statistical significance (0-20 points)
        if p_value < 0.01:  # Highly significant difference
            score += 20
        elif p_value < 0.05:  # Significant difference
            score += 15
        elif p_value < 0.1:  # Marginally significant
            score += 10
        
        # Stability component (0-10 points)
        if stability < 0.3:
            score += 10
        elif stability < 0.5:
            score += 5
        
        return min(100, score)
    
    def _generate_warnings_and_recommendations(
        self,
        performance_ratio: float,
        sharpe_ratio: float,
        p_value: float,
        stability: float,
        overfitting_score: float
    ) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations based on analysis."""
        
        warnings = []
        recommendations = []
        
        # Performance warnings
        if performance_ratio < 0.5:
            warnings.append("Severe performance degradation from training to testing (>50%)")
            recommendations.append("Reduce model complexity and re-validate with fresh data")
        elif performance_ratio < 0.7:
            warnings.append("Significant performance degradation from training to testing")
            recommendations.append("Consider simplifying strategy parameters")
        
        # Sharpe ratio warnings
        if sharpe_ratio < 0.5:
            warnings.append("Risk-adjusted returns significantly worse in testing")
            recommendations.append("Review risk management and position sizing")
        
        # Statistical significance
        if p_value < 0.05:
            warnings.append("Statistically significant difference between train/test performance")
            recommendations.append("Increase out-of-sample testing period")
        
        # Stability warnings
        if stability < 0.3:
            warnings.append("Low performance stability across time periods")
            recommendations.append("Test strategy across different market regimes")
        
        # Overall recommendations
        if overfitting_score >= 70:
            recommendations.append("High overfitting risk - consider complete strategy redesign")
        elif overfitting_score >= 40:
            recommendations.append("Moderate overfitting risk - implement additional validation")
        
        return warnings, recommendations 