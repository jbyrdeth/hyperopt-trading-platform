"""
Parameter Optimization Constraints

This module provides comprehensive constraints on parameter optimization to reduce
the risk of curve fitting and overfitting in trading strategy development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
import json
import os
import hashlib
from scipy import stats
from scipy.stats import pearsonr
import logging
import warnings

try:
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from ..optimization.hyperopt_optimizer import HyperoptOptimizer, OptimizationResult
    from ..utils.logger import get_logger
except ImportError:
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.optimization.hyperopt_optimizer import HyperoptOptimizer, OptimizationResult
    from src.utils.logger import get_logger


@dataclass
class ParameterImportance:
    """Parameter importance analysis results."""
    parameter_name: str
    importance_score: float  # 0-1, higher = more important
    sensitivity: float  # How much performance changes with parameter
    stability: float  # How stable the parameter effect is across time
    correlation_with_performance: float
    recommended_for_optimization: bool
    
    # Detailed analysis
    value_range_tested: Tuple[float, float]
    optimal_value: float
    performance_impact: float  # % change in performance
    
    # Risk indicators
    overfitting_risk: str  # "Low", "Medium", "High"
    warning_flags: List[str] = field(default_factory=list)


@dataclass
class OptimizationBudget:
    """Optimization budget tracking."""
    strategy_name: str
    max_iterations: int
    used_iterations: int
    max_parameters: int
    current_parameters: int
    max_optimization_time_hours: float
    used_optimization_time_hours: float
    
    # Budget status
    iterations_remaining: int = 0
    parameters_remaining: int = 0
    time_remaining_hours: float = 0.0
    budget_exhausted: bool = False
    
    def __post_init__(self):
        self.iterations_remaining = max(0, self.max_iterations - self.used_iterations)
        self.parameters_remaining = max(0, self.max_parameters - self.current_parameters)
        self.time_remaining_hours = max(0, self.max_optimization_time_hours - self.used_optimization_time_hours)
        self.budget_exhausted = (
            self.iterations_remaining <= 0 or 
            self.parameters_remaining <= 0 or 
            self.time_remaining_hours <= 0
        )


@dataclass
class ParameterConstraints:
    """Parameter constraints configuration."""
    max_parameters: int = 5  # Maximum parameters to optimize simultaneously
    max_iterations_per_parameter: int = 50  # Max iterations per parameter
    max_total_iterations: int = 200  # Total optimization budget
    max_optimization_time_hours: float = 24.0  # Maximum optimization time
    
    # Parameter complexity limits
    max_parameter_ranges: int = 10  # Max discrete values per parameter
    min_parameter_impact: float = 0.05  # Minimum 5% performance impact to include
    
    # Overfitting prevention
    require_parameter_stability: bool = True
    min_stability_score: float = 0.6  # Minimum stability across time periods
    max_correlation_between_parameters: float = 0.8  # Max correlation between params
    
    # Performance consistency requirements
    min_consistency_across_periods: float = 0.5
    max_performance_variance: float = 0.3
    
    # Early stopping criteria
    enable_early_stopping: bool = True
    early_stop_patience: int = 20  # Iterations without improvement
    min_improvement_threshold: float = 0.01  # Minimum improvement to continue


class BacktestTracker:
    """
    Tracks backtest iterations and optimization attempts to prevent excessive optimization.
    """
    
    def __init__(self, tracking_file: str = "optimization_tracking.json"):
        """
        Initialize the BacktestTracker.
        
        Args:
            tracking_file: File to store tracking data
        """
        self.tracking_file = tracking_file
        self.logger = get_logger("backtest_tracker")
        
        # Load existing tracking data
        self.tracking_data = self._load_tracking_data()
        
    def _load_tracking_data(self) -> Dict[str, Any]:
        """Load tracking data from file."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load tracking data: {e}")
        
        return {
            'strategies': {},
            'global_stats': {
                'total_backtests': 0,
                'total_optimization_time': 0.0,
                'created_at': datetime.now().isoformat()
            }
        }
    
    def _save_tracking_data(self):
        """Save tracking data to file."""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracking_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save tracking data: {e}")
    
    def _get_strategy_key(self, strategy_name: str, data_hash: str) -> str:
        """Generate unique key for strategy + data combination."""
        return f"{strategy_name}_{data_hash[:8]}"
    
    def record_backtest(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        data_hash: str,
        execution_time: float,
        performance_score: float
    ):
        """
        Record a backtest execution.
        
        Args:
            strategy_name: Name of the strategy
            parameters: Parameters used in backtest
            data_hash: Hash of the data used
            execution_time: Time taken for backtest
            performance_score: Performance score achieved
        """
        strategy_key = self._get_strategy_key(strategy_name, data_hash)
        
        if strategy_key not in self.tracking_data['strategies']:
            self.tracking_data['strategies'][strategy_key] = {
                'strategy_name': strategy_name,
                'data_hash': data_hash,
                'backtests': [],
                'total_iterations': 0,
                'total_time': 0.0,
                'best_score': float('-inf'),
                'best_parameters': {},
                'created_at': datetime.now().isoformat()
            }
        
        strategy_data = self.tracking_data['strategies'][strategy_key]
        
        # Record backtest
        backtest_record = {
            'parameters': parameters,
            'execution_time': execution_time,
            'performance_score': performance_score,
            'timestamp': datetime.now().isoformat()
        }
        
        strategy_data['backtests'].append(backtest_record)
        strategy_data['total_iterations'] += 1
        strategy_data['total_time'] += execution_time
        
        # Update best score
        if performance_score > strategy_data['best_score']:
            strategy_data['best_score'] = performance_score
            strategy_data['best_parameters'] = parameters.copy()
        
        # Update global stats
        self.tracking_data['global_stats']['total_backtests'] += 1
        self.tracking_data['global_stats']['total_optimization_time'] += execution_time
        
        self._save_tracking_data()
        
        self.logger.debug(f"Recorded backtest for {strategy_name}: score={performance_score:.4f}")
    
    def get_optimization_budget(
        self,
        strategy_name: str,
        data_hash: str,
        constraints: ParameterConstraints
    ) -> OptimizationBudget:
        """
        Get current optimization budget for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            data_hash: Hash of the data
            constraints: Parameter constraints configuration
            
        Returns:
            OptimizationBudget with current usage and remaining budget
        """
        strategy_key = self._get_strategy_key(strategy_name, data_hash)
        
        if strategy_key in self.tracking_data['strategies']:
            strategy_data = self.tracking_data['strategies'][strategy_key]
            used_iterations = strategy_data['total_iterations']
            used_time = strategy_data['total_time'] / 3600.0  # Convert to hours
        else:
            used_iterations = 0
            used_time = 0.0
        
        # Estimate current parameters (simplified)
        current_parameters = min(constraints.max_parameters, 3)  # Default estimate
        
        return OptimizationBudget(
            strategy_name=strategy_name,
            max_iterations=constraints.max_total_iterations,
            used_iterations=used_iterations,
            max_parameters=constraints.max_parameters,
            current_parameters=current_parameters,
            max_optimization_time_hours=constraints.max_optimization_time_hours,
            used_optimization_time_hours=used_time
        )
    
    def check_budget_available(
        self,
        strategy_name: str,
        data_hash: str,
        constraints: ParameterConstraints,
        requested_iterations: int = 1
    ) -> Tuple[bool, str]:
        """
        Check if optimization budget is available.
        
        Args:
            strategy_name: Name of the strategy
            data_hash: Hash of the data
            constraints: Parameter constraints
            requested_iterations: Number of iterations requested
            
        Returns:
            Tuple of (budget_available, reason_if_not)
        """
        budget = self.get_optimization_budget(strategy_name, data_hash, constraints)
        
        if budget.budget_exhausted:
            return False, "Optimization budget exhausted"
        
        if budget.iterations_remaining < requested_iterations:
            return False, f"Insufficient iterations remaining: {budget.iterations_remaining} < {requested_iterations}"
        
        return True, "Budget available"
    
    def get_strategy_statistics(self, strategy_name: str, data_hash: str) -> Dict[str, Any]:
        """Get statistics for a specific strategy."""
        strategy_key = self._get_strategy_key(strategy_name, data_hash)
        
        if strategy_key not in self.tracking_data['strategies']:
            return {
                'total_backtests': 0,
                'total_time': 0.0,
                'best_score': None,
                'average_score': None,
                'score_improvement': None
            }
        
        strategy_data = self.tracking_data['strategies'][strategy_key]
        backtests = strategy_data['backtests']
        
        if not backtests:
            return {
                'total_backtests': 0,
                'total_time': 0.0,
                'best_score': None,
                'average_score': None,
                'score_improvement': None
            }
        
        scores = [bt['performance_score'] for bt in backtests]
        
        # Calculate score improvement trend
        if len(scores) >= 2:
            recent_scores = scores[-10:]  # Last 10 scores
            early_scores = scores[:10]   # First 10 scores
            
            if len(early_scores) > 0 and len(recent_scores) > 0:
                score_improvement = np.mean(recent_scores) - np.mean(early_scores)
            else:
                score_improvement = 0.0
        else:
            score_improvement = 0.0
        
        return {
            'total_backtests': len(backtests),
            'total_time': strategy_data['total_time'],
            'best_score': strategy_data['best_score'],
            'average_score': np.mean(scores),
            'score_improvement': score_improvement,
            'score_std': np.std(scores),
            'recent_performance': scores[-5:] if len(scores) >= 5 else scores
        }


class ParameterImportanceAnalyzer:
    """
    Analyzes parameter importance and sensitivity to prioritize optimization efforts.
    """
    
    def __init__(self, backtesting_engine: BacktestingEngine):
        """
        Initialize the ParameterImportanceAnalyzer.
        
        Args:
            backtesting_engine: Backtesting engine for running tests
        """
        self.backtesting_engine = backtesting_engine
        self.logger = get_logger("parameter_importance_analyzer")
    
    def analyze_parameter_importance(
        self,
        strategy_class: type,
        parameter_space: Dict[str, Any],
        data: pd.DataFrame,
        n_samples: int = 50,
        time_splits: int = 3
    ) -> List[ParameterImportance]:
        """
        Analyze the importance of each parameter through sensitivity analysis.
        
        Args:
            strategy_class: Strategy class to analyze
            parameter_space: Parameter space definition
            data: Historical data for testing
            n_samples: Number of samples per parameter
            time_splits: Number of time periods to test stability
            
        Returns:
            List of ParameterImportance objects
        """
        self.logger.info(f"Analyzing parameter importance for {strategy_class.__name__}")
        
        parameter_importance = []
        
        # Split data into time periods for stability analysis
        time_periods = self._split_data_by_time(data, time_splits)
        
        for param_name, param_config in parameter_space.items():
            self.logger.info(f"Analyzing parameter: {param_name}")
            
            try:
                importance = self._analyze_single_parameter(
                    strategy_class, param_name, param_config, 
                    parameter_space, time_periods, n_samples
                )
                parameter_importance.append(importance)
                
            except Exception as e:
                self.logger.error(f"Failed to analyze parameter {param_name}: {e}")
                continue
        
        # Sort by importance score
        parameter_importance.sort(key=lambda x: x.importance_score, reverse=True)
        
        self.logger.info(f"Parameter importance analysis completed for {len(parameter_importance)} parameters")
        return parameter_importance
    
    def _split_data_by_time(self, data: pd.DataFrame, n_splits: int) -> List[pd.DataFrame]:
        """Split data into time periods."""
        split_size = len(data) // n_splits
        periods = []
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < n_splits - 1 else len(data)
            periods.append(data.iloc[start_idx:end_idx].copy())
        
        return periods
    
    def _analyze_single_parameter(
        self,
        strategy_class: type,
        param_name: str,
        param_config: Any,
        full_parameter_space: Dict[str, Any],
        time_periods: List[pd.DataFrame],
        n_samples: int
    ) -> ParameterImportance:
        """Analyze importance of a single parameter."""
        
        # Generate parameter values to test
        param_values = self._generate_parameter_values(param_config, n_samples)
        
        # Default parameters (use middle values for other parameters)
        default_params = self._get_default_parameters(full_parameter_space)
        
        # Test parameter across different values and time periods
        performance_results = []
        
        for period_idx, period_data in enumerate(time_periods):
            period_results = []
            
            for param_value in param_values:
                # Create test parameters
                test_params = default_params.copy()
                test_params[param_name] = param_value
                
                try:
                    # Create strategy instance
                    strategy = strategy_class(
                        name=f"test_{strategy_class.__name__}",
                        parameters=test_params
                    )
                    
                    # Run backtest
                    result = self.backtesting_engine.run_backtest(
                        strategy, period_data, initial_capital=10000.0
                    )
                    
                    period_results.append({
                        'param_value': param_value,
                        'performance': result.sharpe_ratio,  # Use Sharpe as primary metric
                        'total_return': result.total_return,
                        'max_drawdown': result.max_drawdown
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed backtest for {param_name}={param_value}: {e}")
                    continue
            
            performance_results.append(period_results)
        
        # Analyze results
        return self._calculate_parameter_importance(
            param_name, param_values, performance_results
        )
    
    def _generate_parameter_values(self, param_config: Any, n_samples: int) -> List[float]:
        """Generate parameter values to test based on hyperopt configuration."""
        # This is a simplified implementation
        # In practice, you'd need to handle different hyperopt distributions
        
        if hasattr(param_config, 'pos_args'):
            # Handle hyperopt uniform distribution
            if len(param_config.pos_args) >= 2:
                low, high = param_config.pos_args[0], param_config.pos_args[1]
                return np.linspace(low, high, n_samples).tolist()
        
        # Default range
        return np.linspace(0.1, 2.0, n_samples).tolist()
    
    def _get_default_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Get default parameter values (middle of ranges)."""
        defaults = {}
        
        for param_name, param_config in parameter_space.items():
            if hasattr(param_config, 'pos_args') and len(param_config.pos_args) >= 2:
                low, high = param_config.pos_args[0], param_config.pos_args[1]
                defaults[param_name] = (low + high) / 2
            else:
                defaults[param_name] = 1.0  # Default value
        
        return defaults
    
    def _calculate_parameter_importance(
        self,
        param_name: str,
        param_values: List[float],
        performance_results: List[List[Dict[str, Any]]]
    ) -> ParameterImportance:
        """Calculate parameter importance metrics."""
        
        if not performance_results or not any(performance_results):
            return ParameterImportance(
                parameter_name=param_name,
                importance_score=0.0,
                sensitivity=0.0,
                stability=0.0,
                correlation_with_performance=0.0,
                recommended_for_optimization=False,
                value_range_tested=(0.0, 1.0),
                optimal_value=0.5,
                performance_impact=0.0,
                overfitting_risk="High",
                warning_flags=["Insufficient data for analysis"]
            )
        
        # Aggregate performance across time periods
        all_performances = []
        all_param_values = []
        period_correlations = []
        
        for period_results in performance_results:
            if not period_results:
                continue
                
            period_performances = [r['performance'] for r in period_results]
            period_param_values = [r['param_value'] for r in period_results]
            
            all_performances.extend(period_performances)
            all_param_values.extend(period_param_values)
            
            # Calculate correlation for this period
            if len(period_performances) > 2:
                try:
                    corr, _ = pearsonr(period_param_values, period_performances)
                    if not np.isnan(corr):
                        period_correlations.append(corr)
                except:
                    pass
        
        if not all_performances:
            return ParameterImportance(
                parameter_name=param_name,
                importance_score=0.0,
                sensitivity=0.0,
                stability=0.0,
                correlation_with_performance=0.0,
                recommended_for_optimization=False,
                value_range_tested=(min(param_values), max(param_values)),
                optimal_value=np.median(param_values),
                performance_impact=0.0,
                overfitting_risk="High",
                warning_flags=["No valid performance data"]
            )
        
        # Calculate metrics
        performance_range = max(all_performances) - min(all_performances)
        performance_std = np.std(all_performances)
        
        # Sensitivity: how much performance varies with parameter
        sensitivity = performance_std / (np.mean(all_performances) + 1e-8)
        
        # Stability: consistency of correlation across time periods
        if len(period_correlations) > 1:
            stability = 1.0 - np.std(period_correlations)
        else:
            stability = 0.5  # Default moderate stability
        
        # Overall correlation
        if len(all_performances) > 2:
            try:
                overall_correlation, _ = pearsonr(all_param_values, all_performances)
                if np.isnan(overall_correlation):
                    overall_correlation = 0.0
            except:
                overall_correlation = 0.0
        else:
            overall_correlation = 0.0
        
        # Importance score (combination of sensitivity and stability)
        importance_score = (sensitivity * 0.6 + stability * 0.4) * min(1.0, abs(overall_correlation) * 2)
        
        # Find optimal value
        best_idx = np.argmax(all_performances)
        optimal_value = all_param_values[best_idx]
        
        # Performance impact
        performance_impact = performance_range / (np.mean(all_performances) + 1e-8)
        
        # Determine overfitting risk
        if stability < 0.3 or len(period_correlations) < 2:
            overfitting_risk = "High"
        elif stability < 0.6 or performance_impact > 1.0:
            overfitting_risk = "Medium"
        else:
            overfitting_risk = "Low"
        
        # Recommendation
        recommended = (
            importance_score > 0.3 and
            stability > 0.4 and
            abs(overall_correlation) > 0.2 and
            overfitting_risk != "High"
        )
        
        # Warning flags
        warnings = []
        if stability < 0.4:
            warnings.append("Low stability across time periods")
        if abs(overall_correlation) < 0.1:
            warnings.append("Weak correlation with performance")
        if performance_impact < 0.05:
            warnings.append("Minimal performance impact")
        if len(period_correlations) < 2:
            warnings.append("Insufficient time periods for stability analysis")
        
        return ParameterImportance(
            parameter_name=param_name,
            importance_score=importance_score,
            sensitivity=sensitivity,
            stability=stability,
            correlation_with_performance=overall_correlation,
            recommended_for_optimization=recommended,
            value_range_tested=(min(param_values), max(param_values)),
            optimal_value=optimal_value,
            performance_impact=performance_impact,
            overfitting_risk=overfitting_risk,
            warning_flags=warnings
        )


class ParameterConstraintManager:
    """
    Main manager for parameter optimization constraints and anti-overfitting measures.
    """
    
    def __init__(
        self,
        constraints: ParameterConstraints = None,
        backtesting_engine: BacktestingEngine = None,
        tracking_file: str = "optimization_tracking.json"
    ):
        """
        Initialize the ParameterConstraintManager.
        
        Args:
            constraints: Parameter constraints configuration
            backtesting_engine: Backtesting engine for analysis
            tracking_file: File for tracking optimization history
        """
        self.constraints = constraints or ParameterConstraints()
        self.backtesting_engine = backtesting_engine
        self.tracker = BacktestTracker(tracking_file)
        
        if backtesting_engine:
            self.importance_analyzer = ParameterImportanceAnalyzer(backtesting_engine)
        else:
            self.importance_analyzer = None
        
        self.logger = get_logger("parameter_constraint_manager")
    
    def validate_optimization_request(
        self,
        strategy_name: str,
        parameter_space: Dict[str, Any],
        data: pd.DataFrame,
        requested_iterations: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate an optimization request against constraints.
        
        Args:
            strategy_name: Name of the strategy
            parameter_space: Parameter space for optimization
            data: Historical data
            requested_iterations: Number of iterations requested
            
        Returns:
            Tuple of (is_valid, reason_if_invalid, recommendations)
        """
        self.logger.info(f"Validating optimization request for {strategy_name}")
        
        # Generate data hash
        data_hash = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()
        
        # Check budget availability
        budget_available, budget_reason = self.tracker.check_budget_available(
            strategy_name, data_hash, self.constraints, requested_iterations
        )
        
        if not budget_available:
            return False, budget_reason, {'action': 'reduce_iterations'}
        
        # Check parameter count
        if len(parameter_space) > self.constraints.max_parameters:
            return False, f"Too many parameters: {len(parameter_space)} > {self.constraints.max_parameters}", {
                'action': 'reduce_parameters',
                'current_count': len(parameter_space),
                'max_allowed': self.constraints.max_parameters
            }
        
        # Check parameter complexity
        complexity_issues = self._check_parameter_complexity(parameter_space)
        if complexity_issues:
            return False, f"Parameter complexity issues: {complexity_issues}", {
                'action': 'simplify_parameters',
                'issues': complexity_issues
            }
        
        # Get budget status
        budget = self.tracker.get_optimization_budget(strategy_name, data_hash, self.constraints)
        
        recommendations = {
            'budget_status': {
                'iterations_remaining': budget.iterations_remaining,
                'time_remaining_hours': budget.time_remaining_hours,
                'parameters_remaining': budget.parameters_remaining
            },
            'suggested_iterations': min(requested_iterations, budget.iterations_remaining),
            'optimization_tips': self._get_optimization_tips(parameter_space, budget)
        }
        
        return True, "Validation passed", recommendations
    
    def prioritize_parameters(
        self,
        strategy_class: type,
        parameter_space: Dict[str, Any],
        data: pd.DataFrame,
        max_parameters: Optional[int] = None
    ) -> Tuple[Dict[str, Any], List[ParameterImportance]]:
        """
        Prioritize parameters for optimization based on importance analysis.
        
        Args:
            strategy_class: Strategy class to analyze
            parameter_space: Full parameter space
            data: Historical data
            max_parameters: Maximum parameters to include (uses constraint if None)
            
        Returns:
            Tuple of (prioritized_parameter_space, importance_analysis)
        """
        if not self.importance_analyzer:
            self.logger.warning("No importance analyzer available, returning original parameter space")
            return parameter_space, []
        
        max_params = max_parameters or self.constraints.max_parameters
        
        self.logger.info(f"Prioritizing parameters for {strategy_class.__name__}")
        
        # Analyze parameter importance
        importance_results = self.importance_analyzer.analyze_parameter_importance(
            strategy_class, parameter_space, data
        )
        
        # Filter recommended parameters
        recommended_params = [
            result for result in importance_results
            if result.recommended_for_optimization and result.overfitting_risk != "High"
        ]
        
        # Sort by importance and take top parameters
        recommended_params.sort(key=lambda x: x.importance_score, reverse=True)
        top_params = recommended_params[:max_params]
        
        # Create prioritized parameter space
        prioritized_space = {
            param.parameter_name: parameter_space[param.parameter_name]
            for param in top_params
            if param.parameter_name in parameter_space
        }
        
        self.logger.info(f"Prioritized {len(prioritized_space)} parameters from {len(parameter_space)}")
        
        return prioritized_space, importance_results
    
    def track_optimization_iteration(
        self,
        strategy_name: str,
        parameters: Dict[str, Any],
        data: pd.DataFrame,
        execution_time: float,
        performance_score: float
    ):
        """
        Track an optimization iteration.
        
        Args:
            strategy_name: Name of the strategy
            parameters: Parameters used
            data: Data used for backtest
            execution_time: Time taken
            performance_score: Performance achieved
        """
        data_hash = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()
        
        self.tracker.record_backtest(
            strategy_name, parameters, data_hash, execution_time, performance_score
        )
    
    def get_optimization_summary(
        self,
        strategy_name: str,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get optimization summary for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            data: Data used for optimization
            
        Returns:
            Dictionary with optimization summary
        """
        data_hash = hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()
        
        budget = self.tracker.get_optimization_budget(strategy_name, data_hash, self.constraints)
        stats = self.tracker.get_strategy_statistics(strategy_name, data_hash)
        
        return {
            'strategy_name': strategy_name,
            'budget_status': {
                'iterations_used': budget.used_iterations,
                'iterations_remaining': budget.iterations_remaining,
                'time_used_hours': budget.used_optimization_time_hours,
                'time_remaining_hours': budget.time_remaining_hours,
                'budget_exhausted': budget.budget_exhausted
            },
            'performance_stats': stats,
            'constraints': {
                'max_parameters': self.constraints.max_parameters,
                'max_iterations': self.constraints.max_total_iterations,
                'max_time_hours': self.constraints.max_optimization_time_hours
            },
            'recommendations': self._generate_optimization_recommendations(budget, stats)
        }
    
    def _check_parameter_complexity(self, parameter_space: Dict[str, Any]) -> List[str]:
        """Check parameter complexity against constraints."""
        issues = []
        
        for param_name, param_config in parameter_space.items():
            # Check parameter range complexity
            if hasattr(param_config, 'pos_args') and len(param_config.pos_args) >= 2:
                low, high = param_config.pos_args[0], param_config.pos_args[1]
                range_size = high - low
                
                # Estimate number of discrete values
                if range_size > self.constraints.max_parameter_ranges:
                    issues.append(f"Parameter {param_name} has too wide range: {range_size}")
        
        return issues
    
    def _get_optimization_tips(
        self,
        parameter_space: Dict[str, Any],
        budget: OptimizationBudget
    ) -> List[str]:
        """Generate optimization tips based on current state."""
        tips = []
        
        if budget.iterations_remaining < 50:
            tips.append("Consider reducing parameter ranges to focus optimization")
        
        if len(parameter_space) > 3:
            tips.append("Use parameter importance analysis to prioritize key parameters")
        
        if budget.time_remaining_hours < 2:
            tips.append("Time budget is low - consider simpler parameter spaces")
        
        return tips
    
    def _generate_optimization_recommendations(
        self,
        budget: OptimizationBudget,
        stats: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if budget.budget_exhausted:
            recommendations.append("Optimization budget exhausted - consider strategy redesign")
        
        if stats.get('score_improvement', 0) < 0.01 and stats.get('total_backtests', 0) > 20:
            recommendations.append("Limited improvement detected - may have reached optimization limit")
        
        if stats.get('score_std', 0) > 0.5:
            recommendations.append("High performance variance - check for overfitting")
        
        return recommendations 