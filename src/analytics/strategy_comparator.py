"""
Strategy Comparator

This module provides comprehensive strategy comparison and ranking capabilities
with customizable criteria, weighting schemes, and statistical analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from enum import Enum
from scipy import stats
import logging

try:
    from ..strategies.backtesting_engine import BacktestResults, Trade
    from ..utils.logger import get_logger
    from .performance_analyzer import PerformanceReport, AdvancedMetrics, PerformanceBreakdown, RiskAnalysis
    from ..validation import RobustnessScorer
except ImportError:
    from src.strategies.backtesting_engine import BacktestResults, Trade
    from src.utils.logger import get_logger
    from src.analytics.performance_analyzer import PerformanceReport, AdvancedMetrics, PerformanceBreakdown, RiskAnalysis
    from src.validation import RobustnessScorer


class RankingMethod(Enum):
    """Available ranking methods."""
    WEIGHTED_SCORE = "weighted_score"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    TOTAL_RETURN = "total_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    ROBUSTNESS_SCORE = "robustness_score"
    CUSTOM = "custom"


class ComparisonMetric(Enum):
    """Available metrics for comparison."""
    # Return metrics
    TOTAL_RETURN = "total_return"
    ANNUAL_RETURN = "annual_return"
    MONTHLY_RETURN_AVG = "monthly_return_avg"
    
    # Risk-adjusted metrics
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    INFORMATION_RATIO = "information_ratio"
    
    # Risk metrics
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    VAR_95 = "var_95"
    CONDITIONAL_VAR_95 = "conditional_var_95"
    
    # Trade metrics
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TRADE_FREQUENCY = "trade_frequency"
    AVG_TRADE_DURATION = "avg_trade_duration"
    
    # Stability metrics
    RETURN_STABILITY = "return_stability"
    SHARPE_STABILITY = "sharpe_stability"
    
    # Validation metrics
    ROBUSTNESS_SCORE = "robustness_score"


@dataclass
class MetricWeight:
    """Weight configuration for a specific metric."""
    metric: ComparisonMetric
    weight: float
    direction: str = "higher_better"  # "higher_better" or "lower_better"
    normalization: str = "minmax"  # "minmax", "zscore", "rank"


@dataclass
class StrategyRanking:
    """Individual strategy ranking result."""
    strategy_name: str
    rank: int
    score: float
    percentile: float
    metric_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class ComparisonResult:
    """Complete strategy comparison result."""
    rankings: List[StrategyRanking]
    ranking_method: RankingMethod
    metric_weights: Dict[str, float]
    
    # Statistical analysis
    correlation_matrix: pd.DataFrame
    metric_statistics: Dict[str, Dict[str, float]]
    
    # Performance clusters
    performance_clusters: Dict[str, List[str]]
    cluster_characteristics: Dict[str, Dict[str, float]]
    
    # Insights
    top_performers: List[str]
    consistent_performers: List[str]
    high_risk_strategies: List[str]
    
    # Metadata
    comparison_date: datetime = field(default_factory=datetime.now)
    strategies_count: int = 0


@dataclass
class PairwiseComparison:
    """Pairwise strategy comparison result."""
    strategy_a: str
    strategy_b: str
    winner: str
    confidence: float
    significant_differences: Dict[str, Tuple[float, float, bool]]  # metric: (value_a, value_b, is_significant)
    statistical_tests: Dict[str, Dict[str, float]]  # test_name: {statistic, p_value}


class StrategyComparator:
    """
    Comprehensive strategy comparison and ranking system.
    
    Provides advanced comparison capabilities with customizable criteria,
    statistical analysis, and performance clustering.
    """
    
    def __init__(
        self,
        default_weights: Optional[Dict[ComparisonMetric, float]] = None,
        significance_level: float = 0.05
    ):
        """
        Initialize the StrategyComparator.
        
        Args:
            default_weights: Default metric weights for ranking
            significance_level: Statistical significance level for tests
        """
        self.logger = get_logger("strategy_comparator")
        self.significance_level = significance_level
        
        # Default metric weights
        self.default_weights = default_weights or {
            ComparisonMetric.SHARPE_RATIO: 0.25,
            ComparisonMetric.CALMAR_RATIO: 0.20,
            ComparisonMetric.TOTAL_RETURN: 0.15,
            ComparisonMetric.MAX_DRAWDOWN: 0.15,
            ComparisonMetric.WIN_RATE: 0.10,
            ComparisonMetric.RETURN_STABILITY: 0.10,
            ComparisonMetric.VOLATILITY: 0.05
        }
        
        # Metric direction (higher_better or lower_better)
        self.metric_directions = {
            ComparisonMetric.TOTAL_RETURN: "higher_better",
            ComparisonMetric.ANNUAL_RETURN: "higher_better",
            ComparisonMetric.SHARPE_RATIO: "higher_better",
            ComparisonMetric.SORTINO_RATIO: "higher_better",
            ComparisonMetric.CALMAR_RATIO: "higher_better",
            ComparisonMetric.INFORMATION_RATIO: "higher_better",
            ComparisonMetric.WIN_RATE: "higher_better",
            ComparisonMetric.PROFIT_FACTOR: "higher_better",
            ComparisonMetric.RETURN_STABILITY: "higher_better",
            ComparisonMetric.SHARPE_STABILITY: "higher_better",
            ComparisonMetric.ROBUSTNESS_SCORE: "higher_better",
            ComparisonMetric.MAX_DRAWDOWN: "lower_better",
            ComparisonMetric.VOLATILITY: "lower_better",
            ComparisonMetric.VAR_95: "lower_better",
            ComparisonMetric.CONDITIONAL_VAR_95: "lower_better",
            ComparisonMetric.TRADE_FREQUENCY: "neutral",
            ComparisonMetric.AVG_TRADE_DURATION: "neutral"
        }
        
        self.logger.info("StrategyComparator initialized")
    
    def compare_strategies(
        self,
        performance_reports: Dict[str, PerformanceReport],
        ranking_method: RankingMethod = RankingMethod.WEIGHTED_SCORE,
        custom_weights: Optional[Dict[ComparisonMetric, float]] = None,
        include_validation: bool = False
    ) -> ComparisonResult:
        """
        Compare multiple strategies and generate comprehensive ranking.
        
        Args:
            performance_reports: Dictionary of strategy names to performance reports
            ranking_method: Method to use for ranking
            custom_weights: Custom metric weights (overrides defaults)
            include_validation: Whether to include validation scores
            
        Returns:
            Comprehensive comparison result
        """
        self.logger.info(f"Comparing {len(performance_reports)} strategies using {ranking_method.value}")
        
        if len(performance_reports) < 2:
            raise ValueError("At least 2 strategies required for comparison")
        
        # Extract metrics for all strategies
        metrics_df = self._extract_metrics(performance_reports, include_validation)
        
        # Use custom weights if provided
        weights = custom_weights or self.default_weights
        
        # Calculate rankings based on method
        if ranking_method == RankingMethod.WEIGHTED_SCORE:
            rankings = self._calculate_weighted_rankings(metrics_df, weights)
        elif ranking_method == RankingMethod.CUSTOM:
            rankings = self._calculate_custom_rankings(metrics_df, weights)
        else:
            rankings = self._calculate_single_metric_rankings(metrics_df, ranking_method)
        
        # Perform statistical analysis
        correlation_matrix = self._calculate_correlation_matrix(metrics_df)
        metric_statistics = self._calculate_metric_statistics(metrics_df)
        
        # Identify performance clusters
        clusters, cluster_chars = self._identify_performance_clusters(metrics_df, rankings)
        
        # Generate insights
        insights = self._generate_insights(rankings, metrics_df)
        
        return ComparisonResult(
            rankings=rankings,
            ranking_method=ranking_method,
            metric_weights={metric.value: weight for metric, weight in weights.items()},
            correlation_matrix=correlation_matrix,
            metric_statistics=metric_statistics,
            performance_clusters=clusters,
            cluster_characteristics=cluster_chars,
            top_performers=insights['top_performers'],
            consistent_performers=insights['consistent_performers'],
            high_risk_strategies=insights['high_risk_strategies'],
            strategies_count=len(performance_reports)
        )
    
    def pairwise_comparison(
        self,
        strategy_a_report: PerformanceReport,
        strategy_b_report: PerformanceReport,
        metrics: Optional[List[ComparisonMetric]] = None
    ) -> PairwiseComparison:
        """
        Perform detailed pairwise comparison between two strategies.
        
        Args:
            strategy_a_report: Performance report for strategy A
            strategy_b_report: Performance report for strategy B
            metrics: Specific metrics to compare
            
        Returns:
            Detailed pairwise comparison result
        """
        strategy_a = strategy_a_report.basic_results.strategy_name
        strategy_b = strategy_b_report.basic_results.strategy_name
        
        self.logger.info(f"Performing pairwise comparison: {strategy_a} vs {strategy_b}")
        
        # Default metrics for pairwise comparison
        if metrics is None:
            metrics = [
                ComparisonMetric.SHARPE_RATIO,
                ComparisonMetric.CALMAR_RATIO,
                ComparisonMetric.TOTAL_RETURN,
                ComparisonMetric.MAX_DRAWDOWN,
                ComparisonMetric.WIN_RATE,
                ComparisonMetric.VOLATILITY
            ]
        
        # Extract metric values
        values_a = self._extract_strategy_metrics(strategy_a_report)
        values_b = self._extract_strategy_metrics(strategy_b_report)
        
        # Compare each metric
        significant_differences = {}
        for metric in metrics:
            value_a = values_a.get(metric.value, 0)
            value_b = values_b.get(metric.value, 0)
            
            # Determine if difference is significant (simplified)
            diff_pct = abs(value_a - value_b) / (abs(value_a) + abs(value_b) + 1e-8) * 2
            is_significant = diff_pct > 0.1  # 10% difference threshold
            
            significant_differences[metric.value] = (value_a, value_b, is_significant)
        
        # Perform statistical tests on returns
        statistical_tests = self._perform_statistical_tests(
            strategy_a_report, strategy_b_report
        )
        
        # Determine winner based on weighted score
        score_a = self._calculate_strategy_score(values_a, self.default_weights)
        score_b = self._calculate_strategy_score(values_b, self.default_weights)
        
        winner = strategy_a if score_a > score_b else strategy_b
        confidence = abs(score_a - score_b) / max(score_a, score_b, 1e-8)
        
        return PairwiseComparison(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            winner=winner,
            confidence=confidence,
            significant_differences=significant_differences,
            statistical_tests=statistical_tests
        )
    
    def rank_by_custom_function(
        self,
        performance_reports: Dict[str, PerformanceReport],
        ranking_function: Callable[[PerformanceReport], float],
        function_name: str = "Custom Function"
    ) -> List[StrategyRanking]:
        """
        Rank strategies using a custom ranking function.
        
        Args:
            performance_reports: Dictionary of strategy names to performance reports
            ranking_function: Function that takes PerformanceReport and returns score
            function_name: Name of the custom function for display
            
        Returns:
            List of strategy rankings
        """
        self.logger.info(f"Ranking strategies using custom function: {function_name}")
        
        # Calculate scores using custom function
        scores = {}
        for strategy_name, report in performance_reports.items():
            try:
                score = ranking_function(report)
                scores[strategy_name] = score
            except Exception as e:
                self.logger.warning(f"Custom function failed for {strategy_name}: {e}")
                scores[strategy_name] = 0
        
        # Create rankings
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = []
        
        for rank, (strategy_name, score) in enumerate(sorted_strategies, 1):
            percentile = (len(sorted_strategies) - rank + 1) / len(sorted_strategies) * 100
            
            rankings.append(StrategyRanking(
                strategy_name=strategy_name,
                rank=rank,
                score=score,
                percentile=percentile,
                metric_scores={function_name: score},
                normalized_scores={function_name: score},
                strengths=[f"Custom score: {score:.3f}"],
                weaknesses=[]
            ))
        
        return rankings
    
    def create_performance_matrix(
        self,
        performance_reports: Dict[str, PerformanceReport],
        metrics: Optional[List[ComparisonMetric]] = None
    ) -> pd.DataFrame:
        """
        Create a performance matrix for easy comparison.
        
        Args:
            performance_reports: Dictionary of strategy names to performance reports
            metrics: Specific metrics to include
            
        Returns:
            DataFrame with strategies as rows and metrics as columns
        """
        if metrics is None:
            metrics = list(self.default_weights.keys())
        
        matrix_data = {}
        
        for strategy_name, report in performance_reports.items():
            strategy_metrics = self._extract_strategy_metrics(report)
            matrix_data[strategy_name] = {
                metric.value: strategy_metrics.get(metric.value, np.nan)
                for metric in metrics
            }
        
        return pd.DataFrame(matrix_data).T
    
    def _extract_metrics(
        self,
        performance_reports: Dict[str, PerformanceReport],
        include_validation: bool = False
    ) -> pd.DataFrame:
        """Extract metrics from all performance reports into a DataFrame."""
        
        metrics_data = {}
        
        for strategy_name, report in performance_reports.items():
            strategy_metrics = self._extract_strategy_metrics(report)
            
            # Add validation score if available and requested
            if include_validation and report.robustness_score is not None:
                strategy_metrics['robustness_score'] = report.robustness_score
            
            metrics_data[strategy_name] = strategy_metrics
        
        return pd.DataFrame(metrics_data).T
    
    def _extract_strategy_metrics(self, report: PerformanceReport) -> Dict[str, float]:
        """Extract all relevant metrics from a performance report."""
        
        basic = report.basic_results
        advanced = report.advanced_metrics
        risk = report.risk_analysis
        
        return {
            # Basic metrics
            'total_return': basic.total_return,
            'annual_return': basic.annual_return,
            'sharpe_ratio': basic.sharpe_ratio,
            'sortino_ratio': basic.sortino_ratio,
            'calmar_ratio': basic.calmar_ratio,
            'max_drawdown': basic.max_drawdown,
            'volatility': basic.volatility,
            'win_rate': basic.win_rate,
            'profit_factor': basic.profit_factor,
            
            # Advanced metrics
            'information_ratio': advanced.information_ratio,
            'return_stability': advanced.return_stability,
            'sharpe_stability': advanced.sharpe_stability,
            'trade_frequency': advanced.trade_frequency_per_month,
            'avg_trade_duration': advanced.avg_trade_duration_hours,
            'consecutive_wins_max': advanced.consecutive_wins_max,
            'consecutive_losses_max': advanced.consecutive_losses_max,
            'skewness': advanced.skewness,
            'kurtosis': advanced.kurtosis,
            'tail_ratio': advanced.tail_ratio,
            'gain_to_pain_ratio': advanced.gain_to_pain_ratio,
            
            # Risk metrics
            'var_95': risk.var_1d_95,
            'conditional_var_95': risk.es_1d_95,
            'realized_volatility': risk.realized_volatility,
            'tail_expectation_ratio': risk.tail_expectation_ratio,
            'risk_adjusted_return': risk.risk_adjusted_return,
            'return_over_max_dd': risk.return_over_max_dd
        }
    
    def _calculate_weighted_rankings(
        self,
        metrics_df: pd.DataFrame,
        weights: Dict[ComparisonMetric, float]
    ) -> List[StrategyRanking]:
        """Calculate rankings using weighted scoring."""
        
        # Normalize metrics
        normalized_df = self._normalize_metrics(metrics_df, weights.keys())
        
        # Calculate weighted scores
        scores = {}
        for strategy in normalized_df.index:
            score = 0
            for metric, weight in weights.items():
                metric_value = normalized_df.loc[strategy, metric.value]
                if not np.isnan(metric_value):
                    score += metric_value * weight
            scores[strategy] = score
        
        # Create rankings
        return self._create_rankings_from_scores(scores, metrics_df, normalized_df)
    
    def _calculate_single_metric_rankings(
        self,
        metrics_df: pd.DataFrame,
        ranking_method: RankingMethod
    ) -> List[StrategyRanking]:
        """Calculate rankings using a single metric."""
        
        metric_map = {
            RankingMethod.SHARPE_RATIO: 'sharpe_ratio',
            RankingMethod.CALMAR_RATIO: 'calmar_ratio',
            RankingMethod.SORTINO_RATIO: 'sortino_ratio',
            RankingMethod.TOTAL_RETURN: 'total_return',
            RankingMethod.RISK_ADJUSTED_RETURN: 'risk_adjusted_return',
            RankingMethod.ROBUSTNESS_SCORE: 'robustness_score'
        }
        
        metric_name = metric_map.get(ranking_method)
        if metric_name not in metrics_df.columns:
            raise ValueError(f"Metric {metric_name} not available in data")
        
        scores = metrics_df[metric_name].to_dict()
        normalized_df = metrics_df.copy()
        
        return self._create_rankings_from_scores(scores, metrics_df, normalized_df)
    
    def _calculate_custom_rankings(
        self,
        metrics_df: pd.DataFrame,
        weights: Dict[ComparisonMetric, float]
    ) -> List[StrategyRanking]:
        """Calculate rankings using custom weighting scheme."""
        return self._calculate_weighted_rankings(metrics_df, weights)
    
    def _normalize_metrics(
        self,
        metrics_df: pd.DataFrame,
        metrics: List[ComparisonMetric]
    ) -> pd.DataFrame:
        """Normalize metrics to 0-100 scale."""
        
        normalized_df = metrics_df.copy()
        
        for metric in metrics:
            if metric.value not in metrics_df.columns:
                continue
            
            values = metrics_df[metric.value].dropna()
            if len(values) == 0:
                continue
            
            direction = self.metric_directions.get(metric, "higher_better")
            
            if direction == "higher_better":
                # Higher values are better
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    normalized_df[metric.value] = (values - min_val) / (max_val - min_val) * 100
                else:
                    normalized_df[metric.value] = 50  # All values are the same
            elif direction == "lower_better":
                # Lower values are better (invert scale)
                min_val, max_val = values.min(), values.max()
                if max_val > min_val:
                    normalized_df[metric.value] = (max_val - values) / (max_val - min_val) * 100
                else:
                    normalized_df[metric.value] = 50  # All values are the same
            else:  # neutral
                # Normalize around median
                median_val = values.median()
                mad = np.median(np.abs(values - median_val))
                if mad > 0:
                    normalized_df[metric.value] = 50 + (values - median_val) / mad * 25
                else:
                    normalized_df[metric.value] = 50
        
        return normalized_df
    
    def _calculate_strategy_score(
        self,
        metrics: Dict[str, float],
        weights: Dict[ComparisonMetric, float]
    ) -> float:
        """Calculate weighted score for a single strategy."""
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            value = metrics.get(metric.value)
            if value is not None and not np.isnan(value):
                # Simple normalization (would need proper normalization in practice)
                normalized_value = max(0, min(100, value * 100))
                score += normalized_value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0
    
    def _create_rankings_from_scores(
        self,
        scores: Dict[str, float],
        metrics_df: pd.DataFrame,
        normalized_df: pd.DataFrame
    ) -> List[StrategyRanking]:
        """Create StrategyRanking objects from scores."""
        
        # Sort strategies by score
        sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings = []
        
        for rank, (strategy_name, score) in enumerate(sorted_strategies, 1):
            percentile = (len(sorted_strategies) - rank + 1) / len(sorted_strategies) * 100
            
            # Extract metric scores
            metric_scores = metrics_df.loc[strategy_name].to_dict()
            normalized_scores = normalized_df.loc[strategy_name].to_dict()
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(
                strategy_name, normalized_scores
            )
            
            rankings.append(StrategyRanking(
                strategy_name=strategy_name,
                rank=rank,
                score=score,
                percentile=percentile,
                metric_scores=metric_scores,
                normalized_scores=normalized_scores,
                strengths=strengths,
                weaknesses=weaknesses
            ))
        
        return rankings
    
    def _identify_strengths_weaknesses(
        self,
        strategy_name: str,
        normalized_scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Identify strategy strengths and weaknesses based on normalized scores."""
        
        strengths = []
        weaknesses = []
        
        for metric, score in normalized_scores.items():
            if np.isnan(score):
                continue
            
            if score >= 80:
                strengths.append(f"Excellent {metric.replace('_', ' ')}")
            elif score >= 60:
                strengths.append(f"Good {metric.replace('_', ' ')}")
            elif score <= 20:
                weaknesses.append(f"Poor {metric.replace('_', ' ')}")
            elif score <= 40:
                weaknesses.append(f"Below average {metric.replace('_', ' ')}")
        
        return strengths[:5], weaknesses[:5]  # Limit to top 5 each
    
    def _calculate_correlation_matrix(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between metrics."""
        return metrics_df.corr()
    
    def _calculate_metric_statistics(self, metrics_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each metric."""
        
        statistics = {}
        
        for column in metrics_df.columns:
            values = metrics_df[column].dropna()
            if len(values) > 0:
                statistics[column] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'skewness': values.skew(),
                    'kurtosis': values.kurtosis()
                }
        
        return statistics
    
    def _identify_performance_clusters(
        self,
        metrics_df: pd.DataFrame,
        rankings: List[StrategyRanking]
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
        """Identify performance clusters among strategies."""
        
        # Simple clustering based on ranking percentiles
        clusters = {
            'top_performers': [],
            'solid_performers': [],
            'average_performers': [],
            'underperformers': []
        }
        
        for ranking in rankings:
            if ranking.percentile >= 75:
                clusters['top_performers'].append(ranking.strategy_name)
            elif ranking.percentile >= 50:
                clusters['solid_performers'].append(ranking.strategy_name)
            elif ranking.percentile >= 25:
                clusters['average_performers'].append(ranking.strategy_name)
            else:
                clusters['underperformers'].append(ranking.strategy_name)
        
        # Calculate cluster characteristics
        cluster_characteristics = {}
        for cluster_name, strategies in clusters.items():
            if strategies:
                cluster_data = metrics_df.loc[strategies]
                cluster_characteristics[cluster_name] = {
                    'avg_sharpe': cluster_data['sharpe_ratio'].mean(),
                    'avg_return': cluster_data['total_return'].mean(),
                    'avg_drawdown': cluster_data['max_drawdown'].mean(),
                    'count': len(strategies)
                }
        
        return clusters, cluster_characteristics
    
    def _generate_insights(
        self,
        rankings: List[StrategyRanking],
        metrics_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Generate insights from the comparison results."""
        
        insights = {
            'top_performers': [],
            'consistent_performers': [],
            'high_risk_strategies': []
        }
        
        # Top performers (top 25%)
        top_count = max(1, len(rankings) // 4)
        insights['top_performers'] = [r.strategy_name for r in rankings[:top_count]]
        
        # Consistent performers (high return stability)
        if 'return_stability' in metrics_df.columns:
            stable_strategies = metrics_df.nlargest(3, 'return_stability').index.tolist()
            insights['consistent_performers'] = stable_strategies
        
        # High risk strategies (high volatility or large drawdowns)
        high_risk = []
        if 'volatility' in metrics_df.columns:
            high_vol = metrics_df.nlargest(2, 'volatility').index.tolist()
            high_risk.extend(high_vol)
        
        if 'max_drawdown' in metrics_df.columns:
            high_dd = metrics_df.nsmallest(2, 'max_drawdown').index.tolist()
            high_risk.extend(high_dd)
        
        insights['high_risk_strategies'] = list(set(high_risk))
        
        return insights
    
    def _perform_statistical_tests(
        self,
        report_a: PerformanceReport,
        report_b: PerformanceReport
    ) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests between two strategies."""
        
        # Get return series
        returns_a = report_a.basic_results.equity_curve.pct_change().dropna()
        returns_b = report_b.basic_results.equity_curve.pct_change().dropna()
        
        tests = {}
        
        try:
            # T-test for mean difference
            t_stat, t_p_value = stats.ttest_ind(returns_a, returns_b)
            tests['t_test'] = {'statistic': t_stat, 'p_value': t_p_value}
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(returns_a, returns_b, alternative='two-sided')
            tests['mann_whitney'] = {'statistic': u_stat, 'p_value': u_p_value}
            
            # Kolmogorov-Smirnov test for distribution difference
            ks_stat, ks_p_value = stats.ks_2samp(returns_a, returns_b)
            tests['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p_value}
            
        except Exception as e:
            self.logger.warning(f"Statistical tests failed: {e}")
        
        return tests 