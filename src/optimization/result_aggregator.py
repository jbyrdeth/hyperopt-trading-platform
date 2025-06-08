"""
Result Aggregator for Batch Optimization

This module implements sophisticated result aggregation and analysis
for batch optimization results, providing statistical summaries,
performance comparisons, and ranking analysis.
"""

import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# Use absolute imports
try:
    from src.optimization.hyperopt_optimizer import OptimizationResult
    from src.utils.logger import get_logger
except ImportError:
    # Fallback for relative imports
    from ..optimization.hyperopt_optimizer import OptimizationResult
    from ..utils.logger import get_logger


@dataclass
class StrategyPerformanceComparison:
    """Performance comparison for a strategy relative to others."""
    strategy_name: str
    rank: int
    total_strategies: int
    percentile: float
    score: float
    relative_performance: str  # "excellent", "good", "average", "poor"
    
    
@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics across all successful optimizations."""
    # Basic statistics
    count: int
    mean_score: float
    median_score: float
    std_score: float
    min_score: float
    max_score: float
    
    # Performance metrics
    mean_sharpe_ratio: Optional[float] = None
    median_sharpe_ratio: Optional[float] = None
    mean_annual_return: Optional[float] = None
    median_annual_return: Optional[float] = None
    mean_max_drawdown: Optional[float] = None
    median_max_drawdown: Optional[float] = None
    mean_win_rate: Optional[float] = None
    median_win_rate: Optional[float] = None
    
    # Risk-adjusted metrics
    risk_adjusted_return: Optional[float] = None
    return_to_drawdown_ratio: Optional[float] = None
    
    
@dataclass
class BatchAnalysisReport:
    """Comprehensive analysis report for batch optimization results."""
    # Basic info
    batch_id: str
    analysis_timestamp: str
    total_strategies: int
    successful_strategies: int
    failed_strategies: int
    success_rate: float
    
    # Aggregated metrics
    aggregated_metrics: Optional[AggregatedMetrics]
    
    # Rankings and comparisons
    strategy_rankings: List[StrategyPerformanceComparison]
    top_performers: List[str]  # Top 3 strategy names
    poor_performers: List[str]  # Bottom 3 strategy names
    
    # Analysis insights
    performance_distribution: Dict[str, int]  # excellent/good/average/poor counts
    optimization_efficiency: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ResultAggregator:
    """
    Aggregates and analyzes batch optimization results.
    
    Provides sophisticated analysis including statistical summaries,
    performance rankings, and actionable insights.
    """
    
    def __init__(self):
        """Initialize the result aggregator."""
        self.logger = get_logger("result_aggregator")
        
    def aggregate_batch_results(
        self,
        successful_jobs: List[Any],
        failed_jobs: List[Any],
        batch_id: str,
        total_runtime: float
    ) -> BatchAnalysisReport:
        """
        Create a comprehensive analysis report from batch optimization results.
        
        Args:
            successful_jobs: List of successful optimization jobs
            failed_jobs: List of failed optimization jobs  
            batch_id: Unique batch identifier
            total_runtime: Total execution time in seconds
            
        Returns:
            Comprehensive batch analysis report
        """
        self.logger.info(f"Aggregating results for batch {batch_id}: {len(successful_jobs)} successful, {len(failed_jobs)} failed")
        
        total_strategies = len(successful_jobs) + len(failed_jobs)
        success_rate = (len(successful_jobs) / total_strategies * 100) if total_strategies > 0 else 0
        
        # Calculate aggregated metrics
        aggregated_metrics = self._calculate_aggregated_metrics(successful_jobs) if successful_jobs else None
        
        # Create strategy rankings
        strategy_rankings = self._create_strategy_rankings(successful_jobs)
        
        # Identify top and poor performers
        top_performers, poor_performers = self._identify_performance_extremes(strategy_rankings)
        
        # Analyze performance distribution
        performance_distribution = self._analyze_performance_distribution(strategy_rankings)
        
        # Calculate optimization efficiency metrics
        efficiency_metrics = self._calculate_optimization_efficiency(
            successful_jobs, failed_jobs, total_runtime
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            aggregated_metrics, strategy_rankings, efficiency_metrics, success_rate
        )
        
        report = BatchAnalysisReport(
            batch_id=batch_id,
            analysis_timestamp=datetime.utcnow().isoformat(),
            total_strategies=total_strategies,
            successful_strategies=len(successful_jobs),
            failed_strategies=len(failed_jobs),
            success_rate=success_rate,
            aggregated_metrics=aggregated_metrics,
            strategy_rankings=strategy_rankings,
            top_performers=top_performers,
            poor_performers=poor_performers,
            performance_distribution=performance_distribution,
            optimization_efficiency=efficiency_metrics,
            recommendations=recommendations
        )
        
        self.logger.info(f"Analysis complete for batch {batch_id}: success rate {success_rate:.1f}%")
        return report
    
    def _calculate_aggregated_metrics(self, successful_jobs: List[Any]) -> AggregatedMetrics:
        """Calculate aggregated statistical metrics from successful jobs."""
        if not successful_jobs:
            return None
            
        scores = []
        sharpe_ratios = []
        annual_returns = []
        max_drawdowns = []
        win_rates = []
        
        for job in successful_jobs:
            if hasattr(job, 'result') and job.result:
                scores.append(job.result.best_score)
                
                if hasattr(job.result, 'best_metrics') and job.result.best_metrics:
                    metrics = job.result.best_metrics
                    if 'sharpe_ratio' in metrics:
                        sharpe_ratios.append(metrics['sharpe_ratio'])
                    if 'annual_return' in metrics:
                        annual_returns.append(metrics['annual_return'])
                    if 'max_drawdown' in metrics:
                        max_drawdowns.append(metrics['max_drawdown'])
                    if 'win_rate' in metrics:
                        win_rates.append(metrics['win_rate'])
        
        if not scores:
            return None
            
        # Calculate risk-adjusted metrics
        risk_adjusted_return = None
        return_to_drawdown_ratio = None
        
        if sharpe_ratios and annual_returns:
            # Simple risk-adjusted return calculation
            mean_sharpe = statistics.mean(sharpe_ratios)
            mean_return = statistics.mean(annual_returns)
            risk_adjusted_return = mean_return * mean_sharpe
            
        if annual_returns and max_drawdowns:
            # Return to drawdown ratio
            mean_return = statistics.mean(annual_returns)
            mean_drawdown = statistics.mean(max_drawdowns)
            if mean_drawdown > 0:
                return_to_drawdown_ratio = mean_return / mean_drawdown
        
        return AggregatedMetrics(
            count=len(scores),
            mean_score=statistics.mean(scores),
            median_score=statistics.median(scores),
            std_score=statistics.stdev(scores) if len(scores) > 1 else 0,
            min_score=min(scores),
            max_score=max(scores),
            mean_sharpe_ratio=statistics.mean(sharpe_ratios) if sharpe_ratios else None,
            median_sharpe_ratio=statistics.median(sharpe_ratios) if sharpe_ratios else None,
            mean_annual_return=statistics.mean(annual_returns) if annual_returns else None,
            median_annual_return=statistics.median(annual_returns) if annual_returns else None,
            mean_max_drawdown=statistics.mean(max_drawdowns) if max_drawdowns else None,
            median_max_drawdown=statistics.median(max_drawdowns) if max_drawdowns else None,
            mean_win_rate=statistics.mean(win_rates) if win_rates else None,
            median_win_rate=statistics.median(win_rates) if win_rates else None,
            risk_adjusted_return=risk_adjusted_return,
            return_to_drawdown_ratio=return_to_drawdown_ratio
        )
    
    def _create_strategy_rankings(self, successful_jobs: List[Any]) -> List[StrategyPerformanceComparison]:
        """Create performance rankings for strategies."""
        if not successful_jobs:
            return []
            
        # Extract strategy performances
        strategy_scores = []
        for job in successful_jobs:
            if hasattr(job, 'result') and job.result and hasattr(job, 'strategy_name'):
                strategy_scores.append({
                    'strategy_name': job.strategy_name,
                    'score': job.result.best_score
                })
        
        if not strategy_scores:
            return []
            
        # Sort by score (descending - higher is better)
        strategy_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Create rankings
        rankings = []
        total_strategies = len(strategy_scores)
        
        for rank, strategy_data in enumerate(strategy_scores, 1):
            percentile = ((total_strategies - rank) / total_strategies) * 100
            
            # Determine relative performance category
            if percentile >= 75:
                relative_performance = "excellent"
            elif percentile >= 50:
                relative_performance = "good"
            elif percentile >= 25:
                relative_performance = "average"
            else:
                relative_performance = "poor"
                
            rankings.append(StrategyPerformanceComparison(
                strategy_name=strategy_data['strategy_name'],
                rank=rank,
                total_strategies=total_strategies,
                percentile=percentile,
                score=strategy_data['score'],
                relative_performance=relative_performance
            ))
            
        return rankings
    
    def _identify_performance_extremes(
        self, 
        rankings: List[StrategyPerformanceComparison]
    ) -> Tuple[List[str], List[str]]:
        """Identify top and poor performers."""
        if not rankings:
            return [], []
            
        # Top 3 performers (or all if less than 3)
        top_count = min(3, len(rankings))
        top_performers = [r.strategy_name for r in rankings[:top_count]]
        
        # Bottom 3 performers (or all if less than 3, avoiding duplicates with top)
        if len(rankings) > 3:
            poor_performers = [r.strategy_name for r in rankings[-3:]]
        else:
            poor_performers = []
            
        return top_performers, poor_performers
    
    def _analyze_performance_distribution(
        self, 
        rankings: List[StrategyPerformanceComparison]
    ) -> Dict[str, int]:
        """Analyze the distribution of performance categories."""
        distribution = defaultdict(int)
        
        for ranking in rankings:
            distribution[ranking.relative_performance] += 1
            
        return dict(distribution)
    
    def _calculate_optimization_efficiency(
        self,
        successful_jobs: List[Any],
        failed_jobs: List[Any], 
        total_runtime: float
    ) -> Dict[str, Any]:
        """Calculate optimization efficiency metrics."""
        total_jobs = len(successful_jobs) + len(failed_jobs)
        
        # Calculate average optimization time per job
        avg_time_per_job = total_runtime / total_jobs if total_jobs > 0 else 0
        
        # Calculate success rate
        success_rate = len(successful_jobs) / total_jobs * 100 if total_jobs > 0 else 0
        
        # Calculate average evaluations per successful job
        total_evaluations = 0
        evaluation_count = 0
        
        for job in successful_jobs:
            if hasattr(job, 'result') and job.result and hasattr(job.result, 'total_evaluations'):
                total_evaluations += job.result.total_evaluations
                evaluation_count += 1
        
        avg_evaluations = total_evaluations / evaluation_count if evaluation_count > 0 else 0
        
        # Calculate evaluations per second
        evaluations_per_second = total_evaluations / total_runtime if total_runtime > 0 else 0
        
        return {
            "total_runtime_seconds": total_runtime,
            "average_time_per_job": avg_time_per_job,
            "success_rate_percent": success_rate,
            "total_evaluations": total_evaluations,
            "average_evaluations_per_job": avg_evaluations,
            "evaluations_per_second": evaluations_per_second,
            "throughput_jobs_per_minute": (total_jobs / total_runtime * 60) if total_runtime > 0 else 0
        }
    
    def _generate_recommendations(
        self,
        aggregated_metrics: Optional[AggregatedMetrics],
        rankings: List[StrategyPerformanceComparison],
        efficiency_metrics: Dict[str, Any],
        success_rate: float
    ) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        # Success rate recommendations
        if success_rate < 50:
            recommendations.append("Low success rate detected. Review optimization parameters and strategy configurations.")
        elif success_rate > 90:
            recommendations.append("Excellent success rate. Current configuration is well-optimized.")
            
        # Performance spread recommendations
        if aggregated_metrics and aggregated_metrics.std_score > aggregated_metrics.mean_score * 0.5:
            recommendations.append("High performance variation detected. Consider parameter tuning for consistency.")
            
        # Efficiency recommendations
        if efficiency_metrics.get("average_time_per_job", 0) > 300:  # > 5 minutes
            recommendations.append("Long optimization times detected. Consider reducing max_evals or timeout values.")
            
        # Top performer analysis
        if len(rankings) >= 3:
            excellent_count = sum(1 for r in rankings if r.relative_performance == "excellent")
            if excellent_count >= len(rankings) * 0.3:
                recommendations.append("Multiple strategies showing excellent performance. Consider ensemble approaches.")
            elif excellent_count == 1:
                recommendations.append(f"Single standout performer: {rankings[0].strategy_name}. Investigate what makes it effective.")
        
        # Risk metrics recommendations
        if aggregated_metrics:
            if aggregated_metrics.mean_sharpe_ratio and aggregated_metrics.mean_sharpe_ratio > 2.0:
                recommendations.append("Strong risk-adjusted returns detected. Current strategies show good risk management.")
            elif aggregated_metrics.mean_max_drawdown and aggregated_metrics.mean_max_drawdown > 0.3:
                recommendations.append("High drawdown levels detected. Consider implementing additional risk controls.")
                
        # Default recommendation
        if not recommendations:
            recommendations.append("Performance analysis complete. Results are within expected ranges.")
            
        return recommendations 