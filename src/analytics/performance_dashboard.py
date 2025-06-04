"""
Performance Dashboard

This module provides a comprehensive dashboard that aggregates key metrics,
visualizations, and comparison results for rapid evaluation of trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

try:
    from ..strategies.backtesting_engine import BacktestResults, Trade
    from ..utils.logger import get_logger
    from .performance_analyzer import PerformanceAnalyzer, PerformanceReport
    from .visualization_engine import VisualizationEngine
    from .strategy_comparator import StrategyComparator, ComparisonResult, RankingMethod
    from ..validation import RobustnessScorer
except ImportError:
    from src.strategies.backtesting_engine import BacktestResults, Trade
    from src.utils.logger import get_logger
    from src.analytics.performance_analyzer import PerformanceAnalyzer, PerformanceReport
    from src.analytics.visualization_engine import VisualizationEngine
    from src.analytics.strategy_comparator import StrategyComparator, ComparisonResult, RankingMethod
    from src.validation import RobustnessScorer


@dataclass
class DashboardConfig:
    """Configuration for dashboard display and behavior."""
    theme: str = "plotly_white"
    color_palette: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    figure_width: int = 1200
    figure_height: int = 600
    show_validation_scores: bool = True
    max_strategies_display: int = 10
    default_ranking_method: RankingMethod = RankingMethod.WEIGHTED_SCORE


@dataclass
class StrategyInsight:
    """Individual strategy insight for dashboard display."""
    strategy_name: str
    overall_score: float
    rank: int
    percentile: float
    
    # Key metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Risk assessment
    risk_level: str  # "Low", "Medium", "High"
    volatility: float
    
    # Validation
    robustness_score: Optional[float] = None
    
    # Insights
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Status
    deployment_ready: bool = False
    confidence_level: str = "Medium"  # "Low", "Medium", "High"


@dataclass
class DashboardSummary:
    """Complete dashboard summary for all strategies."""
    # Overview
    total_strategies: int
    analysis_date: datetime
    
    # Top performers
    best_strategy: str
    best_sharpe: str
    best_return: str
    most_stable: str
    
    # Portfolio insights
    recommended_strategies: List[str]
    high_risk_strategies: List[str]
    deployment_ready_count: int
    
    # Market insights
    avg_sharpe_ratio: float
    avg_max_drawdown: float
    correlation_insights: List[str]
    
    # Validation summary
    avg_robustness_score: Optional[float] = None
    validation_warnings: List[str] = field(default_factory=list)


class PerformanceDashboard:
    """
    Comprehensive performance dashboard for trading strategy evaluation.
    
    Aggregates analytics from PerformanceAnalyzer, VisualizationEngine, and
    StrategyComparator to provide rapid strategy assessment and insights.
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        include_validation: bool = True
    ):
        """
        Initialize the PerformanceDashboard.
        
        Args:
            config: Dashboard configuration
            include_validation: Whether to include validation scores
        """
        self.logger = get_logger("performance_dashboard")
        self.config = config or DashboardConfig()
        self.include_validation = include_validation
        
        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualization_engine = VisualizationEngine(
            theme=self.config.theme,
            color_palette=self.config.color_palette,
            figure_size=(self.config.figure_width, self.config.figure_height)
        )
        self.strategy_comparator = StrategyComparator()
        
        if self.include_validation:
            self.robustness_scorer = RobustnessScorer()
        
        self.logger.info("PerformanceDashboard initialized")
    
    def create_comprehensive_dashboard(
        self,
        backtest_results: Dict[str, BacktestResults],
        benchmark_data: Optional[pd.Series] = None,
        save_path: Optional[str] = None
    ) -> Tuple[DashboardSummary, Dict[str, go.Figure]]:
        """
        Create a comprehensive dashboard for all strategies.
        
        Args:
            backtest_results: Dictionary of strategy names to backtest results
            benchmark_data: Optional benchmark data for comparison
            save_path: Optional path to save dashboard files
            
        Returns:
            Tuple of dashboard summary and visualization figures
        """
        self.logger.info(f"Creating comprehensive dashboard for {len(backtest_results)} strategies")
        
        # Generate performance reports
        performance_reports = {}
        for strategy_name, results in backtest_results.items():
            report = self.performance_analyzer.analyze_performance(results)
            
            # Add validation score if enabled
            if self.include_validation:
                try:
                    robustness_result = self.robustness_scorer.calculate_robustness_score(results)
                    report.robustness_score = robustness_result.overall_score
                except Exception as e:
                    self.logger.warning(f"Validation failed for {strategy_name}: {e}")
                    report.robustness_score = None
            
            performance_reports[strategy_name] = report
        
        # Generate strategy comparison
        comparison_result = self.strategy_comparator.compare_strategies(
            performance_reports,
            ranking_method=self.config.default_ranking_method,
            include_validation=self.include_validation
        )
        
        # Create strategy insights
        strategy_insights = self._create_strategy_insights(
            performance_reports, comparison_result
        )
        
        # Generate dashboard summary
        dashboard_summary = self._create_dashboard_summary(
            strategy_insights, comparison_result, performance_reports
        )
        
        # Create visualizations
        figures = self._create_dashboard_visualizations(
            performance_reports, comparison_result, benchmark_data
        )
        
        # Save dashboard if path provided
        if save_path:
            self._save_dashboard(figures, dashboard_summary, save_path)
        
        return dashboard_summary, figures
    
    def create_strategy_scorecard(
        self,
        strategy_name: str,
        backtest_results: BacktestResults,
        benchmark_data: Optional[pd.Series] = None
    ) -> Tuple[StrategyInsight, Dict[str, go.Figure]]:
        """
        Create a detailed scorecard for a single strategy.
        
        Args:
            strategy_name: Name of the strategy
            backtest_results: Backtest results for the strategy
            benchmark_data: Optional benchmark data
            
        Returns:
            Tuple of strategy insight and visualization figures
        """
        self.logger.info(f"Creating scorecard for strategy: {strategy_name}")
        
        # Generate performance report
        performance_report = self.performance_analyzer.analyze_performance(backtest_results)
        
        # Add validation score if enabled
        if self.include_validation:
            try:
                robustness_result = self.robustness_scorer.calculate_robustness_score(backtest_results)
                performance_report.robustness_score = robustness_result.overall_score
            except Exception as e:
                self.logger.warning(f"Validation failed for {strategy_name}: {e}")
                performance_report.robustness_score = None
        
        # Create strategy insight
        strategy_insight = self._create_single_strategy_insight(
            strategy_name, performance_report
        )
        
        # Create visualizations
        figures = {
            'equity_curve': self.visualization_engine.create_equity_curve(
                performance_report, benchmark_data
            ),
            'return_distribution': self.visualization_engine.create_return_distribution(
                performance_report
            ),
            'risk_dashboard': self.visualization_engine.create_risk_metrics_dashboard(
                performance_report.risk_analysis, performance_report.advanced_metrics
            ),
            'trade_analysis': self.visualization_engine.create_trade_analysis(
                backtest_results.trades, strategy_name
            ),
            'rolling_metrics': self.visualization_engine.create_rolling_metrics(
                performance_report.performance_breakdown, strategy_name
            )
        }
        
        # Add performance heatmap if sufficient data
        if len(performance_report.performance_breakdown.monthly_returns) > 12:
            figures['performance_heatmap'] = self.visualization_engine.create_performance_heatmap(
                performance_report.performance_breakdown
            )
        
        return strategy_insight, figures
    
    def create_comparison_dashboard(
        self,
        backtest_results: Dict[str, BacktestResults],
        ranking_method: RankingMethod = RankingMethod.WEIGHTED_SCORE
    ) -> Tuple[ComparisonResult, go.Figure]:
        """
        Create a focused comparison dashboard for multiple strategies.
        
        Args:
            backtest_results: Dictionary of strategy names to backtest results
            ranking_method: Method to use for ranking
            
        Returns:
            Tuple of comparison result and comparison visualization
        """
        self.logger.info(f"Creating comparison dashboard for {len(backtest_results)} strategies")
        
        # Generate performance reports
        performance_reports = {}
        for strategy_name, results in backtest_results.items():
            report = self.performance_analyzer.analyze_performance(results)
            performance_reports[strategy_name] = report
        
        # Generate comparison
        comparison_result = self.strategy_comparator.compare_strategies(
            performance_reports,
            ranking_method=ranking_method,
            include_validation=self.include_validation
        )
        
        # Create comparison visualization
        comparison_figure = self.visualization_engine.create_strategy_comparison(
            performance_reports
        )
        
        return comparison_result, comparison_figure
    
    def generate_executive_summary(
        self,
        dashboard_summary: DashboardSummary,
        strategy_insights: List[StrategyInsight]
    ) -> str:
        """
        Generate an executive summary of the analysis.
        
        Args:
            dashboard_summary: Dashboard summary data
            strategy_insights: List of strategy insights
            
        Returns:
            Executive summary as formatted string
        """
        summary_lines = [
            "# TRADING STRATEGY PERFORMANCE EXECUTIVE SUMMARY",
            f"Analysis Date: {dashboard_summary.analysis_date.strftime('%Y-%m-%d %H:%M')}",
            f"Total Strategies Analyzed: {dashboard_summary.total_strategies}",
            "",
            "## KEY FINDINGS",
            f"• Best Overall Strategy: {dashboard_summary.best_strategy}",
            f"• Highest Sharpe Ratio: {dashboard_summary.best_sharpe}",
            f"• Best Total Return: {dashboard_summary.best_return}",
            f"• Most Stable Strategy: {dashboard_summary.most_stable}",
            "",
            "## DEPLOYMENT READINESS",
            f"• Strategies Ready for Deployment: {dashboard_summary.deployment_ready_count}",
            f"• Recommended Strategies: {', '.join(dashboard_summary.recommended_strategies[:3])}",
            f"• High-Risk Strategies: {', '.join(dashboard_summary.high_risk_strategies[:3])}",
            "",
            "## PORTFOLIO METRICS",
            f"• Average Sharpe Ratio: {dashboard_summary.avg_sharpe_ratio:.2f}",
            f"• Average Maximum Drawdown: {dashboard_summary.avg_max_drawdown:.1%}",
        ]
        
        if dashboard_summary.avg_robustness_score is not None:
            summary_lines.extend([
                f"• Average Robustness Score: {dashboard_summary.avg_robustness_score:.1f}/100",
                ""
            ])
        
        # Add top strategy details
        top_strategies = sorted(strategy_insights, key=lambda x: x.overall_score, reverse=True)[:3]
        summary_lines.extend([
            "## TOP 3 STRATEGIES DETAILED ANALYSIS",
            ""
        ])
        
        for i, strategy in enumerate(top_strategies, 1):
            summary_lines.extend([
                f"### {i}. {strategy.strategy_name}",
                f"• Overall Score: {strategy.overall_score:.1f}",
                f"• Total Return: {strategy.total_return:.1%}",
                f"• Sharpe Ratio: {strategy.sharpe_ratio:.2f}",
                f"• Max Drawdown: {strategy.max_drawdown:.1%}",
                f"• Risk Level: {strategy.risk_level}",
                f"• Deployment Ready: {'Yes' if strategy.deployment_ready else 'No'}",
                ""
            ])
            
            if strategy.strengths:
                summary_lines.append(f"  Strengths: {', '.join(strategy.strengths[:2])}")
            if strategy.weaknesses:
                summary_lines.append(f"  Areas for Improvement: {', '.join(strategy.weaknesses[:2])}")
            summary_lines.append("")
        
        # Add warnings if any
        if dashboard_summary.validation_warnings:
            summary_lines.extend([
                "## VALIDATION WARNINGS",
                ""
            ])
            for warning in dashboard_summary.validation_warnings:
                summary_lines.append(f"• {warning}")
            summary_lines.append("")
        
        # Add correlation insights
        if dashboard_summary.correlation_insights:
            summary_lines.extend([
                "## CORRELATION INSIGHTS",
                ""
            ])
            for insight in dashboard_summary.correlation_insights:
                summary_lines.append(f"• {insight}")
        
        return "\n".join(summary_lines)
    
    def _create_strategy_insights(
        self,
        performance_reports: Dict[str, PerformanceReport],
        comparison_result: ComparisonResult
    ) -> List[StrategyInsight]:
        """Create strategy insights from performance reports and comparison."""
        
        insights = []
        
        for ranking in comparison_result.rankings:
            strategy_name = ranking.strategy_name
            report = performance_reports[strategy_name]
            
            # Determine risk level
            risk_level = self._assess_risk_level(
                report.basic_results.volatility,
                report.basic_results.max_drawdown
            )
            
            # Determine deployment readiness
            deployment_ready = self._assess_deployment_readiness(
                ranking.score, report.robustness_score
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                report, ranking, risk_level
            )
            
            insight = StrategyInsight(
                strategy_name=strategy_name,
                overall_score=ranking.score,
                rank=ranking.rank,
                percentile=ranking.percentile,
                total_return=report.basic_results.total_return,
                sharpe_ratio=report.basic_results.sharpe_ratio,
                max_drawdown=report.basic_results.max_drawdown,
                win_rate=report.basic_results.win_rate,
                risk_level=risk_level,
                volatility=report.basic_results.volatility,
                robustness_score=report.robustness_score,
                strengths=ranking.strengths,
                weaknesses=ranking.weaknesses,
                recommendations=recommendations,
                deployment_ready=deployment_ready,
                confidence_level=self._assess_confidence_level(ranking.score, report.robustness_score)
            )
            
            insights.append(insight)
        
        return insights
    
    def _create_single_strategy_insight(
        self,
        strategy_name: str,
        performance_report: PerformanceReport
    ) -> StrategyInsight:
        """Create insight for a single strategy."""
        
        basic = performance_report.basic_results
        
        # Calculate a simple overall score
        overall_score = (
            basic.sharpe_ratio * 30 +\
            basic.calmar_ratio * 25 +\
            (basic.total_return * 100) * 20 +\
            (1 - abs(basic.max_drawdown)) * 100 * 15 +\
            basic.win_rate * 100 * 10
        )
        
        risk_level = self._assess_risk_level(basic.volatility, basic.max_drawdown)
        deployment_ready = self._assess_deployment_readiness(overall_score, performance_report.robustness_score)
        
        # Generate basic strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if basic.sharpe_ratio > 1.5:
            strengths.append("Excellent risk-adjusted returns")
        elif basic.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        
        if abs(basic.max_drawdown) < 0.1:
            strengths.append("Low maximum drawdown")
        elif abs(basic.max_drawdown) > 0.3:
            weaknesses.append("High maximum drawdown")
        
        if basic.win_rate > 0.6:
            strengths.append("High win rate")
        elif basic.win_rate < 0.4:
            weaknesses.append("Low win rate")
        
        recommendations = self._generate_recommendations(
            performance_report, None, risk_level
        )
        
        return StrategyInsight(
            strategy_name=strategy_name,
            overall_score=overall_score,
            rank=1,  # Single strategy
            percentile=100,
            total_return=basic.total_return,
            sharpe_ratio=basic.sharpe_ratio,
            max_drawdown=basic.max_drawdown,
            win_rate=basic.win_rate,
            risk_level=risk_level,
            volatility=basic.volatility,
            robustness_score=performance_report.robustness_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            deployment_ready=deployment_ready,
            confidence_level=self._assess_confidence_level(overall_score, performance_report.robustness_score)
        )
    
    def _create_dashboard_summary(
        self,
        strategy_insights: List[StrategyInsight],
        comparison_result: ComparisonResult,
        performance_reports: Dict[str, PerformanceReport]
    ) -> DashboardSummary:
        """Create dashboard summary from insights and comparison results."""
        
        # Find best performers
        best_strategy = strategy_insights[0].strategy_name if strategy_insights else "N/A"
        
        best_sharpe = max(strategy_insights, key=lambda x: x.sharpe_ratio).strategy_name if strategy_insights else "N/A"
        best_return = max(strategy_insights, key=lambda x: x.total_return).strategy_name if strategy_insights else "N/A"
        
        # Find most stable (highest return stability if available)
        most_stable = "N/A"
        if strategy_insights:
            try:
                most_stable = min(strategy_insights, key=lambda x: x.volatility).strategy_name
            except:
                most_stable = strategy_insights[0].strategy_name
        
        # Calculate averages
        avg_sharpe = np.mean([s.sharpe_ratio for s in strategy_insights]) if strategy_insights else 0
        avg_drawdown = np.mean([abs(s.max_drawdown) for s in strategy_insights]) if strategy_insights else 0
        
        # Robustness score average
        robustness_scores = [s.robustness_score for s in strategy_insights if s.robustness_score is not None]
        avg_robustness = np.mean(robustness_scores) if robustness_scores else None
        
        # Recommendations and warnings
        recommended_strategies = [s.strategy_name for s in strategy_insights if s.deployment_ready][:5]
        high_risk_strategies = [s.strategy_name for s in strategy_insights if s.risk_level == "High"][:5]
        deployment_ready_count = sum(1 for s in strategy_insights if s.deployment_ready)
        
        # Generate correlation insights
        correlation_insights = self._generate_correlation_insights(comparison_result.correlation_matrix)
        
        # Generate validation warnings
        validation_warnings = []
        if avg_robustness and avg_robustness < 60:
            validation_warnings.append("Average robustness score below recommended threshold")
        
        low_confidence_count = sum(1 for s in strategy_insights if s.confidence_level == "Low")
        if low_confidence_count > len(strategy_insights) * 0.3:
            validation_warnings.append(f"{low_confidence_count} strategies have low confidence scores")
        
        return DashboardSummary(
            total_strategies=len(strategy_insights),
            analysis_date=datetime.now(),
            best_strategy=best_strategy,
            best_sharpe=best_sharpe,
            best_return=best_return,
            most_stable=most_stable,
            recommended_strategies=recommended_strategies,
            high_risk_strategies=high_risk_strategies,
            deployment_ready_count=deployment_ready_count,
            avg_sharpe_ratio=avg_sharpe,
            avg_max_drawdown=avg_drawdown,
            correlation_insights=correlation_insights,
            avg_robustness_score=avg_robustness,
            validation_warnings=validation_warnings
        )
    
    def _create_dashboard_visualizations(
        self,
        performance_reports: Dict[str, PerformanceReport],
        comparison_result: ComparisonResult,
        benchmark_data: Optional[pd.Series] = None
    ) -> Dict[str, go.Figure]:
        """Create all dashboard visualizations."""
        
        figures = {}
        
        # Strategy comparison radar chart
        figures['strategy_comparison'] = self.visualization_engine.create_strategy_comparison(
            performance_reports
        )
        
        # Performance overview (top strategies equity curves)
        top_strategies = comparison_result.rankings[:min(5, len(comparison_result.rankings))]
        if top_strategies:
            figures['top_performers_equity'] = self._create_multi_equity_curve(
                {r.strategy_name: performance_reports[r.strategy_name] for r in top_strategies},
                benchmark_data
            )
        
        # Risk-return scatter plot
        figures['risk_return_scatter'] = self._create_risk_return_scatter(
            performance_reports, comparison_result
        )
        
        # Performance metrics heatmap
        figures['metrics_heatmap'] = self._create_metrics_heatmap(
            performance_reports, comparison_result
        )
        
        # Ranking summary chart
        figures['ranking_summary'] = self._create_ranking_summary_chart(
            comparison_result
        )
        
        return figures
    
    def _create_multi_equity_curve(
        self,
        performance_reports: Dict[str, PerformanceReport],
        benchmark_data: Optional[pd.Series] = None
    ) -> go.Figure:
        """Create multi-strategy equity curve comparison."""
        
        fig = go.Figure()
        
        for i, (strategy_name, report) in enumerate(performance_reports.items()):
            equity_curve = report.basic_results.equity_curve
            
            fig.add_trace(go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name=strategy_name,
                line=dict(color=self.config.color_palette[i % len(self.config.color_palette)], width=2),
                hovertemplate=f'<b>{strategy_name}</b><br>' +\
                             'Date: %{x}<br>' +\
                             'Value: $%{y:,.2f}<extra></extra>'\
            ))
        
        # Add benchmark if provided
        if benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=2, dash='dash'),
                hovertemplate='<b>Benchmark</b><br>' +\
                             'Date: %{x}<br>' +\
                             'Value: $%{y:,.2f}<extra></extra>'\
            ))
        
        fig.update_layout(
            title='Top Performers - Equity Curve Comparison',
            template=self.config.theme,
            width=self.config.figure_width,
            height=self.config.figure_height,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified'
        )
        
        return fig
    
    def _create_risk_return_scatter(
        self,
        performance_reports: Dict[str, PerformanceReport],
        comparison_result: ComparisonResult
    ) -> go.Figure:
        """Create risk-return scatter plot."""
        
        # Extract data
        returns = []
        volatilities = []
        strategy_names = []
        scores = []
        
        for ranking in comparison_result.rankings:
            strategy_name = ranking.strategy_name
            report = performance_reports[strategy_name]
            
            returns.append(report.basic_results.total_return * 100)
            volatilities.append(report.basic_results.volatility * 100)
            strategy_names.append(strategy_name)
            scores.append(ranking.score)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            text=strategy_names,
            textposition='top center',
            marker=dict(
                size=[s/5 for s in scores],  # Size based on score
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Overall Score")
            ),
            hovertemplate='<b>%{text}</b><br>' +\
                         'Return: %{y:.1f}%<br>' +\
                         'Volatility: %{x:.1f}%<br>' +\
                         'Score: %{marker.color:.1f}<extra></extra>'\
        ))
        
        fig.update_layout(
            title='Risk-Return Profile',
            template=self.config.theme,
            width=self.config.figure_width,
            height=self.config.figure_height,
            xaxis_title='Volatility (%)',
            yaxis_title='Total Return (%)',
            showlegend=False
        )
        
        return fig
    
    def _create_metrics_heatmap(
        self,
        performance_reports: Dict[str, PerformanceReport],
        comparison_result: ComparisonResult
    ) -> go.Figure:
        """Create performance metrics heatmap."""
        
        # Select key metrics
        metrics = ['sharpe_ratio', 'calmar_ratio', 'total_return', 'max_drawdown', 'win_rate']
        metric_labels = ['Sharpe Ratio', 'Calmar Ratio', 'Total Return', 'Max Drawdown', 'Win Rate']
        
        # Extract data
        data_matrix = []
        strategy_names = []
        
        for ranking in comparison_result.rankings[:self.config.max_strategies_display]:
            strategy_name = ranking.strategy_name
            strategy_names.append(strategy_name)
            
            row = []
            for metric in metrics:
                value = ranking.normalized_scores.get(metric, 0)
                row.append(value)
            data_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data_matrix,
            x=metric_labels,
            y=strategy_names,
            colorscale='RdYlGn',
            hoverongaps=False,
            hovertemplate='Strategy: %{y}<br>Metric: %{x}<br>Score: %{z:.1f}<extra></extra>'\
        ))
        
        fig.update_layout(
            title='Performance Metrics Heatmap (Normalized 0-100)',
            template=self.config.theme,
            width=self.config.figure_width,
            height=max(400, len(strategy_names) * 40),
            xaxis_title='Metrics',
            yaxis_title='Strategies'
        )
        
        return fig
    
    def _create_ranking_summary_chart(
        self,
        comparison_result: ComparisonResult
    ) -> go.Figure:
        """Create ranking summary bar chart."""
        
        # Get top strategies
        top_rankings = comparison_result.rankings[:self.config.max_strategies_display]
        
        strategy_names = [r.strategy_name for r in top_rankings]
        scores = [r.score for r in top_rankings]
        
        # Create bar chart
        fig = go.Figure(data=go.Bar(
            x=strategy_names,
            y=scores,
            marker_color=self.config.color_palette[0],
            hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'\
        ))
        
        fig.update_layout(
            title=f'Strategy Rankings - {comparison_result.ranking_method.value.replace("_", " ").title()}',
            template=self.config.theme,
            width=self.config.figure_width,
            height=self.config.figure_height,
            xaxis_title='Strategy',
            yaxis_title='Overall Score',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def _assess_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """Assess risk level based on volatility and drawdown."""
        
        risk_score = 0
        
        # Volatility component
        if volatility > 0.3:  # 30% annual volatility
            risk_score += 2
        elif volatility > 0.2:  # 20% annual volatility
            risk_score += 1
        
        # Drawdown component
        if abs(max_drawdown) > 0.3:  # 30% max drawdown
            risk_score += 2
        elif abs(max_drawdown) > 0.15:  # 15% max drawdown
            risk_score += 1
        
        if risk_score >= 3:
            return "High"
        elif risk_score >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _assess_deployment_readiness(
        self,
        overall_score: float,
        robustness_score: Optional[float]
    ) -> bool:
        """Assess if strategy is ready for deployment."""
        
        # Basic score threshold
        if overall_score < 50:
            return False
        
        # Robustness score threshold if available
        if robustness_score is not None and robustness_score < 60:
            return False
        
        return True
    
    def _assess_confidence_level(
        self,
        overall_score: float,
        robustness_score: Optional[float]
    ) -> str:
        """Assess confidence level in the strategy."""
        
        confidence_score = overall_score
        
        # Adjust based on robustness score
        if robustness_score is not None:
            confidence_score = (confidence_score + robustness_score) / 2
        
        if confidence_score >= 80:
            return "High"
        elif confidence_score >= 60:
            return "Medium"
        else:
            return "Low"
    
    def _generate_recommendations(
        self,
        performance_report: PerformanceReport,
        ranking: Optional[Any],
        risk_level: str
    ) -> List[str]:
        """Generate actionable recommendations for a strategy."""
        
        recommendations = []
        basic = performance_report.basic_results
        
        # Risk-based recommendations
        if risk_level == "High":
            recommendations.append("Consider position sizing reduction due to high risk")
            recommendations.append("Implement additional risk management measures")
        
        # Performance-based recommendations
        if basic.sharpe_ratio < 1.0:
            recommendations.append("Focus on improving risk-adjusted returns")
        
        if abs(basic.max_drawdown) > 0.2:
            recommendations.append("Implement drawdown control mechanisms")
        
        if basic.win_rate < 0.4:
            recommendations.append("Review entry/exit criteria to improve win rate")
        
        # Validation-based recommendations
        if performance_report.robustness_score and performance_report.robustness_score < 70:
            recommendations.append("Conduct additional validation before deployment")
        
        # Trade frequency recommendations
        if len(basic.trades) < 30:
            recommendations.append("Increase sample size for more reliable statistics")
        
        return recommendations[:3]  # Limit to top 3
    
    def _generate_correlation_insights(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Generate insights from correlation matrix."""
        
        insights = []
        
        try:
            # Find highly correlated metrics
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.8:
                        metric1 = correlation_matrix.columns[i]
                        metric2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((metric1, metric2, corr))
            
            if high_corr_pairs:
                insights.append(f"High correlation detected between {len(high_corr_pairs)} metric pairs")
            
            # Check for negative correlations
            negative_corr = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if corr < -0.5:
                        metric1 = correlation_matrix.columns[i]
                        metric2 = correlation_matrix.columns[j]
                        negative_corr.append((metric1, metric2, corr))
            
            if negative_corr:
                insights.append(f"Strong negative correlations found in {len(negative_corr)} cases")
        
        except Exception as e:
            self.logger.warning(f"Failed to generate correlation insights: {e}")
        
        return insights
    
    def _save_dashboard(
        self,
        figures: Dict[str, go.Figure],
        dashboard_summary: DashboardSummary,
        save_path: str
    ) -> None:
        """Save dashboard figures and summary to files."""
        
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save figures
        for name, fig in figures.items():
            filename = os.path.join(save_path, f"{name}.html")
            fig.write_html(filename)
            self.logger.info(f"Saved figure: {filename}")
        
        # Save summary as JSON
        import json
        summary_dict = {
            'total_strategies': dashboard_summary.total_strategies,
            'analysis_date': dashboard_summary.analysis_date.isoformat(),
            'best_strategy': dashboard_summary.best_strategy,
            'best_sharpe': dashboard_summary.best_sharpe,
            'best_return': dashboard_summary.best_return,
            'most_stable': dashboard_summary.most_stable,
            'recommended_strategies': dashboard_summary.recommended_strategies,
            'high_risk_strategies': dashboard_summary.high_risk_strategies,
            'deployment_ready_count': dashboard_summary.deployment_ready_count,
            'avg_sharpe_ratio': dashboard_summary.avg_sharpe_ratio,
            'avg_max_drawdown': dashboard_summary.avg_max_drawdown,
            'correlation_insights': dashboard_summary.correlation_insights,
            'avg_robustness_score': dashboard_summary.avg_robustness_score,
            'validation_warnings': dashboard_summary.validation_warnings
        }
        
        summary_file = os.path.join(save_path, "dashboard_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        self.logger.info(f"Dashboard saved to: {save_path}") 