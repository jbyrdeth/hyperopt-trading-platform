"""
Visualization Engine

This module provides comprehensive interactive visualizations for trading strategy
performance analysis using Plotly.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
from scipy import stats

try:
    from ..strategies.backtesting_engine import BacktestResults, Trade
    from ..utils.logger import get_logger
    from .performance_analyzer import PerformanceReport, AdvancedMetrics, PerformanceBreakdown, RiskAnalysis
except ImportError:
    from src.strategies.backtesting_engine import BacktestResults, Trade
    from src.utils.logger import get_logger
    from src.analytics.performance_analyzer import PerformanceReport, AdvancedMetrics, PerformanceBreakdown, RiskAnalysis


class VisualizationEngine:
    """
    Comprehensive visualization engine for trading strategy performance analysis.
    
    Provides interactive Plotly visualizations including equity curves, drawdown charts,
    performance heatmaps, return distributions, and comparative analysis.
    """
    
    def __init__(
        self,
        theme: str = "plotly_white",
        color_palette: Optional[List[str]] = None,
        figure_size: Tuple[int, int] = (1200, 600)
    ):
        """
        Initialize the VisualizationEngine.
        
        Args:
            theme: Plotly theme to use for visualizations
            color_palette: Custom color palette for charts
            figure_size: Default figure size (width, height)
        """
        self.logger = get_logger("visualization_engine")
        self.theme = theme
        self.figure_size = figure_size
        
        # Default color palette
        self.color_palette = color_palette or [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Chart styling
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }
        
        self.logger.info("VisualizationEngine initialized")
    
    def create_equity_curve(
        self,
        performance_report: PerformanceReport,
        benchmark_data: Optional[pd.Series] = None,
        show_drawdown: bool = True
    ) -> go.Figure:
        """
        Create interactive equity curve visualization.
        
        Args:
            performance_report: Performance analysis report
            benchmark_data: Optional benchmark data for comparison
            show_drawdown: Whether to show drawdown subplot
            
        Returns:
            Interactive Plotly figure
        """
        results = performance_report.basic_results
        
        # Create subplots
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Equity Curve', 'Drawdown'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=results.equity_curve.index,
                y=results.equity_curve.values,
                mode='lines',
                name=f'{results.strategy_name} Equity',
                line=dict(color=self.color_palette[0], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add benchmark if provided
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_data.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.color_palette[1], width=2, dash='dash'),
                    hovertemplate='<b>Benchmark</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Value: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Drawdown subplot
        if show_drawdown:
            fig.add_trace(
                go.Scatter(
                    x=results.drawdown_curve.index,
                    y=results.drawdown_curve.values * 100,  # Convert to percentage
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='red', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    hovertemplate='<b>Drawdown</b><br>' +
                                 'Date: %{x}<br>' +
                                 'Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line for drawdown
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'Performance Analysis - {results.strategy_name}',
            template=self.theme,
            width=self.figure_size[0],
            height=self.figure_size[1] if not show_drawdown else self.figure_size[1] + 200,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2 if show_drawdown else 1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        if show_drawdown:
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    def create_return_distribution(
        self,
        performance_report: PerformanceReport,
        show_normal_overlay: bool = True
    ) -> go.Figure:
        """
        Create return distribution visualization with statistics.
        
        Args:
            performance_report: Performance analysis report
            show_normal_overlay: Whether to show normal distribution overlay
            
        Returns:
            Interactive Plotly figure
        """
        results = performance_report.basic_results
        returns = results.equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return self._create_empty_figure("No return data available")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Return Distribution', 'Q-Q Plot vs Normal',
                'Rolling Volatility', 'Return Autocorrelation'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Histogram with normal overlay
        fig.add_trace(
            go.Histogram(
                x=returns * 100,  # Convert to percentage
                nbinsx=50,
                name='Returns',
                opacity=0.7,
                marker_color=self.color_palette[0],
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        if show_normal_overlay:
            # Normal distribution overlay
            x_range = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
            normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50
            
            fig.add_trace(
                go.Scatter(
                    x=x_range * 100,
                    y=normal_dist,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2),
                    hovertemplate='Normal PDF<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Q-Q Plot
        theoretical_quantiles, sample_quantiles = stats.probplot(returns, dist="norm")
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles[0],
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(color=self.color_palette[1], size=4),
                hovertemplate='Theoretical: %{x:.4f}<br>Sample: %{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add Q-Q reference line
        qq_min = min(theoretical_quantiles[0].min(), sample_quantiles.min())
        qq_max = max(theoretical_quantiles[0].max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[qq_min, qq_max],
                y=[qq_min, qq_max],
                mode='lines',
                name='Perfect Normal',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Rolling Volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100  # Annualized %
        
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='30-Day Rolling Volatility',
                line=dict(color=self.color_palette[2], width=2),
                hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Return Autocorrelation
        lags = range(1, min(21, len(returns) // 4))  # Up to 20 lags
        autocorr = [returns.autocorr(lag) for lag in lags]
        
        fig.add_trace(
            go.Bar(
                x=list(lags),
                y=autocorr,
                name='Autocorrelation',
                marker_color=self.color_palette[3],
                hovertemplate='Lag: %{x}<br>Autocorr: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Add significance bands for autocorrelation
        significance_level = 1.96 / np.sqrt(len(returns))
        fig.add_hline(y=significance_level, line_dash="dash", line_color="red", 
                     row=2, col=2, annotation_text="95% Confidence")
        fig.add_hline(y=-significance_level, line_dash="dash", line_color="red", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=f'Return Analysis - {results.strategy_name}',
            template=self.theme,
            width=self.figure_size[0],
            height=self.figure_size[1] + 200,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=2)
        fig.update_yaxes(title_text="Autocorrelation", row=2, col=2)
        
        return fig
    
    def create_performance_heatmap(
        self,
        performance_breakdown: PerformanceBreakdown,
        metric: str = 'returns'
    ) -> go.Figure:
        """
        Create performance heatmap by month/year.
        
        Args:
            performance_breakdown: Performance breakdown data
            metric: Metric to display ('returns', 'sharpe', 'win_rate')
            
        Returns:
            Interactive Plotly figure
        """
        if metric == 'returns':
            data = performance_breakdown.monthly_returns
            title_suffix = "Monthly Returns (%)"
            colorscale = 'RdYlGn'
            format_str = '.2%'
        elif metric == 'sharpe':
            data = performance_breakdown.monthly_sharpe
            title_suffix = "Monthly Sharpe Ratio"
            colorscale = 'Viridis'
            format_str = '.2f'
        elif metric == 'win_rate':
            data = performance_breakdown.monthly_win_rate
            title_suffix = "Monthly Win Rate"
            colorscale = 'Blues'
            format_str = '.0%'
        else:
            return self._create_empty_figure(f"Unknown metric: {metric}")
        
        if len(data) == 0:
            return self._create_empty_figure("No monthly data available")
        
        # Prepare data for heatmap
        data_df = data.to_frame('value')
        data_df['year'] = data_df.index.year
        data_df['month'] = data_df.index.month
        
        # Create pivot table
        heatmap_data = data_df.pivot(index='year', columns='month', values='value')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[month_names[i-1] for i in heatmap_data.columns],
            y=heatmap_data.index,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Value: %{z' + format_str + '}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Performance Heatmap - {title_suffix}',
            template=self.theme,
            width=self.figure_size[0],
            height=max(400, len(heatmap_data.index) * 40),
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        return fig
    
    def create_risk_metrics_dashboard(
        self,
        risk_analysis: RiskAnalysis,
        advanced_metrics: AdvancedMetrics
    ) -> go.Figure:
        """
        Create comprehensive risk metrics dashboard.
        
        Args:
            risk_analysis: Risk analysis data
            advanced_metrics: Advanced metrics data
            
        Returns:
            Interactive Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Value at Risk', 'Risk-Return Profile', 'Drawdown Analysis',
                'Tail Risk Metrics', 'Volatility Analysis', 'Risk Ratios'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Value at Risk
        var_data = {
            '1-Day 95%': risk_analysis.var_1d_95 * 100,
            '1-Day 99%': risk_analysis.var_1d_99 * 100,
            '1-Week 95%': risk_analysis.var_1w_95 * 100,
            '1-Week 99%': risk_analysis.var_1w_99 * 100
        }
        
        fig.add_trace(
            go.Bar(
                x=list(var_data.keys()),
                y=list(var_data.values()),
                name='VaR',
                marker_color=self.color_palette[0],
                hovertemplate='%{x}<br>VaR: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Risk-Return Profile (scatter plot)
        fig.add_trace(
            go.Scatter(
                x=[risk_analysis.realized_volatility * 100],
                y=[advanced_metrics.information_ratio],
                mode='markers',
                name='Strategy',
                marker=dict(size=15, color=self.color_palette[1]),
                hovertemplate='Volatility: %{x:.2f}%<br>Info Ratio: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Drawdown Analysis
        drawdown_data = {
            'Expected DD': risk_analysis.expected_drawdown * 100,
            'DD at Risk 95%': risk_analysis.drawdown_at_risk_95 * 100,
            'Avg DD': advanced_metrics.avg_drawdown * 100,
            'Pain Index': advanced_metrics.pain_index * 100
        }
        
        fig.add_trace(
            go.Bar(
                x=list(drawdown_data.keys()),
                y=list(drawdown_data.values()),
                name='Drawdown Metrics',
                marker_color=self.color_palette[2],
                hovertemplate='%{x}<br>Value: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=3
        )
        
        # 4. Tail Risk Metrics
        tail_data = {
            'Skewness': advanced_metrics.skewness,
            'Kurtosis': advanced_metrics.kurtosis,
            'Tail Ratio': advanced_metrics.tail_ratio,
            'Tail Expectation': risk_analysis.tail_expectation_ratio
        }
        
        fig.add_trace(
            go.Bar(
                x=list(tail_data.keys()),
                y=list(tail_data.values()),
                name='Tail Risk',
                marker_color=self.color_palette[3],
                hovertemplate='%{x}<br>Value: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Volatility Analysis
        vol_data = {
            'Realized Vol': risk_analysis.realized_volatility * 100,
            'Vol of Vol': risk_analysis.volatility_of_volatility * 100
        }
        
        fig.add_trace(
            go.Bar(
                x=list(vol_data.keys()),
                y=list(vol_data.values()),
                name='Volatility',
                marker_color=self.color_palette[4],
                hovertemplate='%{x}<br>Value: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Risk Ratios
        ratio_data = {
            'Sterling': advanced_metrics.sterling_ratio,
            'Burke': advanced_metrics.burke_ratio,
            'Martin': advanced_metrics.martin_ratio,
            'Return/MaxDD': risk_analysis.return_over_max_dd
        }
        
        fig.add_trace(
            go.Bar(
                x=list(ratio_data.keys()),
                y=list(ratio_data.values()),
                name='Risk Ratios',
                marker_color=self.color_palette[5],
                hovertemplate='%{x}<br>Ratio: %{y:.2f}<extra></extra>'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Risk Metrics Dashboard',
            template=self.theme,
            width=self.figure_size[0] + 200,
            height=self.figure_size[1] + 200,
            showlegend=False
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="VaR (%)", row=1, col=1)
        fig.update_yaxes(title_text="Information Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Drawdown (%)", row=1, col=3)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=2)
        fig.update_yaxes(title_text="Ratio", row=2, col=3)
        
        return fig
    
    def create_trade_analysis(
        self,
        trades: List[Trade],
        strategy_name: str
    ) -> go.Figure:
        """
        Create comprehensive trade analysis visualization.
        
        Args:
            trades: List of trade objects
            strategy_name: Name of the strategy
            
        Returns:
            Interactive Plotly figure
        """
        if not trades:
            return self._create_empty_figure("No trades available for analysis")
        
        # Prepare trade data
        trade_df = pd.DataFrame([
            {
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': trade.net_pnl,
                'return_pct': trade.return_pct * 100,
                'duration_hours': trade.duration_hours,
                'side': trade.side,
                'size': trade.size
            }
            for trade in trades
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'P&L Distribution', 'Trade Duration', 'Cumulative P&L',
                'Monthly Trade Count', 'Win/Loss Streaks', 'Trade Size Distribution'
            )
        )
        
        # 1. P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=trade_df['pnl'],
                nbinsx=30,
                name='P&L Distribution',
                marker_color=self.color_palette[0],
                opacity=0.7,
                hovertemplate='P&L: $%{x:,.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Trade Duration
        fig.add_trace(
            go.Histogram(
                x=trade_df['duration_hours'],
                nbinsx=20,
                name='Duration Distribution',
                marker_color=self.color_palette[1],
                opacity=0.7,
                hovertemplate='Duration: %{x:.1f} hours<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Cumulative P&L
        cumulative_pnl = trade_df['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=trade_df['exit_time'],
                y=cumulative_pnl,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color=self.color_palette[2], width=2),
                hovertemplate='Date: %{x}<br>Cumulative P&L: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=3
        )
        
        # 4. Monthly Trade Count
        monthly_trades = trade_df.set_index('exit_time').resample('M').size()
        fig.add_trace(
            go.Bar(
                x=monthly_trades.index,
                y=monthly_trades.values,
                name='Monthly Trades',
                marker_color=self.color_palette[3],
                hovertemplate='Month: %{x}<br>Trades: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Win/Loss Streaks
        wins_losses = (trade_df['pnl'] > 0).astype(int)
        streaks = []
        current_streak = 1
        current_type = wins_losses.iloc[0]
        
        for i in range(1, len(wins_losses)):
            if wins_losses.iloc[i] == current_type:
                current_streak += 1
            else:
                streaks.append(current_streak if current_type else -current_streak)
                current_streak = 1
                current_type = wins_losses.iloc[i]
        streaks.append(current_streak if current_type else -current_streak)
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(streaks))),
                y=streaks,
                name='Win/Loss Streaks',
                marker_color=['green' if s > 0 else 'red' for s in streaks],
                hovertemplate='Streak %{x}: %{y} trades<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 6. Trade Size Distribution
        fig.add_trace(
            go.Histogram(
                x=trade_df['size'],
                nbinsx=20,
                name='Size Distribution',
                marker_color=self.color_palette[4],
                opacity=0.7,
                hovertemplate='Size: %{x:,.0f}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title=f'Trade Analysis - {strategy_name}',
            template=self.theme,
            width=self.figure_size[0] + 200,
            height=self.figure_size[1] + 200,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="Duration (hours)", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=1, col=3)
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_xaxes(title_text="Streak #", row=2, col=2)
        fig.update_xaxes(title_text="Trade Size", row=2, col=3)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=3)
        fig.update_yaxes(title_text="Trade Count", row=2, col=1)
        fig.update_yaxes(title_text="Streak Length", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=3)
        
        return fig
    
    def create_strategy_comparison(
        self,
        performance_reports: Dict[str, PerformanceReport],
        metrics: List[str] = None
    ) -> go.Figure:
        """
        Create strategy comparison visualization.
        
        Args:
            performance_reports: Dictionary of strategy names to performance reports
            metrics: List of metrics to compare
            
        Returns:
            Interactive Plotly figure
        """
        if not performance_reports:
            return self._create_empty_figure("No strategies to compare")
        
        default_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'calmar_ratio', 'sortino_ratio', 'volatility'
        ]
        metrics = metrics or default_metrics
        
        # Prepare comparison data
        comparison_data = {}
        strategy_names = list(performance_reports.keys())
        
        for metric in metrics:
            comparison_data[metric] = []
            for strategy_name in strategy_names:
                results = performance_reports[strategy_name].basic_results
                value = getattr(results, metric, 0)
                
                # Convert to percentage for certain metrics
                if metric in ['total_return', 'max_drawdown', 'win_rate', 'volatility']:
                    value *= 100
                
                comparison_data[metric].append(value)
        
        # Create radar chart
        fig = go.Figure()
        
        for i, strategy_name in enumerate(strategy_names):
            # Normalize metrics for radar chart (0-100 scale)
            normalized_values = []
            for metric in metrics:
                values = comparison_data[metric]
                if metric == 'max_drawdown':  # Lower is better
                    normalized = 100 - (abs(values[i]) / max(abs(v) for v in values) * 100)
                else:  # Higher is better
                    max_val = max(values) if max(values) > 0 else 1
                    normalized = (values[i] / max_val) * 100
                normalized_values.append(max(0, min(100, normalized)))
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values + [normalized_values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=strategy_name,
                line_color=self.color_palette[i % len(self.color_palette)],
                hovertemplate=f'<b>{strategy_name}</b><br>' +
                             'Metric: %{theta}<br>' +
                             'Normalized Score: %{r:.1f}<extra></extra>'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title='Strategy Comparison - Normalized Metrics',
            template=self.theme,
            width=self.figure_size[0],
            height=self.figure_size[1],
            showlegend=True
        )
        
        return fig
    
    def create_rolling_metrics(
        self,
        performance_breakdown: PerformanceBreakdown,
        strategy_name: str
    ) -> go.Figure:
        """
        Create rolling metrics visualization.
        
        Args:
            performance_breakdown: Performance breakdown data
            strategy_name: Name of the strategy
            
        Returns:
            Interactive Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                'Rolling 30-Day Sharpe Ratio',
                'Rolling 30-Day Maximum Drawdown',
                'Rolling 30-Day Volatility'
            )
        )
        
        # Rolling Sharpe
        if len(performance_breakdown.rolling_sharpe_30d) > 0:
            fig.add_trace(
                go.Scatter(
                    x=performance_breakdown.rolling_sharpe_30d.index,
                    y=performance_breakdown.rolling_sharpe_30d.values,
                    mode='lines',
                    name='Rolling Sharpe',
                    line=dict(color=self.color_palette[0], width=2),
                    hovertemplate='Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Rolling Max Drawdown
        if len(performance_breakdown.rolling_max_dd_30d) > 0:
            fig.add_trace(
                go.Scatter(
                    x=performance_breakdown.rolling_max_dd_30d.index,
                    y=performance_breakdown.rolling_max_dd_30d.values * 100,
                    mode='lines',
                    name='Rolling Max DD',
                    line=dict(color=self.color_palette[1], width=2),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    hovertemplate='Date: %{x}<br>Max DD: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Rolling Volatility
        if len(performance_breakdown.rolling_volatility_30d) > 0:
            fig.add_trace(
                go.Scatter(
                    x=performance_breakdown.rolling_volatility_30d.index,
                    y=performance_breakdown.rolling_volatility_30d.values * 100,
                    mode='lines',
                    name='Rolling Volatility',
                    line=dict(color=self.color_palette[2], width=2),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'Rolling Performance Metrics - {strategy_name}',
            template=self.theme,
            width=self.figure_size[0],
            height=self.figure_size[1] + 300,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Max Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=3, col=1)
        
        return fig
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template=self.theme,
            width=self.figure_size[0],
            height=self.figure_size[1],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig
    
    def save_figure(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'html'
    ) -> str:
        """
        Save figure to file.
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
            format: Output format ('html', 'png', 'pdf', 'svg')
            
        Returns:
            Path to saved file
        """
        try:
            if format == 'html':
                fig.write_html(filename, config=self.chart_config)
            elif format == 'png':
                fig.write_image(filename, format='png')
            elif format == 'pdf':
                fig.write_image(filename, format='pdf')
            elif format == 'svg':
                fig.write_image(filename, format='svg')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Figure saved to {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Failed to save figure: {e}")
            raise 