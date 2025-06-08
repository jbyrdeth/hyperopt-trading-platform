"""
Report Visualization Engine

This module provides specialized visualization components for PDF report generation,
optimized for print format with professional styling and ReportLab integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import io
import base64
from pathlib import Path

# ReportLab imports for PDF-native charts
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis
from reportlab.graphics.widgets.grids import Grid
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import Image, Spacer
from reportlab.graphics.renderPDF import drawToString

# Plotly imports for advanced charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Matplotlib imports as fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from ..utils.logger import get_logger
    from .data_integration import ReportDataPackage
except ImportError:
    from src.utils.logger import get_logger
    from src.reporting.data_integration import ReportDataPackage


@dataclass
class ChartStyling:
    """Professional chart styling configuration."""
    
    # Colors (matching report theme)
    primary_color: colors.Color = colors.Color(0.1, 0.2, 0.4)  # Dark blue
    secondary_color: colors.Color = colors.Color(0.2, 0.4, 0.6)  # Medium blue
    accent_color: colors.Color = colors.Color(0.0, 0.5, 0.8)  # Light blue
    success_color: colors.Color = colors.Color(0.0, 0.6, 0.0)  # Green
    warning_color: colors.Color = colors.Color(0.8, 0.6, 0.0)  # Orange
    danger_color: colors.Color = colors.Color(0.8, 0.0, 0.0)  # Red
    neutral_color: colors.Color = colors.Color(0.5, 0.5, 0.5)  # Gray
    
    # Chart dimensions
    chart_width: float = 6 * inch
    chart_height: float = 4 * inch
    small_chart_width: float = 4 * inch
    small_chart_height: float = 3 * inch
    
    # Fonts
    title_font: str = 'Helvetica-Bold'
    label_font: str = 'Helvetica'
    title_size: int = 12
    label_size: int = 10
    
    # Grid and styling
    grid_color: colors.Color = colors.Color(0.9, 0.9, 0.9)
    background_color: colors.Color = colors.white
    
    def get_color_palette(self, n_colors: int = 8) -> List[colors.Color]:
        """Get a color palette for multiple series."""
        base_colors = [
            self.primary_color,
            self.secondary_color,
            self.accent_color,
            self.success_color,
            self.warning_color,
            self.danger_color,
            colors.Color(0.6, 0.2, 0.8),  # Purple
            colors.Color(0.8, 0.4, 0.2),  # Brown
        ]
        
        # Extend palette if needed
        while len(base_colors) < n_colors:
            base_colors.extend(base_colors)
        
        return base_colors[:n_colors]


class ReportLabChartGenerator:
    """Generates ReportLab-native charts for direct PDF embedding."""
    
    def __init__(self, styling: ChartStyling = None):
        """
        Initialize the chart generator.
        
        Args:
            styling: Chart styling configuration
        """
        self.styling = styling or ChartStyling()
        self.logger = get_logger("reportlab_chart_generator")
    
    def create_performance_summary_chart(
        self,
        metrics: Dict[str, float],
        title: str = "Performance Summary"
    ) -> Drawing:
        """Create a horizontal bar chart for performance metrics."""
        
        drawing = Drawing(self.styling.chart_width, self.styling.chart_height)
        
        # Prepare data
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Create chart
        chart = HorizontalBarChart()
        chart.x = 50
        chart.y = 50
        chart.width = self.styling.chart_width - 100
        chart.height = self.styling.chart_height - 100
        
        # Set data
        chart.data = [metric_values]
        chart.categoryAxis.categoryNames = metric_names
        
        # Styling
        chart.bars[0].fillColor = self.styling.primary_color
        chart.categoryAxis.labels.fontName = self.styling.label_font
        chart.categoryAxis.labels.fontSize = self.styling.label_size
        chart.valueAxis.labels.fontName = self.styling.label_font
        chart.valueAxis.labels.fontSize = self.styling.label_size
        
        # Add to drawing
        drawing.add(chart)
        
        # Add title
        title_string = String(
            self.styling.chart_width / 2,
            self.styling.chart_height - 30,
            title
        )
        title_string.fontName = self.styling.title_font
        title_string.fontSize = self.styling.title_size
        title_string.textAnchor = 'middle'
        title_string.fillColor = self.styling.primary_color
        drawing.add(title_string)
        
        return drawing
    
    def create_risk_breakdown_pie(
        self,
        risk_components: Dict[str, float],
        title: str = "Risk Breakdown"
    ) -> Drawing:
        """Create a pie chart for risk component breakdown."""
        
        drawing = Drawing(self.styling.chart_width, self.styling.chart_height)
        
        # Prepare data
        labels = list(risk_components.keys())
        values = list(risk_components.values())
        
        # Create pie chart
        pie = Pie()
        pie.x = 50
        pie.y = 50
        pie.width = min(self.styling.chart_width - 200, self.styling.chart_height - 100)
        pie.height = pie.width
        
        # Set data
        pie.data = values
        pie.labels = labels
        
        # Styling
        colors_palette = self.styling.get_color_palette(len(values))
        for i, color in enumerate(colors_palette):
            pie.slices[i].fillColor = color
        
        pie.slices.labelRadius = 1.2
        pie.slices.fontName = self.styling.label_font
        pie.slices.fontSize = self.styling.label_size
        
        # Add to drawing
        drawing.add(pie)
        
        # Add title
        title_string = String(
            self.styling.chart_width / 2,
            self.styling.chart_height - 30,
            title
        )
        title_string.fontName = self.styling.title_font
        title_string.fontSize = self.styling.title_size
        title_string.textAnchor = 'middle'
        title_string.fillColor = self.styling.primary_color
        drawing.add(title_string)
        
        # Add legend
        legend = Legend()
        legend.x = pie.x + pie.width + 20
        legend.y = pie.y + pie.height / 2
        legend.deltay = 18
        legend.dxTextSpace = 5
        legend.columnMaximum = len(labels)
        legend.alignment = 'left'
        legend.colorNamePairs = [(colors_palette[i], labels[i]) for i in range(len(labels))]
        legend.fontName = self.styling.label_font
        legend.fontSize = self.styling.label_size
        drawing.add(legend)
        
        return drawing
    
    def create_validation_scores_chart(
        self,
        validation_scores: Dict[str, float],
        title: str = "Validation Scores"
    ) -> Drawing:
        """Create a vertical bar chart for validation scores."""
        
        drawing = Drawing(self.styling.chart_width, self.styling.chart_height)
        
        # Prepare data
        categories = list(validation_scores.keys())
        scores = list(validation_scores.values())
        
        # Create chart
        chart = VerticalBarChart()
        chart.x = 50
        chart.y = 50
        chart.width = self.styling.chart_width - 100
        chart.height = self.styling.chart_height - 100
        
        # Set data
        chart.data = [scores]
        chart.categoryAxis.categoryNames = categories
        
        # Styling
        chart.bars[0].fillColor = self.styling.secondary_color
        chart.categoryAxis.labels.fontName = self.styling.label_font
        chart.categoryAxis.labels.fontSize = self.styling.label_size
        chart.categoryAxis.labels.angle = 45
        chart.valueAxis.labels.fontName = self.styling.label_font
        chart.valueAxis.labels.fontSize = self.styling.label_size
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = 100
        
        # Add reference lines
        for score_line in [60, 80]:  # Threshold lines
            line = Line(
                chart.x,
                chart.y + (score_line / 100) * chart.height,
                chart.x + chart.width,
                chart.y + (score_line / 100) * chart.height
            )
            line.strokeColor = self.styling.warning_color if score_line == 60 else self.styling.success_color
            line.strokeDashArray = [3, 3]
            drawing.add(line)
        
        # Add to drawing
        drawing.add(chart)
        
        # Add title
        title_string = String(
            self.styling.chart_width / 2,
            self.styling.chart_height - 30,
            title
        )
        title_string.fontName = self.styling.title_font
        title_string.fontSize = self.styling.title_size
        title_string.textAnchor = 'middle'
        title_string.fillColor = self.styling.primary_color
        drawing.add(title_string)
        
        return drawing
    
    def create_correlation_matrix(
        self,
        correlation_data: Dict[str, Dict[str, float]],
        title: str = "Correlation Matrix"
    ) -> Drawing:
        """Create a correlation matrix visualization."""
        
        drawing = Drawing(self.styling.chart_width, self.styling.chart_height)
        
        # Prepare data
        factors = list(correlation_data.keys())
        n_factors = len(factors)
        
        if n_factors == 0:
            return drawing
        
        # Calculate cell size
        matrix_size = min(self.styling.chart_width - 100, self.styling.chart_height - 100)
        cell_size = matrix_size / n_factors
        
        # Draw matrix
        start_x = 50
        start_y = 50
        
        for i, factor1 in enumerate(factors):
            for j, factor2 in enumerate(factors):
                correlation = correlation_data.get(factor1, {}).get(factor2, 0)
                
                # Calculate color intensity based on correlation
                intensity = abs(correlation)
                if correlation > 0:
                    color = colors.Color(intensity, 0, 0)  # Red for positive
                else:
                    color = colors.Color(0, 0, intensity)  # Blue for negative
                
                # Draw cell
                rect = Rect(
                    start_x + j * cell_size,
                    start_y + (n_factors - i - 1) * cell_size,
                    cell_size,
                    cell_size
                )
                rect.fillColor = color
                rect.strokeColor = colors.black
                rect.strokeWidth = 0.5
                drawing.add(rect)
                
                # Add correlation value
                if cell_size > 30:  # Only add text if cell is large enough
                    text = String(
                        start_x + j * cell_size + cell_size / 2,
                        start_y + (n_factors - i - 1) * cell_size + cell_size / 2,
                        f"{correlation:.2f}"
                    )
                    text.fontName = self.styling.label_font
                    text.fontSize = max(6, min(10, cell_size / 4))
                    text.textAnchor = 'middle'
                    text.fillColor = colors.white if intensity > 0.5 else colors.black
                    drawing.add(text)
        
        # Add labels
        for i, factor in enumerate(factors):
            # Y-axis labels
            label = String(
                start_x - 5,
                start_y + (n_factors - i - 1) * cell_size + cell_size / 2,
                factor
            )
            label.fontName = self.styling.label_font
            label.fontSize = self.styling.label_size
            label.textAnchor = 'end'
            drawing.add(label)
            
            # X-axis labels
            label = String(
                start_x + i * cell_size + cell_size / 2,
                start_y - 10,
                factor
            )
            label.fontName = self.styling.label_font
            label.fontSize = self.styling.label_size
            label.textAnchor = 'middle'
            drawing.add(label)
        
        # Add title
        title_string = String(
            self.styling.chart_width / 2,
            self.styling.chart_height - 30,
            title
        )
        title_string.fontName = self.styling.title_font
        title_string.fontSize = self.styling.title_size
        title_string.textAnchor = 'middle'
        title_string.fillColor = self.styling.primary_color
        drawing.add(title_string)
        
        return drawing


class PlotlyChartGenerator:
    """Generates Plotly charts for advanced visualizations."""
    
    def __init__(self, styling: ChartStyling = None):
        """
        Initialize the chart generator.
        
        Args:
            styling: Chart styling configuration
        """
        self.styling = styling or ChartStyling()
        self.logger = get_logger("plotly_chart_generator")
        
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available, some charts may not be generated")
    
    def create_equity_curve(
        self,
        equity_data: pd.Series,
        benchmark_data: pd.Series = None,
        title: str = "Equity Curve"
    ) -> Optional[go.Figure]:
        """Create an equity curve chart."""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Add strategy equity curve
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=equity_data.values,
            mode='lines',
            name='Strategy',
            line=dict(color='#1f4e79', width=2)
        ))
        
        # Add benchmark if provided
        if benchmark_data is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_data.index,
                y=benchmark_data.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='#666666', width=1, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#1f4e79')),
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            width=800,
            height=500,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_drawdown_chart(
        self,
        drawdown_data: pd.Series,
        title: str = "Drawdown Analysis"
    ) -> Optional[go.Figure]:
        """Create a drawdown chart."""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=drawdown_data.index,
            y=drawdown_data.values,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='#cc0000', width=1),
            fillcolor='rgba(204, 0, 0, 0.3)'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#1f4e79')),
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            width=800,
            height=400,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Format y-axis as percentage
        fig.update_yaxes(tickformat='.1%')
        
        return fig
    
    def create_returns_distribution(
        self,
        returns_data: pd.Series,
        title: str = "Returns Distribution"
    ) -> Optional[go.Figure]:
        """Create a returns distribution histogram."""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=returns_data.values,
            nbinsx=50,
            name='Returns',
            marker_color='#1f4e79',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mean_return = returns_data.mean()
        std_return = returns_data.std()
        x_range = np.linspace(returns_data.min(), returns_data.max(), 100)
        normal_dist = (1 / (std_return * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean_return) / std_return) ** 2)
        
        # Scale normal distribution to match histogram
        hist_max = len(returns_data) * (returns_data.max() - returns_data.min()) / 50
        normal_dist = normal_dist * hist_max / normal_dist.max()
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#cc0000', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#1f4e79')),
            xaxis_title="Daily Returns",
            yaxis_title="Frequency",
            template="plotly_white",
            width=800,
            height=400,
            showlegend=True,
            legend=dict(x=0.7, y=0.9),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Format x-axis as percentage
        fig.update_xaxes(tickformat='.1%')
        
        return fig
    
    def create_monthly_returns_heatmap(
        self,
        returns_data: pd.Series,
        title: str = "Monthly Returns Heatmap"
    ) -> Optional[go.Figure]:
        """Create a monthly returns heatmap."""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        # Prepare monthly returns data
        monthly_returns = returns_data.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = monthly_returns.index.to_period('M')
        
        # Create pivot table for heatmap
        monthly_data = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        
        pivot_data = monthly_data.pivot(index='Year', columns='Month', values='Return')
        
        # Month names for columns
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=month_names,
            y=pivot_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_data.values * 100, 1),
            texttemplate="%{text}%",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#1f4e79')),
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_risk_metrics_radar(
        self,
        risk_metrics: Dict[str, float],
        title: str = "Risk Metrics Radar"
    ) -> Optional[go.Figure]:
        """Create a radar chart for risk metrics."""
        
        if not PLOTLY_AVAILABLE:
            return None
        
        # Prepare data
        categories = list(risk_metrics.keys())
        values = list(risk_metrics.values())
        
        # Normalize values to 0-100 scale for radar chart
        normalized_values = [(v / max(values)) * 100 for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='Risk Profile',
            line_color='#1f4e79',
            fillcolor='rgba(31, 78, 121, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title=dict(text=title, font=dict(size=16, color='#1f4e79')),
            template="plotly_white",
            width=600,
            height=600,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig


class ReportVisualizationEngine:
    """Main visualization engine for report generation."""
    
    def __init__(self, styling: ChartStyling = None):
        """
        Initialize the visualization engine.
        
        Args:
            styling: Chart styling configuration
        """
        self.styling = styling or ChartStyling()
        self.logger = get_logger("report_visualization_engine")
        
        # Initialize chart generators
        self.reportlab_generator = ReportLabChartGenerator(self.styling)
        self.plotly_generator = PlotlyChartGenerator(self.styling)
    
    def generate_executive_summary_charts(
        self,
        data_package: ReportDataPackage
    ) -> Dict[str, Any]:
        """Generate charts for executive summary section."""
        
        charts = {}
        
        try:
            # Performance summary chart
            performance_metrics = {
                'Total Return': data_package.performance_metrics.get('total_return', 0) * 100,
                'Sharpe Ratio': data_package.performance_metrics.get('sharpe_ratio', 0),
                'Max Drawdown': abs(data_package.performance_metrics.get('max_drawdown', 0)) * 100,
                'Win Rate': data_package.trading_metrics.get('win_rate', 0) * 100
            }
            
            charts['performance_summary'] = self.reportlab_generator.create_performance_summary_chart(
                performance_metrics, "Key Performance Metrics"
            )
            
            # Risk breakdown pie chart
            risk_components = {
                'Performance Risk': data_package.risk_assessment.get('performance_risk', 0),
                'Overfitting Risk': data_package.risk_assessment.get('overfitting_risk', 0),
                'Validation Risk': data_package.risk_assessment.get('validation_risk', 0)
            }
            
            charts['risk_breakdown'] = self.reportlab_generator.create_risk_breakdown_pie(
                risk_components, "Risk Assessment Breakdown"
            )
            
            # Validation scores chart
            validation_scores = {
                'Multi-Period': data_package.multi_period_validation.get('validation_score', 0) * 100 if data_package.multi_period_validation else 0,
                'Cross-Asset': data_package.cross_asset_validation.get('consistency_score', 0) if data_package.cross_asset_validation else 0,
                'Regime Analysis': data_package.regime_analysis.get('regime_consistency', 0) * 100 if data_package.regime_analysis else 0,
                'Overall': data_package.validation_results.get('overall_score', 0)
            }
            
            charts['validation_scores'] = self.reportlab_generator.create_validation_scores_chart(
                validation_scores, "Validation Framework Results"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary charts: {e}")
        
        return charts
    
    def generate_performance_analysis_charts(
        self,
        data_package: ReportDataPackage,
        equity_data: pd.Series = None,
        returns_data: pd.Series = None
    ) -> Dict[str, Any]:
        """Generate charts for performance analysis section."""
        
        charts = {}
        
        try:
            # Equity curve (if data provided)
            if equity_data is not None and PLOTLY_AVAILABLE:
                charts['equity_curve'] = self.plotly_generator.create_equity_curve(
                    equity_data, title="Strategy Equity Curve"
                )
            
            # Drawdown chart (if data provided)
            if returns_data is not None and PLOTLY_AVAILABLE:
                # Calculate drawdown from returns
                cumulative_returns = (1 + returns_data).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                
                charts['drawdown_chart'] = self.plotly_generator.create_drawdown_chart(
                    drawdown, title="Drawdown Analysis"
                )
                
                # Returns distribution
                charts['returns_distribution'] = self.plotly_generator.create_returns_distribution(
                    returns_data, title="Daily Returns Distribution"
                )
                
                # Monthly returns heatmap
                charts['monthly_returns'] = self.plotly_generator.create_monthly_returns_heatmap(
                    returns_data, title="Monthly Returns Heatmap"
                )
            
            # Risk metrics radar chart
            if PLOTLY_AVAILABLE:
                risk_metrics = {
                    'Volatility': data_package.performance_metrics.get('volatility', 0) * 100,
                    'Max Drawdown': abs(data_package.performance_metrics.get('max_drawdown', 0)) * 100,
                    'VaR (95%)': abs(data_package.performance_metrics.get('var_95', 0)) * 100,
                    'Sharpe Ratio': max(0, data_package.performance_metrics.get('sharpe_ratio', 0)) * 10,
                    'Sortino Ratio': max(0, data_package.performance_metrics.get('sortino_ratio', 0)) * 10
                }
                
                charts['risk_radar'] = self.plotly_generator.create_risk_metrics_radar(
                    risk_metrics, title="Risk Metrics Profile"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance analysis charts: {e}")
        
        return charts
    
    def generate_validation_charts(
        self,
        data_package: ReportDataPackage
    ) -> Dict[str, Any]:
        """Generate charts for validation results section."""
        
        charts = {}
        
        try:
            # Multi-period performance chart
            if data_package.multi_period_validation:
                period_performances = data_package.multi_period_validation.get('period_performances', [])
                if period_performances:
                    period_returns = {
                        p['period_type']: p['total_return'] * 100 
                        for p in period_performances[:8]  # Top 8 periods
                    }
                    
                    charts['period_performance'] = self.reportlab_generator.create_performance_summary_chart(
                        period_returns, "Multi-Period Performance"
                    )
            
            # Cross-asset performance chart
            if data_package.cross_asset_validation:
                asset_results = data_package.cross_asset_validation.get('asset_results', [])
                if asset_results:
                    asset_returns = {
                        a['asset_symbol']: a['total_return'] * 100 
                        for a in asset_results[:6]  # Top 6 assets
                    }
                    
                    charts['cross_asset_performance'] = self.reportlab_generator.create_performance_summary_chart(
                        asset_returns, "Cross-Asset Performance"
                    )
            
            # Market correlation matrix
            if data_package.market_dependency:
                correlations = data_package.market_dependency.get('market_correlations', {})
                if correlations:
                    # Create correlation matrix (simplified for visualization)
                    correlation_matrix = {
                        factor: {factor: 1.0, **{k: v for k, v in correlations.items() if k != factor}}
                        for factor in correlations.keys()
                    }
                    
                    charts['correlation_matrix'] = self.reportlab_generator.create_correlation_matrix(
                        correlation_matrix, "Market Correlation Analysis"
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation charts: {e}")
        
        return charts
    
    def generate_risk_assessment_charts(
        self,
        data_package: ReportDataPackage
    ) -> Dict[str, Any]:
        """Generate charts for risk assessment section."""
        
        charts = {}
        
        try:
            # Anti-overfitting risk breakdown
            if data_package.anti_overfitting:
                overfitting_risks = {
                    'Data Mining': data_package.anti_overfitting.get('data_mining_risk', 0),
                    'Overfitting': data_package.anti_overfitting.get('overfitting_risk', 0),
                    'Complexity': data_package.anti_overfitting.get('complexity_score', 0),
                    'Decay Risk': data_package.anti_overfitting.get('decay_risk', 0)
                }
                
                charts['overfitting_breakdown'] = self.reportlab_generator.create_risk_breakdown_pie(
                    overfitting_risks, "Anti-Overfitting Risk Analysis"
                )
            
            # Overall risk components
            risk_components = {
                'Performance Risk': data_package.risk_assessment.get('performance_risk', 0),
                'Overfitting Risk': data_package.risk_assessment.get('overfitting_risk', 0),
                'Validation Risk': data_package.risk_assessment.get('validation_risk', 0)
            }
            
            charts['overall_risk'] = self.reportlab_generator.create_risk_breakdown_pie(
                risk_components, "Overall Risk Assessment"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate risk assessment charts: {e}")
        
        return charts
    
    def convert_chart_to_image(self, chart: Any, format: str = 'PNG') -> Optional[Image]:
        """Convert a chart to an image for embedding in PDF."""
        
        if chart is None:
            return None
        
        try:
            if hasattr(chart, 'to_image'):  # Plotly figure
                if not PLOTLY_AVAILABLE:
                    self.logger.warning("Plotly not available for chart conversion")
                    return None
                
                # Try to convert to image
                try:
                    img_bytes = chart.to_image(format=format.lower(), width=800, height=600)
                    return Image(io.BytesIO(img_bytes), width=6*inch, height=4*inch)
                except Exception as e:
                    self.logger.warning(f"Failed to convert Plotly chart to image: {e}")
                    return None
                    
            elif hasattr(chart, 'asDrawing'):  # ReportLab drawing
                return chart
            else:
                self.logger.warning(f"Unknown chart type: {type(chart)}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to convert chart to image: {e}")
            return None
    
    def generate_all_charts(
        self,
        data_package: ReportDataPackage,
        equity_data: pd.Series = None,
        returns_data: pd.Series = None
    ) -> Dict[str, Dict[str, Any]]:
        """Generate all charts for the report."""
        
        self.logger.info(f"Generating all charts for strategy: {data_package.strategy_name}")
        
        all_charts = {
            'executive_summary': self.generate_executive_summary_charts(data_package),
            'performance_analysis': self.generate_performance_analysis_charts(data_package, equity_data, returns_data),
            'validation_results': self.generate_validation_charts(data_package),
            'risk_assessment': self.generate_risk_assessment_charts(data_package)
        }
        
        self.logger.info(f"Generated {sum(len(charts) for charts in all_charts.values())} charts")
        
        return all_charts 