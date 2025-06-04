"""
Performance Analytics Engine

This package provides comprehensive performance analytics for trading strategies including:
- Advanced performance metrics calculation
- Interactive visualizations with Plotly
- Strategy comparison and ranking
- Performance dashboards
"""

from .performance_analyzer import PerformanceAnalyzer
from .visualization_engine import VisualizationEngine
from .strategy_comparator import StrategyComparator
from .performance_dashboard import PerformanceDashboard

__all__ = [
    'PerformanceAnalyzer',
    'VisualizationEngine', 
    'StrategyComparator',
    'PerformanceDashboard'
] 