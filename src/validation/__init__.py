"""
Validation Framework

This package provides comprehensive validation tools for trading strategies including:
- Out-of-sample testing with data splitting
- Cross-asset validation
- Monte Carlo simulation
- Statistical significance testing
- Market regime analysis
- Comprehensive robustness scoring
"""

from .data_splitter import DataSplitter
from .cross_asset_validator import CrossAssetValidator
from .monte_carlo_tester import MonteCarloTester
from .statistical_tester import StatisticalTester
from .regime_analyzer import RegimeAnalyzer
from .robustness_scorer import RobustnessScorer

__all__ = [
    'DataSplitter',
    'CrossAssetValidator',
    'MonteCarloTester',
    'StatisticalTester',
    'RegimeAnalyzer',
    'RobustnessScorer'
] 