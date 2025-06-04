"""
Anti-Overfitting Framework

This package provides comprehensive anti-overfitting measures for trading strategy
development and validation, ensuring robust and generalizable strategies.
"""

from .data_splitting_framework import AntiOverfittingDataSplitter, OverfittingDetector

# Optional imports for modules that may not exist yet
try:
    from .trade_requirements import TradeRequirementValidator
except ImportError:
    TradeRequirementValidator = None

try:
    from .win_rate_penalties import WinRatePenaltyCalculator
except ImportError:
    WinRatePenaltyCalculator = None

try:
    from .robustness_scoring import RobustnessScorer as AntiOverfittingRobustnessScorer
except ImportError:
    AntiOverfittingRobustnessScorer = None

try:
    from .walk_forward_analyzer import WalkForwardAnalyzer
except ImportError:
    WalkForwardAnalyzer = None

try:
    from .complexity_penalty import ComplexityPenaltyCalculator
except ImportError:
    ComplexityPenaltyCalculator = None

try:
    from .correlation_analyzer import CorrelationAnalyzer
except ImportError:
    CorrelationAnalyzer = None

from .cross_asset_tester import CrossAssetTester, AssetGroupManager
from .parameter_constraints import (
    ParameterConstraintManager, 
    BacktestTracker, 
    ParameterImportanceAnalyzer,
    ParameterConstraints,
    ParameterImportance,
    OptimizationBudget
)
from .multi_period_validator import (
    MultiPeriodValidator,
    PerformanceConsistencyAnalyzer,
    ValidationPeriodType,
    ConsistencyLevel,
    PeriodPerformance,
    ConsistencyMetrics,
    MultiPeriodValidationResult
)
from .fundamental_inefficiency_analyzer import (
    FundamentalInefficiencyAnalyzer,
    InefficiencyClassifier,
    InefficiencyType,
    StrategyCategory,
    EconomicRationaleStrength,
    PersistenceLevel,
    InefficiencyDocumentation,
    InefficiencyAnalysis,
    FundamentalInefficiencyResult
)

__all__ = [
    'AntiOverfittingDataSplitter',
    'OverfittingDetector',
    'CrossAssetTester',
    'AssetGroupManager',
    'ParameterConstraintManager',
    'BacktestTracker',
    'ParameterImportanceAnalyzer',
    'ParameterConstraints',
    'ParameterImportance',
    'OptimizationBudget',
    'MultiPeriodValidator',
    'PerformanceConsistencyAnalyzer',
    'ValidationPeriodType',
    'ConsistencyLevel',
    'PeriodPerformance',
    'ConsistencyMetrics',
    'MultiPeriodValidationResult',
    'FundamentalInefficiencyAnalyzer',
    'InefficiencyClassifier',
    'InefficiencyType',
    'StrategyCategory',
    'EconomicRationaleStrength',
    'PersistenceLevel',
    'InefficiencyDocumentation',
    'InefficiencyAnalysis',
    'FundamentalInefficiencyResult'
]

# Add optional imports to __all__ if they exist
if TradeRequirementValidator is not None:
    __all__.append('TradeRequirementValidator')
if WinRatePenaltyCalculator is not None:
    __all__.append('WinRatePenaltyCalculator')
if AntiOverfittingRobustnessScorer is not None:
    __all__.append('AntiOverfittingRobustnessScorer')
if WalkForwardAnalyzer is not None:
    __all__.append('WalkForwardAnalyzer')
if ComplexityPenaltyCalculator is not None:
    __all__.append('ComplexityPenaltyCalculator')
if CorrelationAnalyzer is not None:
    __all__.append('CorrelationAnalyzer') 