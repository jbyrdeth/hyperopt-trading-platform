"""
Automated Reporting System

This package provides comprehensive PDF report generation for trading strategy
analysis, performance metrics, validation results, and risk assessments.
"""

from .report_generator import ReportGenerator, ReportConfiguration, ReportCanvas
from .templates import (
    ReportTemplate,
    ReportStyling,
    ExecutiveSummaryTemplate,
    PerformanceAnalysisTemplate,
    ValidationResultsTemplate,
    RiskAssessmentTemplate,
    TechnicalAppendixTemplate
)
from .data_integration import ReportDataCollector, ReportDataPackage
from .visualization import ReportVisualizationEngine, ChartStyling
from .analysis import (
    AutomatedAnalysisEngine,
    ComprehensiveAnalysis,
    PerformanceAnalysis,
    RiskAnalysis,
    ValidationAnalysis,
    MarketAnalysis,
    PerformanceLevel,
    RiskLevel,
    DeploymentRecommendation
)

__all__ = [
    # Main components
    'ReportGenerator',
    'ReportDataCollector',
    'ReportVisualizationEngine',
    'AutomatedAnalysisEngine',
    
    # Configuration
    'ReportConfiguration',
    'ReportStyling',
    'ChartStyling',
    
    # Templates
    'ReportTemplate',
    'ExecutiveSummaryTemplate',
    'PerformanceAnalysisTemplate',
    'ValidationResultsTemplate',
    'RiskAssessmentTemplate',
    'TechnicalAppendixTemplate',
    
    # Data structures
    'ReportDataPackage',
    'ComprehensiveAnalysis',
    'PerformanceAnalysis',
    'RiskAnalysis',
    'ValidationAnalysis',
    'MarketAnalysis',
    
    # Enums
    'PerformanceLevel',
    'RiskLevel',
    'DeploymentRecommendation',
    
    # Advanced components
    'ReportCanvas'
] 