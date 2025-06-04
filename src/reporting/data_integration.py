"""
Data Integration Layer

This module provides comprehensive data collection and processing capabilities
for automated report generation, integrating with all system components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import pickle
import os
import hashlib
from pathlib import Path
import logging

try:
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from ..analytics.performance_analyzer import PerformanceAnalyzer
    from ..analytics.visualization_engine import VisualizationEngine
    from ..analytics.strategy_comparator import StrategyComparator
    from ..validation.regime_analyzer import RegimeAnalyzer, RegimeAnalysisResults
    from ..anti_overfitting.multi_period_validator import MultiPeriodValidator, MultiPeriodValidationResult
    from ..anti_overfitting.cross_asset_tester import CrossAssetTester, CrossAssetTestResult
    from ..anti_overfitting.fundamental_inefficiency_analyzer import FundamentalInefficiencyAnalyzer, FundamentalInefficiencyResult
    from ..anti_overfitting.parameter_constraints import ParameterConstraintManager, OptimizationBudget
    from ..optimization.hyperopt_optimizer import HyperoptOptimizer, OptimizationResult
    from ..utils.logger import get_logger
    # Optional dashboard import
    try:
        from ..analytics.dashboard import PerformanceDashboard
    except ImportError:
        PerformanceDashboard = None
except ImportError:
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.analytics.performance_analyzer import PerformanceAnalyzer
    from src.analytics.visualization_engine import VisualizationEngine
    from src.analytics.strategy_comparator import StrategyComparator
    from src.validation.regime_analyzer import RegimeAnalyzer, RegimeAnalysisResults
    from src.anti_overfitting.multi_period_validator import MultiPeriodValidator, MultiPeriodValidationResult
    from src.anti_overfitting.cross_asset_tester import CrossAssetTester, CrossAssetTestResult
    from src.anti_overfitting.fundamental_inefficiency_analyzer import FundamentalInefficiencyAnalyzer, FundamentalInefficiencyResult
    from src.anti_overfitting.parameter_constraints import ParameterConstraintManager, OptimizationBudget
    from src.optimization.hyperopt_optimizer import HyperoptOptimizer, OptimizationResult
    from src.utils.logger import get_logger
    # Optional dashboard import
    try:
        from src.analytics.dashboard import PerformanceDashboard
    except ImportError:
        PerformanceDashboard = None


@dataclass
class ReportDataCache:
    """Cache for historical report data."""
    strategy_name: str
    data_hash: str
    timestamp: datetime
    report_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'data_hash': self.data_hash,
            'timestamp': self.timestamp.isoformat(),
            'report_data': self.report_data,
            'performance_metrics': self.performance_metrics,
            'validation_results': self.validation_results,
            'risk_assessment': self.risk_assessment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportDataCache':
        """Create from dictionary."""
        return cls(
            strategy_name=data['strategy_name'],
            data_hash=data['data_hash'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            report_data=data['report_data'],
            performance_metrics=data['performance_metrics'],
            validation_results=data['validation_results'],
            risk_assessment=data['risk_assessment']
        )


@dataclass
class ReportDataPackage:
    """Complete data package for report generation."""
    
    # Basic information
    strategy_name: str
    generation_timestamp: datetime
    data_hash: str
    
    # Core data
    strategy_config: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    trading_metrics: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    
    # Validation results
    validation_results: Dict[str, Any]
    multi_period_validation: Optional[Dict[str, Any]] = None
    cross_asset_validation: Optional[Dict[str, Any]] = None
    regime_analysis: Optional[Dict[str, Any]] = None
    
    # Anti-overfitting analysis
    anti_overfitting: Dict[str, Any] = field(default_factory=dict)
    fundamental_inefficiency: Optional[Dict[str, Any]] = None
    
    # Optimization details
    optimization_details: Dict[str, Any] = field(default_factory=dict)
    
    # Data specifications
    data_specifications: Dict[str, Any] = field(default_factory=dict)
    
    # Market dependency analysis
    market_dependency: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations and warnings
    recommendations: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)
    implementation_notes: List[str] = field(default_factory=list)
    
    # Visualizations (base64 encoded images)
    visualizations: Dict[str, str] = field(default_factory=dict)


class PerformanceDataExtractor:
    """Extracts data from performance analytics components."""
    
    def __init__(self):
        self.logger = get_logger("performance_data_extractor")
    
    def extract_performance_metrics(
        self,
        analyzer: PerformanceAnalyzer,
        results: BacktestResults
    ) -> Dict[str, Any]:
        """Extract performance metrics from analyzer."""
        
        try:
            # Calculate comprehensive metrics
            metrics = analyzer.calculate_metrics(results)
            
            # Core performance metrics
            performance_data = {
                'total_return': metrics.get('total_return', 0.0),
                'annual_return': metrics.get('annual_return', 0.0),
                'volatility': metrics.get('volatility', 0.0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
                'sortino_ratio': metrics.get('sortino_ratio', 0.0),
                'calmar_ratio': metrics.get('calmar_ratio', 0.0),
                'max_drawdown': metrics.get('max_drawdown', 0.0),
                'avg_drawdown': metrics.get('avg_drawdown', 0.0),
                'recovery_factor': metrics.get('recovery_factor', 0.0),
                'var_95': metrics.get('var_95', 0.0),
                'cvar_95': metrics.get('cvar_95', 0.0)
            }
            
            return performance_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract performance metrics: {e}")
            return {}
    
    def extract_trading_metrics(self, results: BacktestResults) -> Dict[str, Any]:
        """Extract trading activity metrics."""
        
        try:
            trading_data = {
                'total_trades': len(results.trades),
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'avg_trade': np.mean([t.pnl for t in results.trades]) if results.trades else 0,
                'avg_win': np.mean([t.pnl for t in results.trades if t.pnl > 0]) if results.trades else 0,
                'avg_loss': np.mean([t.pnl for t in results.trades if t.pnl < 0]) if results.trades else 0,
                'expectancy': results.expectancy if hasattr(results, 'expectancy') else 0,
                'avg_trade_duration': np.mean([t.duration for t in results.trades]) if results.trades else 0,
                'max_consecutive_wins': self._calculate_max_consecutive_wins(results.trades),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(results.trades)
            }
            
            return trading_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract trading metrics: {e}")
            return {}
    
    def _calculate_max_consecutive_wins(self, trades: List[Any]) -> int:
        """Calculate maximum consecutive wins."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_max_consecutive_losses(self, trades: List[Any]) -> int:
        """Calculate maximum consecutive losses."""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


class ValidationDataExtractor:
    """Extracts data from validation framework components."""
    
    def __init__(self):
        self.logger = get_logger("validation_data_extractor")
    
    def extract_multi_period_validation(
        self,
        validation_result: MultiPeriodValidationResult
    ) -> Dict[str, Any]:
        """Extract multi-period validation data."""
        
        try:
            # Convert period performances to dictionaries
            period_performances = []
            for period in validation_result.validation_periods:
                period_dict = {
                    'period_type': period.period_type.value,
                    'start_date': period.start_date.isoformat(),
                    'end_date': period.end_date.isoformat(),
                    'duration_days': period.duration_days,
                    'total_return': period.total_return,
                    'annual_return': period.annual_return,
                    'sharpe_ratio': period.sharpe_ratio,
                    'max_drawdown': period.max_drawdown,
                    'win_rate': period.win_rate,
                    'total_trades': period.total_trades,
                    'volatility': period.volatility
                }
                period_performances.append(period_dict)
            
            # Extract consistency metrics
            consistency_data = {
                'overall_consistency_score': validation_result.consistency_metrics.overall_consistency_score,
                'return_consistency_score': validation_result.consistency_metrics.return_consistency_score,
                'risk_adjusted_consistency': validation_result.consistency_metrics.risk_adjusted_consistency,
                'consistency_level': validation_result.consistency_metrics.consistency_level.value,
                'inconsistency_flags': validation_result.consistency_metrics.inconsistency_flags
            }
            
            return {
                'validation_score': validation_result.validation_score,
                'deployment_readiness': validation_result.deployment_readiness,
                'period_performances': period_performances,
                'consistency_score': validation_result.consistency_metrics.overall_consistency_score * 100,
                'consistency_metrics': consistency_data,
                'best_period': validation_result.best_period.value,
                'worst_period': validation_result.worst_period.value,
                'recommendations': validation_result.recommendations,
                'risk_warnings': validation_result.risk_warnings
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract multi-period validation data: {e}")
            return {}
    
    def extract_cross_asset_validation(
        self,
        validation_result: CrossAssetTestResult
    ) -> Dict[str, Any]:
        """Extract cross-asset validation data."""
        
        try:
            # Convert asset results to dictionaries
            asset_results = []
            for asset_result in validation_result.asset_results:
                asset_dict = {
                    'asset_symbol': asset_result.asset_symbol,
                    'total_return': asset_result.performance_metrics.get('total_return', 0),
                    'sharpe_ratio': asset_result.performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': asset_result.performance_metrics.get('max_drawdown', 0),
                    'correlation': asset_result.correlation_metrics.get('price_correlation', 0),
                    'trades': asset_result.performance_metrics.get('total_trades', 0)
                }
                asset_results.append(asset_dict)
            
            return {
                'consistency_score': validation_result.consistency_score,
                'overfitting_probability': validation_result.overfitting_probability,
                'grade': validation_result.grade,
                'deployment_recommendation': validation_result.deployment_recommendation,
                'asset_results': asset_results,
                'correlation_analysis': validation_result.correlation_analysis,
                'warning_flags': validation_result.warning_flags,
                'recommendations': validation_result.recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract cross-asset validation data: {e}")
            return {}
    
    def extract_regime_analysis(
        self,
        regime_result: RegimeAnalysisResults
    ) -> Dict[str, Any]:
        """Extract regime analysis data."""
        
        try:
            # Convert regime performances to dictionaries
            regime_performances = []
            for regime_perf in regime_result.regime_performances:
                regime_dict = {
                    'regime': regime_perf.regime.value,
                    'total_return': regime_perf.total_return,
                    'sharpe_ratio': regime_perf.sharpe_ratio,
                    'max_drawdown': regime_perf.max_drawdown,
                    'win_rate': regime_perf.win_rate,
                    'trades': regime_perf.trades,
                    'duration_days': regime_perf.duration_days
                }
                regime_performances.append(regime_dict)
            
            return {
                'regime_consistency': regime_result.regime_consistency,
                'best_regime': regime_result.best_regime.value,
                'worst_regime': regime_result.worst_regime.value,
                'regime_performances': regime_performances,
                'market_dependency_score': regime_result.market_dependency_score,
                'recommendations': regime_result.recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract regime analysis data: {e}")
            return {}


class AntiOverfittingDataExtractor:
    """Extracts data from anti-overfitting components."""
    
    def __init__(self):
        self.logger = get_logger("anti_overfitting_data_extractor")
    
    def extract_fundamental_inefficiency_analysis(
        self,
        inefficiency_result: FundamentalInefficiencyResult
    ) -> Dict[str, Any]:
        """Extract fundamental inefficiency analysis data."""
        
        try:
            analysis = inefficiency_result.analysis
            
            return {
                'inefficiency_type': analysis.inefficiency_type.value,
                'strategy_category': analysis.strategy_category.value,
                'confidence_score': analysis.confidence_score * 100,
                'rationale_strength': analysis.rationale_strength.value,
                'rationale_score': analysis.rationale_score,
                'persistence_level': analysis.persistence_level.value,
                'persistence_score': analysis.persistence_score,
                'data_mining_risk': analysis.data_mining_risk,
                'overfitting_risk': analysis.overfitting_risk,
                'decay_risk': analysis.decay_risk,
                'overall_score': inefficiency_result.overall_score,
                'deployment_recommendation': inefficiency_result.deployment_recommendation,
                'strengths': inefficiency_result.strengths,
                'weaknesses': inefficiency_result.weaknesses,
                'improvement_suggestions': inefficiency_result.improvement_suggestions,
                'red_flags': analysis.red_flags,
                'warnings': analysis.warnings,
                'recommendations': analysis.recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract fundamental inefficiency analysis: {e}")
            return {}
    
    def extract_optimization_budget(
        self,
        budget: OptimizationBudget
    ) -> Dict[str, Any]:
        """Extract optimization budget data."""
        
        try:
            return {
                'max_iterations': budget.max_iterations,
                'used_iterations': budget.used_iterations,
                'iterations_remaining': budget.iterations_remaining,
                'max_parameters': budget.max_parameters,
                'current_parameters': budget.current_parameters,
                'max_optimization_time_hours': budget.max_optimization_time_hours,
                'used_optimization_time_hours': budget.used_optimization_time_hours,
                'budget_exhausted': budget.budget_exhausted
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract optimization budget: {e}")
            return {}


class VisualizationDataExtractor:
    """Extracts visualization data for reports."""
    
    def __init__(self):
        self.logger = get_logger("visualization_data_extractor")
    
    def extract_visualizations(
        self,
        visualization_engine: VisualizationEngine,
        results: BacktestResults,
        strategy_name: str
    ) -> Dict[str, str]:
        """Extract visualizations as base64 encoded images."""
        
        visualizations = {}
        
        try:
            # Generate equity curve
            equity_fig = visualization_engine.create_equity_curve(results, strategy_name)
            if equity_fig:
                visualizations['equity_curve'] = self._fig_to_base64(equity_fig)
            
            # Generate drawdown chart
            drawdown_fig = visualization_engine.create_drawdown_chart(results, strategy_name)
            if drawdown_fig:
                visualizations['drawdown_chart'] = self._fig_to_base64(drawdown_fig)
            
            # Generate returns distribution
            returns_fig = visualization_engine.create_returns_distribution(results, strategy_name)
            if returns_fig:
                visualizations['returns_distribution'] = self._fig_to_base64(returns_fig)
            
            # Generate monthly returns heatmap
            monthly_fig = visualization_engine.create_monthly_returns_heatmap(results, strategy_name)
            if monthly_fig:
                visualizations['monthly_returns'] = self._fig_to_base64(monthly_fig)
            
        except Exception as e:
            self.logger.error(f"Failed to extract visualizations: {e}")
        
        return visualizations
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string."""
        try:
            import plotly.io as pio
            import base64
            
            # Convert to image bytes
            img_bytes = pio.to_image(fig, format='png', width=800, height=600)
            
            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            self.logger.error(f"Failed to convert figure to base64: {e}")
            return ""


class ReportDataCollector:
    """Main data collector that aggregates information from all system components."""
    
    def __init__(
        self,
        cache_dir: str = "reports/cache",
        enable_caching: bool = True
    ):
        """
        Initialize the ReportDataCollector.
        
        Args:
            cache_dir: Directory for caching report data
            enable_caching: Whether to enable data caching
        """
        self.cache_dir = Path(cache_dir)
        self.enable_caching = enable_caching
        self.logger = get_logger("report_data_collector")
        
        # Create cache directory
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize extractors
        self.performance_extractor = PerformanceDataExtractor()
        self.validation_extractor = ValidationDataExtractor()
        self.anti_overfitting_extractor = AntiOverfittingDataExtractor()
        self.visualization_extractor = VisualizationDataExtractor()
    
    def collect_comprehensive_data(
        self,
        strategy: BaseStrategy,
        backtest_results: BacktestResults,
        data: pd.DataFrame,
        performance_analyzer: PerformanceAnalyzer = None,
        visualization_engine: VisualizationEngine = None,
        multi_period_validation: MultiPeriodValidationResult = None,
        cross_asset_validation: CrossAssetTestResult = None,
        regime_analysis: RegimeAnalysisResults = None,
        fundamental_inefficiency: FundamentalInefficiencyResult = None,
        optimization_result: OptimizationResult = None,
        optimization_budget: OptimizationBudget = None
    ) -> ReportDataPackage:
        """
        Collect comprehensive data from all system components.
        
        Args:
            strategy: Trading strategy
            backtest_results: Backtest results
            data: Historical market data
            performance_analyzer: Performance analyzer instance
            visualization_engine: Visualization engine instance
            multi_period_validation: Multi-period validation results
            cross_asset_validation: Cross-asset validation results
            regime_analysis: Regime analysis results
            fundamental_inefficiency: Fundamental inefficiency analysis
            optimization_result: Optimization results
            optimization_budget: Optimization budget information
            
        Returns:
            ReportDataPackage with all collected data
        """
        self.logger.info(f"Collecting comprehensive data for strategy: {strategy.name}")
        
        # Generate data hash for caching
        data_hash = self._generate_data_hash(strategy, data)
        
        # Check cache first
        if self.enable_caching:
            cached_data = self._load_from_cache(strategy.name, data_hash)
            if cached_data:
                self.logger.info("Loaded data from cache")
                return cached_data
        
        # Extract strategy configuration
        strategy_config = self._extract_strategy_config(strategy)
        
        # Extract performance metrics
        performance_metrics = {}
        trading_metrics = {}
        if performance_analyzer:
            performance_metrics = self.performance_extractor.extract_performance_metrics(
                performance_analyzer, backtest_results
            )
        trading_metrics = self.performance_extractor.extract_trading_metrics(backtest_results)
        
        # Extract validation results
        validation_results = {}
        multi_period_data = None
        cross_asset_data = None
        regime_data = None
        
        if multi_period_validation:
            multi_period_data = self.validation_extractor.extract_multi_period_validation(
                multi_period_validation
            )
            validation_results['multi_period'] = multi_period_data
        
        if cross_asset_validation:
            cross_asset_data = self.validation_extractor.extract_cross_asset_validation(
                cross_asset_validation
            )
            validation_results['cross_asset'] = cross_asset_data
        
        if regime_analysis:
            regime_data = self.validation_extractor.extract_regime_analysis(regime_analysis)
            validation_results['regime_analysis'] = regime_data
        
        # Calculate overall validation score
        validation_results['overall_score'] = self._calculate_overall_validation_score(
            multi_period_data, cross_asset_data, regime_data
        )
        
        # Determine deployment recommendation
        validation_results['deployment_recommendation'] = self._determine_deployment_recommendation(
            validation_results
        )
        
        # Extract anti-overfitting analysis
        anti_overfitting_data = {}
        fundamental_inefficiency_data = None
        
        if fundamental_inefficiency:
            fundamental_inefficiency_data = self.anti_overfitting_extractor.extract_fundamental_inefficiency_analysis(
                fundamental_inefficiency
            )
            anti_overfitting_data.update(fundamental_inefficiency_data)
        
        # Extract optimization details
        optimization_details = {}
        if optimization_result:
            optimization_details = self._extract_optimization_details(optimization_result)
        
        if optimization_budget:
            optimization_details['budget'] = self.anti_overfitting_extractor.extract_optimization_budget(
                optimization_budget
            )
        
        # Extract data specifications
        data_specifications = self._extract_data_specifications(data)
        
        # Calculate risk assessment
        risk_assessment = self._calculate_risk_assessment(
            performance_metrics, anti_overfitting_data, validation_results
        )
        
        # Calculate market dependency
        market_dependency = self._calculate_market_dependency(
            backtest_results, regime_data
        )
        
        # Generate recommendations and warnings
        recommendations, risk_warnings, implementation_notes = self._generate_recommendations(
            performance_metrics, validation_results, anti_overfitting_data
        )
        
        # Extract visualizations
        visualizations = {}
        if visualization_engine:
            visualizations = self.visualization_extractor.extract_visualizations(
                visualization_engine, backtest_results, strategy.name
            )
        
        # Create data package
        data_package = ReportDataPackage(
            strategy_name=strategy.name,
            generation_timestamp=datetime.now(),
            data_hash=data_hash,
            strategy_config=strategy_config,
            performance_metrics=performance_metrics,
            trading_metrics=trading_metrics,
            risk_assessment=risk_assessment,
            validation_results=validation_results,
            multi_period_validation=multi_period_data,
            cross_asset_validation=cross_asset_data,
            regime_analysis=regime_data,
            anti_overfitting=anti_overfitting_data,
            fundamental_inefficiency=fundamental_inefficiency_data,
            optimization_details=optimization_details,
            data_specifications=data_specifications,
            market_dependency=market_dependency,
            recommendations=recommendations,
            risk_warnings=risk_warnings,
            implementation_notes=implementation_notes,
            visualizations=visualizations
        )
        
        # Cache the data
        if self.enable_caching:
            self._save_to_cache(data_package)
        
        self.logger.info(f"Data collection completed for strategy: {strategy.name}")
        return data_package
    
    def _generate_data_hash(self, strategy: BaseStrategy, data: pd.DataFrame) -> str:
        """Generate hash for data caching."""
        strategy_str = f"{strategy.name}_{strategy.parameters}_{strategy.risk_params}"
        data_str = str(data.values.tobytes())
        combined = strategy_str + data_str
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _extract_strategy_config(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """Extract strategy configuration."""
        return {
            'name': strategy.name,
            'parameters': strategy.parameters,
            'risk_params': strategy.risk_params,
            'parameter_descriptions': getattr(strategy, 'parameter_descriptions', {}),
            'strategy_type': strategy.__class__.__name__
        }
    
    def _extract_optimization_details(self, optimization_result: OptimizationResult) -> Dict[str, Any]:
        """Extract optimization details."""
        return {
            'method': 'Hyperopt',
            'iterations': optimization_result.total_iterations,
            'best_score': optimization_result.best_score,
            'best_parameters': optimization_result.best_parameters,
            'duration_hours': optimization_result.optimization_time / 3600.0,
            'parameter_combinations': optimization_result.total_iterations,
            'convergence_achieved': optimization_result.convergence_achieved if hasattr(optimization_result, 'convergence_achieved') else False
        }
    
    def _extract_data_specifications(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract data specifications."""
        return {
            'source': 'Historical Market Data',
            'start_date': data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else 'N/A',
            'end_date': data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else 'N/A',
            'frequency': 'Daily',
            'total_records': len(data),
            'asset_count': 1,  # Assuming single asset for now
            'quality_score': 95,  # Placeholder quality score
            'columns': list(data.columns)
        }
    
    def _calculate_overall_validation_score(
        self,
        multi_period_data: Optional[Dict[str, Any]],
        cross_asset_data: Optional[Dict[str, Any]],
        regime_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall validation score."""
        scores = []
        
        if multi_period_data:
            scores.append(multi_period_data.get('validation_score', 0))
        
        if cross_asset_data:
            scores.append(cross_asset_data.get('consistency_score', 0))
        
        if regime_data:
            scores.append(regime_data.get('regime_consistency', 0) * 100)
        
        return np.mean(scores) if scores else 0.0
    
    def _determine_deployment_recommendation(self, validation_results: Dict[str, Any]) -> str:
        """Determine deployment recommendation."""
        overall_score = validation_results.get('overall_score', 0)
        
        if overall_score >= 80:
            return "Approved"
        elif overall_score >= 60:
            return "Conditional"
        else:
            return "Not Recommended"
    
    def _calculate_risk_assessment(
        self,
        performance_metrics: Dict[str, Any],
        anti_overfitting_data: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment."""
        
        # Calculate risk factors
        performance_risk = self._calculate_performance_risk(performance_metrics)
        overfitting_risk = anti_overfitting_data.get('overfitting_risk', 50)
        validation_risk = 100 - validation_results.get('overall_score', 50)
        
        # Overall risk score (0-100, lower is better)
        risk_score = (performance_risk * 0.3 + overfitting_risk * 0.4 + validation_risk * 0.3)
        
        # Risk level
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'overall_risk_level': risk_level,
            'risk_score': risk_score,
            'performance_risk': performance_risk,
            'overfitting_risk': overfitting_risk,
            'validation_risk': validation_risk,
            'mitigation_recommendations': self._generate_risk_mitigation_recommendations(risk_score)
        }
    
    def _calculate_performance_risk(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate performance-based risk score."""
        risk_factors = []
        
        # High volatility risk
        volatility = performance_metrics.get('volatility', 0)
        if volatility > 0.3:
            risk_factors.append(20)
        elif volatility > 0.2:
            risk_factors.append(10)
        
        # High drawdown risk
        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        if max_drawdown > 0.2:
            risk_factors.append(25)
        elif max_drawdown > 0.1:
            risk_factors.append(15)
        
        # Low Sharpe ratio risk
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < 1.0:
            risk_factors.append(15)
        elif sharpe_ratio < 1.5:
            risk_factors.append(10)
        
        return min(100, sum(risk_factors))
    
    def _calculate_market_dependency(
        self,
        backtest_results: BacktestResults,
        regime_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate market dependency analysis."""
        
        # Placeholder correlations (would be calculated from actual market data)
        market_correlations = {
            'market_direction': 0.3,  # Correlation with overall market direction
            'volatility': 0.2,        # Correlation with market volatility
            'volume': 0.1,            # Correlation with trading volume
            'sector': 0.15            # Correlation with sector performance
        }
        
        return {
            'market_correlations': market_correlations,
            'regime_dependency': regime_data.get('market_dependency_score', 0.5) if regime_data else 0.5
        }
    
    def _generate_recommendations(
        self,
        performance_metrics: Dict[str, Any],
        validation_results: Dict[str, Any],
        anti_overfitting_data: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate recommendations, warnings, and implementation notes."""
        
        recommendations = []
        risk_warnings = []
        implementation_notes = []
        
        # Performance-based recommendations
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < 1.0:
            recommendations.append("Consider improving risk-adjusted returns through better risk management")
        
        max_drawdown = abs(performance_metrics.get('max_drawdown', 0))
        if max_drawdown > 0.15:
            recommendations.append("Implement stricter drawdown controls to limit downside risk")
            risk_warnings.append("High maximum drawdown detected")
        
        # Validation-based recommendations
        overall_score = validation_results.get('overall_score', 0)
        if overall_score < 70:
            recommendations.append("Conduct additional validation testing before deployment")
            risk_warnings.append("Strategy validation score below recommended threshold")
        
        # Anti-overfitting recommendations
        overfitting_risk = anti_overfitting_data.get('overfitting_risk', 0)
        if overfitting_risk > 60:
            recommendations.append("Reduce strategy complexity to minimize overfitting risk")
            risk_warnings.append("High overfitting risk detected")
        
        # Implementation notes
        implementation_notes.extend([
            "Monitor strategy performance continuously after deployment",
            "Implement position sizing based on current market volatility",
            "Regular revalidation recommended every 6 months",
            "Consider portfolio diversification with multiple strategies"
        ])
        
        return recommendations, risk_warnings, implementation_notes
    
    def _generate_risk_mitigation_recommendations(self, risk_score: float) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []
        
        if risk_score > 70:
            recommendations.extend([
                "Implement strict position sizing limits",
                "Add additional risk monitoring systems",
                "Consider reducing strategy allocation",
                "Require manual approval for all trades"
            ])
        elif risk_score > 50:
            recommendations.extend([
                "Monitor performance closely",
                "Implement volatility-based position sizing",
                "Set up automated alerts for drawdown limits"
            ])
        else:
            recommendations.extend([
                "Continue standard risk monitoring",
                "Regular performance reviews"
            ])
        
        return recommendations
    
    def _save_to_cache(self, data_package: ReportDataPackage):
        """Save data package to cache."""
        try:
            cache_file = self.cache_dir / f"{data_package.strategy_name}_{data_package.data_hash}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data_package, f)
            
            self.logger.debug(f"Saved data to cache: {cache_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save data to cache: {e}")
    
    def _load_from_cache(self, strategy_name: str, data_hash: str) -> Optional[ReportDataPackage]:
        """Load data package from cache."""
        try:
            cache_file = self.cache_dir / f"{strategy_name}_{data_hash}.pkl"
            
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data_package = pickle.load(f)
                
                # Check if cache is still valid (less than 24 hours old)
                if datetime.now() - data_package.generation_timestamp < timedelta(hours=24):
                    return data_package
            
        except Exception as e:
            self.logger.error(f"Failed to load data from cache: {e}")
        
        return None
    
    def clear_cache(self, strategy_name: str = None):
        """Clear cached data."""
        try:
            if strategy_name:
                # Clear cache for specific strategy
                pattern = f"{strategy_name}_*.pkl"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                self.logger.info(f"Cleared cache for strategy: {strategy_name}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Cleared all cached data")
                
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}") 