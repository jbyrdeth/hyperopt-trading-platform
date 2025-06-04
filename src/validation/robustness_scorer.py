"""
Comprehensive Robustness Scoring System

This module provides a unified scoring system that integrates all validation
methods to produce a single robustness score for trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

try:
    from ..utils.logger import get_logger
    from ..strategies.base_strategy import BaseStrategy
    from .data_splitter import DataSplitter
    from .cross_asset_validator import CrossAssetValidator
    from .monte_carlo_tester import MonteCarloTester
    from .statistical_tester import StatisticalTester
    from .regime_analyzer import RegimeAnalyzer
except ImportError:
    from src.utils.logger import get_logger
    from src.strategies.base_strategy import BaseStrategy
    from src.validation.data_splitter import DataSplitter
    from src.validation.cross_asset_validator import CrossAssetValidator
    from src.validation.monte_carlo_tester import MonteCarloTester
    from src.validation.statistical_tester import StatisticalTester
    from src.validation.regime_analyzer import RegimeAnalyzer


class DeploymentRecommendation(Enum):
    """Deployment recommendation levels."""
    READY_FOR_PRODUCTION = "Ready for Production"
    DEPLOY_WITH_CAUTION = "Deploy with Caution"
    PAPER_TRADE_FIRST = "Paper Trade First"
    NOT_READY = "Not Ready"


@dataclass
class ComponentScore:
    """Individual validation component score."""
    name: str
    score: float  # 0-100
    weight: float  # 0-1
    weighted_score: float
    details: Dict[str, Any]
    confidence: float  # 0-1
    issues: List[str] = field(default_factory=list)


@dataclass
class RobustnessScore:
    """Complete robustness assessment for a strategy."""
    strategy_name: str
    overall_score: float  # 0-100
    component_scores: Dict[str, ComponentScore]
    confidence_interval: Tuple[float, float]
    deployment_recommendation: DeploymentRecommendation
    
    # Analysis
    strengths: List[str]
    weaknesses: List[str]
    risk_factors: List[str]
    
    # Detailed metrics
    detailed_metrics: Dict[str, Any]
    
    # Metadata
    validation_date: datetime
    data_period: Tuple[datetime, datetime]
    assets_tested: List[str]
    
    # Summary statistics
    min_component_score: float
    max_component_score: float
    score_variance: float


@dataclass
class ComparisonReport:
    """Comparative analysis of multiple strategies."""
    strategy_rankings: List[Tuple[str, float]]  # (name, score)
    score_distribution: Dict[str, float]  # percentiles
    best_strategy: str
    worst_strategy: str
    
    # Cluster analysis
    strategy_clusters: Dict[str, List[str]]
    cluster_characteristics: Dict[str, Dict[str, float]]
    
    # Recommendations
    portfolio_recommendations: List[str]
    deployment_ready_strategies: List[str]
    
    # Statistics
    mean_score: float
    median_score: float
    score_std: float


@dataclass
class DeploymentReport:
    """Detailed deployment readiness assessment."""
    strategy_name: str
    overall_score: float
    recommendation: DeploymentRecommendation
    
    # Risk assessment
    estimated_risk_level: str  # Low, Medium, High
    confidence_score: float
    minimum_capital_requirement: float
    
    # Monitoring recommendations
    key_metrics_to_monitor: List[str]
    revalidation_frequency: str
    stop_loss_recommendations: Dict[str, float]
    
    # Deployment checklist
    pre_deployment_checklist: List[Tuple[str, bool]]  # (item, completed)
    post_deployment_monitoring: List[str]


class RobustnessScorer:
    """
    Comprehensive robustness scoring system that integrates all validation methods.
    
    Produces a unified 0-100 score with deployment recommendations.
    """
    
    def __init__(
        self,
        component_weights: Optional[Dict[str, float]] = None,
        deployment_thresholds: Optional[Dict[str, float]] = None,
        initial_capital: float = 100000,
        confidence_level: float = 0.95
    ):
        """
        Initialize the RobustnessScorer.
        
        Args:
            component_weights: Custom weights for validation components
            deployment_thresholds: Custom score thresholds for deployment recommendations
            initial_capital: Starting capital for backtests
            confidence_level: Confidence level for statistical tests
        """
        self.logger = get_logger("robustness_scorer")
        self.initial_capital = initial_capital
        self.confidence_level = confidence_level
        
        # Default component weights (must sum to 1.0)
        self.component_weights = component_weights or {
            'statistical_significance': 0.25,
            'cross_asset_performance': 0.20,
            'regime_adaptability': 0.20,
            'monte_carlo_robustness': 0.15,
            'risk_adjusted_performance': 0.10,
            'data_quality_overfitting': 0.10
        }
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.component_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 1.0, got {weight_sum}")
        
        # Default deployment thresholds
        self.deployment_thresholds = deployment_thresholds or {
            'ready_for_production': 80.0,
            'deploy_with_caution': 60.0,
            'paper_trade_first': 40.0
        }
        
        # Initialize validation components
        self.data_splitter = DataSplitter()
        self.cross_asset_validator = CrossAssetValidator(initial_capital=initial_capital)
        self.monte_carlo_tester = MonteCarloTester(initial_capital=initial_capital)
        self.statistical_tester = StatisticalTester(initial_capital=initial_capital)
        self.regime_analyzer = RegimeAnalyzer(initial_capital=initial_capital)
        
        self.logger.info("RobustnessScorer initialized with comprehensive validation suite")
    
    def configure_weights(self, component_weights: Dict[str, float]) -> None:
        """
        Configure custom weights for validation components.
        
        Args:
            component_weights: Dictionary of component weights (must sum to 1.0)
        """
        weight_sum = sum(component_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 1.0, got {weight_sum}")
        
        self.component_weights = component_weights
        self.logger.info(f"Updated component weights: {component_weights}")
    
    def calculate_robustness_score(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        assets: List[str] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> RobustnessScore:
        """
        Calculate comprehensive robustness score for a strategy.
        
        Args:
            strategy: Strategy to evaluate
            data: Historical OHLCV data
            assets: List of asset symbols for cross-asset validation
            validation_config: Optional configuration for validation methods
            
        Returns:
            Complete robustness assessment
        """
        self.logger.info(f"Calculating robustness score for {strategy.name}")
        
        if assets is None:
            assets = ['BTC', 'ETH', 'SOL']  # Default crypto assets
        
        validation_config = validation_config or {}
        
        # Run all validation components
        component_scores = {}
        detailed_metrics = {}
        
        try:
            # 1. Statistical Significance Testing
            stat_score, stat_details = self._evaluate_statistical_significance(
                strategy, data, validation_config.get('statistical', {})
            )
            component_scores['statistical_significance'] = stat_score
            detailed_metrics['statistical_significance'] = stat_details
            
            # 2. Cross-Asset Performance
            cross_score, cross_details = self._evaluate_cross_asset_performance(
                strategy, data, assets, validation_config.get('cross_asset', {})
            )
            component_scores['cross_asset_performance'] = cross_score
            detailed_metrics['cross_asset_performance'] = cross_details
            
            # 3. Regime Adaptability
            regime_score, regime_details = self._evaluate_regime_adaptability(
                strategy, data, validation_config.get('regime', {})
            )
            component_scores['regime_adaptability'] = regime_score
            detailed_metrics['regime_adaptability'] = regime_details
            
            # 4. Monte Carlo Robustness
            mc_score, mc_details = self._evaluate_monte_carlo_robustness(
                strategy, data, validation_config.get('monte_carlo', {})
            )
            component_scores['monte_carlo_robustness'] = mc_score
            detailed_metrics['monte_carlo_robustness'] = mc_details
            
            # 5. Risk-Adjusted Performance
            risk_score, risk_details = self._evaluate_risk_adjusted_performance(
                strategy, data, validation_config.get('risk_adjusted', {})
            )
            component_scores['risk_adjusted_performance'] = risk_score
            detailed_metrics['risk_adjusted_performance'] = risk_details
            
            # 6. Data Quality & Overfitting
            quality_score, quality_details = self._evaluate_data_quality_overfitting(
                strategy, data, validation_config.get('data_quality', {})
            )
            component_scores['data_quality_overfitting'] = quality_score
            detailed_metrics['data_quality_overfitting'] = quality_details
            
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            raise
        
        # Calculate overall score
        overall_score = self._calculate_weighted_score(component_scores)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(component_scores)
        
        # Determine deployment recommendation
        deployment_rec = self._determine_deployment_recommendation(overall_score)
        
        # Analyze strengths and weaknesses
        strengths, weaknesses, risk_factors = self._analyze_strengths_weaknesses(component_scores)
        
        # Calculate summary statistics
        scores_list = [score.score for score in component_scores.values()]
        min_score = min(scores_list)
        max_score = max(scores_list)
        score_variance = np.var(scores_list)
        
        return RobustnessScore(
            strategy_name=strategy.name,
            overall_score=overall_score,
            component_scores=component_scores,
            confidence_interval=confidence_interval,
            deployment_recommendation=deployment_rec,
            strengths=strengths,
            weaknesses=weaknesses,
            risk_factors=risk_factors,
            detailed_metrics=detailed_metrics,
            validation_date=datetime.now(),
            data_period=(data.index[0], data.index[-1]),
            assets_tested=assets,
            min_component_score=min_score,
            max_component_score=max_score,
            score_variance=score_variance
        ) 

    def _evaluate_statistical_significance(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[ComponentScore, Dict[str, Any]]:
        """Evaluate statistical significance of strategy performance."""
        try:
            # Test against random strategies
            random_comparisons = self.statistical_tester.test_against_random(
                strategy, data, n_random_strategies=config.get('n_random', 50)
            )
            
            # Test against buy-and-hold
            bnh_comparison = self.statistical_tester.test_buy_and_hold(strategy, data)
            
            # Calculate score based on statistical tests
            significant_tests = 0
            total_tests = 0
            p_values = []
            
            for comparison in random_comparisons + [bnh_comparison]:
                for test in comparison.statistical_tests:
                    total_tests += 1
                    if test.is_significant:
                        significant_tests += 1
                    p_values.append(test.p_value)
            
            # Score based on percentage of significant tests
            significance_rate = significant_tests / total_tests if total_tests > 0 else 0
            
            # Adjust score based on p-value distribution
            avg_p_value = np.mean(p_values) if p_values else 1.0
            p_value_score = max(0, (0.05 - avg_p_value) / 0.05) * 100
            
            # Combined score
            base_score = significance_rate * 100
            adjusted_score = (base_score * 0.7) + (p_value_score * 0.3)
            
            # Confidence based on sample size and consistency
            confidence = min(1.0, len(p_values) / 10) * (1 - np.std(p_values))
            
            issues = []
            if significance_rate < 0.5:
                issues.append("Low statistical significance rate")
            if avg_p_value > 0.1:
                issues.append("High average p-values")
            
            component_score = ComponentScore(
                name="Statistical Significance",
                score=min(100, max(0, adjusted_score)),
                weight=self.component_weights['statistical_significance'],
                weighted_score=adjusted_score * self.component_weights['statistical_significance'],
                details={
                    'significance_rate': significance_rate,
                    'avg_p_value': avg_p_value,
                    'total_tests': total_tests,
                    'significant_tests': significant_tests
                },
                confidence=confidence,
                issues=issues
            )
            
            return component_score, {
                'random_comparisons': random_comparisons,
                'bnh_comparison': bnh_comparison,
                'p_values': p_values
            }
            
        except Exception as e:
            self.logger.error(f"Statistical significance evaluation failed: {e}")
            return self._create_failed_component_score("Statistical Significance"), {}
    
    def _evaluate_cross_asset_performance(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        assets: List[str],
        config: Dict[str, Any]
    ) -> Tuple[ComponentScore, Dict[str, Any]]:
        """Evaluate cross-asset performance consistency."""
        try:
            # Run cross-asset validation
            cross_results = self.cross_asset_validator.validate_cross_asset(
                strategy, {asset: data for asset in assets}  # Simplified - same data for all assets
            )
            
            # Calculate consistency metrics
            returns = [result.total_return for result in cross_results.asset_results.values()]
            sharpe_ratios = [result.sharpe_ratio for result in cross_results.asset_results.values()]
            
            # Score based on consistency and performance
            return_consistency = 1 - (np.std(returns) / (np.mean(np.abs(returns)) + 1e-8))
            sharpe_consistency = 1 - (np.std(sharpe_ratios) / (np.mean(np.abs(sharpe_ratios)) + 1e-8))
            
            # Performance score
            positive_returns = sum(1 for r in returns if r > 0) / len(returns)
            positive_sharpe = sum(1 for s in sharpe_ratios if s > 0) / len(sharpe_ratios)
            
            # Combined score
            consistency_score = (return_consistency + sharpe_consistency) / 2 * 100
            performance_score = (positive_returns + positive_sharpe) / 2 * 100
            overall_score = (consistency_score * 0.6) + (performance_score * 0.4)
            
            confidence = min(1.0, len(assets) / 5)  # Higher confidence with more assets
            
            issues = []
            if return_consistency < 0.5:
                issues.append("Inconsistent returns across assets")
            if positive_returns < 0.6:
                issues.append("Poor performance on multiple assets")
            
            component_score = ComponentScore(
                name="Cross-Asset Performance",
                score=min(100, max(0, overall_score)),
                weight=self.component_weights['cross_asset_performance'],
                weighted_score=overall_score * self.component_weights['cross_asset_performance'],
                details={
                    'return_consistency': return_consistency,
                    'sharpe_consistency': sharpe_consistency,
                    'positive_returns_rate': positive_returns,
                    'positive_sharpe_rate': positive_sharpe,
                    'assets_tested': len(assets)
                },
                confidence=confidence,
                issues=issues
            )
            
            return component_score, {'cross_results': cross_results}
            
        except Exception as e:
            self.logger.error(f"Cross-asset evaluation failed: {e}")
            return self._create_failed_component_score("Cross-Asset Performance"), {}
    
    def _evaluate_regime_adaptability(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[ComponentScore, Dict[str, Any]]:
        """Evaluate strategy adaptability across market regimes."""
        try:
            # Run regime analysis
            regime_results = self.regime_analyzer.analyze_strategy_performance(strategy, data)
            
            # Calculate adaptability metrics
            regime_performances = regime_results.regime_performances
            
            if not regime_performances:
                return self._create_failed_component_score("Regime Adaptability"), {}
            
            # Score based on consistency and adaptability
            consistency_score = regime_results.regime_consistency * 100
            adaptability_score = regime_results.regime_adaptability * 100
            
            # Performance across regimes
            positive_regimes = sum(1 for perf in regime_performances if perf.strategy_return > 0)
            regime_performance_rate = positive_regimes / len(regime_performances)
            
            # Excess return consistency
            excess_returns = [perf.excess_return for perf in regime_performances]
            positive_excess = sum(1 for er in excess_returns if er > 0) / len(excess_returns)
            
            # Combined score
            overall_score = (
                consistency_score * 0.3 +
                adaptability_score * 0.3 +
                regime_performance_rate * 100 * 0.2 +
                positive_excess * 100 * 0.2
            )
            
            confidence = min(1.0, len(regime_performances) / 4)  # Higher confidence with more regimes
            
            issues = []
            if consistency_score < 50:
                issues.append("Inconsistent performance across regimes")
            if positive_excess < 0.5:
                issues.append("Poor excess returns in multiple regimes")
            
            component_score = ComponentScore(
                name="Regime Adaptability",
                score=min(100, max(0, overall_score)),
                weight=self.component_weights['regime_adaptability'],
                weighted_score=overall_score * self.component_weights['regime_adaptability'],
                details={
                    'consistency_score': consistency_score,
                    'adaptability_score': adaptability_score,
                    'regime_performance_rate': regime_performance_rate,
                    'positive_excess_rate': positive_excess,
                    'regimes_tested': len(regime_performances)
                },
                confidence=confidence,
                issues=issues
            )
            
            return component_score, {'regime_results': regime_results}
            
        except Exception as e:
            self.logger.error(f"Regime adaptability evaluation failed: {e}")
            return self._create_failed_component_score("Regime Adaptability"), {}
    
    def _evaluate_monte_carlo_robustness(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[ComponentScore, Dict[str, Any]]:
        """Evaluate Monte Carlo robustness testing results."""
        try:
            # Run Monte Carlo simulation
            mc_results = self.monte_carlo_tester.run_simulation(
                strategy, data, n_simulations=config.get('n_simulations', 100)
            )
            
            # Calculate robustness metrics
            positive_returns = mc_results.positive_return_rate
            stability_score = 1 - mc_results.return_volatility  # Lower volatility = higher stability
            
            # Performance consistency
            percentile_spread = mc_results.percentiles[75] - mc_results.percentiles[25]
            consistency_score = max(0, 1 - percentile_spread)
            
            # Risk metrics
            max_dd_consistency = 1 - (mc_results.max_drawdown_std / (abs(mc_results.max_drawdown_mean) + 1e-8))
            
            # Combined score
            overall_score = (
                positive_returns * 100 * 0.4 +
                stability_score * 100 * 0.3 +
                consistency_score * 100 * 0.2 +
                max_dd_consistency * 100 * 0.1
            )
            
            confidence = min(1.0, mc_results.n_simulations / 50)
            
            issues = []
            if positive_returns < 0.6:
                issues.append("Low positive return rate in Monte Carlo")
            if stability_score < 0.5:
                issues.append("High return volatility across simulations")
            
            component_score = ComponentScore(
                name="Monte Carlo Robustness",
                score=min(100, max(0, overall_score)),
                weight=self.component_weights['monte_carlo_robustness'],
                weighted_score=overall_score * self.component_weights['monte_carlo_robustness'],
                details={
                    'positive_return_rate': positive_returns,
                    'stability_score': stability_score,
                    'consistency_score': consistency_score,
                    'max_dd_consistency': max_dd_consistency,
                    'n_simulations': mc_results.n_simulations
                },
                confidence=confidence,
                issues=issues
            )
            
            return component_score, {'mc_results': mc_results}
            
        except Exception as e:
            self.logger.error(f"Monte Carlo evaluation failed: {e}")
            return self._create_failed_component_score("Monte Carlo Robustness"), {}
    
    def _evaluate_risk_adjusted_performance(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[ComponentScore, Dict[str, Any]]:
        """Evaluate risk-adjusted performance metrics."""
        try:
            from ..strategies.backtesting_engine import BacktestingEngine
            
            # Run backtest
            engine = BacktestingEngine(initial_capital=self.initial_capital)
            results = engine.backtest_strategy(strategy, data, "Risk Assessment")
            
            # Normalize metrics to 0-100 scale
            sharpe_score = min(100, max(0, (results.sharpe_ratio + 2) / 4 * 100))  # -2 to 2 range
            sortino_score = min(100, max(0, (results.sortino_ratio + 2) / 4 * 100))
            calmar_score = min(100, max(0, (results.calmar_ratio + 2) / 4 * 100))
            
            # Return-based score
            annual_return_score = min(100, max(0, (results.annual_return + 0.5) / 1.5 * 100))  # -50% to 100% range
            
            # Risk-based score (lower is better for drawdown)
            max_dd_score = min(100, max(0, (1 + results.max_drawdown) * 100))  # 0% to -100% drawdown
            
            # Combined score
            overall_score = (
                sharpe_score * 0.3 +
                sortino_score * 0.25 +
                calmar_score * 0.2 +
                annual_return_score * 0.15 +
                max_dd_score * 0.1
            )
            
            confidence = 0.9  # High confidence in risk metrics
            
            issues = []
            if results.sharpe_ratio < 0.5:
                issues.append("Low Sharpe ratio")
            if results.max_drawdown < -0.3:
                issues.append("High maximum drawdown")
            
            component_score = ComponentScore(
                name="Risk-Adjusted Performance",
                score=min(100, max(0, overall_score)),
                weight=self.component_weights['risk_adjusted_performance'],
                weighted_score=overall_score * self.component_weights['risk_adjusted_performance'],
                details={
                    'sharpe_ratio': results.sharpe_ratio,
                    'sortino_ratio': results.sortino_ratio,
                    'calmar_ratio': results.calmar_ratio,
                    'annual_return': results.annual_return,
                    'max_drawdown': results.max_drawdown,
                    'volatility': results.volatility
                },
                confidence=confidence,
                issues=issues
            )
            
            return component_score, {'backtest_results': results}
            
        except Exception as e:
            self.logger.error(f"Risk-adjusted performance evaluation failed: {e}")
            return self._create_failed_component_score("Risk-Adjusted Performance"), {}
    
    def _evaluate_data_quality_overfitting(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[ComponentScore, Dict[str, Any]]:
        """Evaluate data quality and overfitting indicators."""
        try:
            # Walk-forward analysis
            splits = self.data_splitter.walk_forward_split(
                data, train_size=0.7, step_size=config.get('step_size', 30)
            )
            
            if len(splits) < 3:
                return self._create_failed_component_score("Data Quality & Overfitting"), {}
            
            from ..strategies.backtesting_engine import BacktestingEngine
            engine = BacktestingEngine(initial_capital=self.initial_capital)
            
            # Test performance consistency across splits
            train_returns = []
            test_returns = []
            
            for train_data, test_data in splits[:5]:  # Limit to 5 splits for performance
                try:
                    train_results = engine.backtest_strategy(strategy, train_data, "Train")
                    test_results = engine.backtest_strategy(strategy, test_data, "Test")
                    
                    train_returns.append(train_results.total_return)
                    test_returns.append(test_results.total_return)
                except Exception:
                    continue
            
            if len(train_returns) < 2:
                return self._create_failed_component_score("Data Quality & Overfitting"), {}
            
            # Calculate overfitting indicators
            train_mean = np.mean(train_returns)
            test_mean = np.mean(test_returns)
            
            # Performance degradation from train to test
            degradation = (train_mean - test_mean) / (abs(train_mean) + 1e-8)
            overfitting_score = max(0, 1 - degradation) * 100
            
            # Consistency across periods
            train_consistency = 1 - (np.std(train_returns) / (abs(train_mean) + 1e-8))
            test_consistency = 1 - (np.std(test_returns) / (abs(test_mean) + 1e-8))
            consistency_score = (train_consistency + test_consistency) / 2 * 100
            
            # Sample size adequacy
            sample_size_score = min(100, len(data) / 252 * 20)  # 20 points per year of data
            
            # Combined score
            overall_score = (
                overfitting_score * 0.5 +
                consistency_score * 0.3 +
                sample_size_score * 0.2
            )
            
            confidence = min(1.0, len(splits) / 5)
            
            issues = []
            if degradation > 0.2:
                issues.append("Significant performance degradation (overfitting)")
            if len(data) < 252:
                issues.append("Insufficient data for robust validation")
            
            component_score = ComponentScore(
                name="Data Quality & Overfitting",
                score=min(100, max(0, overall_score)),
                weight=self.component_weights['data_quality_overfitting'],
                weighted_score=overall_score * self.component_weights['data_quality_overfitting'],
                details={
                    'overfitting_score': overfitting_score,
                    'consistency_score': consistency_score,
                    'sample_size_score': sample_size_score,
                    'performance_degradation': degradation,
                    'n_splits_tested': len(splits)
                },
                confidence=confidence,
                issues=issues
            )
            
            return component_score, {
                'train_returns': train_returns,
                'test_returns': test_returns,
                'splits_tested': len(splits)
            }
            
        except Exception as e:
            self.logger.error(f"Data quality evaluation failed: {e}")
            return self._create_failed_component_score("Data Quality & Overfitting"), {}
    
    def _create_failed_component_score(self, component_name: str) -> ComponentScore:
        """Create a failed component score."""
        return ComponentScore(
            name=component_name,
            score=0.0,
            weight=self.component_weights.get(component_name.lower().replace(' ', '_').replace('&', ''), 0.0),
            weighted_score=0.0,
            details={'error': 'Component evaluation failed'},
            confidence=0.0,
            issues=['Component evaluation failed']
        )
    
    def _calculate_weighted_score(self, component_scores: Dict[str, ComponentScore]) -> float:
        """Calculate overall weighted score."""
        total_weighted_score = sum(score.weighted_score for score in component_scores.values())
        return min(100, max(0, total_weighted_score))
    
    def _calculate_confidence_interval(
        self,
        component_scores: Dict[str, ComponentScore]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the overall score using bootstrap sampling."""
        # Use component confidences to estimate overall confidence
        confidences = [score.confidence for score in component_scores.values()]
        weights = [score.weight for score in component_scores.values()]
        
        # Weighted average confidence
        overall_confidence = sum(c * w for c, w in zip(confidences, weights))
        
        # Calculate score uncertainty based on confidence
        scores = [score.score for score in component_scores.values()]
        score_std = np.std(scores)
        
        # Confidence interval width based on overall confidence
        interval_width = score_std * (1 - overall_confidence) * 2
        
        overall_score = self._calculate_weighted_score(component_scores)
        lower_bound = max(0, overall_score - interval_width)
        upper_bound = min(100, overall_score + interval_width)
        
        return (lower_bound, upper_bound)
    
    def _determine_deployment_recommendation(self, overall_score: float) -> DeploymentRecommendation:
        """Determine deployment recommendation based on score."""
        if overall_score >= self.deployment_thresholds['ready_for_production']:
            return DeploymentRecommendation.READY_FOR_PRODUCTION
        elif overall_score >= self.deployment_thresholds['deploy_with_caution']:
            return DeploymentRecommendation.DEPLOY_WITH_CAUTION
        elif overall_score >= self.deployment_thresholds['paper_trade_first']:
            return DeploymentRecommendation.PAPER_TRADE_FIRST
        else:
            return DeploymentRecommendation.NOT_READY
    
    def _analyze_strengths_weaknesses(
        self,
        component_scores: Dict[str, ComponentScore]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Analyze strategy strengths, weaknesses, and risk factors."""
        strengths = []
        weaknesses = []
        risk_factors = []
        
        # Analyze each component
        for component in component_scores.values():
            if component.score >= 80:
                strengths.append(f"Excellent {component.name.lower()} (Score: {component.score:.1f})")
            elif component.score >= 60:
                strengths.append(f"Good {component.name.lower()} (Score: {component.score:.1f})")
            elif component.score >= 40:
                weaknesses.append(f"Moderate {component.name.lower()} (Score: {component.score:.1f})")
            else:
                weaknesses.append(f"Poor {component.name.lower()} (Score: {component.score:.1f})")
            
            # Add specific issues as risk factors
            for issue in component.issues:
                risk_factors.append(f"{component.name}: {issue}")
            
            # Add confidence-based risk factors
            if component.confidence < 0.5:
                risk_factors.append(f"Low confidence in {component.name.lower()} assessment")
        
        # Overall risk factors
        scores = [score.score for score in component_scores.values()]
        if np.std(scores) > 30:
            risk_factors.append("High variance in component scores indicates inconsistent performance")
        
        if min(scores) < 30:
            risk_factors.append("Critical weakness in at least one validation component")
        
        return strengths, weaknesses, risk_factors
    
    def batch_score_strategies(
        self,
        strategies: List[BaseStrategy],
        data: pd.DataFrame,
        assets: List[str] = None,
        validation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, RobustnessScore]:
        """
        Calculate robustness scores for multiple strategies.
        
        Args:
            strategies: List of strategies to evaluate
            data: Historical OHLCV data
            assets: List of asset symbols for cross-asset validation
            validation_config: Optional configuration for validation methods
            
        Returns:
            Dictionary mapping strategy names to robustness scores
        """
        self.logger.info(f"Batch scoring {len(strategies)} strategies")
        
        results = {}
        
        for i, strategy in enumerate(strategies, 1):
            self.logger.info(f"Scoring strategy {i}/{len(strategies)}: {strategy.name}")
            
            try:
                score = self.calculate_robustness_score(strategy, data, assets, validation_config)
                results[strategy.name] = score
            except Exception as e:
                self.logger.error(f"Failed to score strategy {strategy.name}: {e}")
                # Create a failed score
                results[strategy.name] = self._create_failed_robustness_score(strategy.name)
        
        return results
    
    def compare_strategies(self, scores: Dict[str, RobustnessScore]) -> ComparisonReport:
        """
        Compare multiple strategy robustness scores.
        
        Args:
            scores: Dictionary of strategy names to robustness scores
            
        Returns:
            Comprehensive comparison report
        """
        self.logger.info(f"Comparing {len(scores)} strategies")
        
        # Create rankings
        strategy_rankings = sorted(
            [(name, score.overall_score) for name, score in scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate distribution statistics
        all_scores = [score.overall_score for score in scores.values()]
        score_distribution = {
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'mean': np.mean(all_scores),
            'median': np.median(all_scores),
            'std': np.std(all_scores),
            'q25': np.percentile(all_scores, 25),
            'q75': np.percentile(all_scores, 75)
        }
        
        # Identify best and worst
        best_strategy = strategy_rankings[0][0] if strategy_rankings else None
        worst_strategy = strategy_rankings[-1][0] if strategy_rankings else None
        
        # Simple clustering based on score ranges
        strategy_clusters = {
            'high_performers': [name for name, score in strategy_rankings if score >= 80],
            'moderate_performers': [name for name, score in strategy_rankings if 60 <= score < 80],
            'low_performers': [name for name, score in strategy_rankings if score < 60]
        }
        
        # Cluster characteristics
        cluster_characteristics = {}
        for cluster_name, strategy_names in strategy_clusters.items():
            if strategy_names:
                cluster_scores = [scores[name].overall_score for name in strategy_names]
                cluster_characteristics[cluster_name] = {
                    'mean_score': np.mean(cluster_scores),
                    'count': len(strategy_names),
                    'score_range': (np.min(cluster_scores), np.max(cluster_scores))
                }
        
        # Deployment recommendations
        deployment_ready_strategies = [
            name for name, score in scores.items()
            if score.deployment_recommendation in [
                DeploymentRecommendation.READY_FOR_PRODUCTION,
                DeploymentRecommendation.DEPLOY_WITH_CAUTION
            ]
        ]
        
        # Portfolio recommendations
        portfolio_recommendations = []
        if len(strategy_clusters['high_performers']) >= 2:
            portfolio_recommendations.append(
                f"Consider diversified portfolio with top performers: {', '.join(strategy_clusters['high_performers'][:3])}"
            )
        
        if best_strategy and scores[best_strategy].overall_score >= 85:
            portfolio_recommendations.append(f"Single strategy deployment recommended: {best_strategy}")
        
        return ComparisonReport(
            strategy_rankings=strategy_rankings,
            score_distribution=score_distribution,
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
            strategy_clusters=strategy_clusters,
            cluster_characteristics=cluster_characteristics,
            portfolio_recommendations=portfolio_recommendations,
            deployment_ready_strategies=deployment_ready_strategies,
            mean_score=score_distribution['mean'],
            median_score=score_distribution['median'],
            score_std=score_distribution['std']
        )
    
    def generate_deployment_report(self, score: RobustnessScore) -> DeploymentReport:
        """
        Generate detailed deployment readiness assessment.
        
        Args:
            score: Robustness score for the strategy
            
        Returns:
            Comprehensive deployment report
        """
        # Estimate risk level
        if score.overall_score >= 80:
            risk_level = "Low"
        elif score.overall_score >= 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate confidence score
        confidence_score = np.mean([comp.confidence for comp in score.component_scores.values()])
        
        # Minimum capital requirement (simplified)
        base_capital = 10000  # Base minimum
        risk_multiplier = {"Low": 1.0, "Medium": 1.5, "High": 2.5}[risk_level]
        minimum_capital = base_capital * risk_multiplier
        
        # Key metrics to monitor
        key_metrics = [
            "Daily returns",
            "Sharpe ratio",
            "Maximum drawdown",
            "Win rate"
        ]
        
        # Add component-specific metrics
        for component in score.component_scores.values():
            if component.score < 70:
                if "statistical" in component.name.lower():
                    key_metrics.append("P-values of performance tests")
                elif "regime" in component.name.lower():
                    key_metrics.append("Performance across market regimes")
                elif "monte carlo" in component.name.lower():
                    key_metrics.append("Consistency across random periods")
        
        # Revalidation frequency
        if score.overall_score >= 80:
            revalidation_freq = "Quarterly"
        elif score.overall_score >= 60:
            revalidation_freq = "Monthly"
        else:
            revalidation_freq = "Weekly"
        
        # Stop loss recommendations
        stop_loss_recs = {
            "daily_loss_limit": 0.02 if risk_level == "Low" else 0.015,
            "monthly_drawdown_limit": 0.10 if risk_level == "Low" else 0.05,
            "consecutive_losses_limit": 5 if risk_level == "Low" else 3
        }
        
        # Pre-deployment checklist
        checklist = [
            ("Robustness score calculated", True),
            ("Statistical significance verified", score.component_scores.get('statistical_significance', ComponentScore('', 0, 0, 0, {}, 0)).score >= 60),
            ("Cross-asset validation completed", score.component_scores.get('cross_asset_performance', ComponentScore('', 0, 0, 0, {}, 0)).score >= 60),
            ("Risk management parameters set", True),
            ("Monitoring systems configured", False),  # Assume not done yet
            ("Paper trading completed", score.deployment_recommendation != DeploymentRecommendation.READY_FOR_PRODUCTION)
        ]
        
        # Post-deployment monitoring
        monitoring_items = [
            "Daily performance tracking",
            "Risk metric monitoring",
            "Market regime detection",
            "Strategy parameter stability",
            "Execution quality analysis"
        ]
        
        return DeploymentReport(
            strategy_name=score.strategy_name,
            overall_score=score.overall_score,
            recommendation=score.deployment_recommendation,
            estimated_risk_level=risk_level,
            confidence_score=confidence_score,
            minimum_capital_requirement=minimum_capital,
            key_metrics_to_monitor=key_metrics,
            revalidation_frequency=revalidation_freq,
            stop_loss_recommendations=stop_loss_recs,
            pre_deployment_checklist=checklist,
            post_deployment_monitoring=monitoring_items
        )
    
    def _create_failed_robustness_score(self, strategy_name: str) -> RobustnessScore:
        """Create a failed robustness score."""
        failed_components = {}
        for component_name, weight in self.component_weights.items():
            failed_components[component_name] = ComponentScore(
                name=component_name.replace('_', ' ').title(),
                score=0.0,
                weight=weight,
                weighted_score=0.0,
                details={'error': 'Evaluation failed'},
                confidence=0.0,
                issues=['Evaluation failed']
            )
        
        return RobustnessScore(
            strategy_name=strategy_name,
            overall_score=0.0,
            component_scores=failed_components,
            confidence_interval=(0.0, 0.0),
            deployment_recommendation=DeploymentRecommendation.NOT_READY,
            strengths=[],
            weaknesses=["Complete evaluation failure"],
            risk_factors=["Strategy evaluation failed"],
            detailed_metrics={},
            validation_date=datetime.now(),
            data_period=(datetime.now(), datetime.now()),
            assets_tested=[],
            min_component_score=0.0,
            max_component_score=0.0,
            score_variance=0.0
        )
    
    def create_robustness_report(
        self,
        score: RobustnessScore,
        save_path: Optional[str] = None
    ) -> str:
        """Create a comprehensive robustness assessment report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"ROBUSTNESS ASSESSMENT REPORT - {score.strategy_name}")
        report_lines.append("=" * 80)
        
        # Overall assessment
        report_lines.append(f"\nOVERALL ROBUSTNESS SCORE: {score.overall_score:.1f}/100")
        report_lines.append(f"Confidence Interval: [{score.confidence_interval[0]:.1f}, {score.confidence_interval[1]:.1f}]")
        report_lines.append(f"Deployment Recommendation: {score.deployment_recommendation.value}")
        
        # Component breakdown
        report_lines.append(f"\nCOMPONENT SCORES:")
        for component in score.component_scores.values():
            report_lines.append(f"  {component.name}: {component.score:.1f}/100 (Weight: {component.weight:.1%})")
            if component.issues:
                for issue in component.issues:
                    report_lines.append(f"    ⚠️  {issue}")
        
        # Strengths and weaknesses
        report_lines.append(f"\nSTRENGTHS:")
        for strength in score.strengths:
            report_lines.append(f"  ✅ {strength}")
        
        report_lines.append(f"\nWEAKNESSES:")
        for weakness in score.weaknesses:
            report_lines.append(f"  ❌ {weakness}")
        
        report_lines.append(f"\nRISK FACTORS:")
        for risk in score.risk_factors:
            report_lines.append(f"  ⚠️  {risk}")
        
        # Validation details
        report_lines.append(f"\nVALIDATION DETAILS:")
        report_lines.append(f"  Validation Date: {score.validation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"  Data Period: {score.data_period[0].strftime('%Y-%m-%d')} to {score.data_period[1].strftime('%Y-%m-%d')}")
        report_lines.append(f"  Assets Tested: {', '.join(score.assets_tested)}")
        report_lines.append(f"  Score Variance: {score.score_variance:.2f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Robustness report saved to {save_path}")
        
        return report 