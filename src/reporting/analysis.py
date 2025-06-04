"""
Automated Analysis Engine

This module provides intelligent analysis and interpretation capabilities
for trading strategy reports, generating insights and recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from .data_integration import ReportDataPackage
    from ..utils.logger import get_logger
except ImportError:
    from src.reporting.data_integration import ReportDataPackage
    from src.utils.logger import get_logger


class PerformanceLevel(Enum):
    """Performance level classifications."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    AVERAGE = "Average"
    POOR = "Poor"
    UNACCEPTABLE = "Unacceptable"


class RiskLevel(Enum):
    """Risk level classifications."""
    VERY_LOW = "Very Low"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class DeploymentRecommendation(Enum):
    """Deployment recommendation levels."""
    STRONGLY_RECOMMENDED = "Strongly Recommended"
    RECOMMENDED = "Recommended"
    CONDITIONAL = "Conditional"
    NOT_RECOMMENDED = "Not Recommended"
    REJECTED = "Rejected"


@dataclass
class PerformanceAnalysis:
    """Analysis of strategy performance."""
    performance_level: PerformanceLevel
    key_strengths: List[str]
    key_weaknesses: List[str]
    comparative_analysis: str
    improvement_suggestions: List[str]


@dataclass
class RiskAnalysis:
    """Analysis of strategy risk characteristics."""
    risk_level: RiskLevel
    primary_risk_factors: List[str]
    risk_mitigation_strategies: List[str]
    risk_tolerance_assessment: str
    monitoring_recommendations: List[str]


@dataclass
class ValidationAnalysis:
    """Analysis of validation framework results."""
    validation_confidence: float
    robustness_assessment: str
    generalization_capability: str
    overfitting_assessment: str
    deployment_readiness: DeploymentRecommendation


@dataclass
class MarketAnalysis:
    """Analysis of market conditions and strategy fit."""
    market_dependency: str
    regime_performance: str
    asset_generalization: str
    market_recommendations: List[str]


@dataclass
class ComprehensiveAnalysis:
    """Complete analysis package."""
    executive_summary: str
    performance_analysis: PerformanceAnalysis
    risk_analysis: RiskAnalysis
    validation_analysis: ValidationAnalysis
    market_analysis: MarketAnalysis
    overall_recommendation: DeploymentRecommendation
    confidence_score: float
    key_insights: List[str]
    action_items: List[str]


class PerformanceAnalyzer:
    """Analyzes performance metrics and generates insights."""
    
    def __init__(self):
        self.logger = get_logger("performance_analyzer")
    
    def analyze_performance(self, metrics: Dict[str, Any]) -> PerformanceAnalysis:
        """Analyze performance metrics and generate insights."""
        
        # Extract key metrics
        total_return = metrics.get('total_return', 0)
        annual_return = metrics.get('annual_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        
        # Determine performance level
        performance_level = self._classify_performance(
            annual_return, sharpe_ratio, max_drawdown, win_rate
        )
        
        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(metrics)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(metrics)
        
        # Suggest improvements
        improvement_suggestions = self._generate_improvement_suggestions(metrics, weaknesses)
        
        return PerformanceAnalysis(
            performance_level=performance_level,
            key_strengths=strengths,
            key_weaknesses=weaknesses,
            comparative_analysis=comparative_analysis,
            improvement_suggestions=improvement_suggestions
        )
    
    def _classify_performance(
        self, annual_return: float, sharpe_ratio: float, 
        max_drawdown: float, win_rate: float
    ) -> PerformanceLevel:
        """Classify overall performance level."""
        
        score = 0
        
        # Annual return scoring
        if annual_return > 0.3:
            score += 2
        elif annual_return > 0.15:
            score += 1
        elif annual_return < 0:
            score -= 2
        
        # Sharpe ratio scoring
        if sharpe_ratio > 2.0:
            score += 2
        elif sharpe_ratio > 1.5:
            score += 1
        elif sharpe_ratio < 0.5:
            score -= 1
        elif sharpe_ratio < 0:
            score -= 2
        
        # Drawdown scoring
        if max_drawdown < 0.05:
            score += 1
        elif max_drawdown > 0.2:
            score -= 1
        elif max_drawdown > 0.3:
            score -= 2
        
        # Win rate scoring
        if win_rate > 0.6:
            score += 1
        elif win_rate < 0.4:
            score -= 1
        
        # Classification
        if score >= 4:
            return PerformanceLevel.EXCELLENT
        elif score >= 2:
            return PerformanceLevel.GOOD
        elif score >= 0:
            return PerformanceLevel.AVERAGE
        elif score >= -2:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.UNACCEPTABLE
    
    def _identify_strengths_weaknesses(self, metrics: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Identify key strengths and weaknesses."""
        
        strengths = []
        weaknesses = []
        
        # Analyze individual metrics
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        if sharpe_ratio > 1.5:
            strengths.append(f"Excellent risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        elif sharpe_ratio < 0.8:
            weaknesses.append(f"Poor risk-adjusted returns (Sharpe: {sharpe_ratio:.2f})")
        
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        if max_drawdown < 0.1:
            strengths.append(f"Low maximum drawdown ({max_drawdown:.1%})")
        elif max_drawdown > 0.2:
            weaknesses.append(f"High maximum drawdown ({max_drawdown:.1%})")
        
        win_rate = metrics.get('win_rate', 0)
        if win_rate > 0.6:
            strengths.append(f"High win rate ({win_rate:.1%})")
        elif win_rate < 0.4:
            weaknesses.append(f"Low win rate ({win_rate:.1%})")
        
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor > 1.5:
            strengths.append(f"Strong profit factor ({profit_factor:.2f})")
        elif profit_factor < 1.2:
            weaknesses.append(f"Weak profit factor ({profit_factor:.2f})")
        
        sortino_ratio = metrics.get('sortino_ratio', 0)
        if sortino_ratio > 2.0:
            strengths.append(f"Excellent downside risk management (Sortino: {sortino_ratio:.2f})")
        
        calmar_ratio = metrics.get('calmar_ratio', 0)
        if calmar_ratio > 1.0:
            strengths.append(f"Good return-to-drawdown ratio (Calmar: {calmar_ratio:.2f})")
        
        return strengths, weaknesses
    
    def _generate_comparative_analysis(self, metrics: Dict[str, Any]) -> str:
        """Generate comparative analysis text."""
        
        annual_return = metrics.get('annual_return', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        
        analysis = []
        
        # Benchmark comparisons
        if annual_return > 0.1:
            analysis.append("Strategy significantly outperforms typical market returns")
        elif annual_return > 0.05:
            analysis.append("Strategy moderately outperforms conservative benchmarks")
        else:
            analysis.append("Strategy underperforms compared to passive investment alternatives")
        
        # Risk-adjusted performance
        if sharpe_ratio > 1.5:
            analysis.append("Risk-adjusted performance exceeds institutional standards")
        elif sharpe_ratio > 1.0:
            analysis.append("Risk-adjusted performance meets professional trading criteria")
        else:
            analysis.append("Risk-adjusted performance below institutional expectations")
        
        # Drawdown comparison
        if max_drawdown < 0.1:
            analysis.append("Drawdown characteristics superior to most active strategies")
        elif max_drawdown < 0.2:
            analysis.append("Drawdown levels acceptable for institutional portfolios")
        else:
            analysis.append("Drawdown levels may be concerning for risk-averse investors")
        
        return ". ".join(analysis) + "."
    
    def _generate_improvement_suggestions(
        self, metrics: Dict[str, Any], weaknesses: List[str]
    ) -> List[str]:
        """Generate improvement suggestions based on weaknesses."""
        
        suggestions = []
        
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < 1.0:
            suggestions.append("Consider implementing dynamic position sizing to improve risk-adjusted returns")
            suggestions.append("Evaluate adding volatility filters to reduce exposure during high-risk periods")
        
        max_drawdown = abs(metrics.get('max_drawdown', 0))
        if max_drawdown > 0.15:
            suggestions.append("Implement stricter stop-loss mechanisms to limit downside exposure")
            suggestions.append("Consider portfolio heat models to reduce position sizes during losing streaks")
        
        win_rate = metrics.get('win_rate', 0)
        if win_rate < 0.45:
            suggestions.append("Refine entry signals to improve trade selection accuracy")
            suggestions.append("Consider implementing trend filters to avoid counter-trend trades")
        
        profit_factor = metrics.get('profit_factor', 0)
        if profit_factor < 1.3:
            suggestions.append("Optimize exit strategies to maximize winning trades and minimize losses")
            suggestions.append("Implement trade management techniques like partial profit-taking")
        
        if not suggestions:
            suggestions.append("Strategy shows strong fundamentals; focus on parameter fine-tuning")
            suggestions.append("Consider expanding to additional markets or timeframes")
        
        return suggestions


class RiskAnalyzer:
    """Analyzes risk characteristics and generates risk insights."""
    
    def __init__(self):
        self.logger = get_logger("risk_analyzer")
    
    def analyze_risk(self, risk_data: Dict[str, Any], performance_data: Dict[str, Any]) -> RiskAnalysis:
        """Analyze risk characteristics and generate insights."""
        
        # Determine risk level
        risk_level = self._classify_risk_level(risk_data, performance_data)
        
        # Identify primary risk factors
        risk_factors = self._identify_risk_factors(risk_data, performance_data)
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors, risk_data)
        
        # Assess risk tolerance requirements
        tolerance_assessment = self._assess_risk_tolerance(risk_level, risk_data)
        
        # Generate monitoring recommendations
        monitoring_recs = self._generate_monitoring_recommendations(risk_factors)
        
        return RiskAnalysis(
            risk_level=risk_level,
            primary_risk_factors=risk_factors,
            risk_mitigation_strategies=mitigation_strategies,
            risk_tolerance_assessment=tolerance_assessment,
            monitoring_recommendations=monitoring_recs
        )
    
    def _classify_risk_level(self, risk_data: Dict[str, Any], performance_data: Dict[str, Any]) -> RiskLevel:
        """Classify overall risk level."""
        
        risk_score = risk_data.get('risk_score', 50)
        max_drawdown = abs(performance_data.get('max_drawdown', 0))
        volatility = performance_data.get('volatility', 0)
        
        # Composite risk assessment
        if risk_score < 20 and max_drawdown < 0.05 and volatility < 0.15:
            return RiskLevel.VERY_LOW
        elif risk_score < 40 and max_drawdown < 0.1 and volatility < 0.25:
            return RiskLevel.LOW
        elif risk_score < 60 and max_drawdown < 0.2 and volatility < 0.35:
            return RiskLevel.MODERATE
        elif risk_score < 80 and max_drawdown < 0.3:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _identify_risk_factors(self, risk_data: Dict[str, Any], performance_data: Dict[str, Any]) -> List[str]:
        """Identify primary risk factors."""
        
        factors = []
        
        # Performance-based risk factors
        max_drawdown = abs(performance_data.get('max_drawdown', 0))
        if max_drawdown > 0.2:
            factors.append(f"High maximum drawdown risk ({max_drawdown:.1%})")
        
        volatility = performance_data.get('volatility', 0)
        if volatility > 0.3:
            factors.append(f"High volatility exposure ({volatility:.1%} annual)")
        
        # Anti-overfitting risks
        overfitting_risk = risk_data.get('overfitting_risk', 0)
        if overfitting_risk > 60:
            factors.append("Significant overfitting risk detected")
        
        # Validation risks
        validation_risk = risk_data.get('validation_risk', 0)
        if validation_risk > 50:
            factors.append("Strategy validation concerns identified")
        
        # Market dependency
        market_correlations = risk_data.get('market_correlations', {})
        high_correlations = [k for k, v in market_correlations.items() if abs(v) > 0.7]
        if high_correlations:
            factors.append(f"High market dependency: {', '.join(high_correlations)}")
        
        return factors
    
    def _generate_mitigation_strategies(self, risk_factors: List[str], risk_data: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies."""
        
        strategies = []
        
        # Factor-specific strategies
        for factor in risk_factors:
            if "drawdown" in factor.lower():
                strategies.append("Implement dynamic position sizing based on portfolio heat")
                strategies.append("Add maximum consecutive loss limits")
            
            elif "volatility" in factor.lower():
                strategies.append("Use volatility-adjusted position sizing")
                strategies.append("Implement volatility regime filters")
            
            elif "overfitting" in factor.lower():
                strategies.append("Conduct additional out-of-sample testing")
                strategies.append("Reduce strategy complexity and parameter count")
            
            elif "dependency" in factor.lower():
                strategies.append("Diversify across uncorrelated markets")
                strategies.append("Implement market regime detection")
        
        # General risk management
        strategies.extend([
            "Regular strategy performance monitoring and review",
            "Implement portfolio-level risk limits",
            "Maintain detailed trade logs for performance attribution"
        ])
        
        return list(set(strategies))  # Remove duplicates
    
    def _assess_risk_tolerance(self, risk_level: RiskLevel, risk_data: Dict[str, Any]) -> str:
        """Assess required risk tolerance for strategy deployment."""
        
        if risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            return "Suitable for conservative investors and institutional portfolios with low risk tolerance"
        elif risk_level == RiskLevel.MODERATE:
            return "Appropriate for moderate risk tolerance investors seeking balanced risk-return profiles"
        elif risk_level == RiskLevel.HIGH:
            return "Requires high risk tolerance and sophisticated risk management capabilities"
        else:
            return "Only suitable for aggressive investors with very high risk tolerance and substantial capital"
    
    def _generate_monitoring_recommendations(self, risk_factors: List[str]) -> List[str]:
        """Generate monitoring recommendations."""
        
        recommendations = [
            "Monitor rolling Sharpe ratio on weekly basis",
            "Track maximum drawdown duration and magnitude",
            "Implement real-time performance attribution analysis"
        ]
        
        # Factor-specific monitoring
        if any("overfitting" in factor.lower() for factor in risk_factors):
            recommendations.append("Conduct quarterly out-of-sample validation tests")
        
        if any("dependency" in factor.lower() for factor in risk_factors):
            recommendations.append("Monitor correlation with market factors monthly")
        
        recommendations.extend([
            "Set up automated alerts for risk threshold breaches",
            "Regular review of strategy assumptions and market conditions"
        ])
        
        return recommendations


class AutomatedAnalysisEngine:
    """Main engine that coordinates all analysis components."""
    
    def __init__(self):
        """Initialize the analysis engine."""
        self.logger = get_logger("automated_analysis_engine")
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
    
    def generate_comprehensive_analysis(self, data_package: ReportDataPackage) -> ComprehensiveAnalysis:
        """Generate comprehensive analysis from data package."""
        
        self.logger.info(f"Generating comprehensive analysis for {data_package.strategy_name}")
        
        # Analyze performance
        performance_analysis = self.performance_analyzer.analyze_performance(
            {**data_package.performance_metrics, **data_package.trading_metrics}
        )
        
        # Analyze risk
        risk_analysis = self.risk_analyzer.analyze_risk(
            data_package.risk_assessment,
            data_package.performance_metrics
        )
        
        # Analyze validation
        validation_analysis = self._analyze_validation(data_package)
        
        # Analyze market characteristics
        market_analysis = self._analyze_market_characteristics(data_package)
        
        # Generate overall recommendation
        overall_recommendation = self._determine_overall_recommendation(
            performance_analysis, risk_analysis, validation_analysis
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(data_package, validation_analysis)
        
        # Generate key insights
        key_insights = self._generate_key_insights(
            performance_analysis, risk_analysis, validation_analysis, market_analysis
        )
        
        # Generate action items
        action_items = self._generate_action_items(
            performance_analysis, risk_analysis, overall_recommendation
        )
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            data_package, performance_analysis, risk_analysis, overall_recommendation
        )
        
        return ComprehensiveAnalysis(
            executive_summary=executive_summary,
            performance_analysis=performance_analysis,
            risk_analysis=risk_analysis,
            validation_analysis=validation_analysis,
            market_analysis=market_analysis,
            overall_recommendation=overall_recommendation,
            confidence_score=confidence_score,
            key_insights=key_insights,
            action_items=action_items
        )
    
    def _analyze_validation(self, data_package: ReportDataPackage) -> ValidationAnalysis:
        """Analyze validation framework results."""
        
        overall_score = data_package.validation_results.get('overall_score', 0)
        
        # Validation confidence
        confidence = min(100, overall_score * 1.2) / 100
        
        # Robustness assessment
        if overall_score >= 80:
            robustness = "Strategy demonstrates strong robustness across multiple validation tests"
        elif overall_score >= 60:
            robustness = "Strategy shows adequate robustness with some areas for improvement"
        else:
            robustness = "Strategy robustness is questionable and requires further validation"
        
        # Generalization capability
        cross_asset_score = 0
        if data_package.cross_asset_validation:
            cross_asset_score = data_package.cross_asset_validation.get('consistency_score', 0)
        
        if cross_asset_score >= 70:
            generalization = "Strong generalization capability across different assets"
        elif cross_asset_score >= 50:
            generalization = "Moderate generalization with asset-specific considerations"
        else:
            generalization = "Limited generalization; strategy may be asset-specific"
        
        # Overfitting assessment
        overfitting_risk = data_package.anti_overfitting.get('overfitting_risk', 50)
        if overfitting_risk < 30:
            overfitting = "Low overfitting risk; strategy likely to generalize well"
        elif overfitting_risk < 60:
            overfitting = "Moderate overfitting risk; additional validation recommended"
        else:
            overfitting = "High overfitting risk; strategy may be curve-fitted to historical data"
        
        # Deployment readiness
        if overall_score >= 80 and overfitting_risk < 40:
            deployment = DeploymentRecommendation.STRONGLY_RECOMMENDED
        elif overall_score >= 70 and overfitting_risk < 50:
            deployment = DeploymentRecommendation.RECOMMENDED
        elif overall_score >= 60:
            deployment = DeploymentRecommendation.CONDITIONAL
        elif overall_score >= 40:
            deployment = DeploymentRecommendation.NOT_RECOMMENDED
        else:
            deployment = DeploymentRecommendation.REJECTED
        
        return ValidationAnalysis(
            validation_confidence=confidence,
            robustness_assessment=robustness,
            generalization_capability=generalization,
            overfitting_assessment=overfitting,
            deployment_readiness=deployment
        )
    
    def _analyze_market_characteristics(self, data_package: ReportDataPackage) -> MarketAnalysis:
        """Analyze market characteristics and dependencies."""
        
        # Market dependency analysis
        correlations = data_package.market_dependency.get('market_correlations', {})
        high_deps = [k for k, v in correlations.items() if abs(v) > 0.6]
        
        if not high_deps:
            dependency = "Low market dependency; strategy appears market-neutral"
        elif len(high_deps) == 1:
            dependency = f"Moderate dependency on {high_deps[0]} factor"
        else:
            dependency = f"High market dependency on multiple factors: {', '.join(high_deps)}"
        
        # Regime performance
        regime_data = data_package.regime_analysis
        if regime_data:
            consistency = regime_data.get('regime_consistency', 0) * 100
            if consistency >= 80:
                regime_perf = "Consistent performance across all market regimes"
            elif consistency >= 60:
                regime_perf = "Good performance with some regime-specific variations"
            else:
                regime_perf = "Performance highly dependent on market regime"
        else:
            regime_perf = "Regime analysis not available"
        
        # Asset generalization
        cross_asset_data = data_package.cross_asset_validation
        if cross_asset_data:
            asset_score = cross_asset_data.get('consistency_score', 0)
            if asset_score >= 70:
                asset_gen = "Strong performance across multiple asset classes"
            elif asset_score >= 50:
                asset_gen = "Moderate asset generalization capability"
            else:
                asset_gen = "Limited to specific asset classes"
        else:
            asset_gen = "Cross-asset testing not available"
        
        # Market recommendations
        recommendations = []
        if high_deps:
            recommendations.append("Monitor strategy performance during periods of low correlation with dependent factors")
        
        if regime_data and regime_data.get('regime_consistency', 0) < 0.7:
            recommendations.append("Consider implementing regime-aware position sizing")
        
        recommendations.extend([
            "Regular correlation monitoring with market factors",
            "Diversification across multiple strategies recommended"
        ])
        
        return MarketAnalysis(
            market_dependency=dependency,
            regime_performance=regime_perf,
            asset_generalization=asset_gen,
            market_recommendations=recommendations
        )
    
    def _determine_overall_recommendation(
        self,
        performance_analysis: PerformanceAnalysis,
        risk_analysis: RiskAnalysis,
        validation_analysis: ValidationAnalysis
    ) -> DeploymentRecommendation:
        """Determine overall deployment recommendation."""
        
        # Weight factors
        perf_weight = 0.4
        risk_weight = 0.3
        validation_weight = 0.3
        
        # Performance scoring
        perf_scores = {
            PerformanceLevel.EXCELLENT: 100,
            PerformanceLevel.GOOD: 80,
            PerformanceLevel.AVERAGE: 60,
            PerformanceLevel.POOR: 40,
            PerformanceLevel.UNACCEPTABLE: 0
        }
        perf_score = perf_scores[performance_analysis.performance_level]
        
        # Risk scoring (inverted - lower risk is better)
        risk_scores = {
            RiskLevel.VERY_LOW: 100,
            RiskLevel.LOW: 80,
            RiskLevel.MODERATE: 60,
            RiskLevel.HIGH: 40,
            RiskLevel.VERY_HIGH: 20
        }
        risk_score = risk_scores[risk_analysis.risk_level]
        
        # Validation scoring
        validation_scores = {
            DeploymentRecommendation.STRONGLY_RECOMMENDED: 100,
            DeploymentRecommendation.RECOMMENDED: 80,
            DeploymentRecommendation.CONDITIONAL: 60,
            DeploymentRecommendation.NOT_RECOMMENDED: 40,
            DeploymentRecommendation.REJECTED: 0
        }
        validation_score = validation_scores[validation_analysis.deployment_readiness]
        
        # Weighted overall score
        overall_score = (
            perf_score * perf_weight +
            risk_score * risk_weight +
            validation_score * validation_weight
        )
        
        # Determine recommendation
        if overall_score >= 85:
            return DeploymentRecommendation.STRONGLY_RECOMMENDED
        elif overall_score >= 70:
            return DeploymentRecommendation.RECOMMENDED
        elif overall_score >= 55:
            return DeploymentRecommendation.CONDITIONAL
        elif overall_score >= 35:
            return DeploymentRecommendation.NOT_RECOMMENDED
        else:
            return DeploymentRecommendation.REJECTED
    
    def _calculate_confidence_score(
        self, data_package: ReportDataPackage, validation_analysis: ValidationAnalysis
    ) -> float:
        """Calculate overall confidence score."""
        
        factors = []
        
        # Validation confidence
        factors.append(validation_analysis.validation_confidence)
        
        # Data quality and quantity
        data_quality = data_package.data_specifications.get('quality_score', 95) / 100
        factors.append(data_quality)
        
        # Anti-overfitting measures
        overfitting_risk = data_package.anti_overfitting.get('overfitting_risk', 50)
        overfitting_confidence = max(0, (100 - overfitting_risk) / 100)
        factors.append(overfitting_confidence)
        
        # Performance consistency
        if data_package.multi_period_validation:
            consistency = data_package.multi_period_validation.get('consistency_score', 50) / 100
            factors.append(consistency)
        
        return np.mean(factors)
    
    def _generate_key_insights(
        self,
        performance_analysis: PerformanceAnalysis,
        risk_analysis: RiskAnalysis,
        validation_analysis: ValidationAnalysis,
        market_analysis: MarketAnalysis
    ) -> List[str]:
        """Generate key insights across all analyses."""
        
        insights = []
        
        # Performance insights
        if performance_analysis.performance_level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
            insights.append("Strategy demonstrates strong risk-adjusted performance characteristics")
        
        # Risk insights
        if risk_analysis.risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW]:
            insights.append("Risk profile is suitable for institutional deployment")
        elif risk_analysis.risk_level == RiskLevel.VERY_HIGH:
            insights.append("High risk profile requires sophisticated risk management")
        
        # Validation insights
        if validation_analysis.validation_confidence > 0.8:
            insights.append("High validation confidence supports strategy reliability")
        
        # Market insights
        if "market-neutral" in market_analysis.market_dependency.lower():
            insights.append("Market-neutral characteristics provide portfolio diversification benefits")
        
        return insights
    
    def _generate_action_items(
        self,
        performance_analysis: PerformanceAnalysis,
        risk_analysis: RiskAnalysis,
        overall_recommendation: DeploymentRecommendation
    ) -> List[str]:
        """Generate actionable next steps."""
        
        actions = []
        
        # Performance-based actions
        if performance_analysis.improvement_suggestions:
            actions.append("Implement performance improvement suggestions")
        
        # Risk-based actions
        if risk_analysis.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            actions.append("Develop comprehensive risk management framework")
        
        # Recommendation-based actions
        if overall_recommendation == DeploymentRecommendation.STRONGLY_RECOMMENDED:
            actions.extend([
                "Prepare for production deployment",
                "Set up monitoring and alerting systems"
            ])
        elif overall_recommendation == DeploymentRecommendation.CONDITIONAL:
            actions.extend([
                "Address identified weaknesses before deployment",
                "Conduct additional validation testing"
            ])
        elif overall_recommendation == DeploymentRecommendation.NOT_RECOMMENDED:
            actions.extend([
                "Significant strategy improvements required",
                "Consider alternative strategy development"
            ])
        
        return actions
    
    def _create_executive_summary(
        self,
        data_package: ReportDataPackage,
        performance_analysis: PerformanceAnalysis,
        risk_analysis: RiskAnalysis,
        overall_recommendation: DeploymentRecommendation
    ) -> str:
        """Create executive summary text."""
        
        strategy_name = data_package.strategy_name
        perf_level = performance_analysis.performance_level.value
        risk_level = risk_analysis.risk_level.value
        recommendation = overall_recommendation.value
        
        annual_return = data_package.performance_metrics.get('annual_return', 0)
        sharpe_ratio = data_package.performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = abs(data_package.performance_metrics.get('max_drawdown', 0))
        
        summary = f"""
        The {strategy_name} trading strategy has been comprehensively analyzed and classified as 
        {perf_level.lower()} performance with {risk_level.lower()} risk characteristics. 
        
        Key performance metrics include an annual return of {annual_return:.1%}, Sharpe ratio of 
        {sharpe_ratio:.2f}, and maximum drawdown of {max_drawdown:.1%}. 
        
        Based on the comprehensive analysis across performance, risk, and validation frameworks, 
        the strategy receives a deployment recommendation of: {recommendation}.
        
        {performance_analysis.key_strengths[0] if performance_analysis.key_strengths else 'Strategy shows solid fundamentals.'} 
        However, attention should be paid to {performance_analysis.key_weaknesses[0].lower() if performance_analysis.key_weaknesses else 'ongoing performance monitoring'}.
        """
        
        return " ".join(summary.split())  # Clean up whitespace 