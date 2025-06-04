"""
Fundamental Inefficiency Analysis Framework

This module provides comprehensive analysis to distinguish between strategies that exploit
genuine market inefficiencies versus those relying on statistical artifacts or data mining.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import logging

try:
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from ..utils.logger import get_logger
    from .multi_period_validator import MultiPeriodValidator, MultiPeriodValidationResult
    from .cross_asset_tester import CrossAssetTester, CrossAssetTestResult
except ImportError:
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.utils.logger import get_logger
    from src.anti_overfitting.multi_period_validator import MultiPeriodValidator, MultiPeriodValidationResult
    from src.anti_overfitting.cross_asset_tester import CrossAssetTester, CrossAssetTestResult


class InefficiencyType(Enum):
    """Types of market inefficiencies."""
    FUNDAMENTAL = "fundamental"
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    INFORMATIONAL = "informational"
    TECHNICAL = "technical"
    STATISTICAL_ARTIFACT = "statistical_artifact"
    UNKNOWN = "unknown"


class StrategyCategory(Enum):
    """Strategy categories based on inefficiency exploitation."""
    FUNDAMENTAL_BASED = "fundamental_based"
    BEHAVIORAL_BASED = "behavioral_based"
    STRUCTURAL_BASED = "structural_based"
    TECHNICAL_BASED = "technical_based"
    MIXED_APPROACH = "mixed_approach"
    DATA_MINING = "data_mining"
    UNCLASSIFIED = "unclassified"


class EconomicRationaleStrength(Enum):
    """Strength of economic rationale."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    ABSENT = "absent"


class PersistenceLevel(Enum):
    """Persistence level across conditions."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class InefficiencyDocumentation:
    """Documentation requirements for strategy inefficiency analysis."""
    
    # Required documentation
    economic_rationale: str  # Explanation of why the inefficiency exists
    market_mechanism: str  # How the market mechanism creates the inefficiency
    persistence_reasoning: str  # Why the inefficiency should persist
    
    # Supporting evidence
    academic_references: List[str] = field(default_factory=list)
    empirical_evidence: List[str] = field(default_factory=list)
    market_examples: List[str] = field(default_factory=list)
    
    # Risk factors
    decay_factors: List[str] = field(default_factory=list)  # What could eliminate the inefficiency
    competition_risks: List[str] = field(default_factory=list)  # Competitive threats
    market_evolution_risks: List[str] = field(default_factory=list)  # Market structure changes
    
    # Implementation details
    signal_sources: List[str] = field(default_factory=list)  # Data sources used
    execution_requirements: List[str] = field(default_factory=list)  # Execution constraints
    capacity_limitations: List[str] = field(default_factory=list)  # Scalability limits


@dataclass
class InefficiencyAnalysis:
    """Analysis results for strategy inefficiency classification."""
    
    # Classification results (non-default fields first)
    inefficiency_type: InefficiencyType
    strategy_category: StrategyCategory
    confidence_score: float  # 0-1, confidence in classification
    
    # Economic rationale assessment
    rationale_strength: EconomicRationaleStrength
    rationale_score: float  # 0-100, quality of economic explanation
    
    # Persistence analysis
    persistence_level: PersistenceLevel
    persistence_score: float  # 0-100, persistence across conditions
    cross_asset_persistence: float  # 0-100, persistence across assets
    temporal_persistence: float  # 0-100, persistence over time
    
    # Risk assessment
    data_mining_risk: float  # 0-100, risk of being data mining artifact
    overfitting_risk: float  # 0-100, risk of overfitting
    decay_risk: float  # 0-100, risk of inefficiency decay
    
    # Validation results
    documentation_completeness: float  # 0-100, completeness of documentation
    empirical_support: float  # 0-100, empirical evidence strength
    
    # Supporting metrics (default fields)
    fundamental_indicators: Dict[str, float] = field(default_factory=dict)
    behavioral_indicators: Dict[str, float] = field(default_factory=dict)
    structural_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Flags and warnings (default fields)
    red_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FundamentalInefficiencyResult:
    """Complete fundamental inefficiency analysis result."""
    # Non-default fields first
    strategy_name: str
    analysis: InefficiencyAnalysis
    documentation: InefficiencyDocumentation
    overall_score: float  # 0-100, overall inefficiency exploitation score
    deployment_recommendation: str  # "Approved", "Conditional", "Rejected"
    
    # Default fields
    multi_period_validation: Optional[MultiPeriodValidationResult] = None
    cross_asset_validation: Optional[CrossAssetTestResult] = None
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)


class InefficiencyClassifier:
    """
    Classifies strategies based on the type of market inefficiency they exploit.
    """
    
    def __init__(self):
        """Initialize the InefficiencyClassifier."""
        self.logger = get_logger("inefficiency_classifier")
        
        # Define classification keywords and patterns
        self._fundamental_keywords = {
            'value', 'earnings', 'revenue', 'profit', 'cash_flow', 'dividend',
            'book_value', 'pe_ratio', 'pb_ratio', 'debt', 'equity', 'financial',
            'balance_sheet', 'income_statement', 'dcf', 'intrinsic_value'
        }
        
        self._behavioral_keywords = {
            'sentiment', 'fear', 'greed', 'panic', 'euphoria', 'overreaction',
            'underreaction', 'herding', 'momentum', 'reversal', 'bias',
            'psychology', 'emotion', 'crowd', 'contrarian'
        }
        
        self._structural_keywords = {
            'arbitrage', 'spread', 'basis', 'calendar', 'cross_asset',
            'pairs_trading', 'statistical_arbitrage', 'market_making',
            'liquidity', 'bid_ask', 'order_flow', 'microstructure'
        }
        
        self._technical_keywords = {
            'moving_average', 'rsi', 'macd', 'bollinger', 'stochastic',
            'support', 'resistance', 'trend', 'pattern', 'breakout',
            'oscillator', 'indicator', 'signal', 'crossover'
        }
    
    def classify_strategy(
        self,
        strategy: BaseStrategy,
        documentation: InefficiencyDocumentation,
        performance_results: BacktestResults
    ) -> InefficiencyAnalysis:
        """
        Classify a strategy based on its inefficiency exploitation.
        
        Args:
            strategy: Strategy to classify
            documentation: Strategy documentation
            performance_results: Backtest performance results
            
        Returns:
            InefficiencyAnalysis with classification results
        """
        self.logger.info(f"Classifying strategy: {strategy.name}")
        
        # Analyze documentation content
        doc_analysis = self._analyze_documentation(documentation)
        
        # Analyze strategy implementation
        impl_analysis = self._analyze_implementation(strategy)
        
        # Analyze performance characteristics
        perf_analysis = self._analyze_performance_characteristics(performance_results)
        
        # Combine analyses for classification
        inefficiency_type = self._determine_inefficiency_type(doc_analysis, impl_analysis)
        strategy_category = self._determine_strategy_category(inefficiency_type, doc_analysis)
        confidence_score = self._calculate_confidence_score(doc_analysis, impl_analysis, perf_analysis)
        
        # Assess economic rationale
        rationale_strength, rationale_score = self._assess_economic_rationale(documentation)
        
        # Calculate risk scores
        data_mining_risk = self._calculate_data_mining_risk(impl_analysis, perf_analysis)
        overfitting_risk = self._calculate_overfitting_risk(perf_analysis)
        decay_risk = self._calculate_decay_risk(documentation, inefficiency_type)
        
        # Generate flags and recommendations
        red_flags, warnings, recommendations = self._generate_assessment_items(
            inefficiency_type, rationale_strength, data_mining_risk, overfitting_risk
        )
        
        return InefficiencyAnalysis(
            inefficiency_type=inefficiency_type,
            strategy_category=strategy_category,
            confidence_score=confidence_score,
            rationale_strength=rationale_strength,
            rationale_score=rationale_score,
            persistence_level=PersistenceLevel.MODERATE,  # Will be updated by persistence analyzer
            persistence_score=50.0,  # Will be updated by persistence analyzer
            cross_asset_persistence=50.0,  # Will be updated by cross-asset testing
            temporal_persistence=50.0,  # Will be updated by multi-period testing
            fundamental_indicators=doc_analysis.get('fundamental_score', {}),
            behavioral_indicators=doc_analysis.get('behavioral_score', {}),
            structural_indicators=doc_analysis.get('structural_score', {}),
            data_mining_risk=data_mining_risk,
            overfitting_risk=overfitting_risk,
            decay_risk=decay_risk,
            documentation_completeness=self._assess_documentation_completeness(documentation),
            empirical_support=self._assess_empirical_support(documentation),
            red_flags=red_flags,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _analyze_documentation(self, documentation: InefficiencyDocumentation) -> Dict[str, Any]:
        """Analyze strategy documentation for inefficiency indicators."""
        
        # Combine all text for analysis
        all_text = ' '.join([
            documentation.economic_rationale,
            documentation.market_mechanism,
            documentation.persistence_reasoning,
            ' '.join(documentation.academic_references),
            ' '.join(documentation.empirical_evidence),
            ' '.join(documentation.market_examples)
        ]).lower()
        
        # Count keyword occurrences
        fundamental_score = sum(1 for keyword in self._fundamental_keywords if keyword in all_text)
        behavioral_score = sum(1 for keyword in self._behavioral_keywords if keyword in all_text)
        structural_score = sum(1 for keyword in self._structural_keywords if keyword in all_text)
        technical_score = sum(1 for keyword in self._technical_keywords if keyword in all_text)
        
        # Normalize scores
        total_keywords = len(self._fundamental_keywords) + len(self._behavioral_keywords) + \
                        len(self._structural_keywords) + len(self._technical_keywords)
        
        return {
            'fundamental_score': {'raw': fundamental_score, 'normalized': fundamental_score / len(self._fundamental_keywords)},
            'behavioral_score': {'raw': behavioral_score, 'normalized': behavioral_score / len(self._behavioral_keywords)},
            'structural_score': {'raw': structural_score, 'normalized': structural_score / len(self._structural_keywords)},
            'technical_score': {'raw': technical_score, 'normalized': technical_score / len(self._technical_keywords)},
            'text_length': len(all_text),
            'reference_count': len(documentation.academic_references),
            'evidence_count': len(documentation.empirical_evidence)
        }
    
    def _analyze_implementation(self, strategy: BaseStrategy) -> Dict[str, Any]:
        """Analyze strategy implementation for inefficiency indicators."""
        
        # Analyze parameter names and types
        param_analysis = self._analyze_parameters(strategy.parameters)
        
        # Analyze strategy name and description
        name_analysis = self._analyze_strategy_name(strategy.name)
        
        return {
            'parameter_analysis': param_analysis,
            'name_analysis': name_analysis,
            'parameter_count': len(strategy.parameters),
            'complexity_score': self._calculate_implementation_complexity(strategy)
        }
    
    def _analyze_performance_characteristics(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze performance characteristics for inefficiency indicators."""
        
        return {
            'sharpe_ratio': results.sharpe_ratio,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor,
            'max_drawdown': results.max_drawdown,
            'trade_count': len(results.trades),
            'avg_trade_duration': np.mean([t.duration for t in results.trades]) if results.trades else 0,
            'return_skewness': self._calculate_return_skewness(results),
            'return_kurtosis': self._calculate_return_kurtosis(results),
            'consistency_score': self._calculate_performance_consistency(results)
        }
    
    def _determine_inefficiency_type(
        self,
        doc_analysis: Dict[str, Any],
        impl_analysis: Dict[str, Any]
    ) -> InefficiencyType:
        """Determine the primary inefficiency type."""
        
        scores = {
            'fundamental': doc_analysis['fundamental_score']['normalized'],
            'behavioral': doc_analysis['behavioral_score']['normalized'],
            'structural': doc_analysis['structural_score']['normalized'],
            'technical': doc_analysis['technical_score']['normalized']
        }
        
        # Find highest scoring category
        max_category = max(scores.items(), key=lambda x: x[1])
        
        # Require minimum threshold for classification
        if max_category[1] < 0.1:
            return InefficiencyType.UNKNOWN
        
        # Check for statistical artifact indicators
        if (impl_analysis['parameter_count'] > 10 or 
            impl_analysis['complexity_score'] > 0.8 or
            doc_analysis['reference_count'] == 0):
            return InefficiencyType.STATISTICAL_ARTIFACT
        
        # Map to inefficiency types
        mapping = {
            'fundamental': InefficiencyType.FUNDAMENTAL,
            'behavioral': InefficiencyType.BEHAVIORAL,
            'structural': InefficiencyType.STRUCTURAL,
            'technical': InefficiencyType.TECHNICAL
        }
        
        return mapping.get(max_category[0], InefficiencyType.UNKNOWN)
    
    def _determine_strategy_category(
        self,
        inefficiency_type: InefficiencyType,
        doc_analysis: Dict[str, Any]
    ) -> StrategyCategory:
        """Determine strategy category based on inefficiency type."""
        
        mapping = {
            InefficiencyType.FUNDAMENTAL: StrategyCategory.FUNDAMENTAL_BASED,
            InefficiencyType.BEHAVIORAL: StrategyCategory.BEHAVIORAL_BASED,
            InefficiencyType.STRUCTURAL: StrategyCategory.STRUCTURAL_BASED,
            InefficiencyType.TECHNICAL: StrategyCategory.TECHNICAL_BASED,
            InefficiencyType.STATISTICAL_ARTIFACT: StrategyCategory.DATA_MINING,
            InefficiencyType.UNKNOWN: StrategyCategory.UNCLASSIFIED
        }
        
        # Check for mixed approach
        scores = [
            doc_analysis['fundamental_score']['normalized'],
            doc_analysis['behavioral_score']['normalized'],
            doc_analysis['structural_score']['normalized']
        ]
        
        if sum(1 for score in scores if score > 0.05) >= 2:
            return StrategyCategory.MIXED_APPROACH
        
        return mapping.get(inefficiency_type, StrategyCategory.UNCLASSIFIED)
    
    def _calculate_confidence_score(
        self,
        doc_analysis: Dict[str, Any],
        impl_analysis: Dict[str, Any],
        perf_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence in classification."""
        
        # Documentation quality (40%)
        doc_score = min(1.0, (
            doc_analysis['text_length'] / 1000 * 0.3 +
            doc_analysis['reference_count'] / 5 * 0.4 +
            doc_analysis['evidence_count'] / 3 * 0.3
        ))
        
        # Implementation clarity (30%)
        impl_score = 1.0 - impl_analysis['complexity_score']
        
        # Performance consistency (30%)
        perf_score = perf_analysis['consistency_score']
        
        confidence = doc_score * 0.4 + impl_score * 0.3 + perf_score * 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def _assess_economic_rationale(
        self,
        documentation: InefficiencyDocumentation
    ) -> Tuple[EconomicRationaleStrength, float]:
        """Assess the strength of economic rationale."""
        
        rationale_text = documentation.economic_rationale.lower()
        mechanism_text = documentation.market_mechanism.lower()
        
        # Check for key economic concepts
        economic_concepts = {
            'supply', 'demand', 'equilibrium', 'arbitrage', 'efficiency',
            'information', 'asymmetry', 'friction', 'cost', 'barrier',
            'incentive', 'behavior', 'rational', 'irrational', 'bias'
        }
        
        concept_score = sum(1 for concept in economic_concepts 
                           if concept in rationale_text or concept in mechanism_text)
        
        # Assess explanation depth
        explanation_length = len(rationale_text) + len(mechanism_text)
        
        # Calculate score
        score = min(100, (concept_score / len(economic_concepts) * 60 + 
                         min(explanation_length / 500, 1) * 40))
        
        # Determine strength level
        if score >= 80:
            strength = EconomicRationaleStrength.STRONG
        elif score >= 60:
            strength = EconomicRationaleStrength.MODERATE
        elif score >= 30:
            strength = EconomicRationaleStrength.WEAK
        else:
            strength = EconomicRationaleStrength.ABSENT
        
        return strength, score
    
    def _calculate_data_mining_risk(
        self,
        impl_analysis: Dict[str, Any],
        perf_analysis: Dict[str, Any]
    ) -> float:
        """Calculate risk of strategy being a data mining artifact."""
        
        risk_factors = []
        
        # High parameter count
        if impl_analysis['parameter_count'] > 8:
            risk_factors.append(20)
        elif impl_analysis['parameter_count'] > 5:
            risk_factors.append(10)
        
        # High complexity
        if impl_analysis['complexity_score'] > 0.8:
            risk_factors.append(25)
        elif impl_analysis['complexity_score'] > 0.6:
            risk_factors.append(15)
        
        # Suspicious performance characteristics
        if perf_analysis['sharpe_ratio'] > 3.0:
            risk_factors.append(20)
        elif perf_analysis['sharpe_ratio'] > 2.0:
            risk_factors.append(10)
        
        if perf_analysis['win_rate'] > 0.8:
            risk_factors.append(15)
        
        # Low trade count
        if perf_analysis['trade_count'] < 50:
            risk_factors.append(15)
        elif perf_analysis['trade_count'] < 100:
            risk_factors.append(10)
        
        return min(100, sum(risk_factors))
    
    def _calculate_overfitting_risk(self, perf_analysis: Dict[str, Any]) -> float:
        """Calculate overfitting risk based on performance characteristics."""
        
        risk_factors = []
        
        # Extreme performance metrics
        if perf_analysis['sharpe_ratio'] > 2.5:
            risk_factors.append(20)
        
        if perf_analysis['win_rate'] > 0.75:
            risk_factors.append(15)
        
        if perf_analysis['max_drawdown'] < 0.05:
            risk_factors.append(15)
        
        # Low consistency
        if perf_analysis['consistency_score'] < 0.5:
            risk_factors.append(25)
        
        # Extreme return distribution
        if abs(perf_analysis['return_skewness']) > 2:
            risk_factors.append(10)
        
        if perf_analysis['return_kurtosis'] > 5:
            risk_factors.append(10)
        
        return min(100, sum(risk_factors))
    
    def _calculate_decay_risk(
        self,
        documentation: InefficiencyDocumentation,
        inefficiency_type: InefficiencyType
    ) -> float:
        """Calculate risk of inefficiency decay."""
        
        base_risk = {
            InefficiencyType.FUNDAMENTAL: 20,
            InefficiencyType.BEHAVIORAL: 40,
            InefficiencyType.STRUCTURAL: 30,
            InefficiencyType.TECHNICAL: 60,
            InefficiencyType.STATISTICAL_ARTIFACT: 90,
            InefficiencyType.UNKNOWN: 70
        }
        
        risk = base_risk.get(inefficiency_type, 50)
        
        # Adjust based on documentation
        if len(documentation.decay_factors) > 3:
            risk += 15
        elif len(documentation.decay_factors) > 1:
            risk += 10
        
        if len(documentation.competition_risks) > 2:
            risk += 10
        
        return min(100, risk)
    
    def _generate_assessment_items(
        self,
        inefficiency_type: InefficiencyType,
        rationale_strength: EconomicRationaleStrength,
        data_mining_risk: float,
        overfitting_risk: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate red flags, warnings, and recommendations."""
        
        red_flags = []
        warnings = []
        recommendations = []
        
        # Red flags
        if inefficiency_type == InefficiencyType.STATISTICAL_ARTIFACT:
            red_flags.append("Strategy classified as statistical artifact")
        
        if data_mining_risk > 70:
            red_flags.append("High data mining risk detected")
        
        if overfitting_risk > 70:
            red_flags.append("High overfitting risk detected")
        
        if rationale_strength == EconomicRationaleStrength.ABSENT:
            red_flags.append("No economic rationale provided")
        
        # Warnings
        if data_mining_risk > 50:
            warnings.append("Moderate data mining risk")
        
        if overfitting_risk > 50:
            warnings.append("Moderate overfitting risk")
        
        if rationale_strength == EconomicRationaleStrength.WEAK:
            warnings.append("Weak economic rationale")
        
        # Recommendations
        if rationale_strength in [EconomicRationaleStrength.WEAK, EconomicRationaleStrength.ABSENT]:
            recommendations.append("Strengthen economic rationale documentation")
        
        if data_mining_risk > 40:
            recommendations.append("Reduce strategy complexity and parameter count")
        
        if overfitting_risk > 40:
            recommendations.append("Conduct more extensive out-of-sample testing")
        
        return red_flags, warnings, recommendations
    
    # Helper methods for implementation analysis
    def _analyze_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategy parameters for complexity indicators."""
        return {
            'count': len(parameters),
            'types': [type(v).__name__ for v in parameters.values()],
            'complexity': len(parameters) / 10  # Normalized complexity score
        }
    
    def _analyze_strategy_name(self, name: str) -> Dict[str, Any]:
        """Analyze strategy name for classification hints."""
        name_lower = name.lower()
        
        return {
            'has_fundamental_terms': any(term in name_lower for term in self._fundamental_keywords),
            'has_behavioral_terms': any(term in name_lower for term in self._behavioral_keywords),
            'has_structural_terms': any(term in name_lower for term in self._structural_keywords),
            'has_technical_terms': any(term in name_lower for term in self._technical_keywords)
        }
    
    def _calculate_implementation_complexity(self, strategy: BaseStrategy) -> float:
        """Calculate implementation complexity score."""
        # Simplified complexity based on parameter count
        return min(1.0, len(strategy.parameters) / 15)
    
    def _calculate_return_skewness(self, results: BacktestResults) -> float:
        """Calculate return skewness."""
        if len(results.equity_curve) < 10:
            return 0.0
        
        returns = results.equity_curve.pct_change().dropna()
        if len(returns) < 3:
            return 0.0
        
        return float(stats.skew(returns))
    
    def _calculate_return_kurtosis(self, results: BacktestResults) -> float:
        """Calculate return kurtosis."""
        if len(results.equity_curve) < 10:
            return 0.0
        
        returns = results.equity_curve.pct_change().dropna()
        if len(returns) < 4:
            return 0.0
        
        return float(stats.kurtosis(returns))
    
    def _calculate_performance_consistency(self, results: BacktestResults) -> float:
        """Calculate performance consistency score."""
        if len(results.monthly_returns) < 6:
            return 0.5
        
        # Use coefficient of variation as consistency measure
        monthly_returns = np.array(results.monthly_returns)
        mean_return = np.mean(monthly_returns)
        std_return = np.std(monthly_returns)
        
        if abs(mean_return) < 1e-8:
            return 0.5
        
        cv = std_return / abs(mean_return)
        consistency = max(0, min(1, 1 - cv))
        
        return consistency
    
    def _assess_documentation_completeness(self, documentation: InefficiencyDocumentation) -> float:
        """Assess completeness of strategy documentation."""
        
        required_fields = [
            documentation.economic_rationale,
            documentation.market_mechanism,
            documentation.persistence_reasoning
        ]
        
        completeness_score = 0
        
        # Check required fields
        for field in required_fields:
            if field and len(field.strip()) > 50:
                completeness_score += 25
            elif field and len(field.strip()) > 10:
                completeness_score += 15
        
        # Bonus for supporting evidence
        if documentation.academic_references:
            completeness_score += 10
        
        if documentation.empirical_evidence:
            completeness_score += 10
        
        if documentation.market_examples:
            completeness_score += 5
        
        return min(100, completeness_score)
    
    def _assess_empirical_support(self, documentation: InefficiencyDocumentation) -> float:
        """Assess empirical support strength."""
        
        support_score = 0
        
        # Academic references
        support_score += min(40, len(documentation.academic_references) * 10)
        
        # Empirical evidence
        support_score += min(30, len(documentation.empirical_evidence) * 10)
        
        # Market examples
        support_score += min(20, len(documentation.market_examples) * 5)
        
        # Risk factor analysis (shows understanding)
        if documentation.decay_factors:
            support_score += 5
        
        if documentation.competition_risks:
            support_score += 5
        
        return min(100, support_score)


class FundamentalInefficiencyAnalyzer:
    """
    Main analyzer for evaluating fundamental inefficiency exploitation in strategies.
    """
    
    def __init__(
        self,
        backtesting_engine: BacktestingEngine = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize the FundamentalInefficiencyAnalyzer.
        
        Args:
            backtesting_engine: Backtesting engine for validation
            initial_capital: Initial capital for backtests
        """
        self.backtesting_engine = backtesting_engine or BacktestingEngine(initial_capital)
        self.initial_capital = initial_capital
        
        self.classifier = InefficiencyClassifier()
        self.multi_period_validator = MultiPeriodValidator(backtesting_engine, initial_capital)
        self.cross_asset_tester = CrossAssetTester(backtesting_engine, initial_capital)
        
        self.logger = get_logger("fundamental_inefficiency_analyzer")
    
    def analyze_strategy(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        documentation: InefficiencyDocumentation,
        run_validations: bool = True
    ) -> FundamentalInefficiencyResult:
        """
        Perform comprehensive fundamental inefficiency analysis.
        
        Args:
            strategy: Strategy to analyze
            data: Historical market data
            documentation: Strategy documentation
            run_validations: Whether to run multi-period and cross-asset validations
            
        Returns:
            FundamentalInefficiencyResult with complete analysis
        """
        self.logger.info(f"Starting fundamental inefficiency analysis for {strategy.name}")
        
        # Run initial backtest
        backtest_results = self.backtesting_engine.run_backtest(
            strategy, data, self.initial_capital
        )
        
        # Perform classification analysis
        analysis = self.classifier.classify_strategy(strategy, documentation, backtest_results)
        
        # Run validations if requested
        multi_period_validation = None
        cross_asset_validation = None
        
        if run_validations:
            try:
                multi_period_validation = self.multi_period_validator.validate_strategy(
                    strategy, data
                )
                analysis.temporal_persistence = multi_period_validation.consistency_metrics.overall_consistency_score * 100
                
                # Update persistence level based on validation
                if analysis.temporal_persistence >= 80:
                    analysis.persistence_level = PersistenceLevel.HIGH
                elif analysis.temporal_persistence >= 60:
                    analysis.persistence_level = PersistenceLevel.MODERATE
                elif analysis.temporal_persistence >= 40:
                    analysis.persistence_level = PersistenceLevel.LOW
                else:
                    analysis.persistence_level = PersistenceLevel.VERY_LOW
                
            except Exception as e:
                self.logger.warning(f"Multi-period validation failed: {e}")
            
            try:
                cross_asset_validation = self.cross_asset_tester.test_strategy_cross_asset(
                    strategy, ['BTC-USD', 'ETH-USD', 'ADA-USD']  # Example assets
                )
                analysis.cross_asset_persistence = cross_asset_validation.consistency_score
                
            except Exception as e:
                self.logger.warning(f"Cross-asset validation failed: {e}")
        
        # Calculate overall persistence score
        analysis.persistence_score = (
            analysis.temporal_persistence * 0.6 +
            analysis.cross_asset_persistence * 0.4
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(analysis, multi_period_validation, cross_asset_validation)
        
        # Determine deployment recommendation
        deployment_recommendation = self._determine_deployment_recommendation(analysis, overall_score)
        
        # Generate detailed findings
        strengths, weaknesses, improvements = self._generate_detailed_findings(
            analysis, multi_period_validation, cross_asset_validation
        )
        
        result = FundamentalInefficiencyResult(
            strategy_name=strategy.name,
            analysis=analysis,
            documentation=documentation,
            overall_score=overall_score,
            deployment_recommendation=deployment_recommendation,
            multi_period_validation=multi_period_validation,
            cross_asset_validation=cross_asset_validation,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=improvements
        )
        
        self.logger.info(f"Fundamental inefficiency analysis completed for {strategy.name}")
        return result
    
    def _calculate_overall_score(
        self,
        analysis: InefficiencyAnalysis,
        multi_period_validation: Optional[MultiPeriodValidationResult],
        cross_asset_validation: Optional[CrossAssetTestResult]
    ) -> float:
        """Calculate overall inefficiency exploitation score."""
        
        # Base score from classification (40%)
        base_score = analysis.confidence_score * 40
        
        # Economic rationale score (25%)
        rationale_score = analysis.rationale_score * 0.25
        
        # Persistence score (20%)
        persistence_score = analysis.persistence_score * 0.2
        
        # Risk penalty (15%)
        risk_penalty = (analysis.data_mining_risk + analysis.overfitting_risk) / 2 * 0.15
        
        # Validation bonus
        validation_bonus = 0
        if multi_period_validation and multi_period_validation.validation_score > 70:
            validation_bonus += 5
        if cross_asset_validation and cross_asset_validation.consistency_score > 70:
            validation_bonus += 5
        
        total_score = base_score + rationale_score + persistence_score - risk_penalty + validation_bonus
        
        return min(100, max(0, total_score))
    
    def _determine_deployment_recommendation(self, analysis: InefficiencyAnalysis, overall_score: float) -> str:
        """Determine deployment recommendation."""
        
        # Automatic rejection criteria
        if analysis.red_flags:
            return "Rejected"
        
        if analysis.inefficiency_type == InefficiencyType.STATISTICAL_ARTIFACT:
            return "Rejected"
        
        if analysis.data_mining_risk > 70 or analysis.overfitting_risk > 70:
            return "Rejected"
        
        # Approval criteria
        if (overall_score >= 75 and 
            analysis.rationale_strength in [EconomicRationaleStrength.STRONG, EconomicRationaleStrength.MODERATE] and
            analysis.persistence_level in [PersistenceLevel.HIGH, PersistenceLevel.MODERATE]):
            return "Approved"
        
        # Conditional approval
        if overall_score >= 60:
            return "Conditional"
        
        return "Rejected"
    
    def _generate_detailed_findings(
        self,
        analysis: InefficiencyAnalysis,
        multi_period_validation: Optional[MultiPeriodValidationResult],
        cross_asset_validation: Optional[CrossAssetTestResult]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate detailed strengths, weaknesses, and improvement suggestions."""
        
        strengths = []
        weaknesses = []
        improvements = []
        
        # Analyze classification results
        if analysis.confidence_score > 0.8:
            strengths.append("High confidence in inefficiency classification")
        elif analysis.confidence_score < 0.5:
            weaknesses.append("Low confidence in inefficiency classification")
        
        # Analyze economic rationale
        if analysis.rationale_strength == EconomicRationaleStrength.STRONG:
            strengths.append("Strong economic rationale provided")
        elif analysis.rationale_strength == EconomicRationaleStrength.WEAK:
            weaknesses.append("Weak economic rationale")
            improvements.append("Strengthen economic explanation with more detailed market mechanism analysis")
        elif analysis.rationale_strength == EconomicRationaleStrength.ABSENT:
            weaknesses.append("No economic rationale provided")
            improvements.append("Provide comprehensive economic rationale for the strategy")
        
        # Analyze persistence
        if analysis.persistence_level == PersistenceLevel.HIGH:
            strengths.append("High persistence across different conditions")
        elif analysis.persistence_level == PersistenceLevel.LOW:
            weaknesses.append("Low persistence across different conditions")
            improvements.append("Investigate reasons for performance inconsistency")
        
        # Analyze risks
        if analysis.data_mining_risk < 30:
            strengths.append("Low data mining risk")
        elif analysis.data_mining_risk > 60:
            weaknesses.append("High data mining risk")
            improvements.append("Reduce strategy complexity and parameter count")
        
        if analysis.overfitting_risk < 30:
            strengths.append("Low overfitting risk")
        elif analysis.overfitting_risk > 60:
            weaknesses.append("High overfitting risk")
            improvements.append("Conduct more extensive out-of-sample testing")
        
        # Validation results
        if multi_period_validation:
            if multi_period_validation.validation_score > 80:
                strengths.append("Excellent multi-period validation results")
            elif multi_period_validation.validation_score < 60:
                weaknesses.append("Poor multi-period validation results")
                improvements.append("Improve strategy robustness across different market conditions")
        
        if cross_asset_validation:
            if cross_asset_validation.consistency_score > 80:
                strengths.append("Excellent cross-asset consistency")
            elif cross_asset_validation.consistency_score < 60:
                weaknesses.append("Poor cross-asset consistency")
                improvements.append("Test strategy on more diverse asset classes")
        
        return strengths, weaknesses, improvements 