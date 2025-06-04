"""
Multi-Period Validation System

This module provides comprehensive validation of trading strategies across different
market conditions and time periods to detect overfitting and ensure robustness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ks_2samp
import logging

try:
    from ..validation.regime_analyzer import RegimeAnalyzer, MarketRegime, RegimeAnalysisResults, RegimePerformance
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from ..utils.logger import get_logger
    from .data_splitting_framework import AntiOverfittingDataSplitter
except ImportError:
    from src.validation.regime_analyzer import RegimeAnalyzer, MarketRegime, RegimeAnalysisResults, RegimePerformance
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.utils.logger import get_logger
    from src.anti_overfitting.data_splitting_framework import AntiOverfittingDataSplitter


class ValidationPeriodType(Enum):
    """Types of validation periods."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS_MARKET = "sideways_market"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS_PERIOD = "crisis_period"
    RECOVERY_PERIOD = "recovery_period"
    SEASONAL_Q1 = "seasonal_q1"
    SEASONAL_Q2 = "seasonal_q2"
    SEASONAL_Q3 = "seasonal_q3"
    SEASONAL_Q4 = "seasonal_q4"


class ConsistencyLevel(Enum):
    """Consistency assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class PeriodPerformance:
    """Performance metrics for a specific validation period."""
    period_type: ValidationPeriodType
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Core performance metrics
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Trade metrics
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    
    # Market context
    market_return: float
    market_volatility: float
    beta: float
    alpha: float
    
    # Consistency indicators
    monthly_returns_std: float
    rolling_sharpe_std: float
    drawdown_frequency: int


@dataclass
class ConsistencyMetrics:
    """Metrics measuring performance consistency across periods."""
    
    # Return consistency
    return_consistency_score: float  # 0-1, higher = more consistent
    return_coefficient_variation: float
    return_correlation_across_periods: float
    
    # Risk consistency
    volatility_consistency_score: float
    drawdown_consistency_score: float
    risk_adjusted_consistency: float
    
    # Trade consistency
    win_rate_consistency_score: float
    trade_frequency_consistency: float
    
    # Overall consistency
    overall_consistency_score: float  # Weighted average of all consistency metrics
    consistency_level: ConsistencyLevel
    
    # Statistical tests
    returns_normality_p_value: float
    periods_similarity_p_value: float  # KS test across periods
    
    # Flags and warnings
    inconsistency_flags: List[str] = field(default_factory=list)
    warning_periods: List[ValidationPeriodType] = field(default_factory=list)


@dataclass
class MultiPeriodValidationResult:
    """Complete multi-period validation results."""
    strategy_name: str
    validation_periods: List[PeriodPerformance]
    consistency_metrics: ConsistencyMetrics
    
    # Regime-specific analysis
    regime_analysis: RegimeAnalysisResults
    regime_consistency: float
    
    # Period comparisons
    best_period: ValidationPeriodType
    worst_period: ValidationPeriodType
    most_consistent_metric: str
    least_consistent_metric: str
    
    # Overall assessment
    validation_score: float  # 0-100, higher = better validation
    deployment_readiness: str  # "Ready", "Caution", "Not Ready"
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)


class PerformanceConsistencyAnalyzer:
    """
    Analyzes performance consistency across different time periods and market conditions.
    """
    
    def __init__(
        self,
        consistency_threshold: float = 0.7,
        significance_level: float = 0.05,
        min_trades_per_period: int = 10
    ):
        """
        Initialize the PerformanceConsistencyAnalyzer.
        
        Args:
            consistency_threshold: Minimum consistency score for acceptance
            significance_level: Statistical significance level
            min_trades_per_period: Minimum trades required per period
        """
        self.consistency_threshold = consistency_threshold
        self.significance_level = significance_level
        self.min_trades_per_period = min_trades_per_period
        self.logger = get_logger("performance_consistency_analyzer")
    
    def analyze_consistency(
        self,
        period_performances: List[PeriodPerformance]
    ) -> ConsistencyMetrics:
        """
        Analyze performance consistency across periods.
        
        Args:
            period_performances: List of period performance results
            
        Returns:
            ConsistencyMetrics with detailed analysis
        """
        self.logger.info(f"Analyzing consistency across {len(period_performances)} periods")
        
        if len(period_performances) < 2:
            return self._create_insufficient_data_metrics()
        
        # Extract metrics for analysis
        returns = [p.total_return for p in period_performances]
        volatilities = [p.volatility for p in period_performances]
        sharpe_ratios = [p.sharpe_ratio for p in period_performances]
        drawdowns = [abs(p.max_drawdown) for p in period_performances]
        win_rates = [p.win_rate for p in period_performances]
        trade_counts = [p.total_trades for p in period_performances]
        
        # Calculate consistency scores
        return_consistency = self._calculate_consistency_score(returns)
        volatility_consistency = self._calculate_consistency_score(volatilities)
        drawdown_consistency = self._calculate_consistency_score(drawdowns, lower_is_better=True)
        win_rate_consistency = self._calculate_consistency_score(win_rates)
        
        # Risk-adjusted consistency
        risk_adjusted_consistency = self._calculate_consistency_score(sharpe_ratios)
        
        # Trade frequency consistency
        trade_frequency_consistency = self._calculate_consistency_score(trade_counts)
        
        # Return coefficient of variation
        return_cv = np.std(returns) / (abs(np.mean(returns)) + 1e-8)
        
        # Cross-period correlation
        if len(returns) >= 3:
            # Calculate pairwise correlations (simplified)
            correlations = []
            for i in range(len(returns)):
                for j in range(i + 1, len(returns)):
                    # Use period index as proxy for correlation analysis
                    correlations.append(0.5)  # Placeholder - would need actual time series
            return_correlation = np.mean(correlations) if correlations else 0.5
        else:
            return_correlation = 0.5
        
        # Overall consistency (weighted average)
        overall_consistency = (
            return_consistency * 0.3 +
            risk_adjusted_consistency * 0.25 +
            volatility_consistency * 0.2 +
            drawdown_consistency * 0.15 +
            win_rate_consistency * 0.1
        )
        
        # Determine consistency level
        if overall_consistency >= 0.9:
            consistency_level = ConsistencyLevel.EXCELLENT
        elif overall_consistency >= 0.8:
            consistency_level = ConsistencyLevel.GOOD
        elif overall_consistency >= 0.6:
            consistency_level = ConsistencyLevel.MODERATE
        elif overall_consistency >= 0.4:
            consistency_level = ConsistencyLevel.POOR
        else:
            consistency_level = ConsistencyLevel.VERY_POOR
        
        # Statistical tests
        returns_normality_p = self._test_normality(returns)
        periods_similarity_p = self._test_period_similarity(period_performances)
        
        # Generate flags and warnings
        flags, warning_periods = self._generate_consistency_flags(
            period_performances, overall_consistency
        )
        
        return ConsistencyMetrics(
            return_consistency_score=return_consistency,
            return_coefficient_variation=return_cv,
            return_correlation_across_periods=return_correlation,
            volatility_consistency_score=volatility_consistency,
            drawdown_consistency_score=drawdown_consistency,
            risk_adjusted_consistency=risk_adjusted_consistency,
            win_rate_consistency_score=win_rate_consistency,
            trade_frequency_consistency=trade_frequency_consistency,
            overall_consistency_score=overall_consistency,
            consistency_level=consistency_level,
            returns_normality_p_value=returns_normality_p,
            periods_similarity_p_value=periods_similarity_p,
            inconsistency_flags=flags,
            warning_periods=warning_periods
        )
    
    def _calculate_consistency_score(
        self,
        values: List[float],
        lower_is_better: bool = False
    ) -> float:
        """Calculate consistency score for a metric."""
        if len(values) < 2:
            return 0.5
        
        # Use coefficient of variation as consistency measure
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if abs(mean_val) < 1e-8:
            return 0.5  # Neutral consistency for near-zero values
        
        cv = std_val / abs(mean_val)
        
        # Convert CV to consistency score (0-1, higher = more consistent)
        # Lower CV = higher consistency
        consistency = max(0, min(1, 1 - cv))
        
        return consistency
    
    def _test_normality(self, values: List[float]) -> float:
        """Test normality of returns using Shapiro-Wilk test."""
        if len(values) < 3:
            return 1.0
        
        try:
            _, p_value = stats.shapiro(values)
            return p_value
        except:
            return 1.0
    
    def _test_period_similarity(self, period_performances: List[PeriodPerformance]) -> float:
        """Test similarity between periods using KS test."""
        if len(period_performances) < 2:
            return 1.0
        
        # Compare returns between first and last period (simplified)
        try:
            returns1 = [period_performances[0].total_return]
            returns2 = [period_performances[-1].total_return]
            
            # Need more data points for KS test, so return placeholder
            return 0.5
        except:
            return 1.0
    
    def _generate_consistency_flags(
        self,
        period_performances: List[PeriodPerformance],
        overall_consistency: float
    ) -> Tuple[List[str], List[ValidationPeriodType]]:
        """Generate consistency flags and warning periods."""
        flags = []
        warning_periods = []
        
        if overall_consistency < 0.4:
            flags.append("Very poor consistency across periods")
        elif overall_consistency < 0.6:
            flags.append("Moderate consistency issues detected")
        
        # Check for outlier periods
        returns = [p.total_return for p in period_performances]
        if len(returns) >= 3:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            for i, perf in enumerate(period_performances):
                if abs(perf.total_return - mean_return) > 2 * std_return:
                    flags.append(f"Outlier performance in {perf.period_type.value}")
                    warning_periods.append(perf.period_type)
        
        # Check for insufficient trades
        for perf in period_performances:
            if perf.total_trades < self.min_trades_per_period:
                flags.append(f"Insufficient trades in {perf.period_type.value}: {perf.total_trades}")
                warning_periods.append(perf.period_type)
        
        return flags, warning_periods
    
    def _create_insufficient_data_metrics(self) -> ConsistencyMetrics:
        """Create metrics for insufficient data case."""
        return ConsistencyMetrics(
            return_consistency_score=0.0,
            return_coefficient_variation=float('inf'),
            return_correlation_across_periods=0.0,
            volatility_consistency_score=0.0,
            drawdown_consistency_score=0.0,
            risk_adjusted_consistency=0.0,
            win_rate_consistency_score=0.0,
            trade_frequency_consistency=0.0,
            overall_consistency_score=0.0,
            consistency_level=ConsistencyLevel.VERY_POOR,
            returns_normality_p_value=1.0,
            periods_similarity_p_value=1.0,
            inconsistency_flags=["Insufficient data for consistency analysis"],
            warning_periods=[]
        )


class MultiPeriodValidator:
    """
    Main validator for testing strategies across multiple time periods and market conditions.
    """
    
    def __init__(
        self,
        backtesting_engine: BacktestingEngine = None,
        initial_capital: float = 100000.0,
        min_period_days: int = 90,
        max_periods: int = 20
    ):
        """
        Initialize the MultiPeriodValidator.
        
        Args:
            backtesting_engine: Backtesting engine for running tests
            initial_capital: Initial capital for backtests
            min_period_days: Minimum days per validation period
            max_periods: Maximum number of periods to analyze
        """
        self.backtesting_engine = backtesting_engine or BacktestingEngine(initial_capital)
        self.initial_capital = initial_capital
        self.min_period_days = min_period_days
        self.max_periods = max_periods
        
        self.regime_analyzer = RegimeAnalyzer(initial_capital=initial_capital)
        self.consistency_analyzer = PerformanceConsistencyAnalyzer()
        self.data_splitter = AntiOverfittingDataSplitter()
        
        self.logger = get_logger("multi_period_validator")
    
    def validate_strategy(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        validation_methods: List[str] = None
    ) -> MultiPeriodValidationResult:
        """
        Validate strategy across multiple periods and market conditions.
        
        Args:
            strategy: Strategy to validate
            data: Historical market data
            validation_methods: List of validation methods to apply
            
        Returns:
            MultiPeriodValidationResult with comprehensive analysis
        """
        self.logger.info(f"Starting multi-period validation for {strategy.name}")
        
        if validation_methods is None:
            validation_methods = [
                "regime_based",
                "time_based",
                "volatility_based",
                "seasonal"
            ]
        
        # Perform regime analysis
        regime_analysis = self.regime_analyzer.analyze_strategy_performance(strategy, data)
        
        # Create validation periods
        validation_periods = self._create_validation_periods(data, validation_methods)
        
        # Test strategy on each period
        period_performances = []
        for period_info in validation_periods:
            try:
                performance = self._test_strategy_on_period(
                    strategy, data, period_info
                )
                period_performances.append(performance)
            except Exception as e:
                self.logger.warning(f"Failed to test period {period_info['type']}: {e}")
                continue
        
        if not period_performances:
            raise ValueError("No successful period validations completed")
        
        # Analyze consistency
        consistency_metrics = self.consistency_analyzer.analyze_consistency(period_performances)
        
        # Calculate regime consistency
        regime_consistency = self._calculate_regime_consistency(regime_analysis)
        
        # Identify best/worst periods
        best_period, worst_period = self._identify_best_worst_periods(period_performances)
        
        # Identify most/least consistent metrics
        most_consistent, least_consistent = self._identify_consistency_extremes(consistency_metrics)
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            consistency_metrics, regime_consistency, period_performances
        )
        
        # Determine deployment readiness
        deployment_readiness = self._assess_deployment_readiness(
            validation_score, consistency_metrics
        )
        
        # Generate recommendations
        recommendations, risk_warnings = self._generate_recommendations(
            consistency_metrics, regime_analysis, period_performances
        )
        
        result = MultiPeriodValidationResult(
            strategy_name=strategy.name,
            validation_periods=period_performances,
            consistency_metrics=consistency_metrics,
            regime_analysis=regime_analysis,
            regime_consistency=regime_consistency,
            best_period=best_period,
            worst_period=worst_period,
            most_consistent_metric=most_consistent,
            least_consistent_metric=least_consistent,
            validation_score=validation_score,
            deployment_readiness=deployment_readiness,
            recommendations=recommendations,
            risk_warnings=risk_warnings
        )
        
        self.logger.info(f"Multi-period validation completed for {strategy.name}")
        return result
    
    def _create_validation_periods(
        self,
        data: pd.DataFrame,
        validation_methods: List[str]
    ) -> List[Dict[str, Any]]:
        """Create validation periods based on specified methods."""
        periods = []
        
        if "regime_based" in validation_methods:
            periods.extend(self._create_regime_based_periods(data))
        
        if "time_based" in validation_methods:
            periods.extend(self._create_time_based_periods(data))
        
        if "volatility_based" in validation_methods:
            periods.extend(self._create_volatility_based_periods(data))
        
        if "seasonal" in validation_methods:
            periods.extend(self._create_seasonal_periods(data))
        
        # Limit number of periods
        if len(periods) > self.max_periods:
            periods = periods[:self.max_periods]
        
        return periods
    
    def _create_regime_based_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create periods based on market regimes."""
        regimes = self.regime_analyzer.identify_regimes(data)
        periods = []
        
        # Group consecutive regime periods
        current_regime = None
        start_idx = 0
        
        for i, regime in enumerate(regimes):
            if regime != current_regime:
                if current_regime is not None and i - start_idx >= self.min_period_days:
                    periods.append({
                        'type': self._regime_to_validation_type(current_regime),
                        'start_idx': start_idx,
                        'end_idx': i,
                        'data': data.iloc[start_idx:i]
                    })
                current_regime = regime
                start_idx = i
        
        # Handle last period
        if current_regime is not None and len(regimes) - start_idx >= self.min_period_days:
            periods.append({
                'type': self._regime_to_validation_type(current_regime),
                'start_idx': start_idx,
                'end_idx': len(regimes),
                'data': data.iloc[start_idx:]
            })
        
        return periods
    
    def _create_time_based_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create periods based on time splits."""
        periods = []
        
        # Create yearly periods if enough data
        if len(data) >= 365 * 2:
            years = data.index.year.unique()
            for year in years:
                year_data = data[data.index.year == year]
                if len(year_data) >= self.min_period_days:
                    periods.append({
                        'type': ValidationPeriodType.BULL_MARKET,  # Placeholder
                        'start_idx': 0,
                        'end_idx': len(year_data),
                        'data': year_data
                    })
        
        return periods
    
    def _create_volatility_based_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create periods based on volatility levels."""
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(30).std()
        
        # Define high/low volatility thresholds
        vol_median = volatility.median()
        high_vol_threshold = vol_median * 1.5
        low_vol_threshold = vol_median * 0.5
        
        periods = []
        
        # Find high volatility periods
        high_vol_mask = volatility > high_vol_threshold
        high_vol_periods = self._find_consecutive_periods(high_vol_mask, self.min_period_days)
        
        for start, end in high_vol_periods:
            periods.append({
                'type': ValidationPeriodType.HIGH_VOLATILITY,
                'start_idx': start,
                'end_idx': end,
                'data': data.iloc[start:end]
            })
        
        # Find low volatility periods
        low_vol_mask = volatility < low_vol_threshold
        low_vol_periods = self._find_consecutive_periods(low_vol_mask, self.min_period_days)
        
        for start, end in low_vol_periods:
            periods.append({
                'type': ValidationPeriodType.LOW_VOLATILITY,
                'start_idx': start,
                'end_idx': end,
                'data': data.iloc[start:end]
            })
        
        return periods
    
    def _create_seasonal_periods(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create periods based on seasonal patterns."""
        periods = []
        
        # Group by quarters
        quarters = {
            1: ValidationPeriodType.SEASONAL_Q1,
            2: ValidationPeriodType.SEASONAL_Q2,
            3: ValidationPeriodType.SEASONAL_Q3,
            4: ValidationPeriodType.SEASONAL_Q4
        }
        
        for quarter, period_type in quarters.items():
            quarter_data = data[data.index.quarter == quarter]
            if len(quarter_data) >= self.min_period_days:
                periods.append({
                    'type': period_type,
                    'start_idx': 0,
                    'end_idx': len(quarter_data),
                    'data': quarter_data
                })
        
        return periods
    
    def _test_strategy_on_period(
        self,
        strategy: BaseStrategy,
        full_data: pd.DataFrame,
        period_info: Dict[str, Any]
    ) -> PeriodPerformance:
        """Test strategy on a specific validation period."""
        period_data = period_info['data']
        period_type = period_info['type']
        
        # Create strategy copy
        strategy_copy = self._create_strategy_copy(strategy)
        
        # Run backtest
        results = self.backtesting_engine.run_backtest(
            strategy_copy, period_data, self.initial_capital
        )
        
        # Calculate market metrics
        market_return = (period_data['close'].iloc[-1] / period_data['close'].iloc[0]) - 1
        market_volatility = period_data['close'].pct_change().std() * np.sqrt(252)
        
        # Calculate beta and alpha (simplified)
        strategy_returns = results.equity_curve.pct_change().dropna()
        market_returns = period_data['close'].pct_change().dropna()
        
        # Align returns
        common_index = strategy_returns.index.intersection(market_returns.index)
        if len(common_index) > 10:
            aligned_strategy = strategy_returns[common_index]
            aligned_market = market_returns[common_index]
            
            try:
                beta, alpha, _, _, _ = stats.linregress(aligned_market, aligned_strategy)
            except:
                beta, alpha = 1.0, 0.0
        else:
            beta, alpha = 1.0, 0.0
        
        # Calculate additional metrics
        monthly_returns = self._calculate_monthly_returns(results.equity_curve)
        rolling_sharpe = self._calculate_rolling_sharpe(strategy_returns)
        drawdown_frequency = self._calculate_drawdown_frequency(results.equity_curve)
        
        return PeriodPerformance(
            period_type=period_type,
            start_date=period_data.index[0],
            end_date=period_data.index[-1],
            duration_days=len(period_data),
            total_return=results.total_return,
            annual_return=results.annual_return,
            volatility=results.volatility,
            sharpe_ratio=results.sharpe_ratio,
            sortino_ratio=results.sortino_ratio,
            calmar_ratio=results.calmar_ratio,
            max_drawdown=results.max_drawdown,
            win_rate=results.win_rate,
            profit_factor=results.profit_factor,
            total_trades=len(results.trades),
            avg_trade_duration=np.mean([t.duration for t in results.trades]) if results.trades else 0,
            var_95=np.percentile(strategy_returns, 5) if len(strategy_returns) > 0 else 0,
            cvar_95=strategy_returns[strategy_returns <= np.percentile(strategy_returns, 5)].mean() if len(strategy_returns) > 0 else 0,
            market_return=market_return,
            market_volatility=market_volatility,
            beta=beta,
            alpha=alpha,
            monthly_returns_std=np.std(monthly_returns) if len(monthly_returns) > 1 else 0,
            rolling_sharpe_std=np.std(rolling_sharpe) if len(rolling_sharpe) > 1 else 0,
            drawdown_frequency=drawdown_frequency
        )
    
    def _regime_to_validation_type(self, regime: MarketRegime) -> ValidationPeriodType:
        """Convert market regime to validation period type."""
        mapping = {
            MarketRegime.BULL: ValidationPeriodType.BULL_MARKET,
            MarketRegime.BEAR: ValidationPeriodType.BEAR_MARKET,
            MarketRegime.SIDEWAYS: ValidationPeriodType.SIDEWAYS_MARKET,
            MarketRegime.VOLATILE: ValidationPeriodType.HIGH_VOLATILITY,
            MarketRegime.CRASH: ValidationPeriodType.CRISIS_PERIOD,
            MarketRegime.RALLY: ValidationPeriodType.RECOVERY_PERIOD
        }
        return mapping.get(regime, ValidationPeriodType.SIDEWAYS_MARKET)
    
    def _find_consecutive_periods(
        self,
        mask: pd.Series,
        min_length: int
    ) -> List[Tuple[int, int]]:
        """Find consecutive periods where mask is True."""
        periods = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                if i - start >= min_length:
                    periods.append((start, i))
                start = None
        
        # Handle case where period extends to end
        if start is not None and len(mask) - start >= min_length:
            periods.append((start, len(mask)))
        
        return periods
    
    def _create_strategy_copy(self, strategy: BaseStrategy) -> BaseStrategy:
        """Create a copy of the strategy for testing."""
        strategy_class = type(strategy)
        return strategy_class(
            name=strategy.name,
            parameters=strategy.parameters.copy(),
            risk_params=strategy.risk_params.copy()
        )
    
    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> List[float]:
        """Calculate monthly returns from equity curve."""
        monthly_equity = equity_curve.resample('M').last()
        return monthly_equity.pct_change().dropna().tolist()
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, window: int = 30) -> List[float]:
        """Calculate rolling Sharpe ratio."""
        if len(returns) < window:
            return []
        
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        return rolling_sharpe.dropna().tolist()
    
    def _calculate_drawdown_frequency(self, equity_curve: pd.Series) -> int:
        """Calculate frequency of drawdown periods."""
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Count number of drawdown periods (simplified)
        in_drawdown = drawdown < -0.01  # 1% threshold
        drawdown_periods = 0
        was_in_drawdown = False
        
        for is_in_drawdown in in_drawdown:
            if is_in_drawdown and not was_in_drawdown:
                drawdown_periods += 1
            was_in_drawdown = is_in_drawdown
        
        return drawdown_periods
    
    def _calculate_regime_consistency(self, regime_analysis: RegimeAnalysisResults) -> float:
        """Calculate consistency across market regimes."""
        if not regime_analysis.regime_performances:
            return 0.5
        
        # Use existing regime consistency from analysis
        return regime_analysis.regime_consistency
    
    def _identify_best_worst_periods(
        self,
        period_performances: List[PeriodPerformance]
    ) -> Tuple[ValidationPeriodType, ValidationPeriodType]:
        """Identify best and worst performing periods."""
        if not period_performances:
            return ValidationPeriodType.BULL_MARKET, ValidationPeriodType.BEAR_MARKET
        
        best_period = max(period_performances, key=lambda x: x.sharpe_ratio)
        worst_period = min(period_performances, key=lambda x: x.sharpe_ratio)
        
        return best_period.period_type, worst_period.period_type
    
    def _identify_consistency_extremes(
        self,
        consistency_metrics: ConsistencyMetrics
    ) -> Tuple[str, str]:
        """Identify most and least consistent metrics."""
        metric_scores = {
            'returns': consistency_metrics.return_consistency_score,
            'volatility': consistency_metrics.volatility_consistency_score,
            'drawdown': consistency_metrics.drawdown_consistency_score,
            'win_rate': consistency_metrics.win_rate_consistency_score,
            'risk_adjusted': consistency_metrics.risk_adjusted_consistency
        }
        
        most_consistent = max(metric_scores.items(), key=lambda x: x[1])[0]
        least_consistent = min(metric_scores.items(), key=lambda x: x[1])[0]
        
        return most_consistent, least_consistent
    
    def _calculate_validation_score(
        self,
        consistency_metrics: ConsistencyMetrics,
        regime_consistency: float,
        period_performances: List[PeriodPerformance]
    ) -> float:
        """Calculate overall validation score (0-100)."""
        
        # Base score from consistency
        base_score = consistency_metrics.overall_consistency_score * 60
        
        # Regime consistency bonus
        regime_bonus = regime_consistency * 20
        
        # Performance quality bonus
        avg_sharpe = np.mean([p.sharpe_ratio for p in period_performances])
        performance_bonus = min(20, max(0, avg_sharpe * 10))
        
        total_score = base_score + regime_bonus + performance_bonus
        
        return min(100, max(0, total_score))
    
    def _assess_deployment_readiness(
        self,
        validation_score: float,
        consistency_metrics: ConsistencyMetrics
    ) -> str:
        """Assess deployment readiness based on validation results."""
        
        if (validation_score >= 80 and 
            consistency_metrics.overall_consistency_score >= 0.7 and
            consistency_metrics.consistency_level in [ConsistencyLevel.EXCELLENT, ConsistencyLevel.GOOD]):
            return "Ready"
        elif (validation_score >= 60 and 
              consistency_metrics.overall_consistency_score >= 0.5):
            return "Caution"
        else:
            return "Not Ready"
    
    def _generate_recommendations(
        self,
        consistency_metrics: ConsistencyMetrics,
        regime_analysis: RegimeAnalysisResults,
        period_performances: List[PeriodPerformance]
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations and risk warnings."""
        
        recommendations = []
        risk_warnings = []
        
        # Consistency-based recommendations
        if consistency_metrics.overall_consistency_score < 0.6:
            recommendations.append("Improve strategy consistency across different market conditions")
        
        if consistency_metrics.return_consistency_score < 0.5:
            recommendations.append("Focus on stabilizing return generation")
            risk_warnings.append("High return variability across periods")
        
        if consistency_metrics.drawdown_consistency_score < 0.5:
            recommendations.append("Implement better risk management to control drawdowns")
            risk_warnings.append("Inconsistent drawdown control")
        
        # Regime-based recommendations
        if regime_analysis.regime_consistency < 0.6:
            recommendations.append("Test strategy across more diverse market regimes")
            risk_warnings.append("Poor performance consistency across market regimes")
        
        # Period-specific warnings
        poor_periods = [p for p in period_performances if p.sharpe_ratio < 0]
        if len(poor_periods) > len(period_performances) * 0.3:
            risk_warnings.append("Strategy performs poorly in multiple market conditions")
        
        # Trade frequency warnings
        low_trade_periods = [p for p in period_performances if p.total_trades < 10]
        if len(low_trade_periods) > 0:
            risk_warnings.append("Insufficient trading activity in some periods")
        
        return recommendations, risk_warnings 