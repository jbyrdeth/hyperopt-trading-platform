"""
Performance Analyzer

This module provides comprehensive performance analysis capabilities that extend
the basic backtesting results with advanced institutional-grade metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
import logging

try:
    from ..strategies.backtesting_engine import BacktestResults, Trade
    from ..utils.logger import get_logger
    from ..validation import RobustnessScorer
except ImportError:
    from src.strategies.backtesting_engine import BacktestResults, Trade
    from src.utils.logger import get_logger
    from src.validation import RobustnessScorer


@dataclass
class AdvancedMetrics:
    """Extended performance metrics beyond basic backtesting."""
    
    # Risk-adjusted returns
    information_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    tracking_error: float
    
    # Drawdown analysis
    avg_drawdown: float
    drawdown_duration_avg: float
    drawdown_duration_max: float
    recovery_time_avg: float
    pain_index: float
    
    # Return analysis
    skewness: float
    kurtosis: float
    tail_ratio: float
    gain_to_pain_ratio: float
    
    # Trade analysis
    consecutive_wins_max: float
    consecutive_losses_max: float
    avg_trade_duration_hours: float
    trade_frequency_per_month: float
    
    # Risk metrics
    conditional_var_95: float  # Expected Shortfall
    maximum_adverse_excursion: float
    maximum_favorable_excursion: float
    
    # Stability metrics
    return_stability: float  # Consistency of monthly returns
    sharpe_stability: float  # Rolling Sharpe ratio stability
    
    # Market correlation
    market_correlation: float
    beta: float
    
    # Additional ratios
    sterling_ratio: float
    burke_ratio: float
    martin_ratio: float


@dataclass
class PerformanceBreakdown:
    """Detailed performance breakdown by time periods."""
    
    # Monthly analysis
    monthly_returns: pd.Series
    monthly_sharpe: pd.Series
    monthly_win_rate: pd.Series
    
    # Quarterly analysis
    quarterly_returns: pd.Series
    quarterly_sharpe: pd.Series
    
    # Yearly analysis
    yearly_returns: pd.Series
    yearly_sharpe: pd.Series
    
    # Rolling metrics
    rolling_sharpe_30d: pd.Series
    rolling_max_dd_30d: pd.Series
    rolling_volatility_30d: pd.Series
    
    # Performance attribution
    best_month: Tuple[str, float]
    worst_month: Tuple[str, float]
    best_quarter: Tuple[str, float]
    worst_quarter: Tuple[str, float]


@dataclass
class RiskAnalysis:
    """Comprehensive risk analysis."""
    
    # Value at Risk analysis
    var_1d_95: float
    var_1d_99: float
    var_1w_95: float
    var_1w_99: float
    
    # Expected Shortfall (Conditional VaR)
    es_1d_95: float
    es_1d_99: float
    
    # Tail risk
    tail_expectation_ratio: float
    extreme_tail_loss: float
    
    # Drawdown risk
    expected_drawdown: float
    drawdown_at_risk_95: float
    
    # Volatility analysis
    realized_volatility: float
    volatility_of_volatility: float
    
    # Risk-adjusted metrics
    risk_adjusted_return: float
    return_over_max_dd: float


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    
    # Basic results (from backtesting)
    basic_results: BacktestResults
    
    # Extended analysis
    advanced_metrics: AdvancedMetrics
    performance_breakdown: PerformanceBreakdown
    risk_analysis: RiskAnalysis
    
    # Validation integration
    robustness_score: Optional[float] = None
    validation_details: Optional[Dict[str, Any]] = None
    
    # Analysis metadata
    analysis_date: datetime = field(default_factory=datetime.now)
    benchmark_comparison: Optional[Dict[str, float]] = None
    
    # Summary insights
    key_strengths: List[str] = field(default_factory=list)
    key_weaknesses: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer that extends backtesting results
    with institutional-grade metrics and analysis.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.02,
        benchmark_return: float = 0.0,
        confidence_levels: List[float] = None
    ):
        """
        Initialize the PerformanceAnalyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for calculations
            benchmark_return: Benchmark return for comparison
            confidence_levels: Confidence levels for VaR calculations
        """
        self.logger = get_logger("performance_analyzer")
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        
        # Initialize validation integration
        self.robustness_scorer = None
        
        self.logger.info("PerformanceAnalyzer initialized")
    
    def analyze_performance(
        self,
        backtest_results: BacktestResults,
        market_data: Optional[pd.DataFrame] = None,
        include_validation: bool = False
    ) -> PerformanceReport:
        """
        Perform comprehensive performance analysis.
        
        Args:
            backtest_results: Results from backtesting engine
            market_data: Market data for correlation analysis
            include_validation: Whether to include robustness validation
            
        Returns:
            Comprehensive performance report
        """
        self.logger.info(f"Analyzing performance for {backtest_results.strategy_name}")
        
        try:
            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(backtest_results, market_data)
            
            # Generate performance breakdown
            performance_breakdown = self._generate_performance_breakdown(backtest_results)
            
            # Perform risk analysis
            risk_analysis = self._perform_risk_analysis(backtest_results)
            
            # Generate insights
            strengths, weaknesses, warnings = self._generate_insights(
                backtest_results, advanced_metrics, risk_analysis
            )
            
            # Optional validation integration
            robustness_score = None
            validation_details = None
            if include_validation and market_data is not None:
                robustness_score, validation_details = self._integrate_validation(
                    backtest_results, market_data
                )
            
            # Create comprehensive report
            report = PerformanceReport(
                basic_results=backtest_results,
                advanced_metrics=advanced_metrics,
                performance_breakdown=performance_breakdown,
                risk_analysis=risk_analysis,
                robustness_score=robustness_score,
                validation_details=validation_details,
                key_strengths=strengths,
                key_weaknesses=weaknesses,
                risk_warnings=warnings
            )
            
            self.logger.info("Performance analysis completed successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            raise
    
    def _calculate_advanced_metrics(
        self,
        results: BacktestResults,
        market_data: Optional[pd.DataFrame] = None
    ) -> AdvancedMetrics:
        """Calculate advanced performance metrics."""
        
        # Get returns series
        returns = results.equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return self._create_empty_advanced_metrics()
        
        # Risk-adjusted returns
        information_ratio = self._calculate_information_ratio(returns)
        treynor_ratio = self._calculate_treynor_ratio(returns, market_data)
        jensen_alpha = self._calculate_jensen_alpha(returns, market_data)
        tracking_error = self._calculate_tracking_error(returns, market_data)
        
        # Drawdown analysis
        drawdowns = results.drawdown_curve
        avg_drawdown = drawdowns.mean()
        drawdown_duration_avg, drawdown_duration_max = self._calculate_drawdown_durations(drawdowns)
        recovery_time_avg = self._calculate_recovery_times(drawdowns)
        pain_index = self._calculate_pain_index(drawdowns)
        
        # Return analysis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        tail_ratio = self._calculate_tail_ratio(returns)
        gain_to_pain_ratio = self._calculate_gain_to_pain_ratio(returns)
        
        # Trade analysis
        consecutive_wins_max, consecutive_losses_max = self._calculate_consecutive_trades(results.trades)
        avg_trade_duration_hours = np.mean([trade.duration_hours for trade in results.trades]) if results.trades else 0
        trade_frequency_per_month = self._calculate_trade_frequency(results.trades, results.start_date, results.end_date)
        
        # Risk metrics
        conditional_var_95 = self._calculate_conditional_var(returns, 0.95)
        mae, mfe = self._calculate_mae_mfe(results.trades)
        
        # Stability metrics
        return_stability = self._calculate_return_stability(returns)
        sharpe_stability = self._calculate_sharpe_stability(returns)
        
        # Market correlation
        market_correlation, beta = self._calculate_market_metrics(returns, market_data)
        
        # Additional ratios
        sterling_ratio = self._calculate_sterling_ratio(returns, drawdowns)
        burke_ratio = self._calculate_burke_ratio(returns, drawdowns)
        martin_ratio = self._calculate_martin_ratio(returns, drawdowns)
        
        return AdvancedMetrics(
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            jensen_alpha=jensen_alpha,
            tracking_error=tracking_error,
            avg_drawdown=avg_drawdown,
            drawdown_duration_avg=drawdown_duration_avg,
            drawdown_duration_max=drawdown_duration_max,
            recovery_time_avg=recovery_time_avg,
            pain_index=pain_index,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            gain_to_pain_ratio=gain_to_pain_ratio,
            consecutive_wins_max=consecutive_wins_max,
            consecutive_losses_max=consecutive_losses_max,
            avg_trade_duration_hours=avg_trade_duration_hours,
            trade_frequency_per_month=trade_frequency_per_month,
            conditional_var_95=conditional_var_95,
            maximum_adverse_excursion=mae,
            maximum_favorable_excursion=mfe,
            return_stability=return_stability,
            sharpe_stability=sharpe_stability,
            market_correlation=market_correlation,
            beta=beta,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            martin_ratio=martin_ratio
        )
    
    def _generate_performance_breakdown(self, results: BacktestResults) -> PerformanceBreakdown:
        """Generate detailed performance breakdown by time periods."""
        
        equity_curve = results.equity_curve
        returns = equity_curve.pct_change().dropna()
        
        if len(returns) == 0:
            return self._create_empty_performance_breakdown()
        
        # Monthly analysis
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_sharpe = returns.resample('M').apply(
            lambda x: x.mean() / x.std() * np.sqrt(252/30) if x.std() > 0 else 0
        )
        monthly_win_rate = monthly_returns.apply(lambda x: 1 if x > 0 else 0)
        
        # Quarterly analysis
        quarterly_returns = returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        quarterly_sharpe = returns.resample('Q').apply(
            lambda x: x.mean() / x.std() * np.sqrt(252/90) if x.std() > 0 else 0
        )
        
        # Yearly analysis
        yearly_returns = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        yearly_sharpe = returns.resample('Y').apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Rolling metrics
        rolling_sharpe_30d = returns.rolling(30).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_max_dd_30d = equity_curve.rolling(30).apply(
            lambda x: (x / x.expanding().max() - 1).min()
        )
        rolling_volatility_30d = returns.rolling(30).std() * np.sqrt(252)
        
        # Performance attribution
        best_month = (monthly_returns.idxmax().strftime('%Y-%m'), monthly_returns.max()) if len(monthly_returns) > 0 else ("N/A", 0)
        worst_month = (monthly_returns.idxmin().strftime('%Y-%m'), monthly_returns.min()) if len(monthly_returns) > 0 else ("N/A", 0)
        best_quarter = (quarterly_returns.idxmax().strftime('%Y-Q%q'), quarterly_returns.max()) if len(quarterly_returns) > 0 else ("N/A", 0)
        worst_quarter = (quarterly_returns.idxmin().strftime('%Y-Q%q'), quarterly_returns.min()) if len(quarterly_returns) > 0 else ("N/A", 0)
        
        return PerformanceBreakdown(
            monthly_returns=monthly_returns,
            monthly_sharpe=monthly_sharpe,
            monthly_win_rate=monthly_win_rate,
            quarterly_returns=quarterly_returns,
            quarterly_sharpe=quarterly_sharpe,
            yearly_returns=yearly_returns,
            yearly_sharpe=yearly_sharpe,
            rolling_sharpe_30d=rolling_sharpe_30d,
            rolling_max_dd_30d=rolling_max_dd_30d,
            rolling_volatility_30d=rolling_volatility_30d,
            best_month=best_month,
            worst_month=worst_month,
            best_quarter=best_quarter,
            worst_quarter=worst_quarter
        )
    
    def _perform_risk_analysis(self, results: BacktestResults) -> RiskAnalysis:
        """Perform comprehensive risk analysis."""
        
        returns = results.equity_curve.pct_change().dropna()
        drawdowns = results.drawdown_curve
        
        if len(returns) == 0:
            return self._create_empty_risk_analysis()
        
        # Value at Risk
        var_1d_95 = np.percentile(returns, 5)
        var_1d_99 = np.percentile(returns, 1)
        var_1w_95 = np.percentile(returns.rolling(7).sum().dropna(), 5)
        var_1w_99 = np.percentile(returns.rolling(7).sum().dropna(), 1)
        
        # Expected Shortfall (Conditional VaR)
        es_1d_95 = returns[returns <= var_1d_95].mean()
        es_1d_99 = returns[returns <= var_1d_99].mean()
        
        # Tail risk
        tail_expectation_ratio = abs(returns[returns < 0].mean() / returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0
        extreme_tail_loss = np.percentile(returns, 0.5)  # 99.5% VaR
        
        # Drawdown risk
        expected_drawdown = drawdowns.mean()
        drawdown_at_risk_95 = np.percentile(drawdowns, 5)
        
        # Volatility analysis
        realized_volatility = returns.std() * np.sqrt(252)
        volatility_of_volatility = returns.rolling(30).std().std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        risk_adjusted_return = results.annual_return / realized_volatility if realized_volatility > 0 else 0
        return_over_max_dd = results.annual_return / abs(results.max_drawdown) if results.max_drawdown < 0 else 0
        
        return RiskAnalysis(
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            var_1w_95=var_1w_95,
            var_1w_99=var_1w_99,
            es_1d_95=es_1d_95,
            es_1d_99=es_1d_99,
            tail_expectation_ratio=tail_expectation_ratio,
            extreme_tail_loss=extreme_tail_loss,
            expected_drawdown=expected_drawdown,
            drawdown_at_risk_95=drawdown_at_risk_95,
            realized_volatility=realized_volatility,
            volatility_of_volatility=volatility_of_volatility,
            risk_adjusted_return=risk_adjusted_return,
            return_over_max_dd=return_over_max_dd
        )
    
    def _generate_insights(
        self,
        results: BacktestResults,
        advanced_metrics: AdvancedMetrics,
        risk_analysis: RiskAnalysis
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate key insights about strategy performance."""
        
        strengths = []
        weaknesses = []
        warnings = []
        
        # Analyze Sharpe ratio
        if results.sharpe_ratio > 2.0:
            strengths.append(f"Excellent risk-adjusted returns (Sharpe: {results.sharpe_ratio:.2f})")
        elif results.sharpe_ratio > 1.0:
            strengths.append(f"Good risk-adjusted returns (Sharpe: {results.sharpe_ratio:.2f})")
        elif results.sharpe_ratio < 0.5:
            weaknesses.append(f"Poor risk-adjusted returns (Sharpe: {results.sharpe_ratio:.2f})")
        
        # Analyze drawdown
        if results.max_drawdown > -0.05:
            strengths.append(f"Low maximum drawdown ({results.max_drawdown:.1%})")
        elif results.max_drawdown < -0.30:
            weaknesses.append(f"High maximum drawdown ({results.max_drawdown:.1%})")
            warnings.append("Strategy experiences significant drawdowns")
        
        # Analyze win rate
        if results.win_rate > 0.6:
            strengths.append(f"High win rate ({results.win_rate:.1%})")
        elif results.win_rate < 0.4:
            weaknesses.append(f"Low win rate ({results.win_rate:.1%})")
        
        # Analyze stability
        if advanced_metrics.return_stability > 0.7:
            strengths.append("Consistent monthly returns")
        elif advanced_metrics.return_stability < 0.3:
            weaknesses.append("Inconsistent monthly returns")
            warnings.append("Strategy shows high return volatility")
        
        # Analyze tail risk
        if abs(advanced_metrics.skewness) > 1.0:
            warnings.append(f"High return skewness ({advanced_metrics.skewness:.2f})")
        
        if advanced_metrics.kurtosis > 3.0:
            warnings.append("Returns show fat tails (high kurtosis)")
        
        # Analyze trade frequency
        if advanced_metrics.trade_frequency_per_month > 50:
            warnings.append("Very high trade frequency may increase transaction costs")
        elif advanced_metrics.trade_frequency_per_month < 1:
            warnings.append("Very low trade frequency may miss opportunities")
        
        # Risk warnings
        if risk_analysis.var_1d_99 < -0.05:
            warnings.append("High tail risk: 1% chance of >5% daily loss")
        
        if risk_analysis.volatility_of_volatility > 0.5:
            warnings.append("High volatility clustering detected")
        
        return strengths, weaknesses, warnings
    
    def _integrate_validation(
        self,
        results: BacktestResults,
        market_data: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Integrate with validation framework for robustness scoring."""
        try:
            if self.robustness_scorer is None:
                self.robustness_scorer = RobustnessScorer()
            
            # This would require strategy object - simplified for now
            # In practice, would need to store strategy reference
            return None, None
            
        except Exception as e:
            self.logger.warning(f"Validation integration failed: {e}")
            return None, None
    
    # Helper methods for metric calculations
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    def _calculate_treynor_ratio(self, returns: pd.Series, market_data: Optional[pd.DataFrame]) -> float:
        """Calculate Treynor ratio."""
        if market_data is None:
            return 0
        
        # Simplified - would need proper market return calculation
        excess_returns = returns - self.risk_free_rate / 252
        beta = self._calculate_market_metrics(returns, market_data)[1]
        return excess_returns.mean() * 252 / beta if beta != 0 else 0
    
    def _calculate_jensen_alpha(self, returns: pd.Series, market_data: Optional[pd.DataFrame]) -> float:
        """Calculate Jensen's alpha."""
        if market_data is None:
            return 0
        
        # Simplified implementation
        excess_returns = returns - self.risk_free_rate / 252
        return excess_returns.mean() * 252  # Simplified
    
    def _calculate_tracking_error(self, returns: pd.Series, market_data: Optional[pd.DataFrame]) -> float:
        """Calculate tracking error."""
        if market_data is None:
            return 0
        
        # Simplified - would need proper benchmark returns
        return returns.std() * np.sqrt(252)
    
    def _calculate_drawdown_durations(self, drawdowns: pd.Series) -> Tuple[float, float]:
        """Calculate average and maximum drawdown durations."""
        if len(drawdowns) == 0:
            return 0, 0
        
        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        if not drawdown_periods:
            return 0, 0
        
        return np.mean(drawdown_periods), max(drawdown_periods)
    
    def _calculate_recovery_times(self, drawdowns: pd.Series) -> float:
        """Calculate average recovery time from drawdowns."""
        # Simplified implementation
        return 0  # Would need more complex logic
    
    def _calculate_pain_index(self, drawdowns: pd.Series) -> float:
        """Calculate pain index (average drawdown)."""
        return abs(drawdowns[drawdowns < 0].mean()) if len(drawdowns[drawdowns < 0]) > 0 else 0
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        return abs(p95 / p5) if p5 != 0 else 0
    
    def _calculate_gain_to_pain_ratio(self, returns: pd.Series) -> float:
        """Calculate gain to pain ratio."""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return gains / losses if losses > 0 else 0
    
    def _calculate_consecutive_trades(self, trades: List[Trade]) -> Tuple[float, float]:
        """Calculate maximum consecutive wins and losses."""
        if not trades:
            return 0, 0
        
        consecutive_wins = 0
        consecutive_losses = 0
        max_wins = 0
        max_losses = 0
        
        for trade in trades:
            if trade.net_pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_wins = max(max_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_losses = max(max_losses, consecutive_losses)
        
        return max_wins, max_losses
    
    def _calculate_trade_frequency(self, trades: List[Trade], start_date: datetime, end_date: datetime) -> float:
        """Calculate trade frequency per month."""
        if not trades:
            return 0
        
        total_months = (end_date - start_date).days / 30.44
        return len(trades) / total_months if total_months > 0 else 0
    
    def _calculate_conditional_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        return returns[returns <= var_threshold].mean()
    
    def _calculate_mae_mfe(self, trades: List[Trade]) -> Tuple[float, float]:
        """Calculate Maximum Adverse/Favorable Excursion."""
        # Simplified - would need intra-trade data
        if not trades:
            return 0, 0
        
        losses = [trade.net_pnl for trade in trades if trade.net_pnl < 0]
        gains = [trade.net_pnl for trade in trades if trade.net_pnl > 0]
        
        mae = min(losses) if losses else 0
        mfe = max(gains) if gains else 0
        
        return mae, mfe
    
    def _calculate_return_stability(self, returns: pd.Series) -> float:
        """Calculate return stability (consistency)."""
        if len(returns) < 30:
            return 0
        
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        if len(monthly_returns) < 2:
            return 0
        
        return 1 - (monthly_returns.std() / (abs(monthly_returns.mean()) + 1e-8))
    
    def _calculate_sharpe_stability(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio stability."""
        if len(returns) < 60:
            return 0
        
        rolling_sharpe = returns.rolling(30).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        ).dropna()
        
        if len(rolling_sharpe) < 2:
            return 0
        
        return 1 - (rolling_sharpe.std() / (abs(rolling_sharpe.mean()) + 1e-8))
    
    def _calculate_market_metrics(self, returns: pd.Series, market_data: Optional[pd.DataFrame]) -> Tuple[float, float]:
        """Calculate market correlation and beta."""
        if market_data is None or len(market_data) == 0:
            return 0, 0
        
        # Simplified - would need proper market return calculation
        # For now, return default values
        return 0, 1
    
    def _calculate_sterling_ratio(self, returns: pd.Series, drawdowns: pd.Series) -> float:
        """Calculate Sterling ratio."""
        avg_annual_return = returns.mean() * 252
        avg_drawdown = abs(drawdowns.mean())
        return avg_annual_return / avg_drawdown if avg_drawdown > 0 else 0
    
    def _calculate_burke_ratio(self, returns: pd.Series, drawdowns: pd.Series) -> float:
        """Calculate Burke ratio."""
        avg_annual_return = returns.mean() * 252
        drawdown_squared_sum = (drawdowns ** 2).sum()
        return avg_annual_return / np.sqrt(drawdown_squared_sum) if drawdown_squared_sum > 0 else 0
    
    def _calculate_martin_ratio(self, returns: pd.Series, drawdowns: pd.Series) -> float:
        """Calculate Martin ratio (Ulcer Performance Index)."""
        avg_annual_return = returns.mean() * 252
        ulcer_index = np.sqrt((drawdowns ** 2).mean())
        return avg_annual_return / ulcer_index if ulcer_index > 0 else 0
    
    # Helper methods for empty objects
    def _create_empty_advanced_metrics(self) -> AdvancedMetrics:
        """Create empty advanced metrics for edge cases."""
        return AdvancedMetrics(
            information_ratio=0, treynor_ratio=0, jensen_alpha=0, tracking_error=0,
            avg_drawdown=0, drawdown_duration_avg=0, drawdown_duration_max=0,
            recovery_time_avg=0, pain_index=0, skewness=0, kurtosis=0,
            tail_ratio=0, gain_to_pain_ratio=0, consecutive_wins_max=0,
            consecutive_losses_max=0, avg_trade_duration_hours=0,
            trade_frequency_per_month=0, conditional_var_95=0,
            maximum_adverse_excursion=0, maximum_favorable_excursion=0,
            return_stability=0, sharpe_stability=0, market_correlation=0,
            beta=0, sterling_ratio=0, burke_ratio=0, martin_ratio=0
        )
    
    def _create_empty_performance_breakdown(self) -> PerformanceBreakdown:
        """Create empty performance breakdown for edge cases."""
        empty_series = pd.Series(dtype=float)
        return PerformanceBreakdown(
            monthly_returns=empty_series, monthly_sharpe=empty_series,
            monthly_win_rate=empty_series, quarterly_returns=empty_series,
            quarterly_sharpe=empty_series, yearly_returns=empty_series,
            yearly_sharpe=empty_series, rolling_sharpe_30d=empty_series,
            rolling_max_dd_30d=empty_series, rolling_volatility_30d=empty_series,
            best_month=("N/A", 0), worst_month=("N/A", 0),
            best_quarter=("N/A", 0), worst_quarter=("N/A", 0)
        )
    
    def _create_empty_risk_analysis(self) -> RiskAnalysis:
        """Create empty risk analysis for edge cases."""
        return RiskAnalysis(
            var_1d_95=0, var_1d_99=0, var_1w_95=0, var_1w_99=0,
            es_1d_95=0, es_1d_99=0, tail_expectation_ratio=0,
            extreme_tail_loss=0, expected_drawdown=0, drawdown_at_risk_95=0,
            realized_volatility=0, volatility_of_volatility=0,
            risk_adjusted_return=0, return_over_max_dd=0
        ) 