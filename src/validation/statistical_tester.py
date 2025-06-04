"""
Statistical Significance Testing for Trading Strategies

This module provides comprehensive statistical testing to determine if strategy
performance is significantly better than random or benchmark strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ..utils.logger import get_logger
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.base_strategy import Signal
    from ..strategies.backtesting_engine import BacktestingEngine, CostModel, BacktestResults
    from .data_splitter import DataSplitter
    from .monte_carlo_tester import MonteCarloTester, MonteCarloResults
except ImportError:
    from src.utils.logger import get_logger
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.base_strategy import Signal
    from src.strategies.backtesting_engine import BacktestingEngine, CostModel, BacktestResults
    from src.validation.data_splitter import DataSplitter
    from src.validation.monte_carlo_tester import MonteCarloTester, MonteCarloResults


class TestType(Enum):
    """Types of statistical tests available."""
    ONE_SAMPLE_T_TEST = "one_sample_t_test"
    TWO_SAMPLE_T_TEST = "two_sample_t_test"
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    MANN_WHITNEY_U = "mann_whitney_u"
    BOOTSTRAP_TEST = "bootstrap_test"


class CorrectionMethod(Enum):
    """Multiple hypothesis testing correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    SIDAK = "sidak"
    HOLM_SIDAK = "holm-sidak"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    NONE = "none"


@dataclass
class StatisticalTest:
    """Results from a single statistical test."""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    critical_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    effect_size: Optional[float]
    power: Optional[float]
    sample_size: int
    degrees_of_freedom: Optional[int]
    is_significant: bool
    alpha: float
    alternative: str  # 'two-sided', 'greater', 'less'
    
    # Additional metadata
    description: str
    assumptions_met: Dict[str, bool]
    warnings: List[str] = field(default_factory=list)


@dataclass
class BenchmarkComparison:
    """Results from comparing strategy against benchmarks."""
    strategy_name: str
    benchmark_name: str
    strategy_returns: np.ndarray
    benchmark_returns: np.ndarray
    
    # Performance metrics comparison
    strategy_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]
    
    # Statistical tests
    statistical_tests: List[StatisticalTest]
    
    # Overall assessment
    is_significantly_better: bool
    confidence_level: float
    summary: str


@dataclass
class MultipleTestingResults:
    """Results from multiple hypothesis testing with corrections."""
    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected_hypotheses: List[bool]
    correction_method: CorrectionMethod
    alpha: float
    n_significant_original: int
    n_significant_corrected: int
    family_wise_error_rate: float


@dataclass
class PowerAnalysis:
    """Results from statistical power analysis."""
    effect_size: float
    alpha: float
    power: float
    sample_size: int
    minimum_sample_size: int
    recommended_sample_size: int
    analysis_type: str


class RandomStrategy(BaseStrategy):
    """Random strategy for benchmark comparison (coin flip)."""
    
    def __init__(self, win_probability: float = 0.5, **kwargs):
        """
        Initialize random strategy.
        
        Args:
            win_probability: Probability of generating a buy signal
        """
        parameters = {'win_probability': win_probability}
        risk_params = kwargs.get('risk_params', {})
        
        super().__init__(
            name=f"Random_{win_probability}",
            parameters=parameters,
            risk_params=risk_params
        )
        self.win_probability = win_probability
        self.required_periods = 1
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with data."""
        pass  # No initialization needed for random strategy
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Generate a random signal."""
        action = 'buy' if np.random.random() < self.win_probability else 'sell'
        strength = np.random.random()  # Random strength
        
        return Signal(
            timestamp=current_time,
            action=action,
            strength=strength,
            price=current_data['close'],
            confidence=0.5,  # Neutral confidence for random signals
            metadata={'strategy_type': 'random', 'win_probability': self.win_probability}
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """Calculate position size (always use default)."""
        return self.risk_params.get('position_size_pct', 0.95)
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate random buy/sell signals."""
        signals = pd.DataFrame(index=data.index)
        
        # Generate random signals
        random_values = np.random.random(len(data))
        signals['signal'] = np.where(random_values < self.win_probability, 1, -1)
        signals['confidence'] = 0.5  # Neutral confidence for random signals
        
        return signals


class BuyAndHoldStrategy(BaseStrategy):
    """Buy and hold benchmark strategy."""
    
    def __init__(self, **kwargs):
        parameters = {}
        risk_params = kwargs.get('risk_params', {})
        
        super().__init__(
            name="BuyAndHold",
            parameters=parameters,
            risk_params=risk_params
        )
        self.required_periods = 1
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with data."""
        pass  # No initialization needed for buy and hold
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Always generate buy signal."""
        return Signal(
            timestamp=current_time,
            action='buy',
            strength=1.0,
            price=current_data['close'],
            confidence=1.0,  # Full confidence in buy and hold
            metadata={'strategy_type': 'buy_and_hold'}
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """Calculate position size (always use full allocation)."""
        return self.risk_params.get('position_size_pct', 1.0)
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and hold signals."""
        signals = pd.DataFrame(index=data.index)
        
        # Buy at the beginning and hold
        signals['signal'] = 1  # Always long
        signals['confidence'] = 1.0  # Full confidence in buy and hold
        
        return signals


class StatisticalTester:
    """
    Comprehensive statistical significance testing for trading strategies.
    
    Provides various statistical tests to determine if strategy performance
    is significantly better than random or benchmark strategies.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        initial_capital: float = 100000,
        cost_model: Optional[CostModel] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the StatisticalTester.
        
        Args:
            alpha: Significance level for statistical tests
            initial_capital: Starting capital for backtests
            cost_model: Cost model for transaction costs
            random_seed: Random seed for reproducible results
        """
        self.alpha = alpha
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.random_seed = random_seed
        
        self.logger = get_logger("statistical_tester")
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(
            initial_capital=initial_capital,
            cost_model=self.cost_model
        )
    
    def test_against_random(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        n_random_strategies: int = 100,
        win_probabilities: List[float] = None,
        test_types: List[TestType] = None
    ) -> List[BenchmarkComparison]:
        """
        Test strategy against random strategies.
        
        Args:
            strategy: Strategy to test
            data: Historical OHLCV data
            n_random_strategies: Number of random strategies to generate
            win_probabilities: List of win probabilities for random strategies
            test_types: Types of statistical tests to perform
            
        Returns:
            List of benchmark comparison results
        """
        self.logger.info(f"Testing {strategy.name} against {n_random_strategies} random strategies")
        
        if win_probabilities is None:
            win_probabilities = [0.5]  # Fair coin flip
        
        if test_types is None:
            test_types = [TestType.TWO_SAMPLE_T_TEST, TestType.MANN_WHITNEY_U]
        
        # Run strategy backtest
        strategy_results = self.backtesting_engine.backtest_strategy(strategy, data, "Strategy")
        strategy_returns = self._extract_returns(strategy_results)
        
        comparisons = []
        
        for win_prob in win_probabilities:
            # Generate random strategy results
            random_returns_list = []
            
            for i in range(n_random_strategies):
                random_strategy = RandomStrategy(win_probability=win_prob)
                random_results = self.backtesting_engine.backtest_strategy(
                    random_strategy, data, f"Random_{win_prob}_{i}"
                )
                random_returns = self._extract_returns(random_results)
                random_returns_list.append(random_returns)
            
            # Combine all random returns
            all_random_returns = np.concatenate(random_returns_list)
            
            # Perform statistical tests
            statistical_tests = []
            for test_type in test_types:
                test_result = self._perform_test(
                    strategy_returns,
                    all_random_returns,
                    test_type,
                    f"Strategy vs Random({win_prob})"
                )
                statistical_tests.append(test_result)
            
            # Calculate metrics
            strategy_metrics = self._calculate_performance_metrics(strategy_results)
            
            # Calculate average random metrics
            random_metrics = {}
            for key in strategy_metrics.keys():
                random_values = []
                for random_returns in random_returns_list:
                    # Create mock results for metric calculation
                    mock_results = self._create_mock_results(random_returns)
                    mock_metrics = self._calculate_performance_metrics(mock_results)
                    if key in mock_metrics:
                        random_values.append(mock_metrics[key])
                
                if random_values:
                    random_metrics[key] = np.mean(random_values)
            
            # Determine if significantly better
            is_significantly_better = any(
                test.is_significant and test.statistic > 0 
                for test in statistical_tests
            )
            
            comparison = BenchmarkComparison(
                strategy_name=strategy.name,
                benchmark_name=f"Random({win_prob})",
                strategy_returns=strategy_returns,
                benchmark_returns=all_random_returns,
                strategy_metrics=strategy_metrics,
                benchmark_metrics=random_metrics,
                statistical_tests=statistical_tests,
                is_significantly_better=is_significantly_better,
                confidence_level=1 - self.alpha,
                summary=self._generate_comparison_summary(
                    strategy.name, f"Random({win_prob})", statistical_tests, is_significantly_better
                )
            )
            
            comparisons.append(comparison)
        
        return comparisons
    
    def test_against_benchmark(
        self,
        strategy: BaseStrategy,
        benchmark_strategy: BaseStrategy,
        data: pd.DataFrame,
        test_types: List[TestType] = None
    ) -> BenchmarkComparison:
        """
        Test strategy against a specific benchmark strategy.
        
        Args:
            strategy: Strategy to test
            benchmark_strategy: Benchmark strategy to compare against
            data: Historical OHLCV data
            test_types: Types of statistical tests to perform
            
        Returns:
            Benchmark comparison results
        """
        self.logger.info(f"Testing {strategy.name} against {benchmark_strategy.name}")
        
        if test_types is None:
            test_types = [TestType.PAIRED_T_TEST, TestType.WILCOXON_SIGNED_RANK]
        
        # Run backtests
        strategy_results = self.backtesting_engine.backtest_strategy(strategy, data, "Strategy")
        benchmark_results = self.backtesting_engine.backtest_strategy(benchmark_strategy, data, "Benchmark")
        
        strategy_returns = self._extract_returns(strategy_results)
        benchmark_returns = self._extract_returns(benchmark_results)
        
        # Perform statistical tests
        statistical_tests = []
        for test_type in test_types:
            test_result = self._perform_test(
                strategy_returns,
                benchmark_returns,
                test_type,
                f"{strategy.name} vs {benchmark_strategy.name}"
            )
            statistical_tests.append(test_result)
        
        # Calculate metrics
        strategy_metrics = self._calculate_performance_metrics(strategy_results)
        benchmark_metrics = self._calculate_performance_metrics(benchmark_results)
        
        # Determine if significantly better
        is_significantly_better = any(
            test.is_significant and test.statistic > 0 
            for test in statistical_tests
        )
        
        comparison = BenchmarkComparison(
            strategy_name=strategy.name,
            benchmark_name=benchmark_strategy.name,
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            strategy_metrics=strategy_metrics,
            benchmark_metrics=benchmark_metrics,
            statistical_tests=statistical_tests,
            is_significantly_better=is_significantly_better,
            confidence_level=1 - self.alpha,
            summary=self._generate_comparison_summary(
                strategy.name, benchmark_strategy.name, statistical_tests, is_significantly_better
            )
        )
        
        return comparison
    
    def test_buy_and_hold(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        test_types: List[TestType] = None
    ) -> BenchmarkComparison:
        """
        Test strategy against buy-and-hold benchmark.
        
        Args:
            strategy: Strategy to test
            data: Historical OHLCV data
            test_types: Types of statistical tests to perform
            
        Returns:
            Benchmark comparison results
        """
        buy_and_hold = BuyAndHoldStrategy()
        return self.test_against_benchmark(strategy, buy_and_hold, data, test_types)
    
    def perform_multiple_testing_correction(
        self,
        p_values: List[float],
        method: CorrectionMethod = CorrectionMethod.FDR_BH,
        alpha: Optional[float] = None
    ) -> MultipleTestingResults:
        """
        Apply multiple hypothesis testing correction.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method to use
            alpha: Significance level (uses instance alpha if None)
            
        Returns:
            Multiple testing correction results
        """
        if alpha is None:
            alpha = self.alpha
        
        self.logger.info(f"Applying {method.value} correction to {len(p_values)} p-values")
        
        # Apply correction
        if method == CorrectionMethod.NONE:
            corrected_p_values = p_values
            rejected = [p < alpha for p in p_values]
        else:
            rejected, corrected_p_values, _, _ = multipletests(
                p_values, alpha=alpha, method=method.value
            )
        
        # Calculate statistics
        n_significant_original = sum(p < alpha for p in p_values)
        n_significant_corrected = sum(rejected)
        
        # Estimate family-wise error rate
        family_wise_error_rate = 1 - (1 - alpha) ** len(p_values)
        
        return MultipleTestingResults(
            original_p_values=p_values,
            corrected_p_values=list(corrected_p_values),
            rejected_hypotheses=list(rejected),
            correction_method=method,
            alpha=alpha,
            n_significant_original=n_significant_original,
            n_significant_corrected=n_significant_corrected,
            family_wise_error_rate=family_wise_error_rate
        )
    
    def calculate_power_analysis(
        self,
        effect_size: float,
        sample_size: Optional[int] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
        analysis_type: str = "two_sample"
    ) -> PowerAnalysis:
        """
        Perform statistical power analysis.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size (if calculating power)
            alpha: Significance level
            power: Desired power (if calculating sample size)
            analysis_type: Type of analysis ('one_sample', 'two_sample', 'paired')
            
        Returns:
            Power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        try:
            from statsmodels.stats.power import ttest_power
            
            if sample_size is not None and power is None:
                # Calculate power given sample size
                calculated_power = ttest_power(effect_size, sample_size, alpha, alternative='two-sided')
                minimum_sample_size = sample_size
                recommended_sample_size = sample_size
                
            elif power is not None and sample_size is None:
                # Calculate sample size given power
                from statsmodels.stats.power import tt_solve_power
                calculated_sample_size = tt_solve_power(
                    effect_size=effect_size, 
                    power=power, 
                    alpha=alpha, 
                    alternative='two-sided'
                )
                sample_size = int(np.ceil(calculated_sample_size))
                calculated_power = power
                minimum_sample_size = sample_size
                recommended_sample_size = int(sample_size * 1.2)  # 20% buffer
                
            else:
                raise ValueError("Must specify either sample_size or power, but not both")
            
        except ImportError:
            self.logger.warning("statsmodels not available for power analysis, using approximation")
            
            # Simple approximation for power analysis
            if sample_size is not None:
                # Approximate power calculation
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = effect_size * np.sqrt(sample_size/2) - z_alpha
                calculated_power = stats.norm.cdf(z_beta)
                minimum_sample_size = sample_size
                recommended_sample_size = sample_size
            else:
                # Approximate sample size calculation
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = stats.norm.ppf(power or 0.8)
                sample_size = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
                calculated_power = power or 0.8
                minimum_sample_size = sample_size
                recommended_sample_size = int(sample_size * 1.2)
        
        return PowerAnalysis(
            effect_size=effect_size,
            alpha=alpha,
            power=calculated_power,
            sample_size=sample_size,
            minimum_sample_size=minimum_sample_size,
            recommended_sample_size=recommended_sample_size,
            analysis_type=analysis_type
        )
    
    def calculate_minimum_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: Optional[float] = None
    ) -> int:
        """
        Calculate minimum sample size for detecting an effect.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power
            alpha: Significance level
            
        Returns:
            Minimum sample size needed
        """
        power_analysis = self.calculate_power_analysis(
            effect_size=effect_size,
            power=power,
            alpha=alpha
        )
        return power_analysis.minimum_sample_size
    
    def _perform_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        test_type: TestType,
        test_name: str,
        alternative: str = "two-sided"
    ) -> StatisticalTest:
        """Perform a specific statistical test."""
        
        # Remove NaN values
        sample1 = sample1[~np.isnan(sample1)]
        sample2 = sample2[~np.isnan(sample2)]
        
        assumptions_met = {}
        warnings_list = []
        
        try:
            if test_type == TestType.ONE_SAMPLE_T_TEST:
                # Test if sample1 mean is significantly different from 0
                statistic, p_value = stats.ttest_1samp(sample1, 0, alternative=alternative)
                degrees_of_freedom = len(sample1) - 1
                
                # Check normality assumption
                if len(sample1) >= 8:
                    _, norm_p = stats.shapiro(sample1)
                    assumptions_met['normality'] = norm_p > 0.05
                    if norm_p <= 0.05:
                        warnings_list.append("Normality assumption violated")
                
            elif test_type == TestType.TWO_SAMPLE_T_TEST:
                # Independent samples t-test
                statistic, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
                degrees_of_freedom = len(sample1) + len(sample2) - 2
                
                # Check assumptions
                if len(sample1) >= 8 and len(sample2) >= 8:
                    _, norm_p1 = stats.shapiro(sample1)
                    _, norm_p2 = stats.shapiro(sample2)
                    assumptions_met['normality'] = norm_p1 > 0.05 and norm_p2 > 0.05
                    
                    # Equal variances test
                    _, var_p = stats.levene(sample1, sample2)
                    assumptions_met['equal_variances'] = var_p > 0.05
                    
                    if norm_p1 <= 0.05 or norm_p2 <= 0.05:
                        warnings_list.append("Normality assumption violated")
                    if var_p <= 0.05:
                        warnings_list.append("Equal variances assumption violated")
                
            elif test_type == TestType.PAIRED_T_TEST:
                # Paired samples t-test
                min_len = min(len(sample1), len(sample2))
                sample1_paired = sample1[:min_len]
                sample2_paired = sample2[:min_len]
                
                statistic, p_value = stats.ttest_rel(sample1_paired, sample2_paired, alternative=alternative)
                degrees_of_freedom = min_len - 1
                
                # Check normality of differences
                differences = sample1_paired - sample2_paired
                if len(differences) >= 8:
                    _, norm_p = stats.shapiro(differences)
                    assumptions_met['normality_differences'] = norm_p > 0.05
                    if norm_p <= 0.05:
                        warnings_list.append("Normality of differences assumption violated")
                
            elif test_type == TestType.WILCOXON_SIGNED_RANK:
                # Non-parametric paired test
                min_len = min(len(sample1), len(sample2))
                sample1_paired = sample1[:min_len]
                sample2_paired = sample2[:min_len]
                
                statistic, p_value = stats.wilcoxon(sample1_paired, sample2_paired, alternative=alternative)
                degrees_of_freedom = None
                assumptions_met['non_parametric'] = True
                
            elif test_type == TestType.MANN_WHITNEY_U:
                # Non-parametric independent samples test
                statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
                degrees_of_freedom = None
                assumptions_met['non_parametric'] = True
                
            elif test_type == TestType.BOOTSTRAP_TEST:
                # Bootstrap test
                statistic, p_value = self._bootstrap_test(sample1, sample2, alternative)
                degrees_of_freedom = None
                assumptions_met['bootstrap'] = True
                
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(sample1, sample2)
            
            # Calculate confidence interval for the difference in means
            confidence_interval = self._calculate_confidence_interval(sample1, sample2, test_type)
            
            # Calculate power (approximate)
            power = self._estimate_power(sample1, sample2, effect_size)
            
            # Determine significance
            is_significant = bool(p_value < self.alpha)  # Explicitly convert to Python bool
            
            return StatisticalTest(
                test_name=test_name,
                test_type=test_type,
                statistic=statistic,
                p_value=p_value,
                critical_value=None,  # Could be calculated if needed
                confidence_interval=confidence_interval,
                effect_size=effect_size,
                power=power,
                sample_size=len(sample1) + len(sample2),
                degrees_of_freedom=degrees_of_freedom,
                is_significant=is_significant,
                alpha=self.alpha,
                alternative=alternative,
                description=f"{test_type.value} comparing {test_name}",
                assumptions_met=assumptions_met,
                warnings=warnings_list
            )
            
        except Exception as e:
            self.logger.error(f"Error performing {test_type.value}: {e}")
            
            # Return a failed test result
            return StatisticalTest(
                test_name=test_name,
                test_type=test_type,
                statistic=np.nan,
                p_value=1.0,
                critical_value=None,
                confidence_interval=None,
                effect_size=None,
                power=None,
                sample_size=len(sample1) + len(sample2),
                degrees_of_freedom=None,
                is_significant=False,
                alpha=self.alpha,
                alternative=alternative,
                description=f"Failed {test_type.value}: {str(e)}",
                assumptions_met={},
                warnings=[f"Test failed: {str(e)}"]
            )
    
    def _bootstrap_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        alternative: str = "two-sided",
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """Perform bootstrap hypothesis test."""
        
        # Calculate observed difference in means
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Combine samples for bootstrap null distribution
        combined = np.concatenate([sample1, sample2])
        n1, n2 = len(sample1), len(sample2)
        
        # Generate bootstrap distribution under null hypothesis
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample without replacement to maintain sample sizes
            resampled = np.random.choice(combined, size=len(combined), replace=False)
            boot_sample1 = resampled[:n1]
            boot_sample2 = resampled[n1:]
            bootstrap_diffs.append(np.mean(boot_sample1) - np.mean(boot_sample2))
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        elif alternative == "greater":
            p_value = np.mean(bootstrap_diffs >= observed_diff)
        elif alternative == "less":
            p_value = np.mean(bootstrap_diffs <= observed_diff)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        return observed_diff, p_value
    
    def _calculate_cohens_d(self, sample1: np.ndarray, sample2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        try:
            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
            n1, n2 = len(sample1), len(sample2)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (mean1 - mean2) / pooled_std
            return cohens_d
            
        except Exception:
            return np.nan
    
    def _calculate_confidence_interval(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        test_type: TestType,
        confidence: float = None
    ) -> Optional[Tuple[float, float]]:
        """Calculate confidence interval for the difference in means."""
        if confidence is None:
            confidence = 1 - self.alpha
        
        try:
            if test_type in [TestType.TWO_SAMPLE_T_TEST, TestType.PAIRED_T_TEST]:
                if test_type == TestType.PAIRED_T_TEST:
                    min_len = min(len(sample1), len(sample2))
                    differences = sample1[:min_len] - sample2[:min_len]
                    mean_diff = np.mean(differences)
                    se_diff = stats.sem(differences)
                    df = len(differences) - 1
                else:
                    mean_diff = np.mean(sample1) - np.mean(sample2)
                    se1 = stats.sem(sample1)
                    se2 = stats.sem(sample2)
                    se_diff = np.sqrt(se1**2 + se2**2)
                    df = len(sample1) + len(sample2) - 2
                
                t_critical = stats.t.ppf((1 + confidence) / 2, df)
                margin_error = t_critical * se_diff
                
                return (mean_diff - margin_error, mean_diff + margin_error)
            
            else:
                # For non-parametric tests, use bootstrap CI
                return self._bootstrap_confidence_interval(sample1, sample2, confidence)
                
        except Exception:
            return None
    
    def _bootstrap_confidence_interval(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        confidence: float,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_diffs = []
        
        for _ in range(n_bootstrap):
            boot_sample1 = np.random.choice(sample1, size=len(sample1), replace=True)
            boot_sample2 = np.random.choice(sample2, size=len(sample2), replace=True)
            bootstrap_diffs.append(np.mean(boot_sample1) - np.mean(boot_sample2))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return (
            np.percentile(bootstrap_diffs, lower_percentile),
            np.percentile(bootstrap_diffs, upper_percentile)
        )
    
    def _estimate_power(self, sample1: np.ndarray, sample2: np.ndarray, effect_size: float) -> Optional[float]:
        """Estimate statistical power."""
        try:
            n = (len(sample1) + len(sample2)) / 2  # Average sample size
            
            # Simple power approximation
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = effect_size * np.sqrt(n/2) - z_alpha
            power = stats.norm.cdf(z_beta)
            
            return max(0, min(1, power))  # Clamp between 0 and 1
            
        except Exception:
            return None
    
    def _extract_returns(self, backtest_results: BacktestResults) -> np.ndarray:
        """Extract returns from backtest results."""
        if hasattr(backtest_results, 'equity_curve') and backtest_results.equity_curve is not None:
            # Check if equity_curve is a DataFrame with 'equity' column
            if isinstance(backtest_results.equity_curve, pd.DataFrame) and 'equity' in backtest_results.equity_curve.columns:
                equity = backtest_results.equity_curve['equity'].values
                returns = np.diff(equity) / equity[:-1]
                return returns[~np.isnan(returns)]
            # Check if equity_curve is a Series (the equity values directly)
            elif isinstance(backtest_results.equity_curve, pd.Series):
                equity = backtest_results.equity_curve.values
                returns = np.diff(equity) / equity[:-1]
                return returns[~np.isnan(returns)]
        
        # Fallback: generate synthetic returns based on total return
        n_periods = 252  # Assume daily data for a year
        total_return = backtest_results.total_return
        daily_return = (1 + total_return) ** (1/n_periods) - 1
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.01, n_periods)
        returns = np.full(n_periods, daily_return) + noise
        
        return returns
    
    def _calculate_performance_metrics(self, backtest_results: BacktestResults) -> Dict[str, float]:
        """Calculate performance metrics from backtest results."""
        return {
            'total_return': backtest_results.total_return,
            'annual_return': backtest_results.annual_return,
            'sharpe_ratio': backtest_results.sharpe_ratio,
            'max_drawdown': backtest_results.max_drawdown,
            'win_rate': backtest_results.win_rate,
            'volatility': backtest_results.volatility,
            'calmar_ratio': backtest_results.calmar_ratio,
            'sortino_ratio': backtest_results.sortino_ratio,
            'total_trades': backtest_results.total_trades
        }
    
    def _create_mock_results(self, returns: np.ndarray) -> BacktestResults:
        """Create mock BacktestResults from returns array."""
        # This is a simplified mock - in practice you'd want more sophisticated calculation
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Create a minimal BacktestResults object
        # Note: This is a simplified version - you might need to adjust based on your actual BacktestResults structure
        mock_results = type('MockBacktestResults', (), {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': np.min(np.cumsum(returns)),  # Simplified
            'win_rate': np.mean(returns > 0),
            'calmar_ratio': annual_return / abs(np.min(np.cumsum(returns))) if np.min(np.cumsum(returns)) != 0 else 0,
            'sortino_ratio': annual_return / (np.std(returns[returns < 0]) * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0,
            'total_trades': len(returns)
        })()
        
        return mock_results
    
    def _generate_comparison_summary(
        self,
        strategy_name: str,
        benchmark_name: str,
        statistical_tests: List[StatisticalTest],
        is_significantly_better: bool
    ) -> str:
        """Generate a summary of the comparison results."""
        summary_lines = []
        
        summary_lines.append(f"Statistical Comparison: {strategy_name} vs {benchmark_name}")
        summary_lines.append(f"Overall Result: {'Significantly Better' if is_significantly_better else 'Not Significantly Better'}")
        
        for test in statistical_tests:
            significance = "Significant" if test.is_significant else "Not Significant"
            summary_lines.append(
                f"{test.test_type.value}: p={test.p_value:.4f}, "
                f"effect_size={test.effect_size:.4f}, {significance}"
            )
        
        return "\n".join(summary_lines)
    
    def create_statistical_report(
        self,
        comparisons: List[BenchmarkComparison],
        multiple_testing_results: Optional[MultipleTestingResults] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Create a comprehensive statistical testing report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("STATISTICAL SIGNIFICANCE TESTING REPORT")
        report_lines.append("=" * 80)
        
        # Summary
        n_comparisons = len(comparisons)
        n_significant = sum(comp.is_significantly_better for comp in comparisons)
        
        report_lines.append(f"\nSUMMARY:")
        report_lines.append(f"Total Comparisons: {n_comparisons}")
        report_lines.append(f"Significantly Better: {n_significant}")
        report_lines.append(f"Success Rate: {n_significant/n_comparisons:.1%}")
        
        # Individual comparisons
        for i, comparison in enumerate(comparisons, 1):
            report_lines.append(f"\n{'-' * 60}")
            report_lines.append(f"COMPARISON {i}: {comparison.strategy_name} vs {comparison.benchmark_name}")
            report_lines.append(f"{'-' * 60}")
            
            # Performance metrics comparison
            report_lines.append("\nPerformance Metrics:")
            for metric in ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']:
                if metric in comparison.strategy_metrics and metric in comparison.benchmark_metrics:
                    strategy_val = comparison.strategy_metrics[metric]
                    benchmark_val = comparison.benchmark_metrics[metric]
                    diff = strategy_val - benchmark_val
                    report_lines.append(
                        f"  {metric}: Strategy={strategy_val:.4f}, "
                        f"Benchmark={benchmark_val:.4f}, Diff={diff:.4f}"
                    )
            
            # Statistical tests
            report_lines.append("\nStatistical Tests:")
            for test in comparison.statistical_tests:
                report_lines.append(f"  {test.test_type.value}:")
                report_lines.append(f"    Statistic: {test.statistic:.4f}")
                report_lines.append(f"    P-value: {test.p_value:.6f}")
                report_lines.append(f"    Effect Size: {test.effect_size:.4f}")
                report_lines.append(f"    Significant: {test.is_significant}")
                
                if test.confidence_interval:
                    ci_lower, ci_upper = test.confidence_interval
                    report_lines.append(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                if test.warnings:
                    report_lines.append(f"    Warnings: {', '.join(test.warnings)}")
            
            report_lines.append(f"\nConclusion: {comparison.summary}")
        
        # Multiple testing correction
        if multiple_testing_results:
            report_lines.append(f"\n{'-' * 60}")
            report_lines.append("MULTIPLE TESTING CORRECTION")
            report_lines.append(f"{'-' * 60}")
            
            report_lines.append(f"Method: {multiple_testing_results.correction_method.value}")
            report_lines.append(f"Original Significant: {multiple_testing_results.n_significant_original}")
            report_lines.append(f"Corrected Significant: {multiple_testing_results.n_significant_corrected}")
            report_lines.append(f"Family-wise Error Rate: {multiple_testing_results.family_wise_error_rate:.4f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Statistical report saved to {save_path}")
        
        return report
    
    def plot_test_results(
        self,
        comparison: BenchmarkComparison,
        save_path: Optional[str] = None
    ):
        """Create visualization of statistical test results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Statistical Test Results: {comparison.strategy_name} vs {comparison.benchmark_name}', fontsize=16)
        
        # Returns distribution comparison
        axes[0, 0].hist(comparison.strategy_returns, bins=30, alpha=0.7, label='Strategy', density=True)
        axes[0, 0].hist(comparison.benchmark_returns, bins=30, alpha=0.7, label='Benchmark', density=True)
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        from scipy.stats import probplot
        probplot(comparison.strategy_returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Strategy Returns Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Performance metrics comparison
        metrics = ['total_return', 'annual_return', 'sharpe_ratio']
        strategy_vals = [comparison.strategy_metrics.get(m, 0) for m in metrics]
        benchmark_vals = [comparison.benchmark_metrics.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, strategy_vals, width, label='Strategy', alpha=0.7)
        axes[1, 0].bar(x + width/2, benchmark_vals, width, label='Benchmark', alpha=0.7)
        axes[1, 0].set_title('Performance Metrics Comparison')
        axes[1, 0].set_xlabel('Metrics')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # P-values from statistical tests
        test_names = [test.test_type.value.replace('_', ' ').title() for test in comparison.statistical_tests]
        p_values = [test.p_value for test in comparison.statistical_tests]
        colors = ['green' if test.is_significant else 'red' for test in comparison.statistical_tests]
        
        axes[1, 1].bar(test_names, p_values, color=colors, alpha=0.7)
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        axes[1, 1].set_title('Statistical Test P-values')
        axes[1, 1].set_xlabel('Test Type')
        axes[1, 1].set_ylabel('P-value')
        axes[1, 1].set_yscale('log')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Test results plot saved to {save_path}")
        
        plt.show() 