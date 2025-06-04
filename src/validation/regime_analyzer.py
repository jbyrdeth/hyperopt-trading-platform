"""
Market Regime Analysis for Trading Strategies

This module provides comprehensive market regime identification and analysis
to evaluate strategy performance across different market conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ..utils.logger import get_logger
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from .data_splitter import DataSplitter
except ImportError:
    from src.utils.logger import get_logger
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.validation.data_splitter import DataSplitter


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRASH = "crash"
    RALLY = "rally"


class RegimeMethod(Enum):
    """Methods for regime identification."""
    TREND_BASED = "trend_based"
    VOLATILITY_BASED = "volatility_based"
    MOMENTUM_BASED = "momentum_based"
    COMBINED = "combined"
    CLUSTERING = "clustering"


@dataclass
class RegimeMetrics:
    """Metrics for a specific market regime."""
    regime: MarketRegime
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Price metrics
    total_return: float
    volatility: float
    max_drawdown: float
    
    # Trend metrics
    trend_strength: float
    trend_direction: float  # 1 for up, -1 for down, 0 for sideways
    
    # Momentum metrics
    momentum_score: float
    rsi_avg: float
    
    # Volume metrics
    volume_trend: float
    volume_volatility: float


@dataclass
class RegimePerformance:
    """Strategy performance within a specific regime."""
    regime: MarketRegime
    regime_metrics: RegimeMetrics
    
    # Strategy performance
    strategy_return: float
    strategy_volatility: float
    strategy_sharpe: float
    strategy_max_drawdown: float
    strategy_win_rate: float
    
    # Benchmark comparison
    benchmark_return: float
    excess_return: float
    
    # Trade statistics
    total_trades: int
    avg_trade_duration: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    
    # Additional metrics
    calmar_ratio: float
    sortino_ratio: float


@dataclass
class RegimeAnalysisResults:
    """Complete regime analysis results."""
    regimes: List[RegimeMetrics]
    regime_performances: List[RegimePerformance]
    
    # Summary statistics
    regime_distribution: Dict[MarketRegime, float]  # Percentage of time in each regime
    best_regime: MarketRegime
    worst_regime: MarketRegime
    
    # Overall assessment
    regime_consistency: float  # How consistent performance is across regimes
    regime_adaptability: float  # How well strategy adapts to regime changes
    
    # Transition analysis
    transition_matrix: pd.DataFrame
    transition_performance: Dict[Tuple[MarketRegime, MarketRegime], float]


class RegimeAnalyzer:
    """
    Comprehensive market regime analysis for trading strategies.
    
    Identifies market regimes and evaluates strategy performance
    across different market conditions.
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        volatility_window: int = 20,
        momentum_window: int = 14,
        trend_threshold: float = 0.02,
        volatility_threshold: float = 0.25,
        initial_capital: float = 100000
    ):
        """
        Initialize the RegimeAnalyzer.
        
        Args:
            lookback_window: Days to look back for regime identification
            volatility_window: Window for volatility calculations
            momentum_window: Window for momentum calculations
            trend_threshold: Threshold for trend classification
            volatility_threshold: Threshold for high volatility regimes
            initial_capital: Starting capital for backtests
        """
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.initial_capital = initial_capital
        
        self.logger = get_logger("regime_analyzer")
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(initial_capital=initial_capital)
    
    def identify_regimes(
        self,
        data: pd.DataFrame,
        method: RegimeMethod = RegimeMethod.COMBINED
    ) -> pd.DataFrame:
        """
        Identify market regimes in the data.
        
        Args:
            data: OHLCV data
            method: Method to use for regime identification
            
        Returns:
            DataFrame with regime labels and metrics
        """
        self.logger.info(f"Identifying market regimes using {method.value} method")
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(data)
        
        # Apply regime identification method
        if method == RegimeMethod.TREND_BASED:
            regimes = self._identify_trend_regimes(indicators)
        elif method == RegimeMethod.VOLATILITY_BASED:
            regimes = self._identify_volatility_regimes(indicators)
        elif method == RegimeMethod.MOMENTUM_BASED:
            regimes = self._identify_momentum_regimes(indicators)
        elif method == RegimeMethod.CLUSTERING:
            regimes = self._identify_clustering_regimes(indicators)
        else:  # COMBINED
            regimes = self._identify_combined_regimes(indicators)
        
        # Add regime metrics
        regime_data = self._calculate_regime_metrics(data, regimes, indicators)
        
        return regime_data
    
    def analyze_strategy_performance(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        regime_method: RegimeMethod = RegimeMethod.COMBINED
    ) -> RegimeAnalysisResults:
        """
        Analyze strategy performance across market regimes.
        
        Args:
            strategy: Strategy to analyze
            data: Historical OHLCV data
            regime_method: Method for regime identification
            
        Returns:
            Complete regime analysis results
        """
        self.logger.info(f"Analyzing {strategy.name} performance across market regimes")
        
        # Identify regimes
        regime_data = self.identify_regimes(data, regime_method)
        
        # Run full strategy backtest
        full_results = self.backtesting_engine.backtest_strategy(strategy, data, "Full Period")
        
        # Analyze performance by regime
        regime_performances = []
        regimes = []
        
        for regime_type in MarketRegime:
            regime_periods = regime_data[regime_data['regime'] == regime_type.value]
            
            if len(regime_periods) == 0:
                continue
            
            # Calculate regime metrics
            regime_metrics = self._aggregate_regime_metrics(regime_periods, regime_type)
            regimes.append(regime_metrics)
            
            # Calculate strategy performance in this regime
            regime_performance = self._calculate_regime_performance(
                strategy, data, regime_periods, regime_metrics, full_results
            )
            regime_performances.append(regime_performance)
        
        # Calculate summary statistics
        regime_distribution = self._calculate_regime_distribution(regime_data)
        best_regime, worst_regime = self._identify_best_worst_regimes(regime_performances)
        
        # Calculate consistency and adaptability scores
        regime_consistency = self._calculate_regime_consistency(regime_performances)
        regime_adaptability = self._calculate_regime_adaptability(regime_performances)
        
        # Analyze regime transitions
        transition_matrix = self._calculate_transition_matrix(regime_data)
        transition_performance = self._analyze_transition_performance(
            strategy, data, regime_data
        )
        
        return RegimeAnalysisResults(
            regimes=regimes,
            regime_performances=regime_performances,
            regime_distribution=regime_distribution,
            best_regime=best_regime,
            worst_regime=worst_regime,
            regime_consistency=regime_consistency,
            regime_adaptability=regime_adaptability,
            transition_matrix=transition_matrix,
            transition_performance=transition_performance
        )
    
    def simulate_extreme_conditions(
        self,
        strategy: BaseStrategy,
        base_data: pd.DataFrame,
        scenarios: List[str] = None
    ) -> Dict[str, RegimePerformance]:
        """
        Simulate strategy performance under extreme market conditions.
        
        Args:
            strategy: Strategy to test
            base_data: Base historical data
            scenarios: List of scenarios to simulate
            
        Returns:
            Performance results for each scenario
        """
        if scenarios is None:
            scenarios = ['flash_crash', 'sudden_rally', 'high_volatility', 'trending_market']
        
        self.logger.info(f"Simulating extreme market conditions: {scenarios}")
        
        results = {}
        
        for scenario in scenarios:
            # Generate scenario data
            scenario_data = self._generate_scenario_data(base_data, scenario)
            
            # Run backtest
            scenario_results = self.backtesting_engine.backtest_strategy(
                strategy, scenario_data, f"Scenario_{scenario}"
            )
            
            # Create regime metrics for the scenario
            regime_metrics = RegimeMetrics(
                regime=MarketRegime.CRASH if 'crash' in scenario else MarketRegime.RALLY,
                start_date=scenario_data.index[0],
                end_date=scenario_data.index[-1],
                duration_days=len(scenario_data),
                total_return=(scenario_data['close'].iloc[-1] / scenario_data['close'].iloc[0]) - 1,
                volatility=scenario_data['close'].pct_change().std() * np.sqrt(252),
                max_drawdown=self._calculate_max_drawdown(scenario_data['close']),
                trend_strength=abs(scenario_data['close'].iloc[-1] / scenario_data['close'].iloc[0] - 1),
                trend_direction=1 if scenario_data['close'].iloc[-1] > scenario_data['close'].iloc[0] else -1,
                momentum_score=0.0,  # Simplified for scenarios
                rsi_avg=50.0,  # Simplified for scenarios
                volume_trend=0.0,
                volume_volatility=0.0
            )
            
            # Calculate performance metrics
            performance = RegimePerformance(
                regime=regime_metrics.regime,
                regime_metrics=regime_metrics,
                strategy_return=scenario_results.total_return,
                strategy_volatility=scenario_results.volatility,
                strategy_sharpe=scenario_results.sharpe_ratio,
                strategy_max_drawdown=scenario_results.max_drawdown,
                strategy_win_rate=scenario_results.win_rate,
                benchmark_return=regime_metrics.total_return,
                excess_return=scenario_results.total_return - regime_metrics.total_return,
                total_trades=scenario_results.total_trades,
                avg_trade_duration=0.0,  # Simplified
                var_95=0.0,  # Would need more sophisticated calculation
                cvar_95=0.0,  # Would need more sophisticated calculation
                calmar_ratio=scenario_results.calmar_ratio,
                sortino_ratio=scenario_results.sortino_ratio
            )
            
            results[scenario] = performance
        
        return results
    
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for regime identification."""
        indicators = pd.DataFrame(index=data.index)
        
        # Price-based indicators
        indicators['returns'] = data['close'].pct_change()
        indicators['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        indicators['sma_20'] = data['close'].rolling(20).mean()
        indicators['sma_50'] = data['close'].rolling(50).mean()
        indicators['sma_200'] = data['close'].rolling(200).mean()
        
        # Trend indicators
        indicators['trend_20'] = (data['close'] / indicators['sma_20'] - 1)
        indicators['trend_50'] = (data['close'] / indicators['sma_50'] - 1)
        indicators['trend_200'] = (data['close'] / indicators['sma_200'] - 1)
        
        # Volatility indicators
        indicators['volatility'] = indicators['returns'].rolling(self.volatility_window).std() * np.sqrt(252)
        indicators['atr'] = self._calculate_atr(data, self.volatility_window)
        
        # Momentum indicators
        indicators['rsi'] = self._calculate_rsi(data['close'], self.momentum_window)
        indicators['momentum'] = data['close'] / data['close'].shift(self.momentum_window) - 1
        
        # Volume indicators
        if 'volume' in data.columns:
            indicators['volume_sma'] = data['volume'].rolling(20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
        
        return indicators.dropna()
    
    def _calculate_atr(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window).mean()
        
        # Fill NaN values with 0 for the first few periods
        return atr.fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _identify_trend_regimes(self, indicators: pd.DataFrame) -> pd.Series:
        """Identify regimes based on trend analysis."""
        regimes = pd.Series(index=indicators.index, dtype='object')
        
        # Define conditions
        bull_condition = (
            (indicators['trend_20'] > self.trend_threshold) &
            (indicators['trend_50'] > 0) &
            (indicators['sma_20'] > indicators['sma_50'])
        )
        
        bear_condition = (
            (indicators['trend_20'] < -self.trend_threshold) &
            (indicators['trend_50'] < 0) &
            (indicators['sma_20'] < indicators['sma_50'])
        )
        
        # Assign regimes
        regimes[bull_condition] = MarketRegime.BULL.value
        regimes[bear_condition] = MarketRegime.BEAR.value
        regimes[~(bull_condition | bear_condition)] = MarketRegime.SIDEWAYS.value
        
        return regimes
    
    def _identify_volatility_regimes(self, indicators: pd.DataFrame) -> pd.Series:
        """Identify regimes based on volatility clustering."""
        regimes = pd.Series(index=indicators.index, dtype='object')
        
        # Calculate volatility percentiles
        vol_75 = indicators['volatility'].quantile(0.75)
        vol_25 = indicators['volatility'].quantile(0.25)
        
        high_vol_condition = indicators['volatility'] > vol_75
        low_vol_condition = indicators['volatility'] < vol_25
        
        # Combine with trend for more nuanced classification
        positive_trend = indicators['trend_20'] > 0
        
        regimes[high_vol_condition & positive_trend] = MarketRegime.RALLY.value
        regimes[high_vol_condition & ~positive_trend] = MarketRegime.CRASH.value
        regimes[low_vol_condition] = MarketRegime.SIDEWAYS.value
        regimes[~(high_vol_condition | low_vol_condition)] = MarketRegime.VOLATILE.value
        
        return regimes
    
    def _identify_momentum_regimes(self, indicators: pd.DataFrame) -> pd.Series:
        """Identify regimes based on momentum indicators."""
        regimes = pd.Series(index=indicators.index, dtype='object')
        
        # RSI-based classification
        overbought = indicators['rsi'] > 70
        oversold = indicators['rsi'] < 30
        
        # Momentum-based classification
        strong_momentum = indicators['momentum'] > 0.1
        weak_momentum = indicators['momentum'] < -0.1
        
        regimes[strong_momentum & ~overbought] = MarketRegime.BULL.value
        regimes[weak_momentum & ~oversold] = MarketRegime.BEAR.value
        regimes[overbought | oversold] = MarketRegime.VOLATILE.value
        regimes[~(strong_momentum | weak_momentum | overbought | oversold)] = MarketRegime.SIDEWAYS.value
        
        return regimes
    
    def _identify_clustering_regimes(self, indicators: pd.DataFrame) -> pd.Series:
        """Identify regimes using machine learning clustering."""
        # Select features for clustering
        features = ['trend_20', 'volatility', 'rsi', 'momentum']
        feature_data = indicators[features].dropna()
        
        if len(feature_data) < 100:  # Not enough data for clustering
            return self._identify_combined_regimes(indicators)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Perform K-means clustering
        n_clusters = min(4, len(feature_data) // 50)  # Adaptive number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Map clusters to regime types based on cluster centers
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        regime_mapping = {}
        
        for i, center in enumerate(cluster_centers):
            trend, vol, rsi, momentum = center
            
            if trend > 0.02 and vol < 0.3:
                regime_mapping[i] = MarketRegime.BULL.value
            elif trend < -0.02 and vol < 0.3:
                regime_mapping[i] = MarketRegime.BEAR.value
            elif vol > 0.4:
                regime_mapping[i] = MarketRegime.VOLATILE.value
            else:
                regime_mapping[i] = MarketRegime.SIDEWAYS.value
        
        # Create regime series
        regimes = pd.Series(index=feature_data.index, dtype='object')
        for i, cluster in enumerate(clusters):
            regimes.iloc[i] = regime_mapping[cluster]
        
        # Fill missing values
        regimes = regimes.reindex(indicators.index).fillna(MarketRegime.SIDEWAYS.value)
        
        return regimes
    
    def _identify_combined_regimes(self, indicators: pd.DataFrame) -> pd.Series:
        """Identify regimes using combined approach."""
        # Get individual regime classifications
        trend_regimes = self._identify_trend_regimes(indicators)
        vol_regimes = self._identify_volatility_regimes(indicators)
        momentum_regimes = self._identify_momentum_regimes(indicators)
        
        # Combine using majority voting
        regimes = pd.Series(index=indicators.index, dtype='object')
        
        for idx in indicators.index:
            regime_votes = [
                trend_regimes.loc[idx],
                vol_regimes.loc[idx],
                momentum_regimes.loc[idx]
            ]
            
            # Count votes
            vote_counts = {}
            for vote in regime_votes:
                vote_counts[vote] = vote_counts.get(vote, 0) + 1
            
            # Select majority vote
            regimes.loc[idx] = max(vote_counts, key=vote_counts.get)
        
        return regimes
    
    def _calculate_regime_metrics(
        self,
        data: pd.DataFrame,
        regimes: pd.Series,
        indicators: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate metrics for each regime period."""
        regime_data = pd.DataFrame(index=data.index)
        regime_data['regime'] = regimes
        regime_data['price'] = data['close']
        regime_data['returns'] = indicators['returns']
        regime_data['volatility'] = indicators['volatility']
        regime_data['trend_strength'] = abs(indicators['trend_20'])
        regime_data['momentum'] = indicators['momentum']
        regime_data['rsi'] = indicators['rsi']
        
        return regime_data.dropna()
    
    def _aggregate_regime_metrics(
        self,
        regime_periods: pd.DataFrame,
        regime_type: MarketRegime
    ) -> RegimeMetrics:
        """Aggregate metrics for a specific regime type."""
        if len(regime_periods) == 0:
            # Return default metrics for empty regime
            return RegimeMetrics(
                regime=regime_type,
                start_date=datetime.now(),
                end_date=datetime.now(),
                duration_days=0,
                total_return=0.0,
                volatility=0.0,
                max_drawdown=0.0,
                trend_strength=0.0,
                trend_direction=0.0,
                momentum_score=0.0,
                rsi_avg=50.0,
                volume_trend=0.0,
                volume_volatility=0.0
            )
        
        start_date = regime_periods.index[0]
        end_date = regime_periods.index[-1]
        duration_days = (end_date - start_date).days
        
        # Calculate price metrics
        start_price = regime_periods['price'].iloc[0]
        end_price = regime_periods['price'].iloc[-1]
        total_return = (end_price / start_price) - 1
        
        # Calculate drawdown
        cumulative_returns = (1 + regime_periods['returns']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        return RegimeMetrics(
            regime=regime_type,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            total_return=total_return,
            volatility=regime_periods['volatility'].mean(),
            max_drawdown=max_drawdown,
            trend_strength=regime_periods['trend_strength'].mean(),
            trend_direction=1 if total_return > 0 else -1,
            momentum_score=regime_periods['momentum'].mean(),
            rsi_avg=regime_periods['rsi'].mean(),
            volume_trend=0.0,  # Simplified
            volume_volatility=0.0  # Simplified
        )
    
    def _calculate_regime_performance(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        regime_periods: pd.DataFrame,
        regime_metrics: RegimeMetrics,
        full_results: BacktestResults
    ) -> RegimePerformance:
        """Calculate strategy performance within a specific regime."""
        # Extract regime data
        regime_start = regime_periods.index[0]
        regime_end = regime_periods.index[-1]
        regime_data = data.loc[regime_start:regime_end]
        
        if len(regime_data) < 10:  # Not enough data for meaningful backtest
            return self._create_empty_regime_performance(regime_metrics)
        
        # Run backtest for this regime period
        try:
            regime_results = self.backtesting_engine.backtest_strategy(
                strategy, regime_data, f"Regime_{regime_metrics.regime.value}"
            )
            
            return RegimePerformance(
                regime=regime_metrics.regime,
                regime_metrics=regime_metrics,
                strategy_return=regime_results.total_return,
                strategy_volatility=regime_results.volatility,
                strategy_sharpe=regime_results.sharpe_ratio,
                strategy_max_drawdown=regime_results.max_drawdown,
                strategy_win_rate=regime_results.win_rate,
                benchmark_return=regime_metrics.total_return,
                excess_return=regime_results.total_return - regime_metrics.total_return,
                total_trades=regime_results.total_trades,
                avg_trade_duration=0.0,  # Simplified
                var_95=0.0,  # Would need more sophisticated calculation
                cvar_95=0.0,  # Would need more sophisticated calculation
                calmar_ratio=regime_results.calmar_ratio,
                sortino_ratio=regime_results.sortino_ratio
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to backtest regime {regime_metrics.regime.value}: {e}")
            return self._create_empty_regime_performance(regime_metrics)
    
    def _create_empty_regime_performance(self, regime_metrics: RegimeMetrics) -> RegimePerformance:
        """Create empty performance metrics for regimes with insufficient data."""
        return RegimePerformance(
            regime=regime_metrics.regime,
            regime_metrics=regime_metrics,
            strategy_return=0.0,
            strategy_volatility=0.0,
            strategy_sharpe=0.0,
            strategy_max_drawdown=0.0,
            strategy_win_rate=0.0,
            benchmark_return=regime_metrics.total_return,
            excess_return=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            var_95=0.0,
            cvar_95=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0
        )
    
    def _calculate_regime_distribution(self, regime_data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Calculate the percentage of time spent in each regime."""
        if len(regime_data) == 0:
            # Return empty distribution if no data
            return {regime: 0.0 for regime in MarketRegime}
        
        regime_counts = regime_data['regime'].value_counts()
        total_periods = len(regime_data)
        
        distribution = {}
        for regime in MarketRegime:
            count = regime_counts.get(regime.value, 0)
            distribution[regime] = (count / total_periods) * 100
        
        return distribution
    
    def _identify_best_worst_regimes(
        self,
        regime_performances: List[RegimePerformance]
    ) -> Tuple[MarketRegime, MarketRegime]:
        """Identify the best and worst performing regimes."""
        if not regime_performances:
            return MarketRegime.SIDEWAYS, MarketRegime.SIDEWAYS
        
        # Sort by strategy return
        sorted_performances = sorted(
            regime_performances,
            key=lambda x: x.strategy_return,
            reverse=True
        )
        
        best_regime = sorted_performances[0].regime
        worst_regime = sorted_performances[-1].regime
        
        return best_regime, worst_regime
    
    def _calculate_regime_consistency(self, regime_performances: List[RegimePerformance]) -> float:
        """Calculate how consistent performance is across regimes."""
        if len(regime_performances) < 2:
            return 1.0
        
        returns = [perf.strategy_return for perf in regime_performances]
        return 1.0 - (np.std(returns) / (np.mean(np.abs(returns)) + 1e-8))
    
    def _calculate_regime_adaptability(self, regime_performances: List[RegimePerformance]) -> float:
        """Calculate how well strategy adapts to regime changes."""
        if len(regime_performances) < 2:
            return 1.0
        
        # Calculate excess returns (strategy vs benchmark)
        excess_returns = [perf.excess_return for perf in regime_performances]
        positive_excess = sum(1 for er in excess_returns if er > 0)
        
        return positive_excess / len(excess_returns)
    
    def _calculate_transition_matrix(self, regime_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime transition probability matrix."""
        regimes = regime_data['regime'].values
        unique_regimes = list(set(regimes))
        
        # Initialize transition matrix
        transition_matrix = pd.DataFrame(
            0.0,
            index=unique_regimes,
            columns=unique_regimes
        )
        
        # Count transitions
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            transition_matrix.loc[current_regime, next_regime] += 1
        
        # Convert to probabilities
        for regime in unique_regimes:
            row_sum = transition_matrix.loc[regime].sum()
            if row_sum > 0:
                transition_matrix.loc[regime] = transition_matrix.loc[regime] / row_sum
        
        return transition_matrix
    
    def _analyze_transition_performance(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        regime_data: pd.DataFrame
    ) -> Dict[Tuple[MarketRegime, MarketRegime], float]:
        """Analyze strategy performance during regime transitions."""
        transitions = {}
        regimes = regime_data['regime'].values
        
        # Find transition points
        for i in range(len(regimes) - 1):
            current_regime = regimes[i]
            next_regime = regimes[i + 1]
            
            if current_regime != next_regime:
                transition_key = (
                    MarketRegime(current_regime),
                    MarketRegime(next_regime)
                )
                
                # Calculate performance around transition (simplified)
                transition_start = max(0, i - 5)
                transition_end = min(len(regimes), i + 5)
                
                transition_period = data.iloc[transition_start:transition_end]
                if len(transition_period) > 5:
                    try:
                        transition_results = self.backtesting_engine.backtest_strategy(
                            strategy, transition_period, f"Transition_{current_regime}_{next_regime}"
                        )
                        transitions[transition_key] = transition_results.total_return
                    except Exception:
                        transitions[transition_key] = 0.0
        
        return transitions
    
    def _generate_scenario_data(self, base_data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Generate synthetic data for extreme market scenarios."""
        scenario_data = base_data.copy()
        
        if scenario == 'flash_crash':
            # Simulate a flash crash: sudden 20% drop followed by partial recovery
            crash_day = len(scenario_data) // 2
            crash_factor = 0.8  # 20% drop
            recovery_factor = 1.1  # 10% recovery
            
            scenario_data.loc[scenario_data.index[crash_day]:, 'close'] *= crash_factor
            scenario_data.loc[scenario_data.index[crash_day+1]:, 'close'] *= recovery_factor
            
        elif scenario == 'sudden_rally':
            # Simulate sudden rally: 30% gain over short period
            rally_start = len(scenario_data) // 3
            rally_end = rally_start + 10
            rally_factor = 1.3
            
            daily_factor = rally_factor ** (1/10)
            for i in range(rally_start, min(rally_end, len(scenario_data))):
                scenario_data.loc[scenario_data.index[i]:, 'close'] *= daily_factor
                
        elif scenario == 'high_volatility':
            # Simulate high volatility: increase daily volatility by 3x
            returns = scenario_data['close'].pct_change().dropna()
            high_vol_returns = returns * 3
            
            # Reconstruct prices
            new_prices = [scenario_data['close'].iloc[0]]
            for ret in high_vol_returns:
                new_prices.append(new_prices[-1] * (1 + ret))
            
            scenario_data['close'] = new_prices[:len(scenario_data)]
            
        elif scenario == 'trending_market':
            # Simulate strong trending market: consistent 1% daily gains
            daily_return = 0.01
            for i in range(1, len(scenario_data)):
                scenario_data.loc[scenario_data.index[i], 'close'] = (
                    scenario_data.loc[scenario_data.index[i-1], 'close'] * (1 + daily_return)
                )
        
        # Update OHLV based on new close prices
        scenario_data['open'] = scenario_data['close'].shift(1).fillna(scenario_data['close'].iloc[0])
        scenario_data['high'] = scenario_data[['open', 'close']].max(axis=1) * 1.01
        scenario_data['low'] = scenario_data[['open', 'close']].min(axis=1) * 0.99
        
        return scenario_data
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max) - 1
        return drawdown.min()
    
    def create_regime_report(
        self,
        results: RegimeAnalysisResults,
        strategy_name: str,
        save_path: Optional[str] = None
    ) -> str:
        """Create a comprehensive regime analysis report."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"MARKET REGIME ANALYSIS REPORT - {strategy_name}")
        report_lines.append("=" * 80)
        
        # Summary
        report_lines.append(f"\nREGIME DISTRIBUTION:")
        for regime, percentage in results.regime_distribution.items():
            report_lines.append(f"  {regime.value.title()}: {percentage:.1f}%")
        
        report_lines.append(f"\nOVERALL ASSESSMENT:")
        report_lines.append(f"  Best Regime: {results.best_regime.value.title()}")
        report_lines.append(f"  Worst Regime: {results.worst_regime.value.title()}")
        report_lines.append(f"  Regime Consistency: {results.regime_consistency:.3f}")
        report_lines.append(f"  Regime Adaptability: {results.regime_adaptability:.3f}")
        
        # Detailed performance by regime
        for performance in results.regime_performances:
            report_lines.append(f"\n{'-' * 60}")
            report_lines.append(f"REGIME: {performance.regime.value.upper()}")
            report_lines.append(f"{'-' * 60}")
            
            report_lines.append(f"Strategy Performance:")
            report_lines.append(f"  Total Return: {performance.strategy_return:.2%}")
            report_lines.append(f"  Volatility: {performance.strategy_volatility:.2%}")
            report_lines.append(f"  Sharpe Ratio: {performance.strategy_sharpe:.3f}")
            report_lines.append(f"  Max Drawdown: {performance.strategy_max_drawdown:.2%}")
            report_lines.append(f"  Win Rate: {performance.strategy_win_rate:.1%}")
            
            report_lines.append(f"Market Conditions:")
            report_lines.append(f"  Market Return: {performance.benchmark_return:.2%}")
            report_lines.append(f"  Excess Return: {performance.excess_return:.2%}")
            report_lines.append(f"  Duration: {performance.regime_metrics.duration_days} days")
            
            report_lines.append(f"Risk Metrics:")
            report_lines.append(f"  Calmar Ratio: {performance.calmar_ratio:.3f}")
            report_lines.append(f"  Sortino Ratio: {performance.sortino_ratio:.3f}")
            report_lines.append(f"  Total Trades: {performance.total_trades}")
        
        # Transition analysis
        report_lines.append(f"\n{'-' * 60}")
        report_lines.append("REGIME TRANSITION ANALYSIS")
        report_lines.append(f"{'-' * 60}")
        
        report_lines.append("\nTransition Matrix:")
        report_lines.append(str(results.transition_matrix.round(3)))
        
        if results.transition_performance:
            report_lines.append("\nTransition Performance:")
            for (from_regime, to_regime), performance in results.transition_performance.items():
                report_lines.append(
                    f"  {from_regime.value} → {to_regime.value}: {performance:.2%}"
                )
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Regime analysis report saved to {save_path}")
        
        return report
    
    def plot_regime_analysis(
        self,
        data: pd.DataFrame,
        regime_data: pd.DataFrame,
        results: RegimeAnalysisResults,
        save_path: Optional[str] = None
    ):
        """Create comprehensive visualization of regime analysis."""
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Market Regime Analysis', fontsize=16)
        
        # Price chart with regime coloring
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['close'], linewidth=1, alpha=0.7)
        
        # Color background by regime
        regime_colors = {
            'bull': 'green',
            'bear': 'red',
            'sideways': 'gray',
            'volatile': 'orange',
            'crash': 'darkred',
            'rally': 'darkgreen'
        }
        
        current_regime = None
        start_idx = None
        
        for i, (idx, row) in enumerate(regime_data.iterrows()):
            if row['regime'] != current_regime:
                if current_regime is not None and start_idx is not None:
                    ax1.axvspan(
                        regime_data.index[start_idx], idx,
                        alpha=0.2,
                        color=regime_colors.get(current_regime, 'gray'),
                        label=current_regime if current_regime not in ax1.get_legend_handles_labels()[1] else ""
                    )
                current_regime = row['regime']
                start_idx = i
        
        ax1.set_title('Price Chart with Market Regimes')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regime distribution pie chart
        ax2 = axes[0, 1]
        regime_percentages = [results.regime_distribution[regime] for regime in MarketRegime if results.regime_distribution[regime] > 0]
        regime_labels = [regime.value.title() for regime in MarketRegime if results.regime_distribution[regime] > 0]
        colors = [regime_colors.get(regime.value, 'gray') for regime in MarketRegime if results.regime_distribution[regime] > 0]
        
        ax2.pie(regime_percentages, labels=regime_labels, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Regime Distribution')
        
        # Performance by regime
        ax3 = axes[1, 0]
        regimes = [perf.regime.value.title() for perf in results.regime_performances]
        strategy_returns = [perf.strategy_return * 100 for perf in results.regime_performances]
        benchmark_returns = [perf.benchmark_return * 100 for perf in results.regime_performances]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        ax3.bar(x - width/2, strategy_returns, width, label='Strategy', alpha=0.7)
        ax3.bar(x + width/2, benchmark_returns, width, label='Benchmark', alpha=0.7)
        ax3.set_title('Returns by Regime')
        ax3.set_ylabel('Return (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(regimes, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Sharpe ratios by regime
        ax4 = axes[1, 1]
        sharpe_ratios = [perf.strategy_sharpe for perf in results.regime_performances]
        colors_list = [regime_colors.get(perf.regime.value, 'gray') for perf in results.regime_performances]
        
        ax4.bar(regimes, sharpe_ratios, color=colors_list, alpha=0.7)
        ax4.set_title('Sharpe Ratio by Regime')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Transition matrix heatmap
        ax5 = axes[2, 0]
        sns.heatmap(results.transition_matrix, annot=True, cmap='Blues', ax=ax5, fmt='.2f')
        ax5.set_title('Regime Transition Matrix')
        ax5.set_xlabel('To Regime')
        ax5.set_ylabel('From Regime')
        
        # Risk-return scatter
        ax6 = axes[2, 1]
        returns = [perf.strategy_return * 100 for perf in results.regime_performances]
        volatilities = [perf.strategy_volatility * 100 for perf in results.regime_performances]
        
        scatter = ax6.scatter(volatilities, returns, c=colors_list, s=100, alpha=0.7)
        
        for i, regime in enumerate(regimes):
            ax6.annotate(regime, (volatilities[i], returns[i]), xytext=(5, 5), textcoords='offset points')
        
        ax6.set_title('Risk-Return by Regime')
        ax6.set_xlabel('Volatility (%)')
        ax6.set_ylabel('Return (%)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Regime analysis plot saved to {save_path}")
        
        plt.show() 