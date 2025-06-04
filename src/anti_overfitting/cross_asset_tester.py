"""
Cross-Asset Testing Module

This module provides functionality to test trading strategies across multiple
correlated assets to identify genuine market inefficiencies and detect overfitting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import asyncio
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from ..data.data_fetcher import DataFetcher, FetchResult
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from ..utils.logger import get_logger
    from .data_splitting_framework import AntiOverfittingDataSplitter, OverfittingDetector
except ImportError:
    from src.data.data_fetcher import DataFetcher, FetchResult
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.utils.logger import get_logger
    from src.anti_overfitting.data_splitting_framework import AntiOverfittingDataSplitter, OverfittingDetector


@dataclass
class AssetCorrelation:
    """Asset correlation analysis results."""
    asset1: str
    asset2: str
    price_correlation: float
    return_correlation: float
    volatility_correlation: float
    correlation_stability: float  # How stable correlation is over time
    correlation_strength: str  # "Strong", "Moderate", "Weak"


@dataclass
class CrossAssetPerformance:
    """Performance metrics across multiple assets."""
    asset: str
    backtest_results: BacktestResults
    performance_rank: int
    consistency_score: float  # How consistent performance is vs other assets
    relative_performance: float  # Performance relative to asset group average
    overfitting_risk: str  # "Low", "Medium", "High"
    
    # Detailed metrics
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Cross-asset specific metrics
    correlation_adjusted_return: float
    diversification_benefit: float


@dataclass
class CrossAssetTestResult:
    """Results of cross-asset testing."""
    strategy_name: str
    test_period: Tuple[datetime, datetime]
    assets_tested: List[str]
    
    # Performance across assets
    asset_performances: List[CrossAssetPerformance]
    
    # Correlation analysis
    asset_correlations: List[AssetCorrelation]
    
    # Consistency metrics
    performance_consistency: float  # 0-1, higher = more consistent
    rank_consistency: float  # How consistent rankings are across assets
    risk_adjusted_consistency: float
    
    # Overfitting indicators
    performance_variance: float  # Variance in performance across assets
    outlier_assets: List[str]  # Assets with significantly different performance
    overfitting_probability: float  # 0-1, probability of overfitting
    
    # Statistical analysis
    performance_correlation_with_asset_correlation: float
    statistical_significance: float  # p-value for performance differences
    
    # Summary
    overall_consistency_grade: str  # "A", "B", "C", "D", "F"
    deployment_recommendation: str  # "Recommended", "Caution", "Not Recommended"
    warning_flags: List[str]
    recommendations: List[str]


class AssetGroupManager:
    """Manages groups of correlated assets for testing."""
    
    def __init__(self):
        self.logger = get_logger("asset_group_manager")
        
        # Predefined asset groups
        self.asset_groups = {
            'major_crypto': [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT'
            ],
            'defi_tokens': [
                'UNI/USDT', 'AAVE/USDT', 'COMP/USDT', 'MKR/USDT', 'SNX/USDT',
                'CRV/USDT', 'YFI/USDT', 'SUSHI/USDT', '1INCH/USDT', 'BAL/USDT'
            ],
            'layer1_blockchains': [
                'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 'DOT/USDT',
                'AVAX/USDT', 'ALGO/USDT', 'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT'
            ],
            'meme_coins': [
                'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT'
            ],
            'exchange_tokens': [
                'BNB/USDT', 'FTT/USDT', 'CRO/USDT', 'HT/USDT', 'OKB/USDT'
            ]
        }
    
    def get_asset_group(self, group_name: str) -> List[str]:
        """Get assets in a specific group."""
        return self.asset_groups.get(group_name, [])
    
    def get_all_groups(self) -> Dict[str, List[str]]:
        """Get all asset groups."""
        return self.asset_groups.copy()
    
    def add_custom_group(self, group_name: str, assets: List[str]):
        """Add a custom asset group."""
        self.asset_groups[group_name] = assets
        self.logger.info(f"Added custom asset group '{group_name}' with {len(assets)} assets")
    
    def find_correlated_assets(
        self,
        base_asset: str,
        data_fetcher: DataFetcher,
        min_correlation: float = 0.5,
        lookback_days: int = 90
    ) -> List[str]:
        """
        Find assets correlated with a base asset.
        
        Args:
            base_asset: Base asset to find correlations for
            data_fetcher: Data fetcher instance
            min_correlation: Minimum correlation threshold
            lookback_days: Lookback period for correlation calculation
            
        Returns:
            List of correlated assets
        """
        # This would require implementing correlation analysis
        # For now, return assets from the same group
        for group_name, assets in self.asset_groups.items():
            if base_asset in assets:
                return [asset for asset in assets if asset != base_asset]
        
        return []


class CrossAssetTester:
    """
    Cross-asset testing engine for strategy validation.
    
    Tests strategies across multiple correlated assets to identify
    genuine market inefficiencies and detect overfitting.
    """
    
    def __init__(
        self,
        data_fetcher: DataFetcher,
        backtesting_engine: BacktestingEngine,
        min_correlation: float = 0.3,
        max_assets_per_test: int = 10,
        min_test_period_days: int = 252,
        significance_level: float = 0.05
    ):
        """
        Initialize the CrossAssetTester.
        
        Args:
            data_fetcher: Data fetcher for loading asset data
            backtesting_engine: Backtesting engine for strategy testing
            min_correlation: Minimum correlation for asset inclusion
            max_assets_per_test: Maximum number of assets to test
            min_test_period_days: Minimum test period in days
            significance_level: Statistical significance level
        """
        self.data_fetcher = data_fetcher
        self.backtesting_engine = backtesting_engine
        self.min_correlation = min_correlation
        self.max_assets_per_test = max_assets_per_test
        self.min_test_period_days = min_test_period_days
        self.significance_level = significance_level
        
        self.asset_manager = AssetGroupManager()
        self.overfitting_detector = OverfittingDetector()
        self.logger = get_logger("cross_asset_tester")
        
        # Test results cache
        self.test_results_cache: Dict[str, CrossAssetTestResult] = {}
    
    async def test_strategy_across_assets(
        self,
        strategy: BaseStrategy,
        asset_group: Union[str, List[str]],
        timeframe: str = '1h',
        test_period_days: int = 365,
        use_walk_forward: bool = True,
        correlation_analysis: bool = True
    ) -> CrossAssetTestResult:
        """
        Test a strategy across multiple correlated assets.
        
        Args:
            strategy: Strategy to test
            asset_group: Asset group name or list of assets
            timeframe: Trading timeframe
            test_period_days: Test period in days
            use_walk_forward: Whether to use walk-forward analysis
            correlation_analysis: Whether to perform correlation analysis
            
        Returns:
            CrossAssetTestResult with comprehensive analysis
        """
        self.logger.info(f"Starting cross-asset test for {strategy.name}")
        
        # Get asset list
        if isinstance(asset_group, str):
            assets = self.asset_manager.get_asset_group(asset_group)
            if not assets:
                raise ValueError(f"Unknown asset group: {asset_group}")
        else:
            assets = asset_group
        
        # Limit number of assets
        if len(assets) > self.max_assets_per_test:
            assets = assets[:self.max_assets_per_test]
            self.logger.warning(f"Limited test to {self.max_assets_per_test} assets")
        
        # Set up test period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_period_days)
        
        # Fetch data for all assets
        self.logger.info(f"Fetching data for {len(assets)} assets")
        asset_data = await self.data_fetcher.fetch_multiple_assets(
            symbols=assets,
            timeframes=[timeframe],
            start_date=start_date,
            end_date=end_date
        )
        
        # Filter successful fetches
        valid_assets = []
        valid_data = {}
        for asset in assets:
            if (asset in asset_data and 
                timeframe in asset_data[asset] and 
                asset_data[asset][timeframe].success and
                len(asset_data[asset][timeframe].data) >= self.min_test_period_days):
                valid_assets.append(asset)
                valid_data[asset] = asset_data[asset][timeframe].data
        
        if len(valid_assets) < 2:
            raise ValueError(f"Insufficient valid assets for testing: {len(valid_assets)}")
        
        self.logger.info(f"Testing on {len(valid_assets)} valid assets: {valid_assets}")
        
        # Perform correlation analysis if requested
        correlations = []
        if correlation_analysis:
            correlations = await self._analyze_asset_correlations(valid_data)
        
        # Test strategy on each asset
        asset_performances = []
        
        for i, asset in enumerate(valid_assets):
            self.logger.info(f"Testing {strategy.name} on {asset} ({i+1}/{len(valid_assets)})")
            
            try:
                # Create strategy copy for this asset
                strategy_copy = self._create_strategy_copy(strategy)
                
                # Run backtest
                backtest_result = await self._run_asset_backtest(
                    strategy_copy, asset, valid_data[asset], use_walk_forward
                )
                
                # Calculate cross-asset specific metrics
                performance = self._create_cross_asset_performance(
                    asset, backtest_result, i + 1
                )
                
                asset_performances.append(performance)
                
            except Exception as e:
                self.logger.error(f"Failed to test {asset}: {e}")
                continue
        
        if not asset_performances:
            raise ValueError("No successful asset tests completed")
        
        # Analyze consistency and overfitting
        consistency_metrics = self._analyze_performance_consistency(asset_performances)
        overfitting_analysis = self._analyze_overfitting_indicators(
            asset_performances, correlations
        )
        
        # Create comprehensive result
        result = CrossAssetTestResult(
            strategy_name=strategy.name,
            test_period=(start_date, end_date),
            assets_tested=valid_assets,
            asset_performances=asset_performances,
            asset_correlations=correlations,
            **consistency_metrics,
            **overfitting_analysis
        )
        
        # Cache result
        cache_key = f"{strategy.name}_{asset_group}_{timeframe}_{test_period_days}"
        self.test_results_cache[cache_key] = result
        
        self.logger.info(f"Cross-asset test completed for {strategy.name}")
        return result
    
    async def _analyze_asset_correlations(
        self,
        asset_data: Dict[str, pd.DataFrame]
    ) -> List[AssetCorrelation]:
        """Analyze correlations between assets."""
        correlations = []
        assets = list(asset_data.keys())
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                # Align data by timestamp
                data1 = asset_data[asset1]['close']
                data2 = asset_data[asset2]['close']
                
                # Find common timestamps
                common_index = data1.index.intersection(data2.index)
                if len(common_index) < 30:  # Need minimum data points
                    continue
                
                aligned_data1 = data1[common_index]
                aligned_data2 = data2[common_index]
                
                # Calculate correlations
                price_corr, _ = pearsonr(aligned_data1, aligned_data2)
                
                # Return correlations
                returns1 = aligned_data1.pct_change().dropna()
                returns2 = aligned_data2.pct_change().dropna()
                common_returns = returns1.index.intersection(returns2.index)
                
                if len(common_returns) > 10:
                    return_corr, _ = pearsonr(returns1[common_returns], returns2[common_returns])
                else:
                    return_corr = 0.0
                
                # Volatility correlation
                vol1 = returns1.rolling(30).std()
                vol2 = returns2.rolling(30).std()
                common_vol = vol1.index.intersection(vol2.index)
                
                if len(common_vol) > 10:
                    vol_corr, _ = pearsonr(vol1[common_vol].dropna(), vol2[common_vol].dropna())
                else:
                    vol_corr = 0.0
                
                # Correlation stability (rolling correlation std)
                rolling_corr = returns1.rolling(30).corr(returns2)
                stability = 1.0 - rolling_corr.std() if not rolling_corr.empty else 0.0
                
                # Determine correlation strength
                avg_corr = (abs(price_corr) + abs(return_corr)) / 2
                if avg_corr >= 0.7:
                    strength = "Strong"
                elif avg_corr >= 0.4:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                correlation = AssetCorrelation(
                    asset1=asset1,
                    asset2=asset2,
                    price_correlation=price_corr,
                    return_correlation=return_corr,
                    volatility_correlation=vol_corr,
                    correlation_stability=stability,
                    correlation_strength=strength
                )
                
                correlations.append(correlation)
        
        return correlations
    
    def _create_strategy_copy(self, strategy: BaseStrategy) -> BaseStrategy:
        """Create a copy of the strategy for testing."""
        # This is a simplified implementation
        # In practice, you'd need proper strategy cloning
        strategy_class = type(strategy)
        return strategy_class(
            name=strategy.name,
            parameters=strategy.parameters.copy(),
            risk_params=strategy.risk_params.copy()
        )
    
    async def _run_asset_backtest(
        self,
        strategy: BaseStrategy,
        asset: str,
        data: pd.DataFrame,
        use_walk_forward: bool
    ) -> BacktestResults:
        """Run backtest for a specific asset."""
        
        if use_walk_forward:
            # Use anti-overfitting data splitter for walk-forward analysis
            splitter = AntiOverfittingDataSplitter()
            splits = splitter.create_walk_forward_splits(
                data, window_months=6, step_months=1, min_splits=5
            )
            
            # Run backtest on the most recent split
            if splits:
                split = splits[-1]  # Use most recent split
                test_data = split.test
            else:
                test_data = data
        else:
            # Use full dataset
            test_data = data
        
        # Run backtest
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.backtesting_engine.run_backtest,
            strategy,
            test_data,
            10000.0  # Initial capital
        )
        
        return result
    
    def _create_cross_asset_performance(
        self,
        asset: str,
        backtest_result: BacktestResults,
        rank: int
    ) -> CrossAssetPerformance:
        """Create cross-asset performance metrics."""
        
        # Calculate consistency score (placeholder - would need comparison data)
        consistency_score = 0.8  # Default moderate consistency
        
        # Calculate relative performance (placeholder)
        relative_performance = 1.0  # Neutral relative performance
        
        # Determine overfitting risk based on performance metrics
        if (backtest_result.sharpe_ratio > 3.0 or 
            backtest_result.win_rate > 0.8 or
            backtest_result.total_return > 5.0):
            overfitting_risk = "High"
        elif (backtest_result.sharpe_ratio > 2.0 or 
              backtest_result.win_rate > 0.65):
            overfitting_risk = "Medium"
        else:
            overfitting_risk = "Low"
        
        # Calculate correlation-adjusted return (placeholder)
        correlation_adjusted_return = backtest_result.total_return * 0.9
        
        # Calculate diversification benefit (placeholder)
        diversification_benefit = 0.1
        
        return CrossAssetPerformance(
            asset=asset,
            backtest_results=backtest_result,
            performance_rank=rank,
            consistency_score=consistency_score,
            relative_performance=relative_performance,
            overfitting_risk=overfitting_risk,
            sharpe_ratio=backtest_result.sharpe_ratio,
            total_return=backtest_result.total_return,
            max_drawdown=backtest_result.max_drawdown,
            win_rate=backtest_result.win_rate,
            profit_factor=backtest_result.profit_factor,
            correlation_adjusted_return=correlation_adjusted_return,
            diversification_benefit=diversification_benefit
        )
    
    def _analyze_performance_consistency(
        self,
        performances: List[CrossAssetPerformance]
    ) -> Dict[str, Any]:
        """Analyze performance consistency across assets."""
        
        if len(performances) < 2:
            return {
                'performance_consistency': 1.0,
                'rank_consistency': 1.0,
                'risk_adjusted_consistency': 1.0
            }
        
        # Extract metrics
        returns = [p.total_return for p in performances]
        sharpe_ratios = [p.sharpe_ratio for p in performances]
        drawdowns = [abs(p.max_drawdown) for p in performances]
        
        # Calculate coefficient of variation for consistency
        def cv(values):
            mean_val = np.mean(values)
            if mean_val == 0:
                return 0
            return np.std(values) / abs(mean_val)
        
        # Performance consistency (lower CV = higher consistency)
        return_cv = cv(returns)
        performance_consistency = max(0, 1 - return_cv)
        
        # Rank consistency (how consistent rankings are)
        # For simplicity, use Sharpe ratio consistency as proxy
        sharpe_cv = cv(sharpe_ratios)
        rank_consistency = max(0, 1 - sharpe_cv)
        
        # Risk-adjusted consistency
        risk_adjusted_returns = [p.sharpe_ratio for p in performances]
        risk_cv = cv(risk_adjusted_returns)
        risk_adjusted_consistency = max(0, 1 - risk_cv)
        
        return {
            'performance_consistency': performance_consistency,
            'rank_consistency': rank_consistency,
            'risk_adjusted_consistency': risk_adjusted_consistency
        }
    
    def _analyze_overfitting_indicators(
        self,
        performances: List[CrossAssetPerformance],
        correlations: List[AssetCorrelation]
    ) -> Dict[str, Any]:
        """Analyze overfitting indicators."""
        
        # Performance variance
        returns = [p.total_return for p in performances]
        performance_variance = np.var(returns) if len(returns) > 1 else 0.0
        
        # Identify outlier assets (using IQR method)
        if len(returns) >= 4:
            q1, q3 = np.percentile(returns, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_assets = [
                p.asset for p in performances
                if p.total_return < lower_bound or p.total_return > upper_bound
            ]
        else:
            outlier_assets = []
        
        # Overfitting probability based on performance variance and outliers
        if performance_variance > 1.0 or len(outlier_assets) > len(performances) * 0.3:
            overfitting_probability = 0.8
        elif performance_variance > 0.5 or len(outlier_assets) > 0:
            overfitting_probability = 0.5
        else:
            overfitting_probability = 0.2
        
        # Performance correlation with asset correlation
        if correlations and len(performances) > 2:
            # Calculate average correlation for each asset
            asset_avg_correlations = {}
            for perf in performances:
                asset_corrs = [
                    c.return_correlation for c in correlations
                    if c.asset1 == perf.asset or c.asset2 == perf.asset
                ]
                asset_avg_correlations[perf.asset] = np.mean(asset_corrs) if asset_corrs else 0.0
            
            # Correlate performance with asset correlations
            perf_values = [p.total_return for p in performances]
            corr_values = [asset_avg_correlations.get(p.asset, 0.0) for p in performances]
            
            if len(perf_values) > 2:
                perf_corr_with_asset_corr, _ = pearsonr(perf_values, corr_values)
            else:
                perf_corr_with_asset_corr = 0.0
        else:
            perf_corr_with_asset_corr = 0.0
        
        # Statistical significance (simplified)
        if len(returns) > 2:
            _, p_value = stats.f_oneway(*[[r] for r in returns])
            statistical_significance = p_value
        else:
            statistical_significance = 1.0
        
        # Overall consistency grade
        consistency_score = (
            (1 - performance_variance / 2) * 0.4 +
            (1 - len(outlier_assets) / len(performances)) * 0.3 +
            (1 - overfitting_probability) * 0.3
        )
        
        if consistency_score >= 0.9:
            grade = "A"
        elif consistency_score >= 0.8:
            grade = "B"
        elif consistency_score >= 0.7:
            grade = "C"
        elif consistency_score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        # Deployment recommendation
        if grade in ["A", "B"] and overfitting_probability < 0.4:
            recommendation = "Recommended"
        elif grade == "C" or overfitting_probability < 0.7:
            recommendation = "Caution"
        else:
            recommendation = "Not Recommended"
        
        # Warning flags
        warnings = []
        if performance_variance > 1.0:
            warnings.append("High performance variance across assets")
        if len(outlier_assets) > 0:
            warnings.append(f"Outlier performance detected in: {', '.join(outlier_assets)}")
        if overfitting_probability > 0.7:
            warnings.append("High overfitting probability")
        if abs(perf_corr_with_asset_corr) > 0.7:
            warnings.append("Performance highly correlated with asset correlations")
        
        # Recommendations
        recommendations = []
        if overfitting_probability > 0.5:
            recommendations.append("Increase out-of-sample testing period")
        if len(outlier_assets) > 0:
            recommendations.append("Investigate outlier assets for data quality issues")
        if performance_variance > 0.5:
            recommendations.append("Consider strategy parameter adjustment for consistency")
        
        return {
            'performance_variance': performance_variance,
            'outlier_assets': outlier_assets,
            'overfitting_probability': overfitting_probability,
            'performance_correlation_with_asset_correlation': perf_corr_with_asset_corr,
            'statistical_significance': statistical_significance,
            'overall_consistency_grade': grade,
            'deployment_recommendation': recommendation,
            'warning_flags': warnings,
            'recommendations': recommendations
        }
    
    def get_test_summary(self, result: CrossAssetTestResult) -> Dict[str, Any]:
        """Get a summary of cross-asset test results."""
        
        avg_return = np.mean([p.total_return for p in result.asset_performances])
        avg_sharpe = np.mean([p.sharpe_ratio for p in result.asset_performances])
        avg_drawdown = np.mean([abs(p.max_drawdown) for p in result.asset_performances])
        
        best_asset = max(result.asset_performances, key=lambda x: x.total_return)
        worst_asset = min(result.asset_performances, key=lambda x: x.total_return)
        
        return {
            'strategy_name': result.strategy_name,
            'assets_tested': len(result.assets_tested),
            'test_period_days': (result.test_period[1] - result.test_period[0]).days,
            'overall_grade': result.overall_consistency_grade,
            'deployment_recommendation': result.deployment_recommendation,
            'overfitting_probability': result.overfitting_probability,
            'performance_consistency': result.performance_consistency,
            'average_metrics': {
                'return': avg_return,
                'sharpe_ratio': avg_sharpe,
                'max_drawdown': avg_drawdown
            },
            'best_performing_asset': {
                'asset': best_asset.asset,
                'return': best_asset.total_return,
                'sharpe_ratio': best_asset.sharpe_ratio
            },
            'worst_performing_asset': {
                'asset': worst_asset.asset,
                'return': worst_asset.total_return,
                'sharpe_ratio': worst_asset.sharpe_ratio
            },
            'warning_flags': result.warning_flags,
            'recommendations': result.recommendations
        }
    
    async def batch_test_strategies(
        self,
        strategies: List[BaseStrategy],
        asset_group: Union[str, List[str]],
        timeframe: str = '1h',
        test_period_days: int = 365
    ) -> Dict[str, CrossAssetTestResult]:
        """
        Test multiple strategies across assets in batch.
        
        Args:
            strategies: List of strategies to test
            asset_group: Asset group or list of assets
            timeframe: Trading timeframe
            test_period_days: Test period in days
            
        Returns:
            Dictionary mapping strategy names to test results
        """
        self.logger.info(f"Starting batch cross-asset testing for {len(strategies)} strategies")
        
        results = {}
        
        for i, strategy in enumerate(strategies):
            self.logger.info(f"Testing strategy {i+1}/{len(strategies)}: {strategy.name}")
            
            try:
                result = await self.test_strategy_across_assets(
                    strategy=strategy,
                    asset_group=asset_group,
                    timeframe=timeframe,
                    test_period_days=test_period_days
                )
                results[strategy.name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to test strategy {strategy.name}: {e}")
                continue
        
        self.logger.info(f"Batch testing completed: {len(results)}/{len(strategies)} successful")
        return results 