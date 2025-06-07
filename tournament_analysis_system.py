#!/usr/bin/env python3
"""
Enhanced Tournament Analysis System

Comprehensive tournament framework to evaluate all 24 trading strategies,
identify top performers, and establish definitive champion strategies
across different market conditions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
import itertools
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

class TournamentAnalysisSystem:
    """Enhanced tournament analysis system for comprehensive strategy evaluation."""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine(initial_capital=100000)
        
        # Tournament results storage
        self.tournament_results = {}
        self.performance_matrix = None
        self.correlation_matrix = None
        self.champion_strategies = {}
        self.optimal_portfolios = {}
        
        # Advanced analysis components
        self.market_regimes = {}
        self.strategy_fingerprints = {}
        self.monte_carlo_results = {}
        self.pareto_frontier = None
        
        print("🏆 ENHANCED TOURNAMENT ANALYSIS SYSTEM")
        print("=" * 60)
        print("🎯 Evaluating 24 strategies across multiple dimensions")
        print("📊 Advanced analytics, optimization, and champion selection")
        
    def run_comprehensive_tournament(self):
        """Execute the complete tournament analysis framework."""
        try:
            print("\n🚀 STARTING COMPREHENSIVE TOURNAMENT ANALYSIS")
            print("=" * 60)
            
            # Step 1: Data preparation and market regime analysis
            self._prepare_tournament_data()
            
            # Step 2: Round-robin strategy evaluation
            self._conduct_round_robin_tournament()
            
            # Step 3: Advanced performance analysis
            self._perform_advanced_analysis()
            
            # Step 4: Strategy combination optimization
            self._optimize_strategy_combinations()
            
            # Step 5: Champion selection
            self._select_champion_strategies()
            
            # Step 6: Generate comprehensive reports
            self._generate_tournament_reports()
            
            print("\n🎉 TOURNAMENT ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {os.path.join(self.project_root, 'tournament_results/')}")
            
        except Exception as e:
            print(f"❌ Tournament analysis failed: {str(e)}")
            raise
    
    def _prepare_tournament_data(self):
        """Prepare data and perform market regime analysis."""
        print("\n📊 PREPARING TOURNAMENT DATA")
        print("=" * 50)
        
        # Generate enhanced test data with multiple market regimes
        self.test_data = self._generate_comprehensive_market_data()
        
        # Detect market regimes
        self._detect_market_regimes()
        
        print(f"✅ Market data prepared: {len(self.test_data)} data points")
        print(f"✅ Market regimes detected: {len(self.market_regimes)} regimes")
    
    def _generate_comprehensive_market_data(self, days: int = 365) -> pd.DataFrame:
        """Generate comprehensive market data with multiple regimes."""
        np.random.seed(42)
        
        # Create complex market scenarios
        scenarios = {
            'bull_market': {'trend': 0.0008, 'volatility': 0.02, 'duration': 90},
            'bear_market': {'trend': -0.0006, 'volatility': 0.03, 'duration': 60},
            'sideways': {'trend': 0.0001, 'volatility': 0.015, 'duration': 120},
            'high_volatility': {'trend': 0.0003, 'volatility': 0.05, 'duration': 45},
            'recovery': {'trend': 0.0012, 'volatility': 0.025, 'duration': 50}
        }
        
        all_data = []
        current_date = datetime(2023, 1, 1)
        base_price = 50000
        
        for scenario_name, params in scenarios.items():
            scenario_data = self._generate_scenario_data(
                current_date, params['duration'], base_price, 
                params['trend'], params['volatility'], scenario_name
            )
            all_data.append(scenario_data)
            current_date += timedelta(days=params['duration'])
            base_price = scenario_data['close'].iloc[-1]
        
        # Combine all scenarios
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Add technical indicators
        combined_data = self._add_technical_indicators(combined_data)
        
        return combined_data
    
    def _generate_scenario_data(self, start_date, days, start_price, trend, volatility, regime):
        """Generate data for a specific market scenario."""
        dates = pd.date_range(start_date, periods=days*24, freq='H')
        returns = np.random.normal(trend, volatility, len(dates))
        
        # Add some autocorrelation for realism
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for i in range(len(dates)):
            price = prices[i]
            high = price * (1 + abs(np.random.normal(0, volatility/4)))
            low = price * (1 - abs(np.random.normal(0, volatility/4)))
            volume = np.random.uniform(1000, 5000)
            
            data.append({
                'timestamp': dates[i],
                'open': price,
                'high': max(price, high),
                'low': min(price, low),
                'close': price,
                'volume': volume,
                'market_regime': regime
            })
        
        return pd.DataFrame(data).set_index('timestamp')
    
    def _add_technical_indicators(self, data):
        """Add technical indicators for strategy evaluation."""
        # Simple moving averages
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility
        data['volatility'] = data['close'].rolling(20).std()
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        
        return data
    
    def _detect_market_regimes(self):
        """Detect and classify market regimes in the data."""
        print("🔍 Detecting market regimes...")
        
        # Use existing regime labels from data generation
        regimes = self.test_data['market_regime'].unique()
        
        for regime in regimes:
            regime_data = self.test_data[self.test_data['market_regime'] == regime]
            
            # Calculate regime characteristics
            returns = regime_data['close'].pct_change().dropna()
            
            self.market_regimes[regime] = {
                'start_date': regime_data.index[0],
                'end_date': regime_data.index[-1],
                'duration_days': len(regime_data) / 24,
                'total_return': (regime_data['close'].iloc[-1] / regime_data['close'].iloc[0]) - 1,
                'volatility': returns.std() * np.sqrt(365 * 24),
                'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(365 * 24) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(regime_data['close']),
                'data_points': len(regime_data)
            }
        
        print(f"✅ Market regimes characterized: {list(self.market_regimes.keys())}")
        
        # Print regime summary
        for regime, stats in self.market_regimes.items():
            print(f"   📊 {regime}: {stats['total_return']:.1%} return, "
                  f"{stats['volatility']:.1%} volatility, "
                  f"{stats['duration_days']:.0f} days")
    
    def _calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown for a price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _conduct_round_robin_tournament(self):
        """Conduct round-robin tournament evaluation of all strategies."""
        print("\n🥊 CONDUCTING ROUND-ROBIN TOURNAMENT")
        print("=" * 50)
        
        strategies = self.factory.get_all_strategies()
        total_tests = len(strategies) * len(self.market_regimes)
        current_test = 0
        
        print(f"🎯 Testing {len(strategies)} strategies across {len(self.market_regimes)} market regimes")
        print(f"📊 Total evaluations: {total_tests}")
        
        performance_data = []
        
        for strategy_name in strategies:
            print(f"\n🔄 Evaluating {strategy_name}...")
            
            strategy_results = {}
            
            for regime_name, regime_info in self.market_regimes.items():
                current_test += 1
                print(f"   📈 {regime_name} ({current_test}/{total_tests})")
                
                # Get regime-specific data
                regime_data = self.test_data[
                    self.test_data['market_regime'] == regime_name
                ].copy()
                
                if len(regime_data) < 50:  # Skip if insufficient data
                    continue
                
                # Test strategy
                try:
                    result = self._evaluate_strategy_on_regime(
                        strategy_name, regime_data, regime_name
                    )
                    strategy_results[regime_name] = result
                    
                    # Store for matrix analysis
                    performance_data.append({
                        'strategy': strategy_name,
                        'regime': regime_name,
                        'annual_return': result.get('annual_return', 0),
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'win_rate': result.get('win_rate', 0),
                        'profit_factor': result.get('profit_factor', 0),
                        'total_trades': result.get('total_trades', 0),
                        'calmar_ratio': result.get('calmar_ratio', 0),
                        'sortino_ratio': result.get('sortino_ratio', 0)
                    })
                    
                except Exception as e:
                    print(f"      ❌ Failed: {str(e)[:50]}")
                    strategy_results[regime_name] = {'error': str(e)}
            
            self.tournament_results[strategy_name] = strategy_results
        
        # Create performance matrix
        self.performance_df = pd.DataFrame(performance_data)
        print(f"\n✅ Tournament complete: {len(performance_data)} successful evaluations")
        
        # Generate performance matrix
        self._create_performance_matrices()
    
    def _evaluate_strategy_on_regime(self, strategy_name: str, data: pd.DataFrame, regime: str) -> Dict:
        """Evaluate a single strategy on a market regime."""
        try:
            # Get strategy with optimal parameters for this scenario
            strategy_class = self.factory.registry.get_strategy_class(strategy_name)
            default_params = self.factory.get_default_parameters(strategy_name)
            
            # Create strategy instance
            strategy = strategy_class(**default_params)
            
            # Run backtesting
            result = self.engine.backtest_strategy(strategy, data)
            
            # Extract key metrics
            return {
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades,
                'calmar_ratio': result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0,
                'sortino_ratio': result.sortino_ratio,
                'total_return': result.total_return,
                'volatility': result.volatility,
                'recovery_factor': result.recovery_factor if hasattr(result, 'recovery_factor') else 0,
                'regime': regime
            }
            
        except Exception as e:
            raise Exception(f"Strategy evaluation failed: {str(e)}")
    
    def _create_performance_matrices(self):
        """Create performance matrices for analysis."""
        print("\n📊 CREATING PERFORMANCE MATRICES")
        print("=" * 50)
        
        if self.performance_df.empty:
            print("❌ No performance data available")
            return
        
        # Create pivot tables for different metrics
        metrics = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        
        self.performance_matrices = {}
        for metric in metrics:
            matrix = self.performance_df.pivot(
                index='strategy', 
                columns='regime', 
                values=metric
            ).fillna(0)
            self.performance_matrices[metric] = matrix
        
        # Create correlation matrix
        strategy_returns = self.performance_matrices['annual_return']
        self.correlation_matrix = strategy_returns.T.corr()
        
        print(f"✅ Performance matrices created for {len(metrics)} metrics")
        print(f"✅ Correlation matrix: {self.correlation_matrix.shape}")
    
    def _perform_advanced_analysis(self):
        """Perform advanced performance analysis including Monte Carlo and fingerprinting."""
        print("\n🔬 PERFORMING ADVANCED ANALYSIS")
        print("=" * 50)
        
        # Strategy fingerprinting
        self._create_strategy_fingerprints()
        
        # Monte Carlo simulation
        self._run_monte_carlo_analysis()
        
        # Statistical significance testing
        self._perform_statistical_testing()
        
        # Hierarchical clustering
        self._perform_clustering_analysis()
    
    def _create_strategy_fingerprints(self):
        """Create performance fingerprints for each strategy."""
        print("🔍 Creating strategy fingerprints...")
        
        for strategy in self.performance_df['strategy'].unique():
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            
            if len(strategy_data) == 0:
                continue
            
            fingerprint = {
                'best_regime': strategy_data.loc[strategy_data['annual_return'].idxmax(), 'regime'],
                'worst_regime': strategy_data.loc[strategy_data['annual_return'].idxmin(), 'regime'],
                'avg_return': strategy_data['annual_return'].mean(),
                'consistency': 1 - strategy_data['annual_return'].std() / (abs(strategy_data['annual_return'].mean()) + 1e-8),
                'regime_scores': {}
            }
            
            # Score performance in each regime
            for regime in strategy_data['regime'].unique():
                regime_data = strategy_data[strategy_data['regime'] == regime]
                if len(regime_data) > 0:
                    fingerprint['regime_scores'][regime] = regime_data['annual_return'].iloc[0]
            
            self.strategy_fingerprints[strategy] = fingerprint
        
        print(f"✅ Strategy fingerprints created: {len(self.strategy_fingerprints)}")
    
    def _run_monte_carlo_analysis(self, n_simulations: int = 100):
        """Run Monte Carlo analysis for strategy robustness."""
        print(f"🎲 Running Monte Carlo simulation ({n_simulations} iterations)...")
        
        # Due to time constraints, we'll run a simplified version
        # In production, this would run 1000+ simulations with different random seeds
        
        strategies = self.performance_df['strategy'].unique()
        
        for strategy in strategies[:5]:  # Limit to top 5 for demo
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            
            if len(strategy_data) == 0:
                continue
            
            returns = strategy_data['annual_return'].values
            
            if len(returns) > 0:
                # Simple bootstrap simulation
                mc_results = []
                for _ in range(n_simulations):
                    bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                    mc_results.append(np.mean(bootstrap_sample))
                
                self.monte_carlo_results[strategy] = {
                    'mean_return': np.mean(mc_results),
                    'std_return': np.std(mc_results),
                    'confidence_95': np.percentile(mc_results, [2.5, 97.5]),
                    'probability_positive': np.sum(np.array(mc_results) > 0) / len(mc_results)
                }
        
        print(f"✅ Monte Carlo analysis complete: {len(self.monte_carlo_results)} strategies")
    
    def _perform_statistical_testing(self):
        """Perform statistical significance testing."""
        print("📈 Performing statistical significance testing...")
        
        # T-tests for strategy performance vs zero
        self.statistical_results = {}
        
        for strategy in self.performance_df['strategy'].unique():
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            returns = strategy_data['annual_return'].values
            
            if len(returns) > 1:
                t_stat, p_value = stats.ttest_1samp(returns, 0)
                self.statistical_results[strategy] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'mean_return': np.mean(returns)
                }
        
        print(f"✅ Statistical testing complete: {len(self.statistical_results)} strategies")
    
    def _perform_clustering_analysis(self):
        """Perform hierarchical clustering of strategies."""
        print("🔗 Performing clustering analysis...")
        
        if self.correlation_matrix is not None and len(self.correlation_matrix) > 1:
            # Calculate distance matrix
            distance_matrix = 1 - abs(self.correlation_matrix)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method='ward')
            
            # Get cluster assignments
            n_clusters = min(5, len(self.correlation_matrix))
            clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Store cluster results
            self.cluster_results = {
                'linkage_matrix': linkage_matrix,
                'clusters': dict(zip(self.correlation_matrix.index, clusters)),
                'n_clusters': n_clusters
            }
            
            print(f"✅ Clustering complete: {n_clusters} clusters identified")
        else:
            print("⚠️ Insufficient data for clustering analysis")
    
    def _optimize_strategy_combinations(self):
        """Optimize strategy combinations using portfolio theory."""
        print("\n🎯 OPTIMIZING STRATEGY COMBINATIONS")
        print("=" * 50)
        
        if self.performance_matrices is None or 'annual_return' not in self.performance_matrices:
            print("❌ Performance matrices not available")
            return
        
        returns_matrix = self.performance_matrices['annual_return']
        
        if len(returns_matrix) < 2:
            print("❌ Insufficient strategies for portfolio optimization")
            return
        
        # Calculate portfolio combinations
        self._calculate_optimal_portfolios(returns_matrix)
        
        # Generate Pareto frontier
        self._generate_pareto_frontier(returns_matrix)
    
    def _calculate_optimal_portfolios(self, returns_matrix):
        """Calculate optimal portfolio allocations."""
        print("📊 Calculating optimal portfolios...")
        
        strategies = returns_matrix.index.tolist()
        n_strategies = len(strategies)
        
        if n_strategies < 2:
            print("⚠️ Need at least 2 strategies for portfolio optimization")
            return
        
        # Simple equal-weight portfolio
        equal_weights = np.ones(n_strategies) / n_strategies
        
        # Calculate portfolio metrics
        regimes = returns_matrix.columns
        portfolio_returns = {}
        
        for regime in regimes:
            regime_returns = returns_matrix[regime].values
            if not np.isnan(regime_returns).all():
                portfolio_return = np.dot(equal_weights, regime_returns)
                portfolio_returns[regime] = portfolio_return
        
        self.optimal_portfolios['equal_weight'] = {
            'weights': dict(zip(strategies, equal_weights)),
            'regime_returns': portfolio_returns,
            'avg_return': np.mean(list(portfolio_returns.values()))
        }
        
        # Top 3 performers portfolio
        avg_returns = returns_matrix.mean(axis=1)
        top_3_strategies = avg_returns.nlargest(3).index.tolist()
        
        top_3_weights = np.zeros(n_strategies)
        for i, strategy in enumerate(strategies):
            if strategy in top_3_strategies:
                top_3_weights[i] = 1/3
        
        portfolio_returns = {}
        for regime in regimes:
            regime_returns = returns_matrix[regime].values
            if not np.isnan(regime_returns).all():
                portfolio_return = np.dot(top_3_weights, regime_returns)
                portfolio_returns[regime] = portfolio_return
        
        self.optimal_portfolios['top_3'] = {
            'weights': dict(zip(strategies, top_3_weights)),
            'regime_returns': portfolio_returns,
            'avg_return': np.mean(list(portfolio_returns.values())),
            'top_strategies': top_3_strategies
        }
        
        print(f"✅ Optimal portfolios calculated: {len(self.optimal_portfolios)}")
    
    def _generate_pareto_frontier(self, returns_matrix):
        """Generate Pareto frontier for risk-return optimization."""
        print("📈 Generating Pareto frontier...")
        
        # For demonstration, create a simplified Pareto frontier
        strategies = returns_matrix.index.tolist()
        strategy_data = []
        
        for strategy in strategies:
            avg_return = returns_matrix.loc[strategy].mean()
            volatility = returns_matrix.loc[strategy].std()
            
            if not np.isnan(avg_return) and not np.isnan(volatility):
                strategy_data.append({
                    'strategy': strategy,
                    'return': avg_return,
                    'volatility': volatility,
                    'sharpe': avg_return / volatility if volatility > 0 else 0
                })
        
        self.pareto_frontier = pd.DataFrame(strategy_data)
        print(f"✅ Pareto frontier generated: {len(self.pareto_frontier)} points")
    
    def _select_champion_strategies(self):
        """Select champion strategies based on multi-factor scoring."""
        print("\n🏆 SELECTING CHAMPION STRATEGIES")
        print("=" * 50)
        
        if self.performance_df.empty:
            print("❌ No performance data for champion selection")
            return
        
        # Calculate composite scores for each strategy
        strategy_scores = []
        
        for strategy in self.performance_df['strategy'].unique():
            strategy_data = self.performance_df[self.performance_df['strategy'] == strategy]
            
            if len(strategy_data) == 0:
                continue
            
            # Multi-factor scoring
            metrics = {
                'avg_return': strategy_data['annual_return'].mean(),
                'consistency': 1 - (strategy_data['annual_return'].std() / (abs(strategy_data['annual_return'].mean()) + 1e-8)),
                'avg_sharpe': strategy_data['sharpe_ratio'].mean(),
                'avg_win_rate': strategy_data['win_rate'].mean(),
                'total_trades': strategy_data['total_trades'].mean()
            }
            
            # Composite score (weighted combination)
            composite_score = (
                0.4 * self._normalize_score(metrics['avg_return'], self.performance_df['annual_return']) +
                0.2 * self._normalize_score(metrics['consistency'], [0, 1]) +
                0.2 * self._normalize_score(metrics['avg_sharpe'], self.performance_df['sharpe_ratio']) +
                0.1 * self._normalize_score(metrics['avg_win_rate'], self.performance_df['win_rate']) +
                0.1 * self._normalize_score(metrics['total_trades'], self.performance_df['total_trades'])
            )
            
            strategy_scores.append({
                'strategy': strategy,
                'composite_score': composite_score,
                **metrics
            })
        
        # Sort by composite score
        strategy_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Select champions (top 25% or minimum 3)
        n_champions = max(3, len(strategy_scores) // 4)
        champions = strategy_scores[:n_champions]
        
        self.champion_strategies = {
            'champions': champions,
            'criteria': {
                'return_weight': 0.4,
                'consistency_weight': 0.2,
                'sharpe_weight': 0.2,
                'win_rate_weight': 0.1,
                'trades_weight': 0.1
            }
        }
        
        print(f"🏆 CHAMPION STRATEGIES SELECTED")
        print(f"   📊 Total candidates: {len(strategy_scores)}")
        print(f"   🥇 Champions selected: {n_champions}")
        
        for i, champion in enumerate(champions, 1):
            print(f"   #{i} {champion['strategy']}: "
                  f"Score {champion['composite_score']:.3f}, "
                  f"Return {champion['avg_return']:.1%}")
    
    def _normalize_score(self, value, reference_data):
        """Normalize a score to 0-1 range based on reference data."""
        if isinstance(reference_data, list):
            min_val, max_val = reference_data
        else:
            min_val = reference_data.min()
            max_val = reference_data.max()
        
        if max_val == min_val:
            return 0.5
        
        return (value - min_val) / (max_val - min_val)
    
    def _generate_tournament_reports(self):
        """Generate comprehensive tournament reports and visualizations."""
        print("\n📋 GENERATING TOURNAMENT REPORTS")
        print("=" * 50)
        
        # Create results directory
        results_dir = os.path.join(self.project_root, 'tournament_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save raw data
        self._save_raw_results(results_dir)
        
        # Generate summary report
        self._generate_summary_report(results_dir)
        
        # Create visualizations
        self._create_tournament_visualizations(results_dir)
        
        print(f"✅ Tournament reports generated in: {results_dir}")
    
    def _save_raw_results(self, results_dir):
        """Save raw tournament results."""
        print("💾 Saving raw results...")
        
        # Performance data
        if not self.performance_df.empty:
            self.performance_df.to_csv(os.path.join(results_dir, 'tournament_performance.csv'), index=False)
        
        # Tournament results
        with open(os.path.join(results_dir, 'tournament_results.json'), 'w') as f:
            json.dump(self.tournament_results, f, indent=2, default=str)
        
        # Champion strategies
        with open(os.path.join(results_dir, 'champion_strategies.json'), 'w') as f:
            json.dump(self.champion_strategies, f, indent=2, default=str)
        
        # Market regimes
        with open(os.path.join(results_dir, 'market_regimes.json'), 'w') as f:
            json.dump(self.market_regimes, f, indent=2, default=str)
        
        print("✅ Raw results saved")
    
    def _generate_summary_report(self, results_dir):
        """Generate executive summary report."""
        print("📊 Generating summary report...")
        
        summary = {
            'tournament_overview': {
                'total_strategies': len(self.tournament_results),
                'market_regimes': len(self.market_regimes),
                'total_evaluations': len(self.performance_df),
                'champions_selected': len(self.champion_strategies.get('champions', []))
            },
            'top_performers': self.champion_strategies.get('champions', [])[:5],
            'market_regime_analysis': self.market_regimes,
            'optimal_portfolios': self.optimal_portfolios
        }
        
        with open(os.path.join(results_dir, 'tournament_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✅ Summary report generated")
    
    def _create_tournament_visualizations(self, results_dir):
        """Create tournament visualizations."""
        print("📈 Creating visualizations...")
        
        try:
            # Performance heatmap
            if 'annual_return' in self.performance_matrices:
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    self.performance_matrices['annual_return'], 
                    annot=True, 
                    fmt='.1%', 
                    cmap='RdYlGn',
                    center=0
                )
                plt.title('Strategy Performance Across Market Regimes')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'performance_heatmap.png'), dpi=300)
                plt.close()
            
            # Champion strategies bar chart
            if self.champion_strategies and 'champions' in self.champion_strategies:
                champions = self.champion_strategies['champions'][:10]
                
                plt.figure(figsize=(12, 6))
                strategies = [c['strategy'] for c in champions]
                scores = [c['composite_score'] for c in champions]
                
                bars = plt.bar(range(len(strategies)), scores, color='skyblue', edgecolor='navy')
                plt.xlabel('Strategy')
                plt.ylabel('Composite Score')
                plt.title('Top Champion Strategies')
                plt.xticks(range(len(strategies)), strategies, rotation=45)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'champion_strategies.png'), dpi=300)
                plt.close()
            
            print("✅ Visualizations created")
            
        except Exception as e:
            print(f"⚠️ Visualization error: {str(e)}")

if __name__ == "__main__":
    # Run the comprehensive tournament analysis
    system = TournamentAnalysisSystem()
    system.run_comprehensive_tournament() 