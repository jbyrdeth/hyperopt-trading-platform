#!/usr/bin/env python3
"""
Strategy Ranking and Categorization System

Comprehensive analysis and ranking of all 24 trading strategies based on 
multiple performance dimensions and market conditions.
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

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine

class StrategyRankingSystem:
    """Comprehensive strategy ranking and categorization system."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine(initial_capital=100000)
        self.strategy_names = self._get_all_strategy_names()
        
    def _get_all_strategy_names(self) -> list:
        """Get all available strategy names from the factory."""
        # Get the full list of 24 strategies
        return [
            "MovingAverageCrossover",
            "MACD", 
            "RSI",
            "BollingerBands",
            "Momentum",
            "ROC",
            "Stochastic",
            "WilliamsR", 
            "UltimateOscillator",
            "VWAP",
            "OBV",
            "AD",
            "CMF",
            "ATR",
            "BollingerSqueeze",
            "KeltnerChannel",
            "HistoricalVolatility",
            "SupportResistance",
            "PivotPoints",
            "FibonacciRetracement",
            "DoubleTopBottom",
            "MTFTrendAnalysis",
            "MTFRSI",
            "MTFMACD"
        ]
        
    def create_market_scenarios(self) -> dict:
        """Create different market scenarios for comprehensive testing."""
        scenarios = {}
        np.random.seed(42)
        
        # Scenario 1: Strong Bull Market
        bull_data = self._generate_scenario_data(
            days=30, trend=0.002, volatility=0.015, 
            name="Strong Bull Market"
        )
        scenarios['bull'] = bull_data
        
        # Scenario 2: Bear Market
        bear_data = self._generate_scenario_data(
            days=30, trend=-0.0015, volatility=0.02,
            name="Bear Market"
        )
        scenarios['bear'] = bear_data
        
        # Scenario 3: Sideways/Ranging Market
        sideways_data = self._generate_scenario_data(
            days=30, trend=0.0002, volatility=0.025,
            name="Sideways Market"
        )
        scenarios['sideways'] = sideways_data
        
        # Scenario 4: High Volatility Market
        volatile_data = self._generate_scenario_data(
            days=30, trend=0.0005, volatility=0.035,
            name="High Volatility Market"
        )
        scenarios['volatile'] = volatile_data
        
        # Scenario 5: Mixed Market (Real-world complexity)
        mixed_data = self._generate_mixed_scenario_data(
            days=60, name="Mixed Market Conditions"
        )
        scenarios['mixed'] = mixed_data
        
        return scenarios
        
    def _generate_scenario_data(self, days: int, trend: float, volatility: float, name: str) -> pd.DataFrame:
        """Generate data for a specific market scenario."""
        n_points = days * 24  # Hourly data
        
        # Generate price movements
        price_changes = np.random.normal(trend, volatility, n_points)
        price_changes = np.cumsum(price_changes)
        
        # Generate realistic price series
        base_price = 50000
        prices = base_price * np.exp(price_changes)
        
        # Generate OHLCV
        opens = np.roll(prices, 1)
        opens[0] = base_price
        
        highs = prices * (1 + np.abs(np.random.normal(0, 0.008, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.008, len(prices))))
        volume = np.random.lognormal(15, 0.8, len(prices))
        
        # Create datetime index
        start_date = datetime.now() - timedelta(days=days)
        datetime_index = pd.date_range(start=start_date, periods=len(prices), freq='H')
        
        data = pd.DataFrame({
            'open': opens,
            'high': np.maximum.reduce([opens, prices, highs]),
            'low': np.minimum.reduce([opens, prices, lows]),
            'close': prices,
            'volume': volume
        }, index=datetime_index)
        
        return data
        
    def _generate_mixed_scenario_data(self, days: int, name: str) -> pd.DataFrame:
        """Generate complex mixed market scenario data."""
        np.random.seed(42)
        n_points = days * 24
        
        # Create multiple phases
        bull_points = int(n_points * 0.3)
        sideways_points = int(n_points * 0.25)
        bear_points = int(n_points * 0.2)
        recovery_points = n_points - bull_points - sideways_points - bear_points
        
        # Generate each phase
        bull_trend = np.cumsum(np.random.normal(0.0008, 0.01, bull_points))
        sideways = np.cumsum(np.random.normal(0.0001, 0.02, sideways_points))
        bear_trend = np.cumsum(np.random.normal(-0.0003, 0.015, bear_points))
        recovery_trend = np.cumsum(np.random.normal(0.0006, 0.018, recovery_points))
        
        # Combine phases
        price_changes = np.concatenate([bull_trend, sideways, bear_trend, recovery_trend])
        
        # Generate realistic price series
        base_price = 50000
        prices = base_price * np.exp(price_changes)
        
        # Generate OHLCV
        opens = np.roll(prices, 1)
        opens[0] = base_price
        
        highs = prices * (1 + np.abs(np.random.normal(0, 0.008, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.008, len(prices))))
        volume = np.random.lognormal(15, 0.8, len(prices))
        
        # Create datetime index
        start_date = datetime.now() - timedelta(days=days)
        datetime_index = pd.date_range(start=start_date, periods=len(prices), freq='H')
        
        data = pd.DataFrame({
            'open': opens,
            'high': np.maximum.reduce([opens, prices, highs]),
            'low': np.minimum.reduce([opens, prices, lows]),
            'close': prices,
            'volume': volume
        }, index=datetime_index)
        
        return data
        
    def get_default_parameters(self, strategy_name: str) -> dict:
        """Get reasonable default parameters for each strategy."""
        defaults = {
            "MovingAverageCrossover": {
                'fast_period': 8, 'slow_period': 21, 'ma_type': 'EMA',
                'signal_threshold': 0.002, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "MACD": {
                'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "RSI": {
                'period': 14, 'overbought': 70, 'oversold': 30,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "BollingerBands": {
                'period': 20, 'std_dev': 2, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "Momentum": {
                'period': 10, 'threshold': 0.02, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "ROC": {
                'period': 12, 'threshold': 5, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "Stochastic": {
                'k_period': 14, 'd_period': 3, 'overbought': 80, 'oversold': 20,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "WilliamsR": {
                'period': 14, 'overbought': -20, 'oversold': -80,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "UltimateOscillator": {
                'short_period': 7, 'medium_period': 20, 'long_period': 28,
                'overbought': 70, 'oversold': 30, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "VWAP": {
                'period': 20, 'threshold': 0.005, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "OBV": {
                'signal_line_period': 10, 'threshold': 0.01, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "AD": {
                'signal_line_period': 10, 'threshold': 0.01, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "CMF": {
                'period': 20, 'threshold': 0.1, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "ATR": {
                'period': 14, 'multiplier': 2, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "BollingerSqueeze": {
                'bb_period': 20, 'bb_std': 2, 'kc_period': 20, 'atr_period': 10,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "KeltnerChannel": {
                'period': 20, 'atr_period': 10, 'multiplier': 2,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "HistoricalVolatility": {
                'period': 20, 'threshold': 0.02, 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "SupportResistance": {
                'lookback_period': 50, 'min_touches': 2, 'tolerance': 0.01,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "PivotPoints": {
                'pivot_type': 'standard', 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "FibonacciRetracement": {
                'lookback_period': 50, 'retracement_levels': [0.382, 0.618],
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "DoubleTopBottom": {
                'lookback_period': 50, 'tolerance': 0.02, 'min_distance': 10,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "MTFTrendAnalysis": {
                'short_period': 10, 'medium_period': 20, 'long_period': 50,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "MTFRSI": {
                'rsi_period': 14, 'timeframes': ['1h', '4h'], 'overbought': 70, 'oversold': 30,
                'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            },
            "MTFMACD": {
                'fast_period': 12, 'slow_period': 26, 'signal_period': 9,
                'timeframes': ['1h', '4h'], 'position_size_pct': 0.15,
                'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
            }
        }
        
        return defaults.get(strategy_name, {
            'position_size_pct': 0.15, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.06
        })
        
    def test_strategy_on_scenario(self, strategy_name: str, scenario_data: pd.DataFrame, scenario_name: str) -> dict:
        """Test a single strategy on a market scenario."""
        try:
            # Get default parameters
            params = self.get_default_parameters(strategy_name)
            
            # Create strategy
            strategy = self.factory.create_strategy(strategy_name, **params)
            
            # Run backtest
            result = self.engine.backtest_strategy(strategy, scenario_data)
            
            # Extract key metrics
            return {
                'strategy_name': strategy_name,
                'scenario': scenario_name,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'total_trades': result.total_trades,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'volatility': result.volatility,
                'calmar_ratio': result.calmar_ratio,
                'sortino_ratio': result.sortino_ratio,
                'expectancy': result.expectancy,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy_name,
                'scenario': scenario_name,
                'success': False,
                'error': str(e)[:200],
                'total_return': -999,
                'annual_return': -999,
                'total_trades': 0,
                'sharpe_ratio': -999,
                'max_drawdown': -999,
                'win_rate': 0,
                'profit_factor': 0,
                'volatility': 0,
                'calmar_ratio': -999,
                'sortino_ratio': -999,
                'expectancy': -999
            }
            
    def run_comprehensive_analysis(self) -> pd.DataFrame:
        """Run comprehensive analysis of all strategies across all scenarios."""
        print("🚀 COMPREHENSIVE STRATEGY RANKING ANALYSIS")
        print("=" * 70)
        
        # Create market scenarios
        print("📊 Creating market scenarios...")
        scenarios = self.create_market_scenarios()
        
        for name, data in scenarios.items():
            market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            print(f"   {name}: {len(data)} points, {market_return:.1f}% market return")
        
        # Test all strategies on all scenarios
        print(f"\n🧪 Testing {len(self.strategy_names)} strategies across {len(scenarios)} scenarios...")
        
        results = []
        total_tests = len(self.strategy_names) * len(scenarios)
        completed_tests = 0
        
        for strategy_name in self.strategy_names:
            print(f"\n📈 Testing {strategy_name}...")
            strategy_results = []
            
            for scenario_name, scenario_data in scenarios.items():
                result = self.test_strategy_on_scenario(strategy_name, scenario_data, scenario_name)
                results.append(result)
                strategy_results.append(result)
                completed_tests += 1
                
                # Show progress
                success_icon = "✅" if result['success'] else "❌"
                trades = result['total_trades'] if result['success'] else 0
                annual_return = result['annual_return'] * 100 if result['success'] else 0
                
                print(f"   {success_icon} {scenario_name}: {trades} trades, {annual_return:.1f}% annual")
                
            # Show strategy summary
            successful_scenarios = sum(1 for r in strategy_results if r['success'])
            print(f"   📊 Success rate: {successful_scenarios}/{len(scenarios)} scenarios")
            
        print(f"\n✅ Analysis complete: {completed_tests}/{total_tests} tests")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df
        
    def calculate_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite scores for ranking strategies."""
        print("\n📊 CALCULATING COMPOSITE SCORES")
        print("=" * 50)
        
        # Filter successful results only
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            print("❌ No successful strategy tests to analyze!")
            return pd.DataFrame()
        
        # Get number of scenarios from data
        num_scenarios = len(successful_df['scenario'].unique())
        
        # Clean NaN and infinite values
        successful_df = successful_df.replace([np.inf, -np.inf], np.nan)
        
        # Calculate strategy-level metrics
        strategy_metrics = []
        
        for strategy_name in self.strategy_names:
            strategy_data = successful_df[successful_df['strategy_name'] == strategy_name]
            
            if len(strategy_data) == 0:
                continue
                
            # Clean numeric columns and handle NaN values
            numeric_cols = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades', 'profit_factor']
            for col in numeric_cols:
                if col in strategy_data.columns:
                    # Replace NaN/inf with reasonable defaults
                    if col == 'annual_return':
                        strategy_data.loc[:, col] = strategy_data[col].fillna(-1.0)  # -100% return for NaN
                    elif col == 'sharpe_ratio':
                        strategy_data.loc[:, col] = strategy_data[col].fillna(-10.0)  # Very poor Sharpe for NaN
                    elif col == 'max_drawdown':
                        strategy_data.loc[:, col] = strategy_data[col].fillna(-1.0)  # -100% drawdown for NaN
                    elif col == 'win_rate':
                        strategy_data.loc[:, col] = strategy_data[col].fillna(0.0)  # 0% win rate for NaN
                    elif col == 'profit_factor':
                        strategy_data.loc[:, col] = strategy_data[col].fillna(0.0)  # 0 profit factor for NaN
                    
                    # Cap extreme values to prevent skewing
                    if col == 'annual_return':
                        strategy_data.loc[:, col] = strategy_data[col].clip(-1.0, 10.0)  # -100% to 1000%
                    elif col == 'sharpe_ratio':
                        strategy_data.loc[:, col] = strategy_data[col].clip(-10.0, 10.0)  # Reasonable Sharpe range
                        
            # Calculate averages across scenarios
            avg_annual_return = strategy_data['annual_return'].mean()
            avg_sharpe_ratio = strategy_data['sharpe_ratio'].mean()
            avg_max_drawdown = strategy_data['max_drawdown'].mean()
            avg_win_rate = strategy_data['win_rate'].mean()
            avg_trades = strategy_data['total_trades'].mean()
            avg_profit_factor = strategy_data['profit_factor'].mean()
            success_rate = len(strategy_data) / num_scenarios * 100
            
            # Calculate consistency (lower volatility of returns across scenarios)
            return_std = strategy_data['annual_return'].std()
            consistency_score = 1 / (1 + return_std) if return_std > 0 else 1
            
            # Calculate composite score (weighted average of key metrics)
            # Weights: Return (30%), Sharpe (25%), Drawdown (20%), Win Rate (15%), Consistency (10%)
            composite_score = (
                (avg_annual_return * 0.30) +
                (avg_sharpe_ratio * 0.25) +
                (abs(avg_max_drawdown) * -0.20) +  # Negative because lower drawdown is better
                (avg_win_rate * 0.15) +
                (consistency_score * 0.10)
            )
            
            strategy_metrics.append({
                'strategy_name': strategy_name,
                'avg_annual_return': avg_annual_return,
                'avg_sharpe_ratio': avg_sharpe_ratio,
                'avg_max_drawdown': avg_max_drawdown,
                'avg_win_rate': avg_win_rate,
                'avg_trades': avg_trades,
                'avg_profit_factor': avg_profit_factor,
                'success_rate': success_rate,
                'consistency_score': consistency_score,
                'composite_score': composite_score
            })
            
        # Convert to DataFrame and sort by composite score
        strategy_rankings = pd.DataFrame(strategy_metrics)
        strategy_rankings = strategy_rankings.sort_values('composite_score', ascending=False)
        
        print(f"✅ Calculated composite scores for {len(strategy_rankings)} strategies")
        
        return strategy_rankings
        
    def categorize_strategies(self, df: pd.DataFrame, rankings: pd.DataFrame) -> dict:
        """Categorize strategies by their strengths in different market conditions."""
        print("\n🏷️ CATEGORIZING STRATEGIES BY MARKET CONDITIONS")
        print("=" * 60)
        
        categories = {
            'bull_market_champions': [],
            'bear_market_defenders': [],
            'sideways_market_specialists': [],
            'volatility_masters': [],
            'all_weather_performers': [],
            'high_frequency_traders': [],
            'risk_adjusted_leaders': []
        }
        
        # Filter successful results
        successful_df = df[df['success'] == True].copy()
        
        if len(successful_df) == 0:
            return categories
        
        # Analyze performance by scenario
        scenario_performance = {}
        for scenario in ['bull', 'bear', 'sideways', 'volatile', 'mixed']:
            scenario_data = successful_df[successful_df['scenario'] == scenario]
            if len(scenario_data) > 0:
                scenario_performance[scenario] = scenario_data.groupby('strategy_name').agg({
                    'annual_return': 'mean',
                    'sharpe_ratio': 'mean',
                    'total_trades': 'mean'
                }).reset_index()
        
        # Bull Market Champions (best return in bull market)
        if 'bull' in scenario_performance:
            bull_leaders = scenario_performance['bull'].nlargest(5, 'annual_return')
            categories['bull_market_champions'] = bull_leaders['strategy_name'].tolist()
        
        # Bear Market Defenders (best return in bear market, or least negative)
        if 'bear' in scenario_performance:
            bear_defenders = scenario_performance['bear'].nlargest(5, 'annual_return')
            categories['bear_market_defenders'] = bear_defenders['strategy_name'].tolist()
        
        # Sideways Market Specialists
        if 'sideways' in scenario_performance:
            sideways_specialists = scenario_performance['sideways'].nlargest(5, 'annual_return')
            categories['sideways_market_specialists'] = sideways_specialists['strategy_name'].tolist()
        
        # Volatility Masters (best in high volatility)
        if 'volatile' in scenario_performance:
            volatility_masters = scenario_performance['volatile'].nlargest(5, 'sharpe_ratio')
            categories['volatility_masters'] = volatility_masters['strategy_name'].tolist()
        
        # All Weather Performers (top composite scores)
        if len(rankings) > 0:
            all_weather = rankings.head(8)['strategy_name'].tolist()
            categories['all_weather_performers'] = all_weather
        
        # High Frequency Traders (most trades)
        high_freq_data = successful_df.groupby('strategy_name')['total_trades'].mean().nlargest(5)
        categories['high_frequency_traders'] = high_freq_data.index.tolist()
        
        # Risk Adjusted Leaders (best Sharpe ratios)
        risk_adjusted_data = successful_df.groupby('strategy_name')['sharpe_ratio'].mean().nlargest(5)
        categories['risk_adjusted_leaders'] = risk_adjusted_data.index.tolist()
        
        # Print categorization results
        for category, strategies in categories.items():
            category_name = category.replace('_', ' ').title()
            print(f"🏆 {category_name}: {len(strategies)} strategies")
            for strategy in strategies[:3]:  # Show top 3
                print(f"   • {strategy}")
        
        return categories
        
    def generate_performance_report(self, df: pd.DataFrame, rankings: pd.DataFrame, categories: dict):
        """Generate comprehensive performance report."""
        print("\n📋 GENERATING PERFORMANCE REPORT")
        print("=" * 50)
        
        # Overall statistics
        successful_tests = df[df['success'] == True]
        total_tests = len(df)
        success_rate = len(successful_tests) / total_tests * 100
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"   Total strategy tests: {total_tests}")
        print(f"   Successful tests: {len(successful_tests)} ({success_rate:.1f}%)")
        print(f"   Average trades per test: {successful_tests['total_trades'].mean():.1f}")
        print(f"   Average annual return: {successful_tests['annual_return'].mean()*100:.1f}%")
        
        # Top performers
        if len(rankings) > 0:
            print(f"\n🏆 TOP 10 STRATEGY RANKINGS:")
            print(f"{'Rank':<5} {'Strategy':<25} {'Score':<8} {'Return':<8} {'Sharpe':<8} {'Success%':<8}")
            print("-" * 70)
            
            for i, row in rankings.head(10).iterrows():
                rank = rankings.index.get_loc(i) + 1
                strategy = row['strategy_name'][:23]
                score = row['composite_score']
                annual_return = row['avg_annual_return'] * 100
                sharpe = row['avg_sharpe_ratio']
                success = row['success_rate']
                
                print(f"{rank:<5} {strategy:<25} {score:<8.3f} {annual_return:<8.1f}% {sharpe:<8.2f} {success:<8.0f}%")
        
        # Category highlights
        print(f"\n🎯 STRATEGY CATEGORIES:")
        for category, strategies in categories.items():
            if strategies:
                category_name = category.replace('_', ' ').title()
                print(f"   {category_name}: {strategies[0]}")
        
        return {
            'total_tests': total_tests,
            'successful_tests': len(successful_tests),
            'success_rate': success_rate,
            'top_strategies': rankings.head(10).to_dict('records') if len(rankings) > 0 else [],
            'categories': categories
        }

def main():
    """Run comprehensive strategy ranking analysis."""
    ranking_system = StrategyRankingSystem()
    
    print("🎯 STRATEGY RANKING AND CATEGORIZATION SYSTEM")
    print("=" * 70)
    print("📈 Analyzing all 24 strategies across multiple market conditions")
    print("🏆 Creating comprehensive performance rankings and categories")
    
    # Run comprehensive analysis
    results_df = ranking_system.run_comprehensive_analysis()
    
    # Calculate composite scores and rankings
    rankings_df = ranking_system.calculate_composite_scores(results_df)
    
    # Categorize strategies
    categories = ranking_system.categorize_strategies(results_df, rankings_df)
    
    # Generate final report
    report = ranking_system.generate_performance_report(results_df, rankings_df, categories)
    
    # Save results
    print(f"\n💾 SAVING RESULTS...")
    results_df.to_csv('strategy_analysis_results.csv', index=False)
    rankings_df.to_csv('strategy_rankings.csv', index=False)
    
    with open('strategy_categories.json', 'w') as f:
        json.dump(categories, f, indent=2)
    
    with open('performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   ✅ Results saved to strategy_analysis_results.csv")
    print(f"   ✅ Rankings saved to strategy_rankings.csv")
    print(f"   ✅ Categories saved to strategy_categories.json")
    print(f"   ✅ Report saved to performance_report.json")
    
    print(f"\n🎉 STRATEGY RANKING ANALYSIS COMPLETE!")
    print(f"   🏆 {len(rankings_df)} strategies ranked and categorized")
    print(f"   📊 Ready for production strategy selection")

if __name__ == "__main__":
    main() 