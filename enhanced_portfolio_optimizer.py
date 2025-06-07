#!/usr/bin/env python3
"""
Enhanced Portfolio Optimizer - Address Performance Crisis

This script addresses the critical performance issue where all 24 strategies 
are underperforming the market benchmark. Key improvements:

1. AGGRESSIVE PARAMETER EXPANSION: Dramatically widen constraint ranges
2. PORTFOLIO OPTIMIZATION: Combine strategies for better risk-adjusted returns  
3. CORRELATION ANALYSIS: Identify complementary strategy combinations
4. DYNAMIC WEIGHTING: Performance-based and risk-parity allocation schemes
5. MARKET REGIME ANALYSIS: Test across different market conditions

Target: Transform -3.9% average return to competitive 15-45% annual returns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.optimization.strategy_factory import StrategyFactory
from src.strategies.backtesting_engine import BacktestingEngine
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import itertools

class EnhancedPortfolioOptimizer:
    """Advanced portfolio optimizer targeting performance breakthrough."""
    
    def __init__(self):
        self.factory = StrategyFactory()
        self.engine = BacktestingEngine()
        
        # AGGRESSIVE PARAMETER RANGES - Dramatically expanded
        self.aggressive_params = {
            'MovingAverageCrossover': {
                'fast_period': [2, 3, 4, 5, 8, 10, 12],
                'slow_period': [15, 20, 25, 30, 40, 50, 100],
                'ma_type': ['SMA', 'EMA', 'WMA'],
                'signal_threshold': [0.001, 0.002, 0.005, 0.01, 0.02],
                'position_size_pct': [0.8, 0.9, 1.0],
                'stop_loss_pct': [0.02, 0.03, 0.05, 0.08],
                'take_profit_pct': [0.04, 0.06, 0.08, 0.12, 0.15]
            },
            'MACD': {
                'fast_period': [3, 5, 8, 10, 12],
                'slow_period': [15, 20, 26, 30, 35],
                'signal_period': [5, 7, 9, 12, 15],
                'histogram_threshold': [0.001, 0.003, 0.005, 0.01],
                'position_size_pct': [0.8, 0.9, 1.0],
                'stop_loss_pct': [0.02, 0.03, 0.05],
                'take_profit_pct': [0.04, 0.06, 0.08, 0.12]
            },
            'RSI': {
                'rsi_period': [5, 7, 10, 14, 21],
                'overbought': [65, 70, 75, 80],
                'oversold': [20, 25, 30, 35],
                'exit_signal': ['opposite', 'middle'],
                'position_size_pct': [0.8, 0.9, 1.0],
                'stop_loss_pct': [0.02, 0.03, 0.05],
                'take_profit_pct': [0.04, 0.06, 0.08, 0.12]
            },
            'BollingerBands': {
                'period': [10, 15, 20, 25],
                'std_dev': [1.5, 2.0, 2.5],
                'entry_method': ['breakout', 'mean_reversion'],
                'position_size_pct': [0.8, 0.9, 1.0],
                'stop_loss_pct': [0.02, 0.03, 0.05],
                'take_profit_pct': [0.04, 0.06, 0.08]
            }
        }
        
        # Portfolio optimization settings
        self.portfolio_settings = {
            'max_strategies': 5,  # Maximum strategies in portfolio
            'min_correlation': -0.5,  # Minimum correlation for diversification
            'max_correlation': 0.7,   # Maximum correlation to avoid redundancy
            'rebalance_frequency': 'monthly',
            'transaction_costs': 0.001  # 0.1% per trade
        }
        
    def create_enhanced_test_data(self, days: int = 180) -> pd.DataFrame:
        """Create more comprehensive test data for better strategy evaluation."""
        np.random.seed(42)
        
        # Create different market phases
        n_points = days * 24  # Hourly data
        
        # Phase 1: Bull market (40% of data)
        bull_points = int(n_points * 0.4)
        bull_trend = np.random.normal(0.0008, 0.02, bull_points).cumsum()
        
        # Phase 2: Bear market (30% of data) 
        bear_points = int(n_points * 0.3)
        bear_trend = np.random.normal(-0.0005, 0.025, bear_points).cumsum()
        
        # Phase 3: Sideways market (30% of data)
        sideways_points = n_points - bull_points - bear_points
        sideways_trend = np.random.normal(0.0001, 0.015, sideways_points).cumsum()
        
        # Combine phases
        price_changes = np.concatenate([bull_trend, bear_trend, sideways_trend])
        
        # Generate OHLCV data
        base_price = 50000
        prices = base_price * np.exp(price_changes)
        
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices))))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices))))
        
        # Ensure OHLC logic
        opens = np.roll(prices, 1)
        opens[0] = base_price
        
        volume = np.random.lognormal(15, 1, len(prices))
        
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
        
    def optimize_single_strategy(self, strategy_name: str, data: pd.DataFrame, 
                               max_combinations: int = 50) -> List[Dict[str, Any]]:
        """Aggressively optimize a single strategy with expanded parameters."""
        if strategy_name not in self.aggressive_params:
            return []
            
        results = []
        param_sets = self.aggressive_params[strategy_name]
        
        # Generate parameter combinations
        param_names = list(param_sets.keys())
        param_values = [param_sets[name] for name in param_names]
        
        # Limit combinations for performance
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) > max_combinations:
            # Sample randomly for diversity
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations
            
        print(f"Testing {len(combinations)} parameter combinations for {strategy_name}...")
        
        for i, combo in enumerate(combinations):
            try:
                # Create parameter dict
                params = dict(zip(param_names, combo))
                
                # Create strategy using factory
                strategy = self.factory.create_strategy(strategy_name, **params)
                
                # Run backtest
                result = self.engine.backtest(strategy, data)
                
                if result['total_trades'] > 0 and result['total_return'] > -0.5:
                    results.append({
                        'strategy': strategy_name,
                        'params': params,
                        'total_return': result['total_return'],
                        'total_trades': result['total_trades'],
                        'sharpe_ratio': result.get('sharpe_ratio', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'win_rate': result.get('win_rate', 0),
                        'profit_factor': result.get('profit_factor', 0),
                        'annual_return': result['total_return'] * (365 / (len(data) / 24))
                    })
                    
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(combinations)} combinations tested")
                    
            except Exception as e:
                continue
                
        # Sort by annual return
        results.sort(key=lambda x: x['annual_return'], reverse=True)
        return results[:10]  # Top 10 variations
        
    def calculate_portfolio_metrics(self, strategies: List[Dict], weights: List[float], 
                                  data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate portfolio-level performance metrics."""
        if len(strategies) != len(weights):
            raise ValueError("Strategies and weights must have same length")
            
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Run each strategy
        equity_curves = []
        total_returns = []
        
        for strategy_config in strategies:
            try:
                # Use factory to create strategy
                strategy = self.factory.create_strategy(strategy_config['strategy'], **strategy_config['params'])
                result = self.engine.backtest(strategy, data)
                
                if 'equity_curve' in result:
                    equity_curves.append(result['equity_curve'])
                    total_returns.append(result['total_return'])
                else:
                    # Handle missing equity curve
                    equity_curves.append(pd.Series([1.0] * len(data), index=data.index))
                    total_returns.append(0.0)
                    
            except Exception as e:
                print(f"Error running strategy {strategy_config['strategy']}: {e}")
                equity_curves.append(pd.Series([1.0] * len(data), index=data.index))
                total_returns.append(0.0)
                
        if not equity_curves:
            return {'error': 'No valid strategies in portfolio'}
            
        # Combine equity curves with weights
        portfolio_equity = pd.Series(0.0, index=data.index)
        for curve, weight in zip(equity_curves, weights):
            portfolio_equity += curve * weight
            
        # Calculate portfolio metrics
        portfolio_return = portfolio_equity.iloc[-1] - 1.0
        
        # Calculate volatility
        returns = portfolio_equity.pct_change().dropna()
        volatility = returns.std() * np.sqrt(365 * 24)  # Annualized
        
        # Sharpe ratio
        sharpe = (portfolio_return * (365 * 24 / len(data))) / volatility if volatility > 0 else 0
        
        # Max drawdown
        peak = portfolio_equity.expanding().max()
        drawdown = (portfolio_equity - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'portfolio_return': portfolio_return,
            'annual_return': portfolio_return * (365 * 24 / len(data)),
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'strategies': [s['strategy'] for s in strategies],
            'weights': weights.tolist(),
            'individual_returns': total_returns
        }
        
    def find_optimal_portfolios(self, strategy_results: Dict[str, List], 
                               data: pd.DataFrame, top_n: int = 20) -> List[Dict]:
        """Find optimal multi-strategy portfolios."""
        print("\n🔍 SEARCHING FOR OPTIMAL PORTFOLIOS")
        print("=" * 50)
        
        portfolios = []
        
        # Get top performers from each strategy
        all_configs = []
        for strategy_name, results in strategy_results.items():
            if results:
                # Take top 3 configurations per strategy
                all_configs.extend(results[:3])
                
        if len(all_configs) < 2:
            print("⚠️ Not enough strategies for portfolio optimization")
            return []
            
        print(f"📊 Testing combinations of {len(all_configs)} top strategy configurations")
        
        # Test portfolio combinations
        for portfolio_size in [2, 3, 4, 5]:
            print(f"\n🧪 Testing {portfolio_size}-strategy portfolios...")
            
            # Generate combinations
            from itertools import combinations
            combos = list(combinations(all_configs, portfolio_size))
            
            # Limit for performance
            if len(combos) > 100:
                combos = np.random.choice(combos, 100, replace=False)
                
            for i, strategy_combo in enumerate(combos):
                # Test equal weights
                equal_weights = [1.0/portfolio_size] * portfolio_size
                
                try:
                    metrics = self.calculate_portfolio_metrics(
                        list(strategy_combo), equal_weights, data
                    )
                    
                    if 'error' not in metrics:
                        portfolios.append({
                            'type': f'{portfolio_size}_strategy_equal_weight',
                            'size': portfolio_size,
                            **metrics
                        })
                        
                except Exception as e:
                    continue
                    
                if (i + 1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(combos)} combinations tested")
                    
        # Sort by Sharpe ratio
        portfolios.sort(key=lambda x: x.get('sharpe_ratio', -999), reverse=True)
        
        print(f"\n📈 Generated {len(portfolios)} portfolio combinations")
        return portfolios[:top_n]
        
    def run_enhanced_optimization(self) -> Dict[str, Any]:
        """Run comprehensive optimization addressing performance issues."""
        print("🚀 ENHANCED PORTFOLIO OPTIMIZATION")
        print("=" * 50)
        print("🎯 Target: Transform underperforming strategies into profitable portfolios")
        print("📈 Approach: Aggressive parameter expansion + portfolio combinations")
        
        # Create enhanced test data
        print("\n📊 Creating enhanced test data (180 days, multiple market phases)...")
        data = self.create_enhanced_test_data(days=180)
        market_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        print(f"   📈 Market return: {market_return:.2f}%")
        print(f"   📅 Data points: {len(data)} (hourly)")
        
        # Optimize core strategies
        core_strategies = ['MovingAverageCrossover', 'MACD', 'RSI', 'BollingerBands']
        strategy_results = {}
        
        print(f"\n🔧 AGGRESSIVE PARAMETER OPTIMIZATION")
        print("=" * 50)
        
        for strategy in core_strategies:
            print(f"\n⚡ Optimizing {strategy}...")
            results = self.optimize_single_strategy(strategy, data)
            strategy_results[strategy] = results
            
            if results:
                best = results[0]
                print(f"   🏆 Best: {best['annual_return']:.1f}% annual, {best['total_trades']} trades")
                print(f"   📊 Sharpe: {best['sharpe_ratio']:.2f}, Drawdown: {best['max_drawdown']:.1%}")
            else:
                print(f"   ❌ No profitable configurations found")
                
        # Find optimal portfolios
        optimal_portfolios = self.find_optimal_portfolios(strategy_results, data)
        
        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'market_benchmark': market_return,
            'data_period_days': 180,
            'optimization_results': {
                'individual_strategies': strategy_results,
                'optimal_portfolios': optimal_portfolios[:10]
            },
            'performance_breakthrough': {
                'best_individual_annual_return': max([
                    max([r['annual_return'] for r in results], default=-999) 
                    for results in strategy_results.values()
                ], default=-999),
                'best_portfolio_annual_return': max([
                    p.get('annual_return', -999) for p in optimal_portfolios
                ], default=-999),
                'best_portfolio_sharpe': max([
                    p.get('sharpe_ratio', -999) for p in optimal_portfolios
                ], default=-999)
            }
        }
        
        return summary
        
    def print_optimization_report(self, summary: Dict[str, Any]):
        """Print comprehensive optimization report."""
        print("\n🎯 ENHANCED OPTIMIZATION RESULTS")
        print("=" * 50)
        
        market_return = summary['market_benchmark']
        breakthrough = summary['performance_breakthrough']
        
        print(f"📊 Market Benchmark: {market_return:.2f}%")
        print(f"📈 Best Individual Strategy: {breakthrough['best_individual_annual_return']:.1f}% annual")
        print(f"🏆 Best Portfolio Return: {breakthrough['best_portfolio_annual_return']:.1f}% annual")
        print(f"📊 Best Portfolio Sharpe: {breakthrough['best_portfolio_sharpe']:.2f}")
        
        # Show top portfolios
        portfolios = summary['optimization_results']['optimal_portfolios']
        if portfolios:
            print(f"\n🏆 TOP 5 PORTFOLIO COMBINATIONS:")
            print("-" * 50)
            for i, portfolio in enumerate(portfolios[:5], 1):
                strategies_str = " + ".join(portfolio['strategies'])
                print(f"{i:2d}. {portfolio['annual_return']:6.1f}% | "
                      f"Sharpe: {portfolio['sharpe_ratio']:5.2f} | "
                      f"DD: {portfolio['max_drawdown']:6.1%} | "
                      f"{strategies_str}")
                      
        # Performance breakthrough assessment
        best_portfolio = breakthrough['best_portfolio_annual_return']
        if best_portfolio > market_return:
            print(f"\n🎉 PERFORMANCE BREAKTHROUGH ACHIEVED!")
            print(f"   📈 Portfolio beats market by {best_portfolio - market_return:.1f}%")
        elif best_portfolio > 15:
            print(f"\n✅ COMPETITIVE PERFORMANCE ACHIEVED!")
            print(f"   📈 {best_portfolio:.1f}% annual return is competitive")
        else:
            print(f"\n⚠️ PERFORMANCE STILL BELOW TARGET")
            print(f"   📉 {best_portfolio:.1f}% vs 15%+ target - needs further optimization")

def main():
    """Run enhanced portfolio optimization."""
    optimizer = EnhancedPortfolioOptimizer()
    
    print("🚀 ENHANCED PORTFOLIO OPTIMIZATION - PERFORMANCE BREAKTHROUGH")
    print("=" * 70)
    print("🎯 Mission: Transform underperforming strategies into profitable portfolios")
    print("📊 Method: Aggressive parameter expansion + multi-strategy combinations")
    
    # Run optimization
    summary = optimizer.run_enhanced_optimization()
    
    # Print results
    optimizer.print_optimization_report(summary)
    
    # Save results
    filename = f"enhanced_portfolio_optimization_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📁 Complete results saved to: {filename}")
    
    print(f"\n🎯 OPTIMIZATION COMPLETE")
    print("=" * 50)
    
if __name__ == "__main__":
    main() 