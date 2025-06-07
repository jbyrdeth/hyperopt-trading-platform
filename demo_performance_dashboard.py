#!/usr/bin/env python3
"""
Performance Dashboard Demonstration

This script demonstrates the comprehensive capabilities of the PerformanceDashboard
for trading strategy evaluation and comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analytics import PerformanceDashboard, DashboardConfig
from analytics.strategy_comparator import RankingMethod
from strategies.backtesting_engine import BacktestResults, Trade
from utils.logger import get_logger

def create_sample_backtest_results(strategy_name: str, seed: int = 42) -> BacktestResults:
    """Create sample backtest results for demonstration."""
    
    np.random.seed(seed)
    
    # Generate sample data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate sample price data
    initial_price = 100
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    price_data = pd.Series(prices, index=dates)
    
    # Generate sample trades
    trades = []
    num_trades = np.random.randint(50, 150)
    
    for i in range(num_trades):
        entry_date = np.random.choice(dates[:-10])
        exit_date = entry_date + timedelta(days=np.random.randint(1, 20))
        
        entry_price = price_data[entry_date]
        exit_price = price_data[exit_date] if exit_date in price_data.index else price_data.iloc[-1]
        
        quantity = np.random.randint(10, 100)
        side = np.random.choice(['long', 'short'])
        
        if side == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        trade = Trade(
            entry_time=entry_date,
            exit_time=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side=side,
            pnl=pnl,
            commission=abs(pnl) * 0.001,  # 0.1% commission
            strategy_name=strategy_name
        )
        trades.append(trade)
    
    # Calculate equity curve
    equity_curve = pd.Series(index=dates, dtype=float)
    equity_curve.iloc[0] = 10000  # Starting capital
    
    for i in range(1, len(equity_curve)):
        date = equity_curve.index[i]
        # Add PnL from trades that closed on this date
        daily_pnl = sum(trade.pnl for trade in trades if trade.exit_time.date() == date.date())
        equity_curve.iloc[i] = equity_curve.iloc[i-1] + daily_pnl
    
    # Calculate basic metrics
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
    
    returns_series = equity_curve.pct_change().dropna()
    volatility = returns_series.std() * np.sqrt(252)  # Annualized
    
    sharpe_ratio = (returns_series.mean() * 252) / (returns_series.std() * np.sqrt(252)) if returns_series.std() > 0 else 0
    
    # Calculate drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calculate other metrics
    winning_trades = [t for t in trades if t.pnl > 0]
    win_rate = len(winning_trades) / len(trades) if trades else 0
    
    total_profit = sum(t.pnl for t in trades if t.pnl > 0)
    total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Monthly returns
    monthly_equity = equity_curve.resample('M').last()
    monthly_returns = monthly_equity.pct_change().dropna()
    
    # Sortino ratio
    negative_returns = returns_series[returns_series < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.001
    sortino_ratio = (returns_series.mean() * 252) / downside_deviation
    
    # Calmar ratio
    calmar_ratio = (returns_series.mean() * 252) / abs(max_drawdown) if max_drawdown < 0 else 0
    
    return BacktestResults(
        strategy_name=strategy_name,
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000,
        final_capital=equity_curve.iloc[-1],
        total_return=total_return,
        annual_return=total_return * (365 / (end_date - start_date).days),
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(trades) - len(winning_trades),
        avg_trade_duration=np.mean([(t.exit_time - t.entry_time).days for t in trades]),
        trades=trades,
        equity_curve=equity_curve,
        drawdown_curve=drawdown,
        monthly_returns=monthly_returns,
        commission_paid=sum(t.commission for t in trades),
        slippage_cost=0
    )

def create_sample_benchmark_data() -> pd.Series:
    """Create sample benchmark data."""
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate benchmark returns (market-like)
    np.random.seed(123)
    returns = np.random.normal(0.0003, 0.015, len(dates))  # Slightly lower vol than strategies
    
    benchmark_values = [10000]  # Starting value
    for ret in returns[1:]:
        benchmark_values.append(benchmark_values[-1] * (1 + ret))
    
    return pd.Series(benchmark_values, index=dates)

def demonstrate_comprehensive_dashboard():
    """Demonstrate the comprehensive dashboard functionality."""
    
    logger = get_logger("dashboard_demo")
    logger.info("Starting comprehensive dashboard demonstration")
    
    # Create sample strategies
    strategies = {
        'Momentum Strategy': create_sample_backtest_results('Momentum Strategy', seed=42),
        'Mean Reversion': create_sample_backtest_results('Mean Reversion', seed=123),
        'Trend Following': create_sample_backtest_results('Trend Following', seed=456),
        'Breakout Strategy': create_sample_backtest_results('Breakout Strategy', seed=789),
        'Scalping Strategy': create_sample_backtest_results('Scalping Strategy', seed=101)
    }
    
    # Create benchmark data
    benchmark_data = create_sample_benchmark_data()
    
    # Configure dashboard
    config = DashboardConfig(
        theme="plotly_white",
        figure_width=1400,
        figure_height=700,
        max_strategies_display=8
    )
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(config=config, include_validation=True)
    
    logger.info("Creating comprehensive dashboard...")
    
    # Create comprehensive dashboard
    summary, figures = dashboard.create_comprehensive_dashboard(
        backtest_results=strategies,
        benchmark_data=benchmark_data,
        save_path="dashboard_output"
    )
    
    # Print executive summary
    strategy_insights = dashboard._create_strategy_insights(
        {name: dashboard.performance_analyzer.analyze_performance(results) 
         for name, results in strategies.items()},
        dashboard.strategy_comparator.compare_strategies(
            {name: dashboard.performance_analyzer.analyze_performance(results) 
             for name, results in strategies.items()}
        )
    )
    
    executive_summary = dashboard.generate_executive_summary(summary, strategy_insights)
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DASHBOARD RESULTS")
    print("="*80)
    print(executive_summary)
    print("="*80)
    
    # Display dashboard summary
    print(f"\nDashboard Summary:")
    print(f"- Total Strategies: {summary.total_strategies}")
    print(f"- Best Strategy: {summary.best_strategy}")
    print(f"- Deployment Ready: {summary.deployment_ready_count}")
    print(f"- Average Sharpe: {summary.avg_sharpe_ratio:.2f}")
    print(f"- Average Max DD: {summary.avg_max_drawdown:.1%}")
    
    if summary.avg_robustness_score:
        print(f"- Avg Robustness: {summary.avg_robustness_score:.1f}/100")
    
    print(f"\nGenerated {len(figures)} visualizations:")
    for name in figures.keys():
        print(f"  - {name}")
    
    logger.info("Comprehensive dashboard demonstration completed")
    return summary, figures

def demonstrate_strategy_scorecard():
    """Demonstrate individual strategy scorecard functionality."""
    
    logger = get_logger("scorecard_demo")
    logger.info("Starting strategy scorecard demonstration")
    
    # Create a sample strategy
    strategy_results = create_sample_backtest_results('Advanced Momentum', seed=999)
    benchmark_data = create_sample_benchmark_data()
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(include_validation=True)
    
    logger.info("Creating strategy scorecard...")
    
    # Create strategy scorecard
    insight, figures = dashboard.create_strategy_scorecard(
        strategy_name='Advanced Momentum',
        backtest_results=strategy_results,
        benchmark_data=benchmark_data
    )
    
    print("\n" + "="*60)
    print("STRATEGY SCORECARD RESULTS")
    print("="*60)
    print(f"Strategy: {insight.strategy_name}")
    print(f"Overall Score: {insight.overall_score:.1f}")
    print(f"Risk Level: {insight.risk_level}")
    print(f"Deployment Ready: {'Yes' if insight.deployment_ready else 'No'}")
    print(f"Confidence: {insight.confidence_level}")
    print(f"\nKey Metrics:")
    print(f"  Total Return: {insight.total_return:.1%}")
    print(f"  Sharpe Ratio: {insight.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {insight.max_drawdown:.1%}")
    print(f"  Win Rate: {insight.win_rate:.1%}")
    print(f"  Volatility: {insight.volatility:.1%}")
    
    if insight.robustness_score:
        print(f"  Robustness Score: {insight.robustness_score:.1f}/100")
    
    print(f"\nStrengths:")
    for strength in insight.strengths:
        print(f"  + {strength}")
    
    print(f"\nWeaknesses:")
    for weakness in insight.weaknesses:
        print(f"  - {weakness}")
    
    print(f"\nRecommendations:")
    for rec in insight.recommendations:
        print(f"  → {rec}")
    
    print(f"\nGenerated {len(figures)} detailed visualizations:")
    for name in figures.keys():
        print(f"  - {name}")
    
    print("="*60)
    
    logger.info("Strategy scorecard demonstration completed")
    return insight, figures

def demonstrate_comparison_dashboard():
    """Demonstrate strategy comparison functionality."""
    
    logger = get_logger("comparison_demo")
    logger.info("Starting comparison dashboard demonstration")
    
    # Create sample strategies for comparison
    strategies = {
        'High Sharpe Strategy': create_sample_backtest_results('High Sharpe Strategy', seed=111),
        'Low Risk Strategy': create_sample_backtest_results('Low Risk Strategy', seed=222),
        'High Return Strategy': create_sample_backtest_results('High Return Strategy', seed=333)
    }
    
    # Initialize dashboard
    dashboard = PerformanceDashboard()
    
    logger.info("Creating comparison dashboard...")
    
    # Test different ranking methods
    ranking_methods = [
        RankingMethod.WEIGHTED_SCORE,
        RankingMethod.SHARPE_RATIO,
        RankingMethod.CALMAR_RATIO
    ]
    
    for method in ranking_methods:
        print(f"\n" + "="*50)
        print(f"COMPARISON RESULTS - {method.value.upper()}")
        print("="*50)
        
        comparison_result, figure = dashboard.create_comparison_dashboard(
            backtest_results=strategies,
            ranking_method=method
        )
        
        print(f"Ranking Method: {comparison_result.ranking_method.value}")
        print(f"Strategies Analyzed: {comparison_result.strategies_count}")
        
        print(f"\nRankings:")
        for i, ranking in enumerate(comparison_result.rankings, 1):
            print(f"  {i}. {ranking.strategy_name} (Score: {ranking.score:.1f})")
        
        print(f"\nTop Performers: {', '.join(comparison_result.top_performers)}")
        print(f"Consistent Performers: {', '.join(comparison_result.consistent_performers)}")
        print(f"High Risk Strategies: {', '.join(comparison_result.high_risk_strategies)}")
    
    logger.info("Comparison dashboard demonstration completed")

def main():
    """Main demonstration function."""
    
    print("Performance Dashboard Demonstration")
    print("==================================")
    print("This demo showcases the comprehensive capabilities of the PerformanceDashboard")
    print("for trading strategy evaluation and comparison.\n")
    
    try:
        # Demonstrate comprehensive dashboard
        print("1. Comprehensive Dashboard Demo")
        print("-" * 30)
        summary, figures = demonstrate_comprehensive_dashboard()
        
        # Demonstrate strategy scorecard
        print("\n\n2. Strategy Scorecard Demo")
        print("-" * 25)
        insight, scorecard_figures = demonstrate_strategy_scorecard()
        
        # Demonstrate comparison dashboard
        print("\n\n3. Strategy Comparison Demo")
        print("-" * 27)
        demonstrate_comparison_dashboard()
        
        print("\n\nDemonstration completed successfully!")
        print("Check the 'dashboard_output' directory for saved visualizations.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 