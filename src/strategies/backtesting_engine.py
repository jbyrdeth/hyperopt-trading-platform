"""
Backtesting Engine for Trading Strategy Optimization

This module provides a realistic backtesting engine that can evaluate trading strategies
with proper modeling of transaction costs, slippage, and market impact.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_strategy import BaseStrategy, Signal, Position
try:
    from src.utils.logger import get_logger, log_performance
except ImportError:
    from src.utils.logger import get_logger, log_performance


@dataclass
class Trade:
    """Individual trade representation."""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    duration: timedelta
    entry_reason: str = "signal"
    exit_reason: str = "signal"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def return_pct(self) -> float:
        """Calculate percentage return."""
        if self.side == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:  # short
            return (self.entry_price - self.exit_price) / self.entry_price
    
    @property
    def duration_hours(self) -> float:
        """Get trade duration in hours."""
        return self.duration.total_seconds() / 3600


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    # Basic metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk (95%)
    
    # Advanced metrics
    kelly_criterion: float
    expectancy: float
    recovery_factor: float
    ulcer_index: float
    
    # Detailed data
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    monthly_returns: pd.Series
    
    # Costs breakdown
    total_commission: float
    total_slippage: float
    total_costs: float
    
    # Metadata
    start_date: datetime
    end_date: datetime
    strategy_name: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_trade_duration': self.avg_trade_duration,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'var_95': self.var_95,
            'kelly_criterion': self.kelly_criterion,
            'expectancy': self.expectancy,
            'recovery_factor': self.recovery_factor,
            'ulcer_index': self.ulcer_index,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_costs,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'strategy_name': self.strategy_name,
            'parameters': self.parameters
        }


class CostModel:
    """Model for transaction costs, slippage, and market impact."""
    
    def __init__(
        self,
        commission_rate: float = 0.001,  # 0.1%
        slippage_base: float = 0.0005,   # 0.05%
        slippage_impact: float = 0.0001, # Additional impact based on volatility
        min_commission: float = 0.0,     # Minimum commission
        max_slippage: float = 0.01       # Maximum slippage (1%)
    ):
        """
        Initialize cost model.
        
        Args:
            commission_rate: Commission as percentage of trade value
            slippage_base: Base slippage percentage
            slippage_impact: Additional slippage based on volatility
            min_commission: Minimum commission amount
            max_slippage: Maximum slippage percentage
        """
        self.commission_rate = commission_rate
        self.slippage_base = slippage_base
        self.slippage_impact = slippage_impact
        self.min_commission = min_commission
        self.max_slippage = max_slippage
    
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_slippage(
        self,
        trade_value: float,
        volatility: float,
        volume: float = None
    ) -> float:
        """
        Calculate slippage for a trade.
        
        Args:
            trade_value: Value of the trade
            volatility: Current volatility
            volume: Current volume (optional)
            
        Returns:
            Slippage amount
        """
        # Base slippage
        slippage_pct = self.slippage_base
        
        # Add volatility impact
        slippage_pct += volatility * self.slippage_impact
        
        # Cap at maximum
        slippage_pct = min(slippage_pct, self.max_slippage)
        
        return trade_value * slippage_pct
    
    def get_execution_price(
        self,
        market_price: float,
        side: str,
        volatility: float,
        volume: float = None
    ) -> float:
        """
        Get realistic execution price including slippage.
        
        Args:
            market_price: Market price
            side: 'buy' or 'sell'
            volatility: Current volatility
            volume: Current volume
            
        Returns:
            Execution price
        """
        # Calculate slippage percentage
        slippage_pct = self.slippage_base + (volatility * self.slippage_impact)
        slippage_pct = min(slippage_pct, self.max_slippage)
        
        # Apply slippage
        if side == 'buy':
            return market_price * (1 + slippage_pct)
        else:  # sell
            return market_price * (1 - slippage_pct)


class BacktestingEngine:
    """
    Comprehensive backtesting engine for trading strategies.
    
    Features:
    - Realistic cost modeling (commission, slippage, market impact)
    - Vectorized operations for performance
    - Comprehensive performance metrics
    - Risk management integration
    - Position sizing support
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        cost_model: CostModel = None,
        risk_free_rate: float = 0.02,  # 2% annual
        benchmark_return: float = 0.0
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            initial_capital: Starting capital
            cost_model: Cost model for transaction costs
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            benchmark_return: Benchmark return for comparison
        """
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.risk_free_rate = risk_free_rate
        self.benchmark_return = benchmark_return
        self.logger = get_logger("backtesting_engine")
        
        # State variables
        self.current_capital = initial_capital
        self.invested_capital = 0.0  # Track capital invested in positions
        self.total_transaction_costs = 0.0  # Track cumulative transaction costs
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        
        self.logger.debug(f"Initialized backtesting engine with ${initial_capital:,.2f}")
    
    def reset(self):
        """Reset backtesting engine state."""
        self.current_capital = self.initial_capital
        self.invested_capital = 0.0  # Track capital invested in positions
        self.total_transaction_costs = 0.0  # Track cumulative transaction costs
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
    
    @log_performance("backtest_strategy")
    def backtest_strategy(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT"
    ) -> BacktestResults:
        """
        Run a complete backtest for a strategy.
        
        Args:
            strategy: Strategy to backtest
            data: OHLCV data with datetime index
            symbol: Trading symbol
            
        Returns:
            BacktestResults object
        """
        self.logger.info(f"Starting backtest for {strategy.name}")
        
        # Reset state
        self.reset()
        strategy.reset()
        
        # Initialize strategy
        strategy.initialize(data)
        
        # Prepare data
        data = data.copy()
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        data['volatility'] = data['volatility'].fillna(data['volatility'].mean())
        
        # Track portfolio value
        portfolio_values = []
        
        # Main backtest loop
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            current_volatility = row['volatility']
            
            # Update strategy position with current price
            strategy.update_position(current_price)
            
            # Check for exit conditions
            should_exit, exit_reason = strategy.should_exit_position(
                current_price, timestamp, row
            )
            
            if should_exit and not strategy.current_position.is_flat:
                self._execute_exit(
                    strategy, current_price, timestamp, 
                    current_volatility, symbol, exit_reason
                )
            
            # Generate new signal
            signal = strategy.generate_signal(timestamp, row)
            strategy.signals_generated += 1
            
            # Process signal
            if signal.action in ['buy', 'sell'] and signal.strength > 0:
                self._execute_signal(
                    strategy, signal, current_price, timestamp,
                    current_volatility, symbol
                )
            
            # Calculate portfolio value
            position_value = 0.0
            
            if not strategy.current_position.is_flat:
                position = strategy.current_position
                position_value = position.size * current_price
            
            # CRITICAL FIX: Correct portfolio value calculation
            # Since we only deduct transaction costs from current_capital,
            # current_capital still includes the "cash equivalent" of our position
            # We need to avoid double-counting by using unrealized P&L approach
            
            if not strategy.current_position.is_flat:
                position = strategy.current_position
                # Calculate unrealized P&L
                unrealized_pnl = (current_price - position.entry_price) * position.size
                # Portfolio = Current capital + unrealized P&L
                total_value = self.current_capital + unrealized_pnl
            else:
                # No position, portfolio value = current available capital
                total_value = self.current_capital
            
            portfolio_values.append(total_value)
            self.timestamps.append(timestamp)
        
        # Close any remaining position
        if not strategy.current_position.is_flat:
            final_price = data.iloc[-1]['close']
            final_volatility = data.iloc[-1]['volatility']
            self._execute_exit(
                strategy, final_price, data.index[-1],
                final_volatility, symbol, "end_of_data"
            )
        
        # Calculate results
        results = self._calculate_results(
            strategy, data, portfolio_values, symbol
        )
        
        self.logger.info(
            f"Backtest completed: {len(self.trades)} trades, "
            f"{results.annual_return:.2%} annual return, "
            f"{results.sharpe_ratio:.2f} Sharpe ratio"
        )
        
        return results
    
    def _execute_signal(
        self,
        strategy: BaseStrategy,
        signal: Signal,
        current_price: float,
        timestamp: datetime,
        volatility: float,
        symbol: str
    ):
        """Execute a trading signal."""
        # Calculate position size
        # Reserve some capital for transaction costs (estimate 0.2%)
        available_capital = self.current_capital * 0.998
        position_size = strategy.calculate_position_size(
            signal, current_price, available_capital, volatility
        )
        
        if abs(position_size) < 1e-8:  # Too small position
            return
        
        # Determine trade side
        side = 'buy' if position_size > 0 else 'sell'
        
        # Get execution price with slippage
        execution_price = self.cost_model.get_execution_price(
            current_price, side, volatility
        )
        
        # Calculate trade value and costs
        trade_value = abs(position_size * execution_price)
        commission = self.cost_model.calculate_commission(trade_value)
        slippage = self.cost_model.calculate_slippage(trade_value, volatility)
        
        # Check if we have enough capital
        total_cost = trade_value + commission + slippage
        
        if total_cost > self.current_capital:
            # This should rarely happen now since we reserved capital for costs
            # But if it does, reduce position size to fit
            max_total_cost = self.current_capital * 0.99  # Leave 1% buffer
            max_trade_value = max_total_cost * 0.998  # Estimate for trade value portion
            position_size = (max_trade_value / execution_price) * np.sign(position_size)
            trade_value = abs(position_size * execution_price)
            commission = self.cost_model.calculate_commission(trade_value)
            slippage = self.cost_model.calculate_slippage(trade_value, volatility)
            total_cost = trade_value + commission + slippage
        
        if total_cost <= self.current_capital and abs(position_size) > 1e-8:
            # Execute the trade
            strategy.open_position(symbol, position_size, execution_price, timestamp)
            
            # CRITICAL FIX: Only deduct transaction costs when opening position
            # When buying an asset, we exchange cash for equivalent asset value
            # We only lose the transaction costs (commission + slippage)
            transaction_costs = commission + slippage
            self.current_capital -= transaction_costs
            self.total_transaction_costs += transaction_costs
            
            # Track invested capital for portfolio calculation
            self.invested_capital += trade_value
            
            self.logger.debug(
                f"Opened position: {position_size:.4f} @ {execution_price:.4f}, "
                f"Trade value: {trade_value:.4f}, Commission: {commission:.4f}, "
                f"Slippage: {slippage:.4f}, Transaction costs: {transaction_costs:.4f}, "
                f"Available cash: {self.current_capital:.4f}, Invested: {self.invested_capital:.4f}"
            )
    
    def _execute_exit(
        self,
        strategy: BaseStrategy,
        current_price: float,
        timestamp: datetime,
        volatility: float,
        symbol: str,
        reason: str
    ):
        """Execute position exit."""
        if strategy.current_position.is_flat:
            return
        
        position = strategy.current_position
        
        # Determine exit side
        side = 'sell' if position.size > 0 else 'buy'
        
        # Get execution price with slippage
        execution_price = self.cost_model.get_execution_price(
            current_price, side, volatility
        )
        
        # Calculate trade details
        trade_value = abs(position.size * execution_price)
        commission = self.cost_model.calculate_commission(trade_value)
        slippage = self.cost_model.calculate_slippage(trade_value, volatility)
        
        # Calculate P&L
        gross_pnl = (execution_price - position.entry_price) * position.size
        net_pnl = gross_pnl - commission - slippage
        
        # Create trade record
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=timestamp,
            symbol=symbol,
            side='long' if position.size > 0 else 'short',
            entry_price=position.entry_price,
            exit_price=execution_price,
            size=abs(position.size),
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            net_pnl=net_pnl,
            duration=timestamp - position.entry_time,
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # CRITICAL FIX: Proper capital management to prevent exponential compounding
        transaction_costs = commission + slippage
        self.total_transaction_costs += transaction_costs
        
        # The key insight: When we opened the position, we only deducted transaction costs
        # but the "invested capital" represents the value tied up in the asset.
        # When we exit, we need to:
        # 1. Get back our original invested amount
        # 2. Add/subtract the profit/loss from price movement
        # 3. Subtract the exit transaction costs
        
        original_investment = abs(position.size * position.entry_price)
        proceeds_from_sale = trade_value  # What we get from selling the asset
        
        # Net change in capital = proceeds - original_investment - transaction_costs
        # This equals the net P&L minus transaction costs, which is exactly net_pnl
        capital_change = proceeds_from_sale - original_investment - transaction_costs
        
        # Update capital with the net change (profit/loss minus costs)
        self.current_capital += capital_change
        
        # Remove the invested capital from tracking since position is closed
        self.invested_capital -= original_investment
        
        # Close position in strategy
        strategy.close_position(execution_price, timestamp, reason)
        
        self.logger.debug(
            f"Closed position: {trade.side} {trade.size:.4f}, "
            f"Original investment: {original_investment:.4f}, "
            f"Proceeds: {proceeds_from_sale:.4f}, "
            f"Transaction costs: {transaction_costs:.4f}, "
            f"Net capital change: {capital_change:.4f}, "
            f"Available cash: {self.current_capital:.4f}, "
            f"Invested: {self.invested_capital:.4f}, Reason: {reason}"
        )
    
    def _calculate_results(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        portfolio_values: List[float],
        symbol: str
    ) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        
        # Create equity curve
        equity_curve = pd.Series(portfolio_values, index=self.timestamps)
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.net_pnl > 0]
            losing_trades = [t for t in self.trades if t.net_pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0
            largest_win = max([t.net_pnl for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t.net_pnl for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum([t.net_pnl for t in winning_trades])
            gross_loss = abs(sum([t.net_pnl for t in losing_trades]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Average trade duration
            avg_trade_duration = np.mean([t.duration_hours for t in self.trades])
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            # Kelly criterion
            if avg_loss != 0:
                kelly_criterion = win_rate - ((1 - win_rate) / (avg_win / abs(avg_loss)))
            else:
                kelly_criterion = 0
            
        else:
            win_rate = 0
            avg_win = avg_loss = largest_win = largest_loss = 0
            profit_factor = 0
            avg_trade_duration = 0
            expectancy = 0
            kelly_criterion = 0
        
        # Advanced risk metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # VaR (95%)
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt((drawdown ** 2).mean()) if len(drawdown) > 0 else 0
        
        # Recovery factor
        total_pnl = sum([t.net_pnl for t in self.trades])
        recovery_factor = total_pnl / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Monthly returns
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        # Cost breakdown
        total_commission = sum([t.commission for t in self.trades])
        total_slippage = sum([t.slippage for t in self.trades])
        total_costs = total_commission + total_slippage
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades) if self.trades else 0,
            losing_trades=len(losing_trades) if self.trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            volatility=volatility,
            downside_deviation=downside_deviation,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            kelly_criterion=kelly_criterion,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            trades=self.trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            monthly_returns=monthly_returns,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_costs=total_costs,
            start_date=data.index[0],
            end_date=data.index[-1],
            strategy_name=strategy.name,
            parameters=strategy.parameters
        ) 