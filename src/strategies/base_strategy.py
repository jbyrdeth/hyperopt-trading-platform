"""
Base Strategy Class for Trading Strategy Optimization

This module provides the abstract base class that all trading strategies inherit from.
It defines the common interface and provides utility methods for strategy implementation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import logging

from utils.logger import get_logger


@dataclass
class Signal:
    """Trading signal representation."""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # Signal strength (0-1)
    price: float
    confidence: float  # Confidence level (0-1)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Position representation."""
    symbol: str
    size: float  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat (no position)."""
        return self.size == 0
    
    def update_price(self, new_price: float):
        """Update current price and unrealized PnL."""
        self.current_price = new_price
        if self.size != 0:
            self.unrealized_pnl = (new_price - self.entry_price) * self.size


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all strategies must implement
    and provides common utility methods for strategy development.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Dict[str, Any] = None,
        risk_params: Dict[str, Any] = None
    ):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            risk_params: Risk management parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.risk_params = risk_params or {}
        self.logger = get_logger(f"strategy.{name}")
        
        # Strategy state
        self.is_initialized = False
        self.current_position = Position(
            symbol="",
            size=0.0,
            entry_price=0.0,
            entry_time=datetime.now(),
            current_price=0.0,
            unrealized_pnl=0.0
        )
        
        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.indicators: Dict[str, pd.Series] = {}
        
        self.logger.debug(f"Initialized strategy: {name}")
    
    @abstractmethod
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize the strategy with historical data.
        
        This method should:
        1. Store the data
        2. Calculate any required indicators
        3. Set up initial state
        
        Args:
            data: OHLCV DataFrame with datetime index
        """
        pass
    
    @abstractmethod
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """
        Generate a trading signal for the current time.
        
        Args:
            current_time: Current timestamp
            current_data: Current OHLCV data row
            
        Returns:
            Signal object with trading decision
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal and risk parameters.
        
        Args:
            signal: Trading signal
            current_price: Current asset price
            available_capital: Available capital for trading
            current_volatility: Current asset volatility
            
        Returns:
            Position size (positive for long, negative for short)
        """
        pass
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        # Default implementation - can be overridden
        return True
    
    def get_required_indicators(self) -> List[str]:
        """
        Get list of required technical indicators.
        
        Returns:
            List of indicator names
        """
        # Default implementation - can be overridden
        return []
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary of indicator name -> Series
        """
        # Default implementation - can be overridden
        return {}
    
    def get_stop_loss_price(self, entry_price: float, position_size: float) -> Optional[float]:
        """
        Calculate stop loss price for a position.
        
        Args:
            entry_price: Entry price of the position
            position_size: Size of the position
            
        Returns:
            Stop loss price or None if no stop loss
        """
        stop_loss_pct = self.risk_params.get('stop_loss_pct', 0.05)  # 5% default
        
        if position_size > 0:  # Long position
            return entry_price * (1 - stop_loss_pct)
        elif position_size < 0:  # Short position
            return entry_price * (1 + stop_loss_pct)
        
        return None
    
    def get_take_profit_price(self, entry_price: float, position_size: float) -> Optional[float]:
        """
        Calculate take profit price for a position.
        
        Args:
            entry_price: Entry price of the position
            position_size: Size of the position
            
        Returns:
            Take profit price or None if no take profit
        """
        take_profit_pct = self.risk_params.get('take_profit_pct', 0.10)  # 10% default
        
        if position_size > 0:  # Long position
            return entry_price * (1 + take_profit_pct)
        elif position_size < 0:  # Short position
            return entry_price * (1 - take_profit_pct)
        
        return None
    
    def should_exit_position(
        self,
        current_price: float,
        current_time: datetime,
        current_data: pd.Series
    ) -> Tuple[bool, str]:
        """
        Check if current position should be exited.
        
        Args:
            current_price: Current asset price
            current_time: Current timestamp
            current_data: Current OHLCV data
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.current_position.is_flat:
            return False, "no_position"
        
        # Check stop loss
        stop_loss_price = self.get_stop_loss_price(
            self.current_position.entry_price,
            self.current_position.size
        )
        
        if stop_loss_price:
            if self.current_position.is_long and current_price <= stop_loss_price:
                return True, "stop_loss"
            elif self.current_position.is_short and current_price >= stop_loss_price:
                return True, "stop_loss"
        
        # Check take profit
        take_profit_price = self.get_take_profit_price(
            self.current_position.entry_price,
            self.current_position.size
        )
        
        if take_profit_price:
            if self.current_position.is_long and current_price >= take_profit_price:
                return True, "take_profit"
            elif self.current_position.is_short and current_price <= take_profit_price:
                return True, "take_profit"
        
        return False, "hold"
    
    def get_current_position(self) -> Position:
        """Get the current position."""
        return self.current_position
    
    def update_position(self, new_price: float):
        """Update current position with new price."""
        self.current_position.update_price(new_price)
    
    def close_position(self, exit_price: float, exit_time: datetime, reason: str = "signal"):
        """
        Close the current position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing
        """
        if not self.current_position.is_flat:
            # Calculate realized PnL
            pnl = (exit_price - self.current_position.entry_price) * self.current_position.size
            self.current_position.realized_pnl += pnl
            self.total_pnl += pnl
            
            self.logger.debug(
                f"Closed position: {self.current_position.size:.4f} @ {exit_price:.4f}, "
                f"PnL: {pnl:.4f}, Reason: {reason}"
            )
            
            # Reset position
            self.current_position.size = 0.0
            self.current_position.unrealized_pnl = 0.0
            self.trades_executed += 1
    
    def open_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        entry_time: datetime
    ):
        """
        Open a new position.
        
        Args:
            symbol: Trading symbol
            size: Position size
            entry_price: Entry price
            entry_time: Entry timestamp
        """
        # Close existing position if any
        if not self.current_position.is_flat:
            self.close_position(entry_price, entry_time, "new_signal")
        
        # Open new position
        self.current_position = Position(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=entry_price,
            unrealized_pnl=0.0
        )
        
        self.logger.debug(
            f"Opened position: {size:.4f} @ {entry_price:.4f}"
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state."""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'risk_params': self.risk_params,
            'is_initialized': self.is_initialized,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'total_pnl': self.total_pnl,
            'current_position': {
                'size': self.current_position.size,
                'entry_price': self.current_position.entry_price,
                'unrealized_pnl': self.current_position.unrealized_pnl,
                'realized_pnl': self.current_position.realized_pnl
            }
        }
    
    def reset(self):
        """Reset strategy state for new backtest."""
        self.current_position = Position(
            symbol="",
            size=0.0,
            entry_price=0.0,
            entry_time=datetime.now(),
            current_price=0.0,
            unrealized_pnl=0.0
        )
        self.signals_generated = 0
        self.trades_executed = 0
        self.total_pnl = 0.0
        self.indicators = {}
        self.is_initialized = False
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        return f"BaseStrategy(name='{self.name}', parameters={self.parameters})" 