"""
Simple Moving Average Crossover Strategy

This is a basic implementation of a moving average crossover strategy
that serves as an example and test case for the strategy framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from .base_strategy import BaseStrategy, Signal


class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Generates buy signals when fast MA crosses above slow MA,
    and sell signals when fast MA crosses below slow MA.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        position_size_pct: float = 0.95,
        **kwargs
    ):
        """
        Initialize the Simple MA strategy.
        
        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            position_size_pct: Percentage of capital to use per trade
        """
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'position_size_pct': position_size_pct
        }
        
        super().__init__(
            name="SimpleMA",
            parameters=parameters,
            **kwargs
        )
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size_pct = position_size_pct
        
        # Indicator storage
        self.fast_ma = None
        self.slow_ma = None
        self.previous_fast_ma = None
        self.previous_slow_ma = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.fast_period >= self.slow_period:
            self.logger.error("Fast period must be less than slow period")
            return False
        
        if self.fast_period < 2 or self.slow_period < 2:
            self.logger.error("Periods must be at least 2")
            return False
        
        if not 0 < self.position_size_pct <= 1:
            self.logger.error("Position size percentage must be between 0 and 1")
            return False
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.data = data.copy()
        
        # Calculate moving averages
        self.indicators['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
        self.indicators['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
        
        self.is_initialized = True
        self.logger.info(
            f"Initialized SimpleMA strategy: fast={self.fast_period}, slow={self.slow_period}"
        )
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on MA crossover."""
        if not self.is_initialized:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Get current MA values
        try:
            current_idx = self.data.index.get_loc(current_time)
        except KeyError:
            # If exact timestamp not found, use nearest
            current_idx = self.data.index.get_indexer([current_time], method='nearest')[0]
        
        if current_idx < self.slow_period:
            # Not enough data for signal
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Current MA values
        fast_ma_current = self.indicators['fast_ma'].iloc[current_idx]
        slow_ma_current = self.indicators['slow_ma'].iloc[current_idx]
        
        # Previous MA values
        if current_idx > 0:
            fast_ma_prev = self.indicators['fast_ma'].iloc[current_idx - 1]
            slow_ma_prev = self.indicators['slow_ma'].iloc[current_idx - 1]
        else:
            fast_ma_prev = fast_ma_current
            slow_ma_prev = slow_ma_current
        
        # Check for crossover
        action = 'hold'
        strength = 0.0
        confidence = 0.0
        
        # Bullish crossover: fast MA crosses above slow MA
        if (fast_ma_prev <= slow_ma_prev and fast_ma_current > slow_ma_current):
            action = 'buy'
            strength = 1.0
            confidence = 0.8
            
        # Bearish crossover: fast MA crosses below slow MA
        elif (fast_ma_prev >= slow_ma_prev and fast_ma_current < slow_ma_current):
            action = 'sell'
            strength = 1.0
            confidence = 0.8
        
        return Signal(
            timestamp=current_time,
            action=action,
            strength=strength,
            price=current_data['close'],
            confidence=confidence,
            metadata={
                'fast_ma': fast_ma_current,
                'slow_ma': slow_ma_current,
                'fast_ma_prev': fast_ma_prev,
                'slow_ma_prev': slow_ma_prev
            }
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """Calculate position size based on available capital."""
        if signal.action == 'hold':
            return 0.0
        
        # Use fixed percentage of available capital
        target_value = available_capital * self.position_size_pct
        
        if signal.action == 'buy':
            # Long position
            position_size = target_value / current_price
        elif signal.action == 'sell':
            # For this simple strategy, we'll close long positions rather than go short
            # In a more sophisticated strategy, this could be a short position
            if not self.current_position.is_flat and self.current_position.is_long:
                position_size = -self.current_position.size  # Close long position
            else:
                position_size = 0.0  # Don't go short
        else:
            position_size = 0.0
        
        return position_size
    
    def get_required_indicators(self) -> list:
        """Get list of required indicators."""
        return ['fast_ma', 'slow_ma'] 