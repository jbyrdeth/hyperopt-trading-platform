"""
Moving Average Crossover Strategy

This strategy generates buy signals when a fast moving average crosses above a slow moving average,
and sell signals when the fast MA crosses below the slow MA. Supports multiple MA types.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy.
    
    Generates signals based on crossovers between fast and slow moving averages.
    Supports SMA, EMA, WMA, and TEMA.
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        ma_type: str = "SMA",
        signal_threshold: float = 0.005,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Moving Average Crossover Strategy.
        
        Args:
            fast_period: Period for fast moving average (5-50)
            slow_period: Period for slow moving average (20-200)
            ma_type: Type of moving average ("SMA", "EMA", "WMA", "TEMA")
            signal_threshold: Minimum crossover strength to generate signal (0.001-0.02)
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'ma_type': ma_type.upper(),
            'signal_threshold': signal_threshold
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"MA_Crossover_{ma_type}_{fast_period}_{slow_period}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.ma_type = ma_type.upper()
        self.signal_threshold = signal_threshold
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
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
        
        if self.ma_type not in ["SMA", "EMA", "WMA", "TEMA"]:
            self.logger.error(f"Unsupported MA type: {self.ma_type}")
            return False
        
        if not (2 <= self.fast_period <= 50):
            self.logger.error(f"Fast period {self.fast_period} outside valid range [2, 50]")
            return False
        
        if not (5 <= self.slow_period <= 200):
            self.logger.error(f"Slow period {self.slow_period} outside valid range [5, 200]")
            return False
        
        if not (0.0005 <= self.signal_threshold <= 0.02):
            self.logger.error(f"Signal threshold {self.signal_threshold} outside valid range [0.0005, 0.02]")
            return False
        
        return True
    
    def calculate_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate moving average based on type."""
        if self.ma_type == "SMA":
            return ta.trend.sma_indicator(data, window=period)
        elif self.ma_type == "EMA":
            return ta.trend.ema_indicator(data, window=period)
        elif self.ma_type == "WMA":
            return ta.trend.wma_indicator(data, window=period)
        elif self.ma_type == "TEMA":
            # Implement TEMA manually (Triple Exponential Moving Average)
            ema1 = ta.trend.ema_indicator(data, window=period)
            ema2 = ta.trend.ema_indicator(ema1, window=period)
            ema3 = ta.trend.ema_indicator(ema2, window=period)
            return 3 * ema1 - 3 * ema2 + ema3
        else:
            raise ValueError(f"Unsupported MA type: {self.ma_type}")
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        if len(data) < self.slow_period + 10:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {self.slow_period + 10} periods")
            self.is_initialized = False
            return
        
        # Calculate moving averages for the entire dataset
        self.fast_ma = self.calculate_moving_average(data['close'], self.fast_period)
        self.slow_ma = self.calculate_moving_average(data['close'], self.slow_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"Fast MA ({self.ma_type}): {self.fast_period}, Slow MA: {self.slow_period}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return self.slow_period + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on MA crossover."""
        if not self.is_initialized:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Strategy not initialized"}
            )
        
        # Get current index in the data
        try:
            current_idx = self.data.index.get_loc(timestamp)
        except KeyError:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Timestamp not found in data"}
            )
        
        # Need at least slow_period + 1 data points
        if current_idx < self.slow_period:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Insufficient historical data"}
            )
        
        # Get current and previous MA values
        current_fast = self.fast_ma.iloc[current_idx]
        current_slow = self.slow_ma.iloc[current_idx]
        
        if current_idx > 0:
            previous_fast = self.fast_ma.iloc[current_idx - 1]
            previous_slow = self.slow_ma.iloc[current_idx - 1]
        else:
            previous_fast = current_fast
            previous_slow = current_slow
        
        # Check for NaN values
        if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(previous_fast) or pd.isna(previous_slow):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Calculate crossover signals
        current_diff = (current_fast - current_slow) / current_slow
        previous_diff = (previous_fast - previous_slow) / previous_slow
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "fast_ma": current_fast,
            "slow_ma": current_slow,
            "diff": current_diff,
            "previous_diff": previous_diff
        }
        
        # Bullish crossover: fast MA crosses above slow MA
        if previous_diff <= 0 and current_diff > 0 and abs(current_diff) > self.signal_threshold:
            action = "buy"
            strength = min(abs(current_diff) / self.signal_threshold, 1.0)
            confidence = min(strength * 2, 1.0)  # Higher confidence for stronger signals
            metadata["signal_type"] = "bullish_crossover"
            
        # Bearish crossover: fast MA crosses below slow MA
        elif previous_diff >= 0 and current_diff < 0 and abs(current_diff) > self.signal_threshold:
            action = "sell"
            strength = min(abs(current_diff) / self.signal_threshold, 1.0)
            confidence = min(strength * 2, 1.0)
            metadata["signal_type"] = "bearish_crossover"
        
        # Additional signal strength based on MA separation
        if action != "hold":
            # Increase strength if MAs are well separated
            separation = abs(current_diff)
            if separation > self.signal_threshold * 2:
                strength = min(strength * 1.5, 1.0)
                confidence = min(confidence * 1.2, 1.0)
        
        return Signal(
            timestamp=timestamp,
            action=action,
            strength=strength,
            price=current_data['close'],
            confidence=confidence,
            metadata=metadata
        )
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get hyperopt parameter space for optimization."""
        from hyperopt import hp
        
        return {
            'fast_period': hp.choice('fast_period', list(range(2, 51))),
            'slow_period': hp.choice('slow_period', list(range(5, 201))),
            'ma_type': hp.choice('ma_type', ["SMA", "EMA", "WMA", "TEMA"]),
            'signal_threshold': hp.uniform('signal_threshold', 0.0005, 0.02),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.fast_ma is None or self.slow_ma is None:
            return {}
        
        return {
            'fast_ma': float(self.fast_ma.iloc[-1]) if not pd.isna(self.fast_ma.iloc[-1]) else 0.0,
            'slow_ma': float(self.slow_ma.iloc[-1]) if not pd.isna(self.slow_ma.iloc[-1]) else 0.0,
            'ma_diff_pct': float((self.fast_ma.iloc[-1] - self.slow_ma.iloc[-1]) / self.slow_ma.iloc[-1]) if not pd.isna(self.fast_ma.iloc[-1]) and not pd.isna(self.slow_ma.iloc[-1]) else 0.0
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"MovingAverageCrossover(fast={self.fast_period}, slow={self.slow_period}, type={self.ma_type})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MovingAverageCrossoverStrategy(fast_period={self.fast_period}, slow_period={self.slow_period}, ma_type='{self.ma_type}', signal_threshold={self.signal_threshold})"
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            signal: Trading signal
            current_price: Current asset price
            available_capital: Available capital for trading
            current_volatility: Current asset volatility
            
        Returns:
            Position size (positive for long, negative for short)
        """
        if signal.action == 'hold':
            return 0.0
        
        # Base position size as percentage of available capital
        base_size = available_capital * self.position_size_pct / current_price
        
        # Adjust based on signal strength
        adjusted_size = base_size * signal.strength
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size 