"""
MACD Strategy

This strategy uses the Moving Average Convergence Divergence (MACD) indicator
to generate trading signals based on line crossovers and histogram analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class MACDStrategy(BaseStrategy):
    """
    MACD Strategy.
    
    Generates signals based on MACD line crossovers with signal line
    and histogram analysis for momentum confirmation.
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        histogram_threshold: float = 0.1,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize MACD Strategy.
        
        Args:
            fast_period: Fast EMA period (8-20)
            slow_period: Slow EMA period (20-35)
            signal_period: Signal line EMA period (5-15)
            histogram_threshold: Minimum histogram value for signal generation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'histogram_threshold': histogram_threshold
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"MACD_{fast_period}_{slow_period}_{signal_period}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.histogram_threshold = histogram_threshold
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.macd_line = None
        self.macd_signal = None
        self.macd_histogram = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.fast_period >= self.slow_period:
            self.logger.error("Fast period must be less than slow period")
            return False
        
        if not (3 <= self.fast_period <= 20):
            self.logger.error(f"Fast period {self.fast_period} outside valid range [3, 20]")
            return False
        
        if not (10 <= self.slow_period <= 50):
            self.logger.error(f"Slow period {self.slow_period} outside valid range [10, 50]")
            return False
        
        if not (3 <= self.signal_period <= 15):
            self.logger.error(f"Signal period {self.signal_period} outside valid range [3, 15]")
            return False
        
        if not (0.0 <= self.histogram_threshold <= 0.1):
            self.logger.error(f"Histogram threshold {self.histogram_threshold} outside valid range [0.0, 0.1]")
            return False
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        super().initialize(data)
        
        min_periods = self.slow_period + self.signal_period + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            return
        
        # Calculate MACD components
        self.macd_line = ta.trend.macd_diff(
            data['close'], 
            window_slow=self.slow_period, 
            window_fast=self.fast_period
        )
        
        self.macd_signal = ta.trend.macd_signal(
            data['close'], 
            window_slow=self.slow_period, 
            window_fast=self.fast_period, 
            window_sign=self.signal_period
        )
        
        self.macd_histogram = self.macd_line - self.macd_signal
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"MACD periods: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on MACD analysis."""
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
        
        # Need sufficient data for MACD calculation
        min_idx = self.slow_period + self.signal_period
        if current_idx < min_idx:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Insufficient historical data"}
            )
        
        # Get current and previous MACD values
        current_macd = self.macd_line.iloc[current_idx]
        current_signal = self.macd_signal.iloc[current_idx]
        current_histogram = self.macd_histogram.iloc[current_idx]
        
        if current_idx > 0:
            previous_macd = self.macd_line.iloc[current_idx - 1]
            previous_signal = self.macd_signal.iloc[current_idx - 1]
            previous_histogram = self.macd_histogram.iloc[current_idx - 1]
        else:
            previous_macd = current_macd
            previous_signal = current_signal
            previous_histogram = current_histogram
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_macd, current_signal, current_histogram, 
                                       previous_macd, previous_signal, previous_histogram]):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "macd_line": current_macd,
            "macd_signal": current_signal,
            "macd_histogram": current_histogram,
            "previous_histogram": previous_histogram
        }
        
        # MACD line crossover signals
        macd_above_signal = current_macd > current_signal
        previous_macd_above_signal = previous_macd > previous_signal
        
        # Bullish signal: MACD crosses above signal line
        if not previous_macd_above_signal and macd_above_signal:
            if current_histogram >= self.histogram_threshold:
                action = "buy"
                # Signal strength based on histogram magnitude and crossover strength
                crossover_strength = abs(current_macd - current_signal) / max(abs(current_signal), 0.001)
                histogram_strength = max(current_histogram, 0) / max(abs(current_macd), 0.001)
                strength = min((crossover_strength + histogram_strength) / 2, 1.0)
                confidence = min(strength * 1.5, 1.0)
                metadata["signal_type"] = "bullish_crossover"
        
        # Bearish signal: MACD crosses below signal line
        elif previous_macd_above_signal and not macd_above_signal:
            if current_histogram <= -self.histogram_threshold:
                action = "sell"
                # Signal strength based on histogram magnitude and crossover strength
                crossover_strength = abs(current_macd - current_signal) / max(abs(current_signal), 0.001)
                histogram_strength = abs(min(current_histogram, 0)) / max(abs(current_macd), 0.001)
                strength = min((crossover_strength + histogram_strength) / 2, 1.0)
                confidence = min(strength * 1.5, 1.0)
                metadata["signal_type"] = "bearish_crossover"
        
        # Additional momentum confirmation
        if action != "hold":
            # Check histogram momentum (increasing/decreasing)
            histogram_momentum = current_histogram - previous_histogram
            
            if action == "buy" and histogram_momentum > 0:
                # Increasing positive histogram strengthens buy signal
                strength = min(strength * 1.3, 1.0)
                confidence = min(confidence * 1.2, 1.0)
                metadata["momentum_confirmation"] = "positive"
            elif action == "sell" and histogram_momentum < 0:
                # Decreasing negative histogram strengthens sell signal
                strength = min(strength * 1.3, 1.0)
                confidence = min(confidence * 1.2, 1.0)
                metadata["momentum_confirmation"] = "negative"
            
            # Zero line crossover adds confidence
            if action == "buy" and current_macd > 0 and previous_macd <= 0:
                confidence = min(confidence * 1.4, 1.0)
                metadata["zero_line_cross"] = "bullish"
            elif action == "sell" and current_macd < 0 and previous_macd >= 0:
                confidence = min(confidence * 1.4, 1.0)
                metadata["zero_line_cross"] = "bearish"
        
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
            'fast_period': hp.choice('fast_period', list(range(3, 21))),
            'slow_period': hp.choice('slow_period', list(range(10, 51))),
            'signal_period': hp.choice('signal_period', list(range(3, 16))),
            'histogram_threshold': hp.uniform('histogram_threshold', 0.0, 0.1),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or any(x is None for x in [self.macd_line, self.macd_signal, self.macd_histogram]):
            return {}
        
        return {
            'macd_line': float(self.macd_line.iloc[-1]) if not pd.isna(self.macd_line.iloc[-1]) else 0.0,
            'macd_signal': float(self.macd_signal.iloc[-1]) if not pd.isna(self.macd_signal.iloc[-1]) else 0.0,
            'macd_histogram': float(self.macd_histogram.iloc[-1]) if not pd.isna(self.macd_histogram.iloc[-1]) else 0.0,
            'macd_above_signal': bool(self.macd_line.iloc[-1] > self.macd_signal.iloc[-1]) if not any(pd.isna(x.iloc[-1]) for x in [self.macd_line, self.macd_signal]) else False
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"MACD({self.fast_period}, {self.slow_period}, {self.signal_period})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MACDStrategy(fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}, histogram_threshold={self.histogram_threshold})"
    
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
        
        # Adjust based on signal strength and confidence
        adjusted_size = base_size * signal.strength * signal.confidence
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size 