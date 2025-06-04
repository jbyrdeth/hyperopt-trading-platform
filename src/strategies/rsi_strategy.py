"""
RSI Strategy

This strategy uses the Relative Strength Index (RSI) to identify overbought and oversold
conditions for mean reversion trading opportunities.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import get_logger


class RSIStrategy(BaseStrategy):
    """
    RSI Strategy.
    
    Generates signals based on RSI overbought/oversold levels with
    configurable exit strategies (opposite signal, middle, trailing).
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        exit_signal: str = "opposite",
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize RSI Strategy.
        
        Args:
            rsi_period: RSI calculation period (5-50)
            oversold: Oversold threshold (10-40)
            overbought: Overbought threshold (60-90)
            exit_signal: Exit signal type ("opposite", "middle", "trailing")
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
            'exit_signal': exit_signal
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"RSI_{rsi_period}_{oversold}_{overbought}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_signal = exit_signal
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.rsi = None
        self.middle_level = 50.0  # RSI middle line
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.rsi_period <= 50):
            self.logger.error(f"RSI period {self.rsi_period} outside valid range [5, 50]")
            return False
        
        if not (10 <= self.oversold <= 40):
            self.logger.error(f"Oversold level {self.oversold} outside valid range [10, 40]")
            return False
        
        if not (60 <= self.overbought <= 90):
            self.logger.error(f"Overbought level {self.overbought} outside valid range [60, 90]")
            return False
        
        if self.oversold >= self.overbought:
            self.logger.error("Oversold level must be less than overbought level")
            return False
        
        if self.exit_signal not in ["opposite", "middle", "trailing"]:
            self.logger.error(f"Invalid exit signal type: {self.exit_signal}")
            return False
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = self.rsi_period + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate RSI
        self.rsi = ta.momentum.rsi(data['close'], window=self.rsi_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"RSI period: {self.rsi_period}, Oversold: {self.oversold}, Overbought: {self.overbought}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return self.rsi_period + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on RSI analysis."""
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
        
        # Need sufficient data for RSI calculation
        if current_idx < self.rsi_period:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Insufficient historical data"}
            )
        
        # Get current and previous RSI values
        current_rsi = self.rsi.iloc[current_idx]
        
        if current_idx > 0:
            previous_rsi = self.rsi.iloc[current_idx - 1]
        else:
            previous_rsi = current_rsi
        
        # Check for NaN values
        if pd.isna(current_rsi) or pd.isna(previous_rsi):
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
            "rsi": current_rsi,
            "previous_rsi": previous_rsi,
            "oversold_level": self.oversold,
            "overbought_level": self.overbought
        }
        
        # Check current position to determine if we should exit
        current_position = self.get_current_position()
        
        # Exit logic based on exit_signal type
        if current_position and current_position.size != 0:
            should_exit = False
            exit_reason = ""
            
            if self.exit_signal == "opposite":
                # Exit on opposite signal
                if current_position.size > 0 and current_rsi >= self.overbought:
                    should_exit = True
                    exit_reason = "RSI overbought - exit long"
                elif current_position.size < 0 and current_rsi <= self.oversold:
                    should_exit = True
                    exit_reason = "RSI oversold - exit short"
            
            elif self.exit_signal == "middle":
                # Exit when RSI returns to middle (50)
                if current_position.size > 0 and previous_rsi < self.middle_level and current_rsi >= self.middle_level:
                    should_exit = True
                    exit_reason = "RSI crossed above middle - exit long"
                elif current_position.size < 0 and previous_rsi > self.middle_level and current_rsi <= self.middle_level:
                    should_exit = True
                    exit_reason = "RSI crossed below middle - exit short"
            
            elif self.exit_signal == "trailing":
                # Exit when RSI starts moving against us after being in our favor
                if current_position.size > 0:
                    # Long position: exit if RSI starts declining from high levels
                    if current_rsi < previous_rsi and current_rsi > 60:
                        should_exit = True
                        exit_reason = "RSI declining from high levels - exit long"
                elif current_position.size < 0:
                    # Short position: exit if RSI starts rising from low levels
                    if current_rsi > previous_rsi and current_rsi < 40:
                        should_exit = True
                        exit_reason = "RSI rising from low levels - exit short"
            
            if should_exit:
                action = "sell" if current_position.size > 0 else "buy"
                strength = 0.8  # High strength for exit signals
                confidence = 0.9
                metadata["signal_type"] = "exit"
                metadata["exit_reason"] = exit_reason
                
                return Signal(
                    timestamp=timestamp,
                    action=action,
                    strength=strength,
                    price=current_data['close'],
                    confidence=confidence,
                    metadata=metadata
                )
        
        # Entry signals (only if no current position)
        if not current_position or current_position.size == 0:
            # Oversold condition - potential buy signal
            if current_rsi <= self.oversold:
                # Check for RSI divergence or momentum
                if previous_rsi > current_rsi:  # RSI still declining
                    strength = min((self.oversold - current_rsi) / self.oversold, 1.0)
                else:  # RSI starting to rise from oversold
                    strength = min((self.oversold - current_rsi) / self.oversold * 1.5, 1.0)
                
                if strength > 0.1:  # Minimum threshold
                    action = "buy"
                    confidence = min(strength * 1.2, 1.0)
                    metadata["signal_type"] = "oversold_entry"
                    
                    # Additional strength for extreme oversold
                    if current_rsi <= self.oversold * 0.7:  # Very oversold
                        strength = min(strength * 1.4, 1.0)
                        confidence = min(confidence * 1.3, 1.0)
                        metadata["extreme_oversold"] = True
            
            # Overbought condition - potential sell signal
            elif current_rsi >= self.overbought:
                # Check for RSI divergence or momentum
                if previous_rsi < current_rsi:  # RSI still rising
                    strength = min((current_rsi - self.overbought) / (100 - self.overbought), 1.0)
                else:  # RSI starting to fall from overbought
                    strength = min((current_rsi - self.overbought) / (100 - self.overbought) * 1.5, 1.0)
                
                if strength > 0.1:  # Minimum threshold
                    action = "sell"
                    confidence = min(strength * 1.2, 1.0)
                    metadata["signal_type"] = "overbought_entry"
                    
                    # Additional strength for extreme overbought
                    if current_rsi >= self.overbought + (100 - self.overbought) * 0.3:  # Very overbought
                        strength = min(strength * 1.4, 1.0)
                        confidence = min(confidence * 1.3, 1.0)
                        metadata["extreme_overbought"] = True
        
        # RSI momentum confirmation
        if action != "hold":
            rsi_momentum = current_rsi - previous_rsi
            
            # For buy signals, prefer when RSI is starting to rise
            if action == "buy" and rsi_momentum > 0:
                confidence = min(confidence * 1.2, 1.0)
                metadata["momentum_confirmation"] = "positive"
            
            # For sell signals, prefer when RSI is starting to fall
            elif action == "sell" and rsi_momentum < 0:
                confidence = min(confidence * 1.2, 1.0)
                metadata["momentum_confirmation"] = "negative"
        
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
            'rsi_period': hp.choice('rsi_period', list(range(5, 51))),
            'oversold': hp.uniform('oversold', 10, 40),
            'overbought': hp.uniform('overbought', 60, 90),
            'exit_signal': hp.choice('exit_signal', ["opposite", "middle", "trailing"]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.rsi is None:
            return {}
        
        current_rsi = self.rsi.iloc[-1] if not pd.isna(self.rsi.iloc[-1]) else 50.0
        
        return {
            'rsi': float(current_rsi),
            'is_oversold': bool(current_rsi <= self.oversold),
            'is_overbought': bool(current_rsi >= self.overbought),
            'distance_to_oversold': float(current_rsi - self.oversold),
            'distance_to_overbought': float(self.overbought - current_rsi)
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"RSI({self.rsi_period}, {self.oversold}/{self.overbought}, {self.exit_signal})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"RSIStrategy(rsi_period={self.rsi_period}, oversold={self.oversold}, overbought={self.overbought}, exit_signal='{self.exit_signal}')"
    
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
        
        # Adjust based on signal strength (RSI distance from extreme levels)
        adjusted_size = base_size * signal.strength
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size 