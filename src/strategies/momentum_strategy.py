"""
Momentum Strategy

This strategy uses the Rate of Change (ROC) indicator to identify momentum trends
and generate trading signals based on momentum strength and direction.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.
    
    Generates signals based on Rate of Change (ROC) momentum indicator
    with configurable thresholds and momentum confirmation.
    """
    
    def __init__(
        self,
        period: int = 14,
        threshold: float = 0.02,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Momentum Strategy.
        
        Args:
            period: Momentum calculation period (5-30)
            threshold: Minimum momentum threshold for signal generation (0.01-0.1)
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'period': period,
            'threshold': threshold
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"Momentum_{period}_{threshold}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.period = period
        self.threshold = threshold
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.momentum = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.period <= 30):
            self.logger.error(f"Period {self.period} outside valid range [5, 30]")
            return False
        
        if not (0.01 <= self.threshold <= 0.1):
            self.logger.error(f"Threshold {self.threshold} outside valid range [0.01, 0.1]")
            return False
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        super().initialize(data)
        
        min_periods = self.period + 20  # Extra periods for smoothing
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            return
        
        # Calculate Rate of Change
        self.momentum = data['close'].pct_change(periods=self.period)
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"ROC period: {self.period}, threshold: {self.threshold}")
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on momentum analysis."""
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
        
        # Need sufficient data for ROC calculation
        if current_idx < self.period + 5:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Insufficient historical data"}
            )
        
        # Get current values
        current_momentum = self.momentum.iloc[current_idx]
        
        # Get previous values
        if current_idx > 0:
            previous_momentum = self.momentum.iloc[current_idx - 1]
        else:
            previous_momentum = current_momentum
        
        # Check for NaN values
        if pd.isna(current_momentum):
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
            "momentum": current_momentum,
            "threshold": self.threshold
        }
        
        # Momentum direction and strength analysis
        is_positive_momentum = current_momentum > self.threshold
        is_negative_momentum = current_momentum < -self.threshold
        
        metadata.update({
            "positive_momentum": is_positive_momentum,
            "negative_momentum": is_negative_momentum
        })
        
        # Generate signals based on momentum conditions
        
        # Strong positive momentum - buy signal
        if is_positive_momentum and current_momentum > self.threshold:
            action = "buy"
            # Base strength on how much momentum exceeds threshold
            strength = min(current_momentum / self.threshold, 3.0) / 3.0
            confidence = min(strength * 1.5, 1.0)
            metadata["signal_type"] = "positive_momentum"
            
            # Additional confirmation factors
            if previous_momentum > self.threshold:
                confidence = min(confidence * 1.3, 1.0)
                metadata["trend_confirmation"] = "bullish"
            
            # Check for momentum crossover (ROC crossing above threshold)
            if previous_momentum <= self.threshold and current_momentum > self.threshold:
                strength = min(strength * 1.4, 1.0)
                confidence = min(confidence * 1.3, 1.0)
                metadata["momentum_breakout"] = "bullish"
        
        # Strong negative momentum - sell signal
        elif is_negative_momentum and current_momentum > self.threshold:
            action = "sell"
            # Base strength on how much momentum exceeds threshold
            strength = min(current_momentum / self.threshold, 3.0) / 3.0
            confidence = min(strength * 1.5, 1.0)
            metadata["signal_type"] = "negative_momentum"
            
            # Additional confirmation factors
            if previous_momentum < -self.threshold:
                confidence = min(confidence * 1.3, 1.0)
                metadata["trend_confirmation"] = "bearish"
            
            # Check for momentum crossover (ROC crossing below negative threshold)
            if previous_momentum >= -self.threshold and current_momentum < -self.threshold:
                strength = min(strength * 1.4, 1.0)
                confidence = min(confidence * 1.3, 1.0)
                metadata["momentum_breakout"] = "bearish"
        
        # Momentum reversal signals (when strong momentum starts to weaken)
        elif action == "hold":
            # Check for momentum exhaustion after strong moves
            if (previous_momentum > self.threshold * 2 and 
                current_momentum < previous_momentum * 0.7):
                
                # Momentum exhaustion after strong positive move
                if previous_momentum > self.threshold:
                    action = "sell"
                    strength = min((previous_momentum - current_momentum) / self.threshold, 1.0)
                    confidence = min(strength * 1.2, 1.0)
                    metadata["signal_type"] = "momentum_exhaustion_bearish"
                
                # Momentum exhaustion after strong negative move
                elif previous_momentum < -self.threshold:
                    action = "buy"
                    strength = min((previous_momentum - current_momentum) / self.threshold, 1.0)
                    confidence = min(strength * 1.2, 1.0)
                    metadata["signal_type"] = "momentum_exhaustion_bullish"
        
        # Additional signal enhancements
        if action != "hold":
            # Volume confirmation (if available)
            if 'volume' in current_data and 'volume' in self.data.columns:
                current_volume = current_data['volume']
                avg_volume = self.data['volume'].iloc[max(0, current_idx-10):current_idx].mean()
                
                if current_volume > avg_volume * 1.5:  # High volume
                    confidence = min(confidence * 1.2, 1.0)
                    metadata["volume_confirmation"] = True
            
            # Momentum persistence check
            if current_idx >= 3:
                recent_momentum = self.momentum.iloc[current_idx-2:current_idx+1]
                if action == "buy" and all(recent_momentum > 0):
                    confidence = min(confidence * 1.1, 1.0)
                    metadata["momentum_persistence"] = "bullish"
                elif action == "sell" and all(recent_momentum < 0):
                    confidence = min(confidence * 1.1, 1.0)
                    metadata["momentum_persistence"] = "bearish"
            
            # Extreme momentum conditions
            if current_momentum > self.threshold * 3:
                # Very strong momentum - increase strength but reduce confidence slightly
                # (extreme moves can reverse quickly)
                strength = min(strength * 1.3, 1.0)
                confidence = min(confidence * 0.9, 1.0)
                metadata["extreme_momentum"] = True
            
            # Momentum divergence with price
            if current_idx >= self.period:
                price_change = (current_data['close'] - self.data['close'].iloc[current_idx - self.period]) / self.data['close'].iloc[current_idx - self.period]
                
                # Bullish divergence: price down but momentum improving
                if (action == "buy" and price_change < 0 and 
                    current_momentum > previous_momentum and current_momentum > -self.threshold * 0.5):
                    confidence = min(confidence * 1.3, 1.0)
                    metadata["bullish_divergence"] = True
                
                # Bearish divergence: price up but momentum weakening
                elif (action == "sell" and price_change > 0 and 
                      current_momentum < previous_momentum and current_momentum < self.threshold * 0.5):
                    confidence = min(confidence * 1.3, 1.0)
                    metadata["bearish_divergence"] = True
        
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
            'period': hp.choice('period', list(range(5, 31))),
            'threshold': hp.uniform('threshold', 0.01, 0.1),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.momentum is None:
            return {}
        
        return {
            'momentum': float(self.momentum.iloc[-1]) if not pd.isna(self.momentum.iloc[-1]) else 0.0,
            'above_threshold': bool(abs(self.momentum.iloc[-1]) > self.threshold) if not pd.isna(self.momentum.iloc[-1]) else False,
            'momentum_direction': 'positive' if self.momentum.iloc[-1] > 0 else 'negative' if not pd.isna(self.momentum.iloc[-1]) else 'neutral'
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"Momentum({self.period}, {self.threshold})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MomentumStrategy(period={self.period}, threshold={self.threshold})"
    
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
        
        # Adjust based on signal strength (momentum magnitude)
        adjusted_size = base_size * signal.strength
        
        # For momentum strategies, we can be more aggressive in trending markets
        # but still reduce size in high volatility
        volatility_adjustment = max(0.4, 1.0 - (current_volatility * 1.5))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size 