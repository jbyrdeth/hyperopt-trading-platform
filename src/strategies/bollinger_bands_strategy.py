"""
Bollinger Bands Strategy

This strategy uses Bollinger Bands to identify volatility conditions and trade
both breakouts and mean reversion opportunities with squeeze detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy.
    
    Generates signals based on Bollinger Band breakouts, mean reversion,
    and volatility squeeze conditions.
    """
    
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        squeeze_threshold: float = 0.2,
        entry_method: str = "breakout",
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Bollinger Bands Strategy.
        
        Args:
            period: Moving average period (10-50)
            std_dev: Standard deviation multiplier (1.0-3.0)
            squeeze_threshold: Threshold for detecting squeezes (0.1-0.5)
            entry_method: Entry method ("breakout" or "mean_reversion")
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'period': period,
            'std_dev': std_dev,
            'squeeze_threshold': squeeze_threshold,
            'entry_method': entry_method
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"BollingerBands_{period}_{std_dev}_{entry_method}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.period = period
        self.std_dev = std_dev
        self.squeeze_threshold = squeeze_threshold
        self.entry_method = entry_method
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.bb_upper = None
        self.bb_middle = None
        self.bb_lower = None
        self.bb_width = None
        self.bb_percent = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (10 <= self.period <= 50):
            self.logger.error(f"Period {self.period} outside valid range [10, 50]")
            return False
        
        if not (1.0 <= self.std_dev <= 3.0):
            self.logger.error(f"Standard deviation {self.std_dev} outside valid range [1.0, 3.0]")
            return False
        
        if not (0.1 <= self.squeeze_threshold <= 0.5):
            self.logger.error(f"Squeeze threshold {self.squeeze_threshold} outside valid range [0.1, 0.5]")
            return False
        
        if self.entry_method not in ["breakout", "reversion", "both"]:
            self.logger.error(f"Invalid entry method: {self.entry_method}")
            return False
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        super().initialize(data)
        
        min_periods = self.period + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            return
        
        # Calculate Bollinger Bands
        self.bb_upper = ta.volatility.bollinger_hband(data['close'], window=self.period, window_dev=self.std_dev)
        self.bb_middle = ta.volatility.bollinger_mavg(data['close'], window=self.period)
        self.bb_lower = ta.volatility.bollinger_lband(data['close'], window=self.period, window_dev=self.std_dev)
        
        # Calculate additional indicators
        self.bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_percent = (data['close'] - self.bb_lower) / (self.bb_upper - self.bb_lower)
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"BB period: {self.period}, std_dev: {self.std_dev}, method: {self.entry_method}")
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on Bollinger Bands analysis."""
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
        
        # Need sufficient data for BB calculation
        if current_idx < self.period:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Insufficient historical data"}
            )
        
        # Get current values
        current_price = current_data['close']
        current_upper = self.bb_upper.iloc[current_idx]
        current_middle = self.bb_middle.iloc[current_idx]
        current_lower = self.bb_lower.iloc[current_idx]
        current_width = self.bb_width.iloc[current_idx]
        current_percent = self.bb_percent.iloc[current_idx]
        
        # Get previous values
        if current_idx > 0:
            previous_price = self.data['close'].iloc[current_idx - 1]
            previous_upper = self.bb_upper.iloc[current_idx - 1]
            previous_lower = self.bb_lower.iloc[current_idx - 1]
            previous_width = self.bb_width.iloc[current_idx - 1]
            previous_percent = self.bb_percent.iloc[current_idx - 1]
        else:
            previous_price = current_price
            previous_upper = current_upper
            previous_lower = current_lower
            previous_width = current_width
            previous_percent = current_percent
        
        # Check for NaN values
        if any(pd.isna(val) for val in [current_upper, current_middle, current_lower, 
                                       current_width, current_percent]):
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
            "bb_upper": current_upper,
            "bb_middle": current_middle,
            "bb_lower": current_lower,
            "bb_width": current_width,
            "bb_percent": current_percent,
            "price_position": "middle"
        }
        
        # Determine price position relative to bands
        if current_price >= current_upper:
            metadata["price_position"] = "above_upper"
        elif current_price <= current_lower:
            metadata["price_position"] = "below_lower"
        elif current_price > current_middle:
            metadata["price_position"] = "upper_half"
        else:
            metadata["price_position"] = "lower_half"
        
        # Detect squeeze condition
        is_squeeze = current_width < self.squeeze_threshold
        was_squeeze = previous_width < self.squeeze_threshold
        squeeze_breakout = was_squeeze and not is_squeeze
        
        metadata["is_squeeze"] = is_squeeze
        metadata["squeeze_breakout"] = squeeze_breakout
        
        # Trading logic based on entry method
        if self.entry_method in ["breakout", "both"]:
            # Breakout signals
            
            # Upper band breakout (bullish)
            if (previous_price <= previous_upper and current_price > current_upper):
                action = "buy"
                strength = min((current_price - current_upper) / current_upper, 1.0)
                confidence = min(strength * 1.5, 1.0)
                metadata["signal_type"] = "upper_breakout"
                
                # Stronger signal if coming out of squeeze
                if squeeze_breakout:
                    strength = min(strength * 1.5, 1.0)
                    confidence = min(confidence * 1.3, 1.0)
                    metadata["squeeze_breakout_signal"] = True
            
            # Lower band breakout (bearish)
            elif (previous_price >= previous_lower and current_price < current_lower):
                action = "sell"
                strength = min((current_lower - current_price) / current_lower, 1.0)
                confidence = min(strength * 1.5, 1.0)
                metadata["signal_type"] = "lower_breakout"
                
                # Stronger signal if coming out of squeeze
                if squeeze_breakout:
                    strength = min(strength * 1.5, 1.0)
                    confidence = min(confidence * 1.3, 1.0)
                    metadata["squeeze_breakout_signal"] = True
        
        if self.entry_method in ["reversion", "both"] and action == "hold":
            # Mean reversion signals
            
            # Price touching upper band - potential sell
            if current_percent >= 0.95:  # Very close to upper band
                action = "sell"
                strength = min((current_percent - 0.95) / 0.05, 1.0)
                confidence = min(strength * 1.2, 1.0)
                metadata["signal_type"] = "upper_reversion"
                
                # Stronger signal if RSI-like conditions apply
                if current_percent > previous_percent:  # Still moving up
                    strength = min(strength * 0.8, 1.0)  # Reduce strength
                else:  # Starting to reverse
                    strength = min(strength * 1.3, 1.0)
                    confidence = min(confidence * 1.2, 1.0)
            
            # Price touching lower band - potential buy
            elif current_percent <= 0.05:  # Very close to lower band
                action = "buy"
                strength = min((0.05 - current_percent) / 0.05, 1.0)
                confidence = min(strength * 1.2, 1.0)
                metadata["signal_type"] = "lower_reversion"
                
                # Stronger signal if RSI-like conditions apply
                if current_percent < previous_percent:  # Still moving down
                    strength = min(strength * 0.8, 1.0)  # Reduce strength
                else:  # Starting to reverse
                    strength = min(strength * 1.3, 1.0)
                    confidence = min(confidence * 1.2, 1.0)
        
        # Additional signal enhancements
        if action != "hold":
            # Volume confirmation (if available)
            if 'volume' in current_data and 'volume' in self.data.columns:
                current_volume = current_data['volume']
                avg_volume = self.data['volume'].iloc[max(0, current_idx-10):current_idx].mean()
                
                if current_volume > avg_volume * 1.5:  # High volume
                    confidence = min(confidence * 1.2, 1.0)
                    metadata["volume_confirmation"] = True
            
            # Volatility context
            if current_width > previous_width:  # Expanding volatility
                if metadata.get("signal_type", "").endswith("breakout"):
                    confidence = min(confidence * 1.2, 1.0)
                    metadata["expanding_volatility"] = True
            else:  # Contracting volatility
                if metadata.get("signal_type", "").endswith("reversion"):
                    confidence = min(confidence * 1.1, 1.0)
                    metadata["contracting_volatility"] = True
            
            # Middle line as support/resistance
            if action == "buy" and current_price > current_middle:
                confidence = min(confidence * 1.1, 1.0)
                metadata["above_middle"] = True
            elif action == "sell" and current_price < current_middle:
                confidence = min(confidence * 1.1, 1.0)
                metadata["below_middle"] = True
        
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
            'period': hp.choice('period', list(range(10, 51))),
            'std_dev': hp.uniform('std_dev', 1.0, 3.0),
            'squeeze_threshold': hp.uniform('squeeze_threshold', 0.1, 0.5),
            'entry_method': hp.choice('entry_method', ["breakout", "reversion", "both"]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or any(x is None for x in [self.bb_upper, self.bb_middle, self.bb_lower]):
            return {}
        
        return {
            'bb_upper': float(self.bb_upper.iloc[-1]) if not pd.isna(self.bb_upper.iloc[-1]) else 0.0,
            'bb_middle': float(self.bb_middle.iloc[-1]) if not pd.isna(self.bb_middle.iloc[-1]) else 0.0,
            'bb_lower': float(self.bb_lower.iloc[-1]) if not pd.isna(self.bb_lower.iloc[-1]) else 0.0,
            'bb_width': float(self.bb_width.iloc[-1]) if not pd.isna(self.bb_width.iloc[-1]) else 0.0,
            'bb_percent': float(self.bb_percent.iloc[-1]) if not pd.isna(self.bb_percent.iloc[-1]) else 0.5,
            'is_squeeze': bool(self.bb_width.iloc[-1] < self.squeeze_threshold) if not pd.isna(self.bb_width.iloc[-1]) else False
        }
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"BollingerBands({self.period}, {self.std_dev}, {self.entry_method})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"BollingerBandsStrategy(period={self.period}, std_dev={self.std_dev}, squeeze_threshold={self.squeeze_threshold}, entry_method='{self.entry_method}')"
    
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
        
        # For Bollinger Bands, consider band width (volatility)
        # Reduce position size when bands are very wide (high volatility)
        volatility_adjustment = max(0.3, 1.0 - (current_volatility * 3))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size 