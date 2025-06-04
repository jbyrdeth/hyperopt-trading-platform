"""
Keltner Channel Strategy

This strategy uses Keltner Channels for breakout and mean reversion modes,
with ATR-based dynamic channel width and multi-timeframe analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import get_logger


class KeltnerChannelStrategy(BaseStrategy):
    """
    Keltner Channel Strategy.
    
    Uses Keltner Channels for both breakout and mean reversion trading,
    with dynamic ATR-based channel width and trend confirmation.
    """
    
    def __init__(
        self,
        kc_period: int = 20,
        atr_multiplier: float = 2.0,
        trading_mode: str = "breakout",  # "breakout" or "mean_reversion"
        channel_position: str = "middle",  # "upper", "middle", "lower"
        trend_filter: bool = True,
        trend_period: int = 50,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Keltner Channel Strategy.
        
        Args:
            kc_period: Period for Keltner Channel calculation (10-50)
            atr_multiplier: ATR multiplier for channel width (1.0-4.0)
            trading_mode: "breakout" or "mean_reversion"
            channel_position: Reference for signals - "upper", "middle", "lower"
            trend_filter: Whether to use trend filter
            trend_period: Period for trend filter (20-100)
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'kc_period': kc_period,
            'atr_multiplier': atr_multiplier,
            'trading_mode': trading_mode,
            'channel_position': channel_position,
            'trend_filter': trend_filter,
            'trend_period': trend_period,
            'volume_confirmation': volume_confirmation
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"KeltnerChannel_{kc_period}_{trading_mode}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.kc_period = kc_period
        self.atr_multiplier = atr_multiplier
        self.trading_mode = trading_mode
        self.channel_position = channel_position
        self.trend_filter = trend_filter
        self.trend_period = trend_period
        self.volume_confirmation = volume_confirmation
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.kc_upper = None
        self.kc_lower = None
        self.kc_middle = None
        self.atr = None
        self.trend_ma = None
        self.volume_ma = None
        self.channel_width = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (10 <= self.kc_period <= 50):
            self.logger.error(f"KC period {self.kc_period} outside valid range [10, 50]")
            return False
        
        if not (1.0 <= self.atr_multiplier <= 4.0):
            self.logger.error(f"ATR multiplier {self.atr_multiplier} outside valid range [1.0, 4.0]")
            return False
        
        if self.trading_mode not in ["breakout", "mean_reversion"]:
            self.logger.error(f"Invalid trading mode: {self.trading_mode}")
            return False
        
        if self.channel_position not in ["upper", "middle", "lower"]:
            self.logger.error(f"Invalid channel position: {self.channel_position}")
            return False
        
        if not (20 <= self.trend_period <= 100):
            self.logger.error(f"Trend period {self.trend_period} outside valid range [20, 100]")
            return False
        
        return True
    
    def calculate_keltner_channels(self, data: pd.DataFrame) -> tuple:
        """Calculate Keltner Channels."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate EMA (middle line)
        middle = close.ewm(span=self.kc_period).mean()
        
        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=self.kc_period).mean()
        
        # Calculate upper and lower channels
        upper = middle + (atr * self.atr_multiplier)
        lower = middle - (atr * self.atr_multiplier)
        
        return upper, lower, middle, atr
    
    def get_channel_position_value(self, price: float, upper: float, middle: float, lower: float) -> float:
        """Get normalized position within channel (0 = lower, 0.5 = middle, 1 = upper)."""
        if upper == lower:
            return 0.5
        return (price - lower) / (upper - lower)
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.kc_period, self.trend_period) + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate Keltner Channels
        self.kc_upper, self.kc_lower, self.kc_middle, self.atr = self.calculate_keltner_channels(data)
        
        # Calculate channel width for volatility analysis
        self.channel_width = (self.kc_upper - self.kc_lower) / self.kc_middle
        
        # Calculate trend filter if enabled
        if self.trend_filter:
            self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_period)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.kc_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"KC period: {self.kc_period}, ATR multiplier: {self.atr_multiplier}, Mode: {self.trading_mode}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.kc_period, self.trend_period) + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on Keltner Channel analysis."""
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
        
        # Need sufficient data
        min_required = max(self.kc_period, self.trend_period)
        if current_idx < min_required:
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
        current_kc_upper = self.kc_upper.iloc[current_idx]
        current_kc_lower = self.kc_lower.iloc[current_idx]
        current_kc_middle = self.kc_middle.iloc[current_idx]
        current_atr = self.atr.iloc[current_idx]
        current_channel_width = self.channel_width.iloc[current_idx]
        
        # Get previous values for breakout detection
        prev_price = self.data['close'].iloc[current_idx - 1] if current_idx > 0 else current_price
        prev_kc_upper = self.kc_upper.iloc[current_idx - 1] if current_idx > 0 else current_kc_upper
        prev_kc_lower = self.kc_lower.iloc[current_idx - 1] if current_idx > 0 else current_kc_lower
        
        # Trend filter
        trend_bullish = True
        trend_bearish = True
        if self.trend_filter:
            current_trend_ma = self.trend_ma.iloc[current_idx]
            if not pd.isna(current_trend_ma):
                trend_bullish = current_price > current_trend_ma
                trend_bearish = current_price < current_trend_ma
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_confirmation:
            current_volume = current_data['volume']
            avg_volume = self.volume_ma.iloc[current_idx]
            volume_confirmed = current_volume >= avg_volume
        
        # Check for NaN values
        if pd.isna(current_kc_upper) or pd.isna(current_kc_lower) or pd.isna(current_kc_middle):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Calculate channel position
        channel_position = self.get_channel_position_value(
            current_price, current_kc_upper, current_kc_middle, current_kc_lower
        )
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "kc_upper": current_kc_upper,
            "kc_lower": current_kc_lower,
            "kc_middle": current_kc_middle,
            "channel_position": channel_position,
            "channel_width": current_channel_width,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish,
            "volume_confirmed": volume_confirmed
        }
        
        # Generate signals based on trading mode
        if self.trading_mode == "breakout":
            # Breakout trading
            if volume_confirmed:
                # Bullish breakout
                if (current_price > current_kc_upper and 
                    prev_price <= prev_kc_upper and 
                    trend_bullish):
                    action = "buy"
                    strength = min((current_price - current_kc_upper) / current_atr, 1.0)
                    confidence = 0.8
                    metadata["signal_type"] = "kc_breakout_buy"
                    
                # Bearish breakout
                elif (current_price < current_kc_lower and 
                      prev_price >= prev_kc_lower and 
                      trend_bearish):
                    action = "sell"
                    strength = min((current_kc_lower - current_price) / current_atr, 1.0)
                    confidence = 0.8
                    metadata["signal_type"] = "kc_breakout_sell"
                    
        elif self.trading_mode == "mean_reversion":
            # Mean reversion trading
            if volume_confirmed:
                # Buy when price near lower channel
                if (channel_position <= 0.2 and trend_bullish):
                    action = "buy"
                    strength = (0.2 - channel_position) * 5  # Scale to 0-1
                    confidence = 0.7
                    metadata["signal_type"] = "kc_mean_reversion_buy"
                    
                # Sell when price near upper channel
                elif (channel_position >= 0.8 and trend_bearish):
                    action = "sell"
                    strength = (channel_position - 0.8) * 5  # Scale to 0-1
                    confidence = 0.7
                    metadata["signal_type"] = "kc_mean_reversion_sell"
        
        # Adjust strength based on channel width (volatility)
        if action != "hold":
            # Higher volatility (wider channels) = stronger signals
            volatility_boost = min(current_channel_width * 10, 1.5)
            strength = min(strength * volatility_boost, 1.0)
            
            # Boost confidence for extreme positions
            if channel_position <= 0.1 or channel_position >= 0.9:
                confidence = min(confidence + 0.1, 1.0)
        
        return Signal(
            timestamp=timestamp,
            action=action,
            strength=strength,
            price=current_price,
            confidence=confidence,
            metadata=metadata
        )
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get hyperopt parameter space for optimization."""
        from hyperopt import hp
        
        return {
            'kc_period': hp.choice('kc_period', list(range(10, 51))),
            'atr_multiplier': hp.uniform('atr_multiplier', 1.0, 4.0),
            'trading_mode': hp.choice('trading_mode', ["breakout", "mean_reversion"]),
            'channel_position': hp.choice('channel_position', ["upper", "middle", "lower"]),
            'trend_filter': hp.choice('trend_filter', [True, False]),
            'trend_period': hp.choice('trend_period', list(range(20, 101, 10))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.kc_upper is None:
            return {}
        
        current_price = self.data['close'].iloc[-1] if len(self.data) > 0 else 0
        current_upper = self.kc_upper.iloc[-1] if not pd.isna(self.kc_upper.iloc[-1]) else 0
        current_lower = self.kc_lower.iloc[-1] if not pd.isna(self.kc_lower.iloc[-1]) else 0
        current_middle = self.kc_middle.iloc[-1] if not pd.isna(self.kc_middle.iloc[-1]) else 0
        
        channel_position = self.get_channel_position_value(
            current_price, current_upper, current_middle, current_lower
        ) if current_upper != current_lower else 0.5
        
        return {
            'kc_upper': float(current_upper),
            'kc_lower': float(current_lower),
            'kc_middle': float(current_middle),
            'channel_position': float(channel_position),
            'channel_width': float(self.channel_width.iloc[-1]) if not pd.isna(self.channel_width.iloc[-1]) else 0.0
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and channel position.
        
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
        signal_adjustment = (signal.strength + signal.confidence) / 2
        adjusted_size = base_size * signal_adjustment
        
        # Channel-specific adjustments
        channel_width = signal.metadata.get('channel_width', 0.02)
        
        # Adjust based on channel width (volatility proxy)
        if channel_width > 0.03:  # High volatility
            adjusted_size = adjusted_size * 0.8
        elif channel_width < 0.015:  # Low volatility
            adjusted_size = adjusted_size * 1.1
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"KeltnerChannel({self.kc_period}, {self.atr_multiplier}, {self.trading_mode})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"KeltnerChannelStrategy(kc_period={self.kc_period}, atr_multiplier={self.atr_multiplier}, trading_mode='{self.trading_mode}')" 