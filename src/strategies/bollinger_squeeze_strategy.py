"""
Bollinger Squeeze Strategy

This strategy detects periods of low volatility (squeeze) using Bollinger Bands and
Keltner Channels, then anticipates breakout direction and trades the expansion.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import get_logger


class BollingerSqueezeStrategy(BaseStrategy):
    """
    Bollinger Squeeze Strategy.
    
    Detects low volatility squeeze periods when Bollinger Bands are inside Keltner Channels,
    then trades the breakout direction with volume confirmation.
    """
    
    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_mult: float = 1.5,
        squeeze_threshold: float = 0.95,
        breakout_bars: int = 3,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Bollinger Squeeze Strategy.
        
        Args:
            bb_period: Period for Bollinger Bands (10-50)
            bb_std: Standard deviation for Bollinger Bands (1.5-3.0)
            kc_period: Period for Keltner Channels (10-50)
            kc_atr_mult: ATR multiplier for Keltner Channels (1.0-3.0)
            squeeze_threshold: Threshold for squeeze detection (0.8-1.0)
            breakout_bars: Number of bars to confirm breakout (1-5)
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'bb_period': bb_period,
            'bb_std': bb_std,
            'kc_period': kc_period,
            'kc_atr_mult': kc_atr_mult,
            'squeeze_threshold': squeeze_threshold,
            'breakout_bars': breakout_bars,
            'volume_confirmation': volume_confirmation
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"BollingerSqueeze_{bb_period}_{squeeze_threshold}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_mult = kc_atr_mult
        self.squeeze_threshold = squeeze_threshold
        self.breakout_bars = breakout_bars
        self.volume_confirmation = volume_confirmation
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.bb_upper = None
        self.bb_lower = None
        self.bb_middle = None
        self.kc_upper = None
        self.kc_lower = None
        self.kc_middle = None
        self.squeeze_on = None
        self.momentum = None
        self.volume_ma = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (10 <= self.bb_period <= 50):
            self.logger.error(f"BB period {self.bb_period} outside valid range [10, 50]")
            return False
        
        if not (1.5 <= self.bb_std <= 3.0):
            self.logger.error(f"BB std {self.bb_std} outside valid range [1.5, 3.0]")
            return False
        
        if not (10 <= self.kc_period <= 50):
            self.logger.error(f"KC period {self.kc_period} outside valid range [10, 50]")
            return False
        
        if not (1.0 <= self.kc_atr_mult <= 3.0):
            self.logger.error(f"KC ATR multiplier {self.kc_atr_mult} outside valid range [1.0, 3.0]")
            return False
        
        if not (0.8 <= self.squeeze_threshold <= 1.0):
            self.logger.error(f"Squeeze threshold {self.squeeze_threshold} outside valid range [0.8, 1.0]")
            return False
        
        if not (1 <= self.breakout_bars <= 5):
            self.logger.error(f"Breakout bars {self.breakout_bars} outside valid range [1, 5]")
            return False
        
        return True
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> tuple:
        """Calculate Bollinger Bands."""
        close = data['close']
        
        # Calculate moving average (middle band)
        middle = close.rolling(window=self.bb_period).mean()
        
        # Calculate standard deviation
        std = close.rolling(window=self.bb_period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return upper, lower, middle
    
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
        upper = middle + (atr * self.kc_atr_mult)
        lower = middle - (atr * self.kc_atr_mult)
        
        return upper, lower, middle
    
    def detect_squeeze(self) -> pd.Series:
        """Detect squeeze conditions (BB inside KC)."""
        # Squeeze occurs when BB bands are inside KC bands
        squeeze_on = (self.bb_upper <= self.kc_upper) & (self.bb_lower >= self.kc_lower)
        
        # Apply threshold for more sensitive detection
        bb_width = self.bb_upper - self.bb_lower
        kc_width = self.kc_upper - self.kc_lower
        width_ratio = bb_width / kc_width
        
        squeeze_on = squeeze_on | (width_ratio <= self.squeeze_threshold)
        
        return squeeze_on
    
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate momentum oscillator for breakout direction."""
        close = data['close']
        
        # Linear regression of close prices
        momentum_values = []
        for i in range(len(close)):
            if i < self.bb_period:
                momentum_values.append(0)
                continue
            
            # Get window of prices
            window = close.iloc[i-self.bb_period+1:i+1]
            x = np.arange(len(window))
            
            # Calculate linear regression
            if len(window) > 1:
                slope, intercept = np.polyfit(x, window, 1)
                # Momentum is the difference between current price and regression line
                regression_value = slope * (len(window) - 1) + intercept
                momentum = close.iloc[i] - regression_value
            else:
                momentum = 0
            
            momentum_values.append(momentum)
        
        return pd.Series(momentum_values, index=close.index)
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.bb_period, self.kc_period) + self.breakout_bars + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate Bollinger Bands
        self.bb_upper, self.bb_lower, self.bb_middle = self.calculate_bollinger_bands(data)
        
        # Calculate Keltner Channels
        self.kc_upper, self.kc_lower, self.kc_middle = self.calculate_keltner_channels(data)
        
        # Detect squeeze conditions
        self.squeeze_on = self.detect_squeeze()
        
        # Calculate momentum
        self.momentum = self.calculate_momentum(data)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.bb_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"BB period: {self.bb_period}, KC period: {self.kc_period}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.bb_period, self.kc_period) + self.breakout_bars + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on squeeze breakout."""
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
        min_required = max(self.bb_period, self.kc_period) + self.breakout_bars
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
        current_squeeze = self.squeeze_on.iloc[current_idx]
        current_momentum = self.momentum.iloc[current_idx]
        current_bb_upper = self.bb_upper.iloc[current_idx]
        current_bb_lower = self.bb_lower.iloc[current_idx]
        
        # Check for recent squeeze
        recent_squeeze = self.squeeze_on.iloc[current_idx-self.breakout_bars:current_idx].any()
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.volume_confirmation:
            current_volume = current_data['volume']
            avg_volume = self.volume_ma.iloc[current_idx]
            volume_confirmed = current_volume >= avg_volume
        
        # Check for NaN values
        if pd.isna(current_momentum) or pd.isna(current_bb_upper) or pd.isna(current_bb_lower):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "squeeze_on": current_squeeze,
            "recent_squeeze": recent_squeeze,
            "momentum": current_momentum,
            "volume_confirmed": volume_confirmed,
            "bb_upper": current_bb_upper,
            "bb_lower": current_bb_lower
        }
        
        # Generate signals based on squeeze breakout
        if not current_squeeze and recent_squeeze and volume_confirmed:
            # Squeeze has ended, look for breakout direction
            
            # Bullish breakout
            if current_momentum > 0 and current_price > current_bb_upper:
                action = "buy"
                strength = min(abs(current_momentum) / (current_price * 0.01), 1.0)
                confidence = 0.8
                metadata["signal_type"] = "squeeze_breakout_buy"
                
            # Bearish breakout
            elif current_momentum < 0 and current_price < current_bb_lower:
                action = "sell"
                strength = min(abs(current_momentum) / (current_price * 0.01), 1.0)
                confidence = 0.8
                metadata["signal_type"] = "squeeze_breakout_sell"
                
        # Alternative signals for momentum direction during squeeze
        elif current_squeeze and volume_confirmed:
            # During squeeze, prepare for breakout based on momentum
            if abs(current_momentum) > current_price * 0.005:  # Significant momentum
                if current_momentum > 0:
                    action = "buy"
                    strength = min(abs(current_momentum) / (current_price * 0.01), 0.6)
                    confidence = 0.6
                    metadata["signal_type"] = "squeeze_momentum_buy"
                    
                elif current_momentum < 0:
                    action = "sell"
                    strength = min(abs(current_momentum) / (current_price * 0.01), 0.6)
                    confidence = 0.6
                    metadata["signal_type"] = "squeeze_momentum_sell"
        
        # Boost strength for strong momentum
        if action != "hold" and abs(current_momentum) > current_price * 0.01:
            strength = min(strength * 1.3, 1.0)
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
            'bb_period': hp.choice('bb_period', list(range(10, 51))),
            'bb_std': hp.uniform('bb_std', 1.5, 3.0),
            'kc_period': hp.choice('kc_period', list(range(10, 51))),
            'kc_atr_mult': hp.uniform('kc_atr_mult', 1.0, 3.0),
            'squeeze_threshold': hp.uniform('squeeze_threshold', 0.8, 1.0),
            'breakout_bars': hp.choice('breakout_bars', list(range(1, 6))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.squeeze_on is None:
            return {}
        
        return {
            'squeeze_on': bool(self.squeeze_on.iloc[-1]) if not pd.isna(self.squeeze_on.iloc[-1]) else False,
            'momentum': float(self.momentum.iloc[-1]) if not pd.isna(self.momentum.iloc[-1]) else 0.0,
            'bb_upper': float(self.bb_upper.iloc[-1]) if not pd.isna(self.bb_upper.iloc[-1]) else 0.0,
            'bb_lower': float(self.bb_lower.iloc[-1]) if not pd.isna(self.bb_lower.iloc[-1]) else 0.0
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and squeeze conditions.
        
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
        
        # Squeeze-specific adjustments
        if "squeeze_breakout" in signal.metadata.get("signal_type", ""):
            # Increase size for confirmed breakouts
            adjusted_size = adjusted_size * 1.2
        elif "squeeze_momentum" in signal.metadata.get("signal_type", ""):
            # Reduce size for momentum-only signals during squeeze
            adjusted_size = adjusted_size * 0.8
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"BollingerSqueeze({self.bb_period}, {self.squeeze_threshold})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"BollingerSqueezeStrategy(bb_period={self.bb_period}, squeeze_threshold={self.squeeze_threshold}, breakout_bars={self.breakout_bars})" 