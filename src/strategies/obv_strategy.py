"""
OBV (On-Balance Volume) Strategy

This strategy uses On-Balance Volume to identify buying and selling pressure,
generating signals based on OBV trends and divergences with price.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class OBVStrategy(BaseStrategy):
    """
    OBV (On-Balance Volume) Strategy.
    
    Generates signals based on OBV trend analysis and price-volume divergences.
    Uses OBV moving averages and momentum to identify buying/selling pressure.
    """
    
    def __init__(
        self,
        obv_ma_period: int = 20,
        divergence_lookback: int = 10,
        signal_threshold: float = 0.02,
        volume_threshold: float = 1.2,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize OBV Strategy.
        
        Args:
            obv_ma_period: Period for OBV moving average (5-50)
            divergence_lookback: Lookback period for divergence detection (5-20)
            signal_threshold: Minimum OBV change to generate signal (0.01-0.05)
            volume_threshold: Volume confirmation multiplier (1.0-2.0)
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'obv_ma_period': obv_ma_period,
            'divergence_lookback': divergence_lookback,
            'signal_threshold': signal_threshold,
            'volume_threshold': volume_threshold
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"OBV_{obv_ma_period}_{divergence_lookback}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.obv_ma_period = obv_ma_period
        self.divergence_lookback = divergence_lookback
        self.signal_threshold = signal_threshold
        self.volume_threshold = volume_threshold
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.obv = None
        self.obv_ma = None
        self.obv_momentum = None
        self.volume_ma = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.obv_ma_period <= 50):
            self.logger.error(f"OBV MA period {self.obv_ma_period} outside valid range [5, 50]")
            return False
        
        if not (5 <= self.divergence_lookback <= 20):
            self.logger.error(f"Divergence lookback {self.divergence_lookback} outside valid range [5, 20]")
            return False
        
        if not (0.01 <= self.signal_threshold <= 0.05):
            self.logger.error(f"Signal threshold {self.signal_threshold} outside valid range [0.01, 0.05]")
            return False
        
        if not (1.0 <= self.volume_threshold <= 2.0):
            self.logger.error(f"Volume threshold {self.volume_threshold} outside valid range [1.0, 2.0]")
            return False
        
        return True
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv_values = [0]  # Start with 0
        
        for i in range(1, len(data)):
            prev_close = data['close'].iloc[i-1]
            curr_close = data['close'].iloc[i]
            curr_volume = data['volume'].iloc[i]
            
            if curr_close > prev_close:
                # Price up, add volume
                obv_values.append(obv_values[-1] + curr_volume)
            elif curr_close < prev_close:
                # Price down, subtract volume
                obv_values.append(obv_values[-1] - curr_volume)
            else:
                # Price unchanged, OBV unchanged
                obv_values.append(obv_values[-1])
        
        return pd.Series(obv_values, index=data.index)
    
    def detect_divergence(self, price_data: pd.Series, obv_data: pd.Series, lookback: int) -> str:
        """Detect bullish or bearish divergence between price and OBV."""
        if len(price_data) < lookback or len(obv_data) < lookback:
            return "none"
        
        # Get recent data
        recent_price = price_data.iloc[-lookback:]
        recent_obv = obv_data.iloc[-lookback:]
        
        # Calculate trends (simple linear regression slope)
        price_trend = np.polyfit(range(len(recent_price)), recent_price, 1)[0]
        obv_trend = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
        
        # Normalize trends for comparison
        price_trend_norm = price_trend / recent_price.mean() if recent_price.mean() != 0 else 0
        obv_trend_norm = obv_trend / abs(recent_obv.mean()) if recent_obv.mean() != 0 else 0
        
        # Detect divergence
        threshold = 0.001  # Minimum trend difference for divergence
        
        if price_trend_norm > threshold and obv_trend_norm < -threshold:
            return "bearish"  # Price up, OBV down
        elif price_trend_norm < -threshold and obv_trend_norm > threshold:
            return "bullish"  # Price down, OBV up
        else:
            return "none"
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.obv_ma_period, self.divergence_lookback) + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate OBV
        self.obv = self.calculate_obv(data)
        
        # Calculate OBV moving average
        self.obv_ma = ta.trend.sma_indicator(self.obv, window=self.obv_ma_period)
        
        # Calculate OBV momentum (rate of change)
        self.obv_momentum = self.obv.pct_change(periods=5)
        
        # Calculate volume moving average for confirmation
        self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.obv_ma_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"OBV MA period: {self.obv_ma_period}, Divergence lookback: {self.divergence_lookback}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.obv_ma_period, self.divergence_lookback) + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on OBV analysis."""
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
        min_required = max(self.obv_ma_period, self.divergence_lookback)
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
        current_obv = self.obv.iloc[current_idx]
        current_obv_ma = self.obv_ma.iloc[current_idx]
        current_obv_momentum = self.obv_momentum.iloc[current_idx]
        current_volume = current_data['volume']
        avg_volume = self.volume_ma.iloc[current_idx]
        
        # Check for NaN values
        if pd.isna(current_obv) or pd.isna(current_obv_ma) or pd.isna(current_obv_momentum):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Detect divergence
        price_window = self.data['close'].iloc[current_idx-self.divergence_lookback+1:current_idx+1]
        obv_window = self.obv.iloc[current_idx-self.divergence_lookback+1:current_idx+1]
        divergence = self.detect_divergence(price_window, obv_window, self.divergence_lookback)
        
        # Volume confirmation
        volume_confirmed = current_volume >= (avg_volume * self.volume_threshold)
        
        # OBV trend analysis
        obv_above_ma = current_obv > current_obv_ma
        obv_momentum_strong = abs(current_obv_momentum) >= self.signal_threshold
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "obv": current_obv,
            "obv_ma": current_obv_ma,
            "obv_momentum": current_obv_momentum,
            "divergence": divergence,
            "volume_confirmed": volume_confirmed,
            "obv_above_ma": obv_above_ma
        }
        
        # Generate signals based on OBV analysis
        if obv_momentum_strong and volume_confirmed:
            # Bullish signals
            if (obv_above_ma and current_obv_momentum > self.signal_threshold) or divergence == "bullish":
                action = "buy"
                strength = min(abs(current_obv_momentum) / self.signal_threshold, 1.0)
                confidence = 0.7
                
                if divergence == "bullish":
                    confidence += 0.2
                    metadata["signal_type"] = "bullish_divergence"
                else:
                    metadata["signal_type"] = "obv_momentum_buy"
                    
            # Bearish signals
            elif (not obv_above_ma and current_obv_momentum < -self.signal_threshold) or divergence == "bearish":
                action = "sell"
                strength = min(abs(current_obv_momentum) / self.signal_threshold, 1.0)
                confidence = 0.7
                
                if divergence == "bearish":
                    confidence += 0.2
                    metadata["signal_type"] = "bearish_divergence"
                else:
                    metadata["signal_type"] = "obv_momentum_sell"
        
        # Adjust strength and confidence based on volume
        if action != "hold":
            volume_boost = min(current_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
            strength = min(strength * volume_boost, 1.0)
            confidence = min(confidence * (0.8 + 0.2 * volume_boost), 1.0)
        
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
            'obv_ma_period': hp.choice('obv_ma_period', list(range(5, 51))),
            'divergence_lookback': hp.choice('divergence_lookback', list(range(5, 21))),
            'signal_threshold': hp.uniform('signal_threshold', 0.01, 0.05),
            'volume_threshold': hp.uniform('volume_threshold', 1.0, 2.0),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.obv is None:
            return {}
        
        return {
            'obv': float(self.obv.iloc[-1]) if not pd.isna(self.obv.iloc[-1]) else 0.0,
            'obv_ma': float(self.obv_ma.iloc[-1]) if not pd.isna(self.obv_ma.iloc[-1]) else 0.0,
            'obv_momentum': float(self.obv_momentum.iloc[-1]) if not pd.isna(self.obv_momentum.iloc[-1]) else 0.0
        }
    
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
        signal_adjustment = (signal.strength + signal.confidence) / 2
        adjusted_size = base_size * signal_adjustment
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"OBV({self.obv_ma_period}, {self.divergence_lookback})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"OBVStrategy(obv_ma_period={self.obv_ma_period}, divergence_lookback={self.divergence_lookback}, signal_threshold={self.signal_threshold})" 