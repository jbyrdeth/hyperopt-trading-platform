"""
Enhanced Stochastic Oscillator Strategy

This strategy uses the full stochastic oscillator with %K and %D lines for momentum
analysis, including overbought/oversold signals, divergence detection, and trend confirmation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import get_logger


class StochasticStrategy(BaseStrategy):
    """
    Enhanced Stochastic Oscillator Strategy.
    
    Uses full stochastic oscillator with %K and %D lines for momentum reversal signals,
    divergence detection, and adaptive signal levels based on market volatility.
    """
    
    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
        overbought: int = 80,
        oversold: int = 20,
        trend_filter_period: int = 50,
        divergence_lookback: int = 20,
        adaptive_levels: bool = True,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Enhanced Stochastic Strategy.
        
        Args:
            k_period: %K calculation period (5-30)
            d_period: %D smoothing period (3-10)
            overbought: Overbought level (70-90)
            oversold: Oversold level (10-30)
            trend_filter_period: Period for trend confirmation MA (20-100)
            divergence_lookback: Lookback for divergence detection (10-50)
            adaptive_levels: Whether to use adaptive overbought/oversold levels
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'k_period': k_period,
            'd_period': d_period,
            'overbought': overbought,
            'oversold': oversold,
            'trend_filter_period': trend_filter_period,
            'divergence_lookback': divergence_lookback,
            'adaptive_levels': adaptive_levels,
            'volume_confirmation': volume_confirmation
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"Stochastic_{k_period}_{overbought}_{oversold}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.k_period = k_period
        self.d_period = d_period
        self.overbought = overbought
        self.oversold = oversold
        self.trend_filter_period = trend_filter_period
        self.divergence_lookback = divergence_lookback
        self.adaptive_levels = adaptive_levels
        self.volume_confirmation = volume_confirmation
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.stoch_k = None
        self.stoch_d = None
        self.trend_ma = None
        self.volume_ma = None
        self.adaptive_overbought = None
        self.adaptive_oversold = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.k_period <= 30):
            self.logger.error(f"K period {self.k_period} outside valid range [5, 30]")
            return False
        
        if not (3 <= self.d_period <= 10):
            self.logger.error(f"D period {self.d_period} outside valid range [3, 10]")
            return False
        
        if not (70 <= self.overbought <= 90):
            self.logger.error(f"Overbought level {self.overbought} outside valid range [70, 90]")
            return False
        
        if not (10 <= self.oversold <= 30):
            self.logger.error(f"Oversold level {self.oversold} outside valid range [10, 30]")
            return False
        
        if self.oversold >= self.overbought:
            self.logger.error("Oversold level must be less than overbought level")
            return False
        
        if not (20 <= self.trend_filter_period <= 100):
            self.logger.error(f"Trend filter period {self.trend_filter_period} outside valid range [20, 100]")
            return False
        
        if not (10 <= self.divergence_lookback <= 50):
            self.logger.error(f"Divergence lookback {self.divergence_lookback} outside valid range [10, 50]")
            return False
        
        return True
    
    def calculate_stochastic(self, data: pd.DataFrame) -> tuple:
        """Calculate Stochastic %K and %D."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate %K
        lowest_low = low.rolling(window=self.k_period).min()
        highest_high = high.rolling(window=self.k_period).max()
        
        stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (smoothed %K)
        stoch_d = stoch_k.rolling(window=self.d_period).mean()
        
        return stoch_k, stoch_d
    
    def calculate_adaptive_levels(self, stoch_k: pd.Series) -> tuple:
        """Calculate adaptive overbought/oversold levels based on volatility."""
        if not self.adaptive_levels:
            return pd.Series([self.overbought] * len(stoch_k), index=stoch_k.index), \
                   pd.Series([self.oversold] * len(stoch_k), index=stoch_k.index)
        
        # Calculate rolling volatility of stochastic
        stoch_volatility = stoch_k.rolling(window=self.trend_filter_period).std()
        
        # Adjust levels based on volatility
        volatility_adjustment = stoch_volatility / stoch_volatility.rolling(window=self.trend_filter_period * 2).mean()
        
        # Higher volatility = more extreme levels
        adaptive_overbought = self.overbought + (volatility_adjustment - 1) * 10
        adaptive_oversold = self.oversold - (volatility_adjustment - 1) * 10
        
        # Clamp to reasonable ranges
        adaptive_overbought = adaptive_overbought.clip(75, 95)
        adaptive_oversold = adaptive_oversold.clip(5, 25)
        
        return adaptive_overbought, adaptive_oversold
    
    def detect_divergence(self, price: pd.Series, stoch: pd.Series, current_idx: int) -> Dict[str, bool]:
        """Detect bullish and bearish divergences between price and stochastic."""
        if current_idx < self.divergence_lookback:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        # Get lookback window
        start_idx = max(0, current_idx - self.divergence_lookback)
        price_window = price.iloc[start_idx:current_idx + 1]
        stoch_window = stoch.iloc[start_idx:current_idx + 1]
        
        if len(price_window) < 5 or len(stoch_window) < 5:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        # Find peaks and troughs
        price_peaks = []
        price_troughs = []
        stoch_peaks = []
        stoch_troughs = []
        
        for i in range(2, len(price_window) - 2):
            # Price peaks and troughs
            if (price_window.iloc[i] > price_window.iloc[i-1] and 
                price_window.iloc[i] > price_window.iloc[i+1] and
                price_window.iloc[i] > price_window.iloc[i-2] and 
                price_window.iloc[i] > price_window.iloc[i+2]):
                price_peaks.append((i, price_window.iloc[i]))
            
            if (price_window.iloc[i] < price_window.iloc[i-1] and 
                price_window.iloc[i] < price_window.iloc[i+1] and
                price_window.iloc[i] < price_window.iloc[i-2] and 
                price_window.iloc[i] < price_window.iloc[i+2]):
                price_troughs.append((i, price_window.iloc[i]))
            
            # Stochastic peaks and troughs
            if (stoch_window.iloc[i] > stoch_window.iloc[i-1] and 
                stoch_window.iloc[i] > stoch_window.iloc[i+1] and
                stoch_window.iloc[i] > stoch_window.iloc[i-2] and 
                stoch_window.iloc[i] > stoch_window.iloc[i+2]):
                stoch_peaks.append((i, stoch_window.iloc[i]))
            
            if (stoch_window.iloc[i] < stoch_window.iloc[i-1] and 
                stoch_window.iloc[i] < stoch_window.iloc[i+1] and
                stoch_window.iloc[i] < stoch_window.iloc[i-2] and 
                stoch_window.iloc[i] < stoch_window.iloc[i+2]):
                stoch_troughs.append((i, stoch_window.iloc[i]))
        
        bullish_divergence = False
        bearish_divergence = False
        
        # Bullish divergence: Price makes lower lows, Stochastic makes higher lows
        if len(price_troughs) >= 2 and len(stoch_troughs) >= 2:
            recent_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            # Find corresponding stochastic troughs
            recent_stoch_trough = None
            prev_stoch_trough = None
            
            for stoch_trough in stoch_troughs:
                if abs(stoch_trough[0] - recent_price_trough[0]) <= 3:
                    recent_stoch_trough = stoch_trough
                    break
            
            for stoch_trough in stoch_troughs:
                if abs(stoch_trough[0] - prev_price_trough[0]) <= 3:
                    prev_stoch_trough = stoch_trough
                    break
            
            if (recent_stoch_trough and prev_stoch_trough and
                recent_price_trough[1] < prev_price_trough[1] and
                recent_stoch_trough[1] > prev_stoch_trough[1]):
                bullish_divergence = True
        
        # Bearish divergence: Price makes higher highs, Stochastic makes lower highs
        if len(price_peaks) >= 2 and len(stoch_peaks) >= 2:
            recent_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            # Find corresponding stochastic peaks
            recent_stoch_peak = None
            prev_stoch_peak = None
            
            for stoch_peak in stoch_peaks:
                if abs(stoch_peak[0] - recent_price_peak[0]) <= 3:
                    recent_stoch_peak = stoch_peak
                    break
            
            for stoch_peak in stoch_peaks:
                if abs(stoch_peak[0] - prev_price_peak[0]) <= 3:
                    prev_stoch_peak = stoch_peak
                    break
            
            if (recent_stoch_peak and prev_stoch_peak and
                recent_price_peak[1] > prev_price_peak[1] and
                recent_stoch_peak[1] < prev_stoch_peak[1]):
                bearish_divergence = True
        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence
        }
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.k_period, self.trend_filter_period, self.divergence_lookback) + self.d_period + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate Stochastic %K and %D
        self.stoch_k, self.stoch_d = self.calculate_stochastic(data)
        
        # Calculate adaptive levels
        self.adaptive_overbought, self.adaptive_oversold = self.calculate_adaptive_levels(self.stoch_k)
        
        # Calculate trend filter
        self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_filter_period)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.k_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"K period: {self.k_period}, D period: {self.d_period}, Levels: {self.oversold}-{self.overbought}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.k_period, self.trend_filter_period, self.divergence_lookback) + self.d_period + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on stochastic oscillator analysis."""
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
        min_required = max(self.k_period, self.trend_filter_period, self.divergence_lookback) + self.d_period
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
        current_k = self.stoch_k.iloc[current_idx]
        current_d = self.stoch_d.iloc[current_idx]
        current_overbought = self.adaptive_overbought.iloc[current_idx]
        current_oversold = self.adaptive_oversold.iloc[current_idx]
        
        # Get previous values for crossover detection
        prev_k = self.stoch_k.iloc[current_idx - 1] if current_idx > 0 else current_k
        prev_d = self.stoch_d.iloc[current_idx - 1] if current_idx > 0 else current_d
        
        # Trend filter
        current_trend_ma = self.trend_ma.iloc[current_idx]
        trend_bullish = current_price > current_trend_ma if not pd.isna(current_trend_ma) else True
        trend_bearish = current_price < current_trend_ma if not pd.isna(current_trend_ma) else True
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_confirmation:
            current_volume = current_data['volume']
            avg_volume = self.volume_ma.iloc[current_idx]
            volume_confirmed = current_volume >= avg_volume
        
        # Check for NaN values
        if pd.isna(current_k) or pd.isna(current_d):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Detect divergences
        divergence_info = self.detect_divergence(self.data['close'], self.stoch_k, current_idx)
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "stoch_k": current_k,
            "stoch_d": current_d,
            "overbought_level": current_overbought,
            "oversold_level": current_oversold,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish,
            "volume_confirmed": volume_confirmed,
            "bullish_divergence": divergence_info["bullish_divergence"],
            "bearish_divergence": divergence_info["bearish_divergence"]
        }
        
        # Generate signals based on stochastic crossovers and levels
        if volume_confirmed:
            # Bullish signals: %K crosses above %D in oversold zone
            if (current_k > current_d and prev_k <= prev_d and 
                current_k <= current_oversold + 10 and trend_bullish):
                action = "buy"
                strength = min((current_oversold + 10 - current_k) / 10, 1.0)
                confidence = 0.8
                metadata["signal_type"] = "stochastic_oversold_crossover_buy"
                
                # Boost for divergence
                if divergence_info["bullish_divergence"]:
                    strength = min(strength * 1.3, 1.0)
                    confidence = min(confidence + 0.1, 1.0)
                    metadata["signal_type"] = "stochastic_divergence_buy"
            
            # Bearish signals: %K crosses below %D in overbought zone
            elif (current_k < current_d and prev_k >= prev_d and 
                  current_k >= current_overbought - 10 and trend_bearish):
                action = "sell"
                strength = min((current_k - (current_overbought - 10)) / 10, 1.0)
                confidence = 0.8
                metadata["signal_type"] = "stochastic_overbought_crossover_sell"
                
                # Boost for divergence
                if divergence_info["bearish_divergence"]:
                    strength = min(strength * 1.3, 1.0)
                    confidence = min(confidence + 0.1, 1.0)
                    metadata["signal_type"] = "stochastic_divergence_sell"
            
            # Divergence-only signals (weaker)
            elif divergence_info["bullish_divergence"] and current_k < 50 and trend_bullish:
                action = "buy"
                strength = 0.6
                confidence = 0.6
                metadata["signal_type"] = "stochastic_divergence_only_buy"
                
            elif divergence_info["bearish_divergence"] and current_k > 50 and trend_bearish:
                action = "sell"
                strength = 0.6
                confidence = 0.6
                metadata["signal_type"] = "stochastic_divergence_only_sell"
            
            # Extreme level signals (additional confirmation)
            elif current_k <= current_oversold and current_d <= current_oversold and trend_bullish:
                action = "buy"
                strength = min((current_oversold - current_k) / current_oversold, 0.7)
                confidence = 0.6
                metadata["signal_type"] = "stochastic_extreme_oversold_buy"
                
            elif current_k >= current_overbought and current_d >= current_overbought and trend_bearish:
                action = "sell"
                strength = min((current_k - current_overbought) / (100 - current_overbought), 0.7)
                confidence = 0.6
                metadata["signal_type"] = "stochastic_extreme_overbought_sell"
        
        # Adjust strength based on distance from extreme levels
        if action != "hold":
            if action == "buy":
                distance_factor = max(0.5, (current_oversold + 20 - current_k) / 20)
            else:
                distance_factor = max(0.5, (current_k - (current_overbought - 20)) / 20)
            
            strength = min(strength * distance_factor, 1.0)
        
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
            'k_period': hp.choice('k_period', list(range(5, 31))),
            'd_period': hp.choice('d_period', list(range(3, 11))),
            'overbought': hp.choice('overbought', list(range(70, 91))),
            'oversold': hp.choice('oversold', list(range(10, 31))),
            'trend_filter_period': hp.choice('trend_filter_period', list(range(20, 101, 10))),
            'divergence_lookback': hp.choice('divergence_lookback', list(range(10, 51))),
            'adaptive_levels': hp.choice('adaptive_levels', [True, False]),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.stoch_k is None:
            return {}
        
        return {
            'stoch_k': float(self.stoch_k.iloc[-1]) if not pd.isna(self.stoch_k.iloc[-1]) else 0.0,
            'stoch_d': float(self.stoch_d.iloc[-1]) if not pd.isna(self.stoch_d.iloc[-1]) else 0.0,
            'overbought_level': float(self.adaptive_overbought.iloc[-1]) if not pd.isna(self.adaptive_overbought.iloc[-1]) else float(self.overbought),
            'oversold_level': float(self.adaptive_oversold.iloc[-1]) if not pd.isna(self.adaptive_oversold.iloc[-1]) else float(self.oversold)
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and stochastic levels.
        
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
        
        # Stochastic-specific adjustments
        stoch_k = signal.metadata.get('stoch_k', 50)
        
        # Increase size for extreme stochastic levels
        if signal.action == 'buy' and stoch_k < 20:
            adjusted_size = adjusted_size * 1.2
        elif signal.action == 'sell' and stoch_k > 80:
            adjusted_size = adjusted_size * 1.2
        
        # Boost for divergence signals
        if "divergence" in signal.metadata.get("signal_type", ""):
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
        return f"Stochastic({self.k_period}, {self.d_period}, {self.oversold}-{self.overbought})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"StochasticStrategy(k_period={self.k_period}, d_period={self.d_period}, overbought={self.overbought}, oversold={self.oversold})" 