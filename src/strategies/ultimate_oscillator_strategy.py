"""
Ultimate Oscillator Strategy

This strategy uses the Ultimate Oscillator for multi-period momentum analysis with
weighted momentum calculation across timeframes, bullish/bearish divergence detection, overbought/oversold with trend confirmation, and signal strength scoring system.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class UltimateOscillatorStrategy(BaseStrategy):
    """
    Ultimate Oscillator Strategy.
    
    Uses Ultimate Oscillator for multi-period momentum analysis with weighted
    momentum calculation, divergence detection, and comprehensive signal strength scoring.
    """
    
    def __init__(
        self,
        short_period: int = 7,
        medium_period: int = 21,
        long_period: int = 28,
        overbought: int = 70,
        oversold: int = 30,
        signal_threshold: int = 50,
        trend_confirmation: int = 50,
        divergence_lookback: int = 20,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Ultimate Oscillator Strategy.
        
        Args:
            short_period: Short period for UO calculation (5-15)
            medium_period: Medium period for UO calculation (15-35)
            long_period: Long period for UO calculation (25-50)
            overbought: Overbought level (65-80)
            oversold: Oversold level (20-35)
            signal_threshold: Minimum signal strength (30-70)
            trend_confirmation: Period for trend confirmation (20-100)
            divergence_lookback: Lookback for divergence detection (10-50)
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'short_period': short_period,
            'medium_period': medium_period,
            'long_period': long_period,
            'overbought': overbought,
            'oversold': oversold,
            'signal_threshold': signal_threshold,
            'trend_confirmation': trend_confirmation,
            'divergence_lookback': divergence_lookback,
            'volume_confirmation': volume_confirmation
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"UltimateOscillator_{short_period}_{medium_period}_{long_period}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self.overbought = overbought
        self.oversold = oversold
        self.signal_threshold = signal_threshold
        self.trend_confirmation = trend_confirmation
        self.divergence_lookback = divergence_lookback
        self.volume_confirmation = volume_confirmation
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.ultimate_oscillator = None
        self.uo_short = None
        self.uo_medium = None
        self.uo_long = None
        self.trend_ma = None
        self.volume_ma = None
        self.signal_strength = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.short_period <= 15):
            self.logger.error(f"Short period {self.short_period} outside valid range [5, 15]")
            return False
        
        if not (15 <= self.medium_period <= 35):
            self.logger.error(f"Medium period {self.medium_period} outside valid range [15, 35]")
            return False
        
        if not (25 <= self.long_period <= 50):
            self.logger.error(f"Long period {self.long_period} outside valid range [25, 50]")
            return False
        
        if not (self.short_period < self.medium_period < self.long_period):
            self.logger.error("Periods must be in ascending order: short < medium < long")
            return False
        
        if not (65 <= self.overbought <= 80):
            self.logger.error(f"Overbought level {self.overbought} outside valid range [65, 80]")
            return False
        
        if not (20 <= self.oversold <= 35):
            self.logger.error(f"Oversold level {self.oversold} outside valid range [20, 35]")
            return False
        
        if self.oversold >= self.overbought:
            self.logger.error("Oversold level must be less than overbought level")
            return False
        
        if not (30 <= self.signal_threshold <= 70):
            self.logger.error(f"Signal threshold {self.signal_threshold} outside valid range [30, 70]")
            return False
        
        if not (20 <= self.trend_confirmation <= 100):
            self.logger.error(f"Trend confirmation {self.trend_confirmation} outside valid range [20, 100]")
            return False
        
        if not (10 <= self.divergence_lookback <= 50):
            self.logger.error(f"Divergence lookback {self.divergence_lookback} outside valid range [10, 50]")
            return False
        
        return True
    
    def calculate_buying_pressure(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate buying pressure for Ultimate Oscillator."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Low = min(Low, Previous Close)
        true_low = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        
        # Buying Pressure = Close - True Low
        buying_pressure = close - true_low
        
        return buying_pressure.rolling(window=period).sum()
    
    def calculate_true_range_sum(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate True Range sum for Ultimate Oscillator."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True High = max(High, Previous Close)
        true_high = pd.concat([high, close.shift(1)], axis=1).max(axis=1)
        
        # True Low = min(Low, Previous Close)
        true_low = pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        
        # True Range = True High - True Low
        true_range = true_high - true_low
        
        return true_range.rolling(window=period).sum()
    
    def calculate_ultimate_oscillator(self, data: pd.DataFrame) -> tuple:
        """Calculate Ultimate Oscillator and component values."""
        # Calculate buying pressure for each period
        bp_short = self.calculate_buying_pressure(data, self.short_period)
        bp_medium = self.calculate_buying_pressure(data, self.medium_period)
        bp_long = self.calculate_buying_pressure(data, self.long_period)
        
        # Calculate true range sum for each period
        tr_short = self.calculate_true_range_sum(data, self.short_period)
        tr_medium = self.calculate_true_range_sum(data, self.medium_period)
        tr_long = self.calculate_true_range_sum(data, self.long_period)
        
        # Calculate raw values for each period
        raw_short = (bp_short / tr_short) * 100
        raw_medium = (bp_medium / tr_medium) * 100
        raw_long = (bp_long / tr_long) * 100
        
        # Calculate Ultimate Oscillator with weights (4:2:1)
        ultimate_oscillator = ((4 * raw_short) + (2 * raw_medium) + raw_long) / 7
        
        return ultimate_oscillator, raw_short, raw_medium, raw_long
    
    def calculate_signal_strength(self, uo: pd.Series, uo_short: pd.Series, uo_medium: pd.Series, uo_long: pd.Series) -> pd.Series:
        """Calculate comprehensive signal strength score."""
        # Base strength from UO level
        base_strength = pd.Series(index=uo.index, dtype=float)
        
        # Strength based on UO position
        base_strength = abs(uo - 50) * 2  # Scale 0-50 to 0-100
        
        # Timeframe alignment bonus
        alignment_bonus = pd.Series(0.0, index=uo.index)
        
        # Check if all timeframes are aligned
        bullish_alignment = (uo_short > 50) & (uo_medium > 50) & (uo_long > 50)
        bearish_alignment = (uo_short < 50) & (uo_medium < 50) & (uo_long < 50)
        
        alignment_bonus[bullish_alignment | bearish_alignment] = 20
        
        # Momentum bonus (rate of change)
        momentum_bonus = abs(uo.diff()) * 5
        momentum_bonus = momentum_bonus.clip(0, 15)
        
        # Combine all components
        total_strength = base_strength + alignment_bonus + momentum_bonus
        total_strength = total_strength.clip(0, 100)
        
        return total_strength
    
    def detect_divergence(self, price: pd.Series, uo: pd.Series, current_idx: int) -> Dict[str, bool]:
        """Detect bullish and bearish divergences between price and Ultimate Oscillator."""
        if current_idx < self.divergence_lookback:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        # Get lookback window
        start_idx = max(0, current_idx - self.divergence_lookback)
        price_window = price.iloc[start_idx:current_idx + 1]
        uo_window = uo.iloc[start_idx:current_idx + 1]
        
        if len(price_window) < 5 or len(uo_window) < 5:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        # Find peaks and troughs
        price_peaks = []
        price_troughs = []
        uo_peaks = []
        uo_troughs = []
        
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
            
            # UO peaks and troughs
            if (uo_window.iloc[i] > uo_window.iloc[i-1] and 
                uo_window.iloc[i] > uo_window.iloc[i+1] and
                uo_window.iloc[i] > uo_window.iloc[i-2] and 
                uo_window.iloc[i] > uo_window.iloc[i+2]):
                uo_peaks.append((i, uo_window.iloc[i]))
            
            if (uo_window.iloc[i] < uo_window.iloc[i-1] and 
                uo_window.iloc[i] < uo_window.iloc[i+1] and
                uo_window.iloc[i] < uo_window.iloc[i-2] and 
                uo_window.iloc[i] < uo_window.iloc[i+2]):
                uo_troughs.append((i, uo_window.iloc[i]))
        
        bullish_divergence = False
        bearish_divergence = False
        
        # Bullish divergence: Price makes lower lows, UO makes higher lows
        if len(price_troughs) >= 2 and len(uo_troughs) >= 2:
            recent_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            # Find corresponding UO troughs
            recent_uo_trough = None
            prev_uo_trough = None
            
            for uo_trough in uo_troughs:
                if abs(uo_trough[0] - recent_price_trough[0]) <= 3:
                    recent_uo_trough = uo_trough
                    break
            
            for uo_trough in uo_troughs:
                if abs(uo_trough[0] - prev_price_trough[0]) <= 3:
                    prev_uo_trough = uo_trough
                    break
            
            if (recent_uo_trough and prev_uo_trough and
                recent_price_trough[1] < prev_price_trough[1] and
                recent_uo_trough[1] > prev_uo_trough[1]):
                bullish_divergence = True
        
        # Bearish divergence: Price makes higher highs, UO makes lower highs
        if len(price_peaks) >= 2 and len(uo_peaks) >= 2:
            recent_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            # Find corresponding UO peaks
            recent_uo_peak = None
            prev_uo_peak = None
            
            for uo_peak in uo_peaks:
                if abs(uo_peak[0] - recent_price_peak[0]) <= 3:
                    recent_uo_peak = uo_peak
                    break
            
            for uo_peak in uo_peaks:
                if abs(uo_peak[0] - prev_price_peak[0]) <= 3:
                    prev_uo_peak = uo_peak
                    break
            
            if (recent_uo_peak and prev_uo_peak and
                recent_price_peak[1] > prev_price_peak[1] and
                recent_uo_peak[1] < prev_uo_peak[1]):
                bearish_divergence = True
        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence
        }
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.long_period, self.trend_confirmation, self.divergence_lookback) + 20
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate Ultimate Oscillator and components
        self.ultimate_oscillator, self.uo_short, self.uo_medium, self.uo_long = self.calculate_ultimate_oscillator(data)
        
        # Calculate signal strength
        self.signal_strength = self.calculate_signal_strength(
            self.ultimate_oscillator, self.uo_short, self.uo_medium, self.uo_long
        )
        
        # Calculate trend confirmation
        self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_confirmation)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.medium_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"Periods: {self.short_period}/{self.medium_period}/{self.long_period}, Levels: {self.oversold}-{self.overbought}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.long_period, self.trend_confirmation, self.divergence_lookback) + 20
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on Ultimate Oscillator analysis."""
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
        min_required = max(self.long_period, self.trend_confirmation, self.divergence_lookback)
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
        current_uo = self.ultimate_oscillator.iloc[current_idx]
        current_uo_short = self.uo_short.iloc[current_idx]
        current_uo_medium = self.uo_medium.iloc[current_idx]
        current_uo_long = self.uo_long.iloc[current_idx]
        current_signal_strength = self.signal_strength.iloc[current_idx]
        
        # Get previous values for momentum detection
        prev_uo = self.ultimate_oscillator.iloc[current_idx - 1] if current_idx > 0 else current_uo
        
        # Trend confirmation
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
        if pd.isna(current_uo) or pd.isna(current_signal_strength):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Detect divergences
        divergence_info = self.detect_divergence(self.data['close'], self.ultimate_oscillator, current_idx)
        
        # Check timeframe alignment
        timeframes_aligned_bullish = (current_uo_short > 50 and current_uo_medium > 50 and current_uo_long > 50)
        timeframes_aligned_bearish = (current_uo_short < 50 and current_uo_medium < 50 and current_uo_long < 50)
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "ultimate_oscillator": current_uo,
            "uo_short": current_uo_short,
            "uo_medium": current_uo_medium,
            "uo_long": current_uo_long,
            "signal_strength": current_signal_strength,
            "overbought_level": self.overbought,
            "oversold_level": self.oversold,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish,
            "volume_confirmed": volume_confirmed,
            "timeframes_aligned_bullish": timeframes_aligned_bullish,
            "timeframes_aligned_bearish": timeframes_aligned_bearish,
            "bullish_divergence": divergence_info["bullish_divergence"],
            "bearish_divergence": divergence_info["bearish_divergence"]
        }
        
        # Generate signals based on Ultimate Oscillator analysis
        if volume_confirmed and current_signal_strength >= self.signal_threshold:
            # Bullish signals: UO rises from oversold with trend confirmation
            if (current_uo > self.oversold and prev_uo <= self.oversold and 
                trend_bullish):
                action = "buy"
                strength = min(current_signal_strength / 100, 1.0)
                confidence = 0.8
                metadata["signal_type"] = "uo_oversold_recovery_buy"
                
                # Boost for divergence
                if divergence_info["bullish_divergence"]:
                    strength = min(strength * 1.3, 1.0)
                    confidence = min(confidence + 0.1, 1.0)
                    metadata["signal_type"] = "uo_divergence_buy"
                
                # Boost for timeframe alignment
                if timeframes_aligned_bullish:
                    strength = min(strength * 1.2, 1.0)
                    confidence = min(confidence + 0.05, 1.0)
                    metadata["signal_type"] = "uo_aligned_buy"
            
            # Bearish signals: UO falls from overbought with trend confirmation
            elif (current_uo < self.overbought and prev_uo >= self.overbought and 
                  trend_bearish):
                action = "sell"
                strength = min(current_signal_strength / 100, 1.0)
                confidence = 0.8
                metadata["signal_type"] = "uo_overbought_decline_sell"
                
                # Boost for divergence
                if divergence_info["bearish_divergence"]:
                    strength = min(strength * 1.3, 1.0)
                    confidence = min(confidence + 0.1, 1.0)
                    metadata["signal_type"] = "uo_divergence_sell"
                
                # Boost for timeframe alignment
                if timeframes_aligned_bearish:
                    strength = min(strength * 1.2, 1.0)
                    confidence = min(confidence + 0.05, 1.0)
                    metadata["signal_type"] = "uo_aligned_sell"
            
            # Divergence-only signals (weaker)
            elif divergence_info["bullish_divergence"] and current_uo < 60 and trend_bullish:
                action = "buy"
                strength = min(current_signal_strength / 150, 0.8)
                confidence = 0.7
                metadata["signal_type"] = "uo_divergence_only_buy"
                
            elif divergence_info["bearish_divergence"] and current_uo > 40 and trend_bearish:
                action = "sell"
                strength = min(current_signal_strength / 150, 0.8)
                confidence = 0.7
                metadata["signal_type"] = "uo_divergence_only_sell"
            
            # Strong momentum signals with timeframe alignment
            elif (timeframes_aligned_bullish and current_uo > 50 and 
                  current_uo > prev_uo + 2 and trend_bullish):
                action = "buy"
                strength = min(current_signal_strength / 120, 0.9)
                confidence = 0.75
                metadata["signal_type"] = "uo_momentum_aligned_buy"
                
            elif (timeframes_aligned_bearish and current_uo < 50 and 
                  current_uo < prev_uo - 2 and trend_bearish):
                action = "sell"
                strength = min(current_signal_strength / 120, 0.9)
                confidence = 0.75
                metadata["signal_type"] = "uo_momentum_aligned_sell"
        
        # Adjust strength based on UO momentum
        if action != "hold":
            uo_momentum = abs(current_uo - prev_uo)
            momentum_boost = min(uo_momentum / 10, 0.2)
            strength = min(strength + momentum_boost, 1.0)
        
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
            'short_period': hp.choice('short_period', list(range(5, 16))),
            'medium_period': hp.choice('medium_period', list(range(15, 36))),
            'long_period': hp.choice('long_period', list(range(25, 51))),
            'overbought': hp.choice('overbought', list(range(65, 81))),
            'oversold': hp.choice('oversold', list(range(20, 36))),
            'signal_threshold': hp.choice('signal_threshold', list(range(30, 71))),
            'trend_confirmation': hp.choice('trend_confirmation', list(range(20, 101, 10))),
            'divergence_lookback': hp.choice('divergence_lookback', list(range(10, 51))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.ultimate_oscillator is None:
            return {}
        
        return {
            'ultimate_oscillator': float(self.ultimate_oscillator.iloc[-1]) if not pd.isna(self.ultimate_oscillator.iloc[-1]) else 0.0,
            'uo_short': float(self.uo_short.iloc[-1]) if not pd.isna(self.uo_short.iloc[-1]) else 0.0,
            'uo_medium': float(self.uo_medium.iloc[-1]) if not pd.isna(self.uo_medium.iloc[-1]) else 0.0,
            'uo_long': float(self.uo_long.iloc[-1]) if not pd.isna(self.uo_long.iloc[-1]) else 0.0,
            'signal_strength': float(self.signal_strength.iloc[-1]) if not pd.isna(self.signal_strength.iloc[-1]) else 0.0,
            'overbought_level': float(self.overbought),
            'oversold_level': float(self.oversold)
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and Ultimate Oscillator analysis.
        
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
        
        # Ultimate Oscillator specific adjustments
        signal_strength_score = signal.metadata.get('signal_strength', 50)
        timeframes_aligned = (signal.metadata.get('timeframes_aligned_bullish', False) or 
                            signal.metadata.get('timeframes_aligned_bearish', False))
        
        # Increase size for high signal strength
        if signal_strength_score > 80:
            adjusted_size = adjusted_size * 1.3
        elif signal_strength_score > 60:
            adjusted_size = adjusted_size * 1.1
        
        # Boost for timeframe alignment
        if timeframes_aligned:
            adjusted_size = adjusted_size * 1.2
        
        # Boost for divergence signals
        if "divergence" in signal.metadata.get("signal_type", ""):
            adjusted_size = adjusted_size * 1.15
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"UltimateOscillator({self.short_period}, {self.medium_period}, {self.long_period})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"UltimateOscillatorStrategy(short_period={self.short_period}, medium_period={self.medium_period}, long_period={self.long_period})" 