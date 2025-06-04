"""
Rate of Change (ROC) Strategy

This strategy uses Rate of Change indicator for momentum analysis with multi-timeframe
ROC periods, crossover signals, persistence filters, and divergence detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import get_logger


class ROCStrategy(BaseStrategy):
    """
    Rate of Change (ROC) Strategy.
    
    Uses ROC indicator for momentum analysis with crossover signals,
    persistence filters, and divergence detection for early momentum shifts.
    """
    
    def __init__(
        self,
        roc_period: int = 12,
        signal_threshold: float = 2.0,
        persistence_bars: int = 3,
        ma_smoothing: int = 5,
        divergence_lookback: int = 20,
        trend_filter: bool = True,
        trend_period: int = 50,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize ROC Strategy.
        
        Args:
            roc_period: Period for ROC calculation (5-50)
            signal_threshold: ROC level for signals in % (0.5-5.0)
            persistence_bars: Confirmation requirement in bars (1-10)
            ma_smoothing: ROC smoothing period (3-21)
            divergence_lookback: Lookback for divergence detection (10-50)
            trend_filter: Whether to use trend filter
            trend_period: Period for trend filter (20-100)
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'roc_period': roc_period,
            'signal_threshold': signal_threshold,
            'persistence_bars': persistence_bars,
            'ma_smoothing': ma_smoothing,
            'divergence_lookback': divergence_lookback,
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
            name=f"ROC_{roc_period}_{signal_threshold}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.roc_period = roc_period
        self.signal_threshold = signal_threshold
        self.persistence_bars = persistence_bars
        self.ma_smoothing = ma_smoothing
        self.divergence_lookback = divergence_lookback
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
        self.roc = None
        self.roc_ma = None
        self.trend_ma = None
        self.volume_ma = None
        self.price_peaks = None
        self.roc_peaks = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.roc_period <= 50):
            self.logger.error(f"ROC period {self.roc_period} outside valid range [5, 50]")
            return False
        
        if not (0.5 <= self.signal_threshold <= 5.0):
            self.logger.error(f"Signal threshold {self.signal_threshold} outside valid range [0.5, 5.0]")
            return False
        
        if not (1 <= self.persistence_bars <= 10):
            self.logger.error(f"Persistence bars {self.persistence_bars} outside valid range [1, 10]")
            return False
        
        if not (3 <= self.ma_smoothing <= 21):
            self.logger.error(f"MA smoothing {self.ma_smoothing} outside valid range [3, 21]")
            return False
        
        if not (10 <= self.divergence_lookback <= 50):
            self.logger.error(f"Divergence lookback {self.divergence_lookback} outside valid range [10, 50]")
            return False
        
        if not (20 <= self.trend_period <= 100):
            self.logger.error(f"Trend period {self.trend_period} outside valid range [20, 100]")
            return False
        
        return True
    
    def calculate_roc(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Rate of Change."""
        close = data['close']
        
        # ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
        roc = ((close - close.shift(self.roc_period)) / close.shift(self.roc_period)) * 100
        
        return roc
    
    def detect_divergence(self, price: pd.Series, roc: pd.Series, current_idx: int) -> Dict[str, bool]:
        """Detect bullish and bearish divergences."""
        if current_idx < self.divergence_lookback:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        # Get lookback window
        start_idx = max(0, current_idx - self.divergence_lookback)
        price_window = price.iloc[start_idx:current_idx + 1]
        roc_window = roc.iloc[start_idx:current_idx + 1]
        
        if len(price_window) < 5 or len(roc_window) < 5:
            return {"bullish_divergence": False, "bearish_divergence": False}
        
        # Find peaks and troughs
        price_peaks = []
        price_troughs = []
        roc_peaks = []
        roc_troughs = []
        
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
            
            # ROC peaks and troughs
            if (roc_window.iloc[i] > roc_window.iloc[i-1] and 
                roc_window.iloc[i] > roc_window.iloc[i+1] and
                roc_window.iloc[i] > roc_window.iloc[i-2] and 
                roc_window.iloc[i] > roc_window.iloc[i+2]):
                roc_peaks.append((i, roc_window.iloc[i]))
            
            if (roc_window.iloc[i] < roc_window.iloc[i-1] and 
                roc_window.iloc[i] < roc_window.iloc[i+1] and
                roc_window.iloc[i] < roc_window.iloc[i-2] and 
                roc_window.iloc[i] < roc_window.iloc[i+2]):
                roc_troughs.append((i, roc_window.iloc[i]))
        
        bullish_divergence = False
        bearish_divergence = False
        
        # Bullish divergence: Price makes lower lows, ROC makes higher lows
        if len(price_troughs) >= 2 and len(roc_troughs) >= 2:
            recent_price_trough = price_troughs[-1]
            prev_price_trough = price_troughs[-2]
            
            # Find corresponding ROC troughs
            for roc_trough in roc_troughs:
                if abs(roc_trough[0] - recent_price_trough[0]) <= 2:
                    recent_roc_trough = roc_trough
                    break
            else:
                recent_roc_trough = None
            
            for roc_trough in roc_troughs:
                if abs(roc_trough[0] - prev_price_trough[0]) <= 2:
                    prev_roc_trough = roc_trough
                    break
            else:
                prev_roc_trough = None
            
            if (recent_roc_trough and prev_roc_trough and
                recent_price_trough[1] < prev_price_trough[1] and
                recent_roc_trough[1] > prev_roc_trough[1]):
                bullish_divergence = True
        
        # Bearish divergence: Price makes higher highs, ROC makes lower highs
        if len(price_peaks) >= 2 and len(roc_peaks) >= 2:
            recent_price_peak = price_peaks[-1]
            prev_price_peak = price_peaks[-2]
            
            # Find corresponding ROC peaks
            for roc_peak in roc_peaks:
                if abs(roc_peak[0] - recent_price_peak[0]) <= 2:
                    recent_roc_peak = roc_peak
                    break
            else:
                recent_roc_peak = None
            
            for roc_peak in roc_peaks:
                if abs(roc_peak[0] - prev_price_peak[0]) <= 2:
                    prev_roc_peak = roc_peak
                    break
            else:
                prev_roc_peak = None
            
            if (recent_roc_peak and prev_roc_peak and
                recent_price_peak[1] > prev_price_peak[1] and
                recent_roc_peak[1] < prev_roc_peak[1]):
                bearish_divergence = True
        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence
        }
    
    def check_persistence(self, roc: pd.Series, current_idx: int, signal_type: str) -> bool:
        """Check if signal persists for required number of bars."""
        if current_idx < self.persistence_bars:
            return False
        
        # Check last persistence_bars periods
        for i in range(self.persistence_bars):
            idx = current_idx - i
            if idx < 0:
                return False
            
            current_roc = roc.iloc[idx]
            
            if signal_type == "buy" and current_roc <= self.signal_threshold:
                return False
            elif signal_type == "sell" and current_roc >= -self.signal_threshold:
                return False
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.roc_period, self.trend_period, self.divergence_lookback) + self.persistence_bars + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate ROC
        self.roc = self.calculate_roc(data)
        
        # Calculate smoothed ROC
        self.roc_ma = self.roc.rolling(window=self.ma_smoothing).mean()
        
        # Calculate trend filter if enabled
        if self.trend_filter:
            self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_period)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.roc_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"ROC period: {self.roc_period}, Signal threshold: {self.signal_threshold}%")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.roc_period, self.trend_period, self.divergence_lookback) + self.persistence_bars + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on ROC momentum analysis."""
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
        min_required = max(self.roc_period, self.trend_period, self.divergence_lookback) + self.persistence_bars
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
        current_roc = self.roc.iloc[current_idx]
        current_roc_ma = self.roc_ma.iloc[current_idx]
        
        # Get previous values for crossover detection
        prev_roc = self.roc.iloc[current_idx - 1] if current_idx > 0 else current_roc
        
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
        if pd.isna(current_roc) or pd.isna(current_roc_ma):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Detect divergences
        divergence_info = self.detect_divergence(self.data['close'], self.roc, current_idx)
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "roc": current_roc,
            "roc_ma": current_roc_ma,
            "signal_threshold": self.signal_threshold,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish,
            "volume_confirmed": volume_confirmed,
            "bullish_divergence": divergence_info["bullish_divergence"],
            "bearish_divergence": divergence_info["bearish_divergence"]
        }
        
        # Generate signals based on ROC crossovers and persistence
        if volume_confirmed:
            # Bullish signals
            if (current_roc > self.signal_threshold and 
                prev_roc <= self.signal_threshold and 
                trend_bullish):
                
                # Check persistence
                if self.check_persistence(self.roc, current_idx, "buy"):
                    action = "buy"
                    strength = min(current_roc / (self.signal_threshold * 2), 1.0)
                    confidence = 0.8
                    metadata["signal_type"] = "roc_crossover_buy"
                    
                    # Boost for divergence
                    if divergence_info["bullish_divergence"]:
                        strength = min(strength * 1.3, 1.0)
                        confidence = min(confidence + 0.1, 1.0)
                        metadata["signal_type"] = "roc_divergence_buy"
            
            # Bearish signals
            elif (current_roc < -self.signal_threshold and 
                  prev_roc >= -self.signal_threshold and 
                  trend_bearish):
                
                # Check persistence
                if self.check_persistence(self.roc, current_idx, "sell"):
                    action = "sell"
                    strength = min(abs(current_roc) / (self.signal_threshold * 2), 1.0)
                    confidence = 0.8
                    metadata["signal_type"] = "roc_crossover_sell"
                    
                    # Boost for divergence
                    if divergence_info["bearish_divergence"]:
                        strength = min(strength * 1.3, 1.0)
                        confidence = min(confidence + 0.1, 1.0)
                        metadata["signal_type"] = "roc_divergence_sell"
            
            # Divergence-only signals (weaker)
            elif divergence_info["bullish_divergence"] and trend_bullish:
                action = "buy"
                strength = 0.6
                confidence = 0.6
                metadata["signal_type"] = "roc_divergence_only_buy"
                
            elif divergence_info["bearish_divergence"] and trend_bearish:
                action = "sell"
                strength = 0.6
                confidence = 0.6
                metadata["signal_type"] = "roc_divergence_only_sell"
        
        # Adjust strength based on ROC momentum
        if action != "hold":
            # Stronger signals for extreme ROC values
            roc_strength = abs(current_roc) / 10.0  # Normalize to 0-1 range
            strength = min(strength * (1 + roc_strength), 1.0)
        
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
            'roc_period': hp.choice('roc_period', list(range(5, 51))),
            'signal_threshold': hp.uniform('signal_threshold', 0.5, 5.0),
            'persistence_bars': hp.choice('persistence_bars', list(range(1, 11))),
            'ma_smoothing': hp.choice('ma_smoothing', list(range(3, 22))),
            'divergence_lookback': hp.choice('divergence_lookback', list(range(10, 51))),
            'trend_filter': hp.choice('trend_filter', [True, False]),
            'trend_period': hp.choice('trend_period', list(range(20, 101, 10))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.roc is None:
            return {}
        
        return {
            'roc': float(self.roc.iloc[-1]) if not pd.isna(self.roc.iloc[-1]) else 0.0,
            'roc_ma': float(self.roc_ma.iloc[-1]) if not pd.isna(self.roc_ma.iloc[-1]) else 0.0,
            'signal_threshold': float(self.signal_threshold)
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and ROC momentum.
        
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
        
        # ROC-specific adjustments
        current_roc = signal.metadata.get('roc', 0)
        roc_strength = min(abs(current_roc) / 10.0, 1.0)  # Normalize ROC strength
        
        # Increase size for strong ROC momentum
        if abs(current_roc) > self.signal_threshold * 2:
            adjusted_size = adjusted_size * (1 + roc_strength * 0.3)
        
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
        return f"ROC({self.roc_period}, {self.signal_threshold}%)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ROCStrategy(roc_period={self.roc_period}, signal_threshold={self.signal_threshold}, persistence_bars={self.persistence_bars})" 