"""
Accumulation/Distribution (A/D) Strategy

This strategy uses the Accumulation/Distribution Line to measure money flow
and identify buying/selling pressure based on price and volume relationships.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class ADStrategy(BaseStrategy):
    """
    Accumulation/Distribution (A/D) Strategy.
    
    Generates signals based on A/D line trends and divergences with price.
    Uses A/D momentum and moving averages to identify money flow patterns.
    """
    
    def __init__(
        self,
        ad_ma_period: int = 20,
        trend_threshold: float = 0.02,
        volume_filter: float = 1.0,
        divergence_lookback: int = 10,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize A/D Strategy.
        
        Args:
            ad_ma_period: Period for A/D moving average (5-50)
            trend_threshold: Minimum A/D change to generate signal (0.01-0.05)
            volume_filter: Minimum volume multiplier for signal confirmation (0.5-2.0)
            divergence_lookback: Lookback period for divergence detection (5-20)
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'ad_ma_period': ad_ma_period,
            'trend_threshold': trend_threshold,
            'volume_filter': volume_filter,
            'divergence_lookback': divergence_lookback
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"AD_{ad_ma_period}_{trend_threshold}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.ad_ma_period = ad_ma_period
        self.trend_threshold = trend_threshold
        self.volume_filter = volume_filter
        self.divergence_lookback = divergence_lookback
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.ad_line = None
        self.ad_ma = None
        self.ad_momentum = None
        self.volume_ma = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.ad_ma_period <= 50):
            self.logger.error(f"A/D MA period {self.ad_ma_period} outside valid range [5, 50]")
            return False
        
        if not (0.01 <= self.trend_threshold <= 0.05):
            self.logger.error(f"Trend threshold {self.trend_threshold} outside valid range [0.01, 0.05]")
            return False
        
        if not (0.5 <= self.volume_filter <= 2.0):
            self.logger.error(f"Volume filter {self.volume_filter} outside valid range [0.5, 2.0]")
            return False
        
        if not (5 <= self.divergence_lookback <= 20):
            self.logger.error(f"Divergence lookback {self.divergence_lookback} outside valid range [5, 20]")
            return False
        
        return True
    
    def calculate_ad_line(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        # Money Flow Volume = Money Flow Multiplier * Volume
        # A/D Line = Previous A/D Line + Current Period's Money Flow Volume
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Calculate Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        
        # Handle division by zero (when high == low)
        clv = clv.fillna(0)
        
        # Calculate Money Flow Volume
        money_flow_volume = clv * volume
        
        # Calculate cumulative A/D Line
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
    
    def detect_divergence(self, price_data: pd.Series, ad_data: pd.Series, lookback: int) -> str:
        """Detect bullish or bearish divergence between price and A/D line."""
        if len(price_data) < lookback or len(ad_data) < lookback:
            return "none"
        
        # Get recent data
        recent_price = price_data.iloc[-lookback:]
        recent_ad = ad_data.iloc[-lookback:]
        
        # Calculate trends (simple linear regression slope)
        price_trend = np.polyfit(range(len(recent_price)), recent_price, 1)[0]
        ad_trend = np.polyfit(range(len(recent_ad)), recent_ad, 1)[0]
        
        # Normalize trends for comparison
        price_trend_norm = price_trend / recent_price.mean() if recent_price.mean() != 0 else 0
        ad_trend_norm = ad_trend / abs(recent_ad.mean()) if recent_ad.mean() != 0 else 0
        
        # Detect divergence
        threshold = 0.001  # Minimum trend difference for divergence
        
        if price_trend_norm > threshold and ad_trend_norm < -threshold:
            return "bearish"  # Price up, A/D down
        elif price_trend_norm < -threshold and ad_trend_norm > threshold:
            return "bullish"  # Price down, A/D up
        else:
            return "none"
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.ad_ma_period, self.divergence_lookback) + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate A/D Line
        self.ad_line = self.calculate_ad_line(data)
        
        # Calculate A/D moving average
        self.ad_ma = ta.trend.sma_indicator(self.ad_line, window=self.ad_ma_period)
        
        # Calculate A/D momentum (rate of change)
        self.ad_momentum = self.ad_line.pct_change(periods=5)
        
        # Calculate volume moving average for filtering
        self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.ad_ma_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"A/D MA period: {self.ad_ma_period}, Trend threshold: {self.trend_threshold}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.ad_ma_period, self.divergence_lookback) + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on A/D line analysis."""
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
        min_required = max(self.ad_ma_period, self.divergence_lookback)
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
        current_ad = self.ad_line.iloc[current_idx]
        current_ad_ma = self.ad_ma.iloc[current_idx]
        current_ad_momentum = self.ad_momentum.iloc[current_idx]
        current_volume = current_data['volume']
        avg_volume = self.volume_ma.iloc[current_idx]
        
        # Check for NaN values
        if pd.isna(current_ad) or pd.isna(current_ad_ma) or pd.isna(current_ad_momentum):
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
        ad_window = self.ad_line.iloc[current_idx-self.divergence_lookback+1:current_idx+1]
        divergence = self.detect_divergence(price_window, ad_window, self.divergence_lookback)
        
        # Volume filtering
        volume_confirmed = current_volume >= (avg_volume * self.volume_filter)
        
        # A/D trend analysis
        ad_above_ma = current_ad > current_ad_ma
        ad_momentum_strong = abs(current_ad_momentum) >= self.trend_threshold
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "ad_line": current_ad,
            "ad_ma": current_ad_ma,
            "ad_momentum": current_ad_momentum,
            "divergence": divergence,
            "volume_confirmed": volume_confirmed,
            "ad_above_ma": ad_above_ma
        }
        
        # Generate signals based on A/D analysis
        if volume_confirmed:
            # Bullish signals
            if (ad_above_ma and current_ad_momentum > self.trend_threshold) or divergence == "bullish":
                action = "buy"
                strength = min(abs(current_ad_momentum) / self.trend_threshold, 1.0)
                confidence = 0.7
                
                if divergence == "bullish":
                    confidence += 0.2
                    metadata["signal_type"] = "bullish_divergence"
                else:
                    metadata["signal_type"] = "ad_momentum_buy"
                    
            # Bearish signals
            elif (not ad_above_ma and current_ad_momentum < -self.trend_threshold) or divergence == "bearish":
                action = "sell"
                strength = min(abs(current_ad_momentum) / self.trend_threshold, 1.0)
                confidence = 0.7
                
                if divergence == "bearish":
                    confidence += 0.2
                    metadata["signal_type"] = "bearish_divergence"
                else:
                    metadata["signal_type"] = "ad_momentum_sell"
        
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
            'ad_ma_period': hp.choice('ad_ma_period', list(range(5, 51))),
            'trend_threshold': hp.uniform('trend_threshold', 0.01, 0.05),
            'volume_filter': hp.uniform('volume_filter', 0.5, 2.0),
            'divergence_lookback': hp.choice('divergence_lookback', list(range(5, 21))),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.ad_line is None:
            return {}
        
        return {
            'ad_line': float(self.ad_line.iloc[-1]) if not pd.isna(self.ad_line.iloc[-1]) else 0.0,
            'ad_ma': float(self.ad_ma.iloc[-1]) if not pd.isna(self.ad_ma.iloc[-1]) else 0.0,
            'ad_momentum': float(self.ad_momentum.iloc[-1]) if not pd.isna(self.ad_momentum.iloc[-1]) else 0.0
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
        return f"AD({self.ad_ma_period}, {self.trend_threshold})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ADStrategy(ad_ma_period={self.ad_ma_period}, trend_threshold={self.trend_threshold}, volume_filter={self.volume_filter})" 