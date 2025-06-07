"""
ATR (Average True Range) Strategy

This strategy uses Average True Range to identify volatility breakouts and implement
dynamic position sizing and stop losses based on market volatility.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class ATRStrategy(BaseStrategy):
    """
    ATR (Average True Range) Strategy.
    
    Generates signals based on ATR volatility breakouts and implements
    dynamic position sizing and stop losses based on current volatility.
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        breakout_multiplier: float = 2.0,
        stop_multiplier: float = 1.5,
        trend_filter: bool = True,
        trend_period: int = 50,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize ATR Strategy.
        
        Args:
            atr_period: Period for ATR calculation (5-30)
            breakout_multiplier: ATR multiplier for breakout signals (1.0-4.0)
            stop_multiplier: ATR multiplier for stop loss (0.5-3.0)
            trend_filter: Whether to use trend filter for signals
            trend_period: Period for trend filter (20-100)
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'atr_period': atr_period,
            'breakout_multiplier': breakout_multiplier,
            'stop_multiplier': stop_multiplier,
            'trend_filter': trend_filter,
            'trend_period': trend_period
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"ATR_{atr_period}_{breakout_multiplier}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
        self.stop_multiplier = stop_multiplier
        self.trend_filter = trend_filter
        self.trend_period = trend_period
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.atr = None
        self.atr_ma = None
        self.trend_ma = None
        self.volatility_regime = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.atr_period <= 30):
            self.logger.error(f"ATR period {self.atr_period} outside valid range [5, 30]")
            return False
        
        if not (1.0 <= self.breakout_multiplier <= 4.0):
            self.logger.error(f"Breakout multiplier {self.breakout_multiplier} outside valid range [1.0, 4.0]")
            return False
        
        if not (0.5 <= self.stop_multiplier <= 3.0):
            self.logger.error(f"Stop multiplier {self.stop_multiplier} outside valid range [0.5, 3.0]")
            return False
        
        if not (20 <= self.trend_period <= 100):
            self.logger.error(f"Trend period {self.trend_period} outside valid range [20, 100]")
            return False
        
        return True
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as moving average of True Range
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def detect_volatility_regime(self, atr: pd.Series) -> pd.Series:
        """Detect volatility regime (low/normal/high)."""
        # Calculate ATR percentiles over longer period
        atr_rolling = atr.rolling(window=self.trend_period * 2, min_periods=self.trend_period)
        
        low_threshold = atr_rolling.quantile(0.25)
        high_threshold = atr_rolling.quantile(0.75)
        
        regime = pd.Series(index=atr.index, dtype=str)
        regime[atr <= low_threshold] = "low"
        regime[atr >= high_threshold] = "high"
        regime[(atr > low_threshold) & (atr < high_threshold)] = "normal"
        
        return regime
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.atr_period, self.trend_period) + 20
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate ATR
        self.atr = self.calculate_atr(data)
        
        # Calculate ATR moving average for smoothing
        self.atr_ma = self.atr.rolling(window=5).mean()
        
        # Calculate trend filter if enabled
        if self.trend_filter:
            self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_period)
        
        # Detect volatility regime
        self.volatility_regime = self.detect_volatility_regime(self.atr)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"ATR period: {self.atr_period}, Breakout multiplier: {self.breakout_multiplier}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.atr_period, self.trend_period) + 20
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on ATR volatility breakouts."""
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
        min_required = max(self.atr_period, self.trend_period)
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
        current_high = current_data['high']
        current_low = current_data['low']
        current_atr = self.atr.iloc[current_idx]
        current_atr_ma = self.atr_ma.iloc[current_idx]
        current_regime = self.volatility_regime.iloc[current_idx]
        
        # Get previous values for breakout detection
        prev_high = self.data['high'].iloc[current_idx - 1] if current_idx > 0 else current_high
        prev_low = self.data['low'].iloc[current_idx - 1] if current_idx > 0 else current_low
        prev_close = self.data['close'].iloc[current_idx - 1] if current_idx > 0 else current_price
        
        # Trend filter
        trend_bullish = True
        trend_bearish = True
        if self.trend_filter:
            current_trend_ma = self.trend_ma.iloc[current_idx]
            if not pd.isna(current_trend_ma):
                trend_bullish = current_price > current_trend_ma
                trend_bearish = current_price < current_trend_ma
        
        # Check for NaN values
        if pd.isna(current_atr) or pd.isna(current_atr_ma):
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
            "atr": current_atr,
            "atr_ma": current_atr_ma,
            "volatility_regime": current_regime,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish
        }
        
        # Calculate breakout thresholds
        breakout_threshold = current_atr * self.breakout_multiplier
        
        # Detect volatility breakouts
        upside_breakout = current_high > (prev_high + breakout_threshold)
        downside_breakout = current_low < (prev_low - breakout_threshold)
        
        # Price movement relative to ATR
        price_move = abs(current_price - prev_close)
        relative_move = price_move / current_atr if current_atr > 0 else 0
        
        # Generate signals based on volatility breakouts
        if upside_breakout and trend_bullish:
            action = "buy"
            strength = min(relative_move / self.breakout_multiplier, 1.0)
            confidence = 0.8
            metadata["signal_type"] = "volatility_breakout_buy"
            
            # Boost confidence in high volatility regime
            if current_regime == "high":
                confidence += 0.1
                
        elif downside_breakout and trend_bearish:
            action = "sell"
            strength = min(relative_move / self.breakout_multiplier, 1.0)
            confidence = 0.8
            metadata["signal_type"] = "volatility_breakout_sell"
            
            # Boost confidence in high volatility regime
            if current_regime == "high":
                confidence += 0.1
        
        # Alternative signals based on ATR expansion
        elif current_atr > current_atr_ma * 1.2:  # ATR expanding
            if current_price > prev_close and trend_bullish:
                action = "buy"
                strength = min((current_atr / current_atr_ma - 1) * 2, 0.7)
                confidence = 0.6
                metadata["signal_type"] = "atr_expansion_buy"
                
            elif current_price < prev_close and trend_bearish:
                action = "sell"
                strength = min((current_atr / current_atr_ma - 1) * 2, 0.7)
                confidence = 0.6
                metadata["signal_type"] = "atr_expansion_sell"
        
        # Adjust strength based on volatility regime
        if action != "hold":
            if current_regime == "high":
                strength = min(strength * 1.2, 1.0)
            elif current_regime == "low":
                strength = strength * 0.8
        
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
            'atr_period': hp.choice('atr_period', list(range(5, 31))),
            'breakout_multiplier': hp.uniform('breakout_multiplier', 1.0, 4.0),
            'stop_multiplier': hp.uniform('stop_multiplier', 0.5, 3.0),
            'trend_filter': hp.choice('trend_filter', [True, False]),
            'trend_period': hp.choice('trend_period', list(range(20, 101, 10))),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.atr is None:
            return {}
        
        return {
            'atr': float(self.atr.iloc[-1]) if not pd.isna(self.atr.iloc[-1]) else 0.0,
            'atr_ma': float(self.atr_ma.iloc[-1]) if not pd.isna(self.atr_ma.iloc[-1]) else 0.0,
            'volatility_regime': str(self.volatility_regime.iloc[-1]) if not pd.isna(self.volatility_regime.iloc[-1]) else "unknown"
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and ATR-based volatility.
        
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
        
        # Get current ATR for volatility-based sizing
        current_atr = signal.metadata.get('atr', current_volatility * current_price)
        
        # Base position size as percentage of available capital
        base_size = available_capital * self.position_size_pct / current_price
        
        # Adjust based on signal strength and confidence
        signal_adjustment = (signal.strength + signal.confidence) / 2
        adjusted_size = base_size * signal_adjustment
        
        # ATR-based volatility adjustment (inverse relationship)
        atr_pct = current_atr / current_price if current_price > 0 else 0.02
        volatility_adjustment = max(0.3, 1.0 - (atr_pct * 10))  # Reduce size in high ATR
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"ATR({self.atr_period}, {self.breakout_multiplier})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ATRStrategy(atr_period={self.atr_period}, breakout_multiplier={self.breakout_multiplier}, stop_multiplier={self.stop_multiplier})" 