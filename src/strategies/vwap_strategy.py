"""
VWAP (Volume Weighted Average Price) Strategy

This strategy uses VWAP to identify when price deviates significantly from the volume-weighted
average price, generating signals based on mean reversion or breakout logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class VWAPStrategy(BaseStrategy):
    """
    VWAP (Volume Weighted Average Price) Strategy.
    
    Generates signals based on price deviation from VWAP and volume confirmation.
    Supports multiple VWAP periods and trading modes (mean reversion vs breakout).
    """
    
    def __init__(
        self,
        vwap_period: int = 20,
        deviation_threshold: float = 0.02,
        volume_multiplier: float = 1.5,
        trading_mode: str = "mean_reversion",  # "mean_reversion" or "breakout"
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize VWAP Strategy.
        
        Args:
            vwap_period: Period for VWAP calculation (5-50)
            deviation_threshold: Minimum deviation from VWAP to generate signal (0.005-0.05)
            volume_multiplier: Volume confirmation multiplier (1.0-3.0)
            trading_mode: "mean_reversion" or "breakout"
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'vwap_period': vwap_period,
            'deviation_threshold': deviation_threshold,
            'volume_multiplier': volume_multiplier,
            'trading_mode': trading_mode
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"VWAP_{vwap_period}_{trading_mode}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.vwap_period = vwap_period
        self.deviation_threshold = deviation_threshold
        self.volume_multiplier = volume_multiplier
        self.trading_mode = trading_mode
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.vwap = None
        self.volume_ma = None
        self.price_deviation = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (5 <= self.vwap_period <= 50):
            self.logger.error(f"VWAP period {self.vwap_period} outside valid range [5, 50]")
            return False
        
        if not (0.005 <= self.deviation_threshold <= 0.05):
            self.logger.error(f"Deviation threshold {self.deviation_threshold} outside valid range [0.005, 0.05]")
            return False
        
        if not (1.0 <= self.volume_multiplier <= 3.0):
            self.logger.error(f"Volume multiplier {self.volume_multiplier} outside valid range [1.0, 3.0]")
            return False
        
        if self.trading_mode not in ["mean_reversion", "breakout"]:
            self.logger.error(f"Invalid trading mode: {self.trading_mode}")
            return False
        
        return True
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (Volume Weighted Average Price)."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Rolling VWAP calculation
        vwap_values = []
        for i in range(len(data)):
            start_idx = max(0, i - self.vwap_period + 1)
            end_idx = i + 1
            
            window_tp = typical_price.iloc[start_idx:end_idx]
            window_volume = data['volume'].iloc[start_idx:end_idx]
            
            if window_volume.sum() > 0:
                vwap = (window_tp * window_volume).sum() / window_volume.sum()
            else:
                vwap = window_tp.mean()
            
            vwap_values.append(vwap)
        
        return pd.Series(vwap_values, index=data.index)
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = self.vwap_period + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate VWAP
        self.vwap = self.calculate_vwap(data)
        
        # Calculate volume moving average for confirmation
        self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.vwap_period)
        
        # Calculate price deviation from VWAP
        self.price_deviation = (data['close'] - self.vwap) / self.vwap
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"VWAP period: {self.vwap_period}, Mode: {self.trading_mode}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return self.vwap_period + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on VWAP deviation."""
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
        if current_idx < self.vwap_period:
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
        current_vwap = self.vwap.iloc[current_idx]
        current_volume = current_data['volume']
        avg_volume = self.volume_ma.iloc[current_idx]
        current_deviation = self.price_deviation.iloc[current_idx]
        
        # Check for NaN values
        if pd.isna(current_vwap) or pd.isna(current_deviation) or pd.isna(avg_volume):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Volume confirmation
        volume_confirmed = current_volume >= (avg_volume * self.volume_multiplier)
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "vwap": current_vwap,
            "price_deviation": current_deviation,
            "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 0,
            "volume_confirmed": volume_confirmed
        }
        
        # Calculate signal strength based on deviation magnitude
        deviation_strength = min(abs(current_deviation) / self.deviation_threshold, 2.0)
        
        if abs(current_deviation) >= self.deviation_threshold and volume_confirmed:
            if self.trading_mode == "mean_reversion":
                # Mean reversion: buy when price below VWAP, sell when above
                if current_deviation < -self.deviation_threshold:
                    action = "buy"
                    strength = deviation_strength
                    confidence = min(strength * 0.8, 1.0)
                    metadata["signal_type"] = "mean_reversion_buy"
                elif current_deviation > self.deviation_threshold:
                    action = "sell"
                    strength = deviation_strength
                    confidence = min(strength * 0.8, 1.0)
                    metadata["signal_type"] = "mean_reversion_sell"
                    
            elif self.trading_mode == "breakout":
                # Breakout: buy when price breaks above VWAP, sell when breaks below
                if current_deviation > self.deviation_threshold:
                    action = "buy"
                    strength = deviation_strength
                    confidence = min(strength * 0.7, 1.0)
                    metadata["signal_type"] = "breakout_buy"
                elif current_deviation < -self.deviation_threshold:
                    action = "sell"
                    strength = deviation_strength
                    confidence = min(strength * 0.7, 1.0)
                    metadata["signal_type"] = "breakout_sell"
        
        # Adjust strength based on volume confirmation
        if action != "hold":
            volume_boost = min(current_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
            strength = min(strength * (0.5 + 0.5 * volume_boost), 1.0)
            confidence = min(confidence * volume_boost, 1.0)
        
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
            'vwap_period': hp.choice('vwap_period', list(range(5, 51))),
            'deviation_threshold': hp.uniform('deviation_threshold', 0.005, 0.05),
            'volume_multiplier': hp.uniform('volume_multiplier', 1.0, 3.0),
            'trading_mode': hp.choice('trading_mode', ["mean_reversion", "breakout"]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.vwap is None:
            return {}
        
        return {
            'vwap': float(self.vwap.iloc[-1]) if not pd.isna(self.vwap.iloc[-1]) else 0.0,
            'price_deviation': float(self.price_deviation.iloc[-1]) if not pd.isna(self.price_deviation.iloc[-1]) else 0.0,
            'volume_ma': float(self.volume_ma.iloc[-1]) if not pd.isna(self.volume_ma.iloc[-1]) else 0.0
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
        return f"VWAP({self.vwap_period}, {self.trading_mode})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"VWAPStrategy(vwap_period={self.vwap_period}, deviation_threshold={self.deviation_threshold}, trading_mode='{self.trading_mode}')" 