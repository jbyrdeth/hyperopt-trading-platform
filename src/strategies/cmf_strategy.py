"""
Chaikin Money Flow (CMF) Strategy

This strategy uses the Chaikin Money Flow oscillator to identify buying and selling pressure
based on volume-weighted price movements and overbought/oversold conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class CMFStrategy(BaseStrategy):
    """
    Chaikin Money Flow (CMF) Strategy.
    
    Generates signals based on CMF oscillator levels and crossovers.
    Uses overbought/oversold thresholds and momentum analysis.
    """
    
    def __init__(
        self,
        cmf_period: int = 20,
        buy_threshold: float = 0.1,
        sell_threshold: float = -0.1,
        momentum_period: int = 5,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize CMF Strategy.
        
        Args:
            cmf_period: Period for CMF calculation (10-50)
            buy_threshold: CMF level for buy signals (0.05-0.3)
            sell_threshold: CMF level for sell signals (-0.3 to -0.05)
            momentum_period: Period for CMF momentum calculation (3-10)
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'cmf_period': cmf_period,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'momentum_period': momentum_period,
            'volume_confirmation': volume_confirmation
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"CMF_{cmf_period}_{buy_threshold}_{sell_threshold}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.cmf_period = cmf_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.momentum_period = momentum_period
        self.volume_confirmation = volume_confirmation
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.cmf = None
        self.cmf_momentum = None
        self.volume_ma = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (10 <= self.cmf_period <= 50):
            self.logger.error(f"CMF period {self.cmf_period} outside valid range [10, 50]")
            return False
        
        if not (0.05 <= self.buy_threshold <= 0.3):
            self.logger.error(f"Buy threshold {self.buy_threshold} outside valid range [0.05, 0.3]")
            return False
        
        if not (-0.3 <= self.sell_threshold <= -0.05):
            self.logger.error(f"Sell threshold {self.sell_threshold} outside valid range [-0.3, -0.05]")
            return False
        
        if not (3 <= self.momentum_period <= 10):
            self.logger.error(f"Momentum period {self.momentum_period} outside valid range [3, 10]")
            return False
        
        if self.buy_threshold <= 0 or self.sell_threshold >= 0:
            self.logger.error("Buy threshold must be positive and sell threshold must be negative")
            return False
        
        return True
    
    def calculate_cmf(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Chaikin Money Flow."""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        clv = ((close - low) - (high - close)) / (high - low)
        
        # Handle division by zero (when high == low)
        clv = clv.fillna(0)
        
        # Money Flow Volume = Money Flow Multiplier * Volume
        money_flow_volume = clv * volume
        
        # CMF = Sum of Money Flow Volume over period / Sum of Volume over period
        cmf_values = []
        for i in range(len(data)):
            start_idx = max(0, i - self.cmf_period + 1)
            end_idx = i + 1
            
            period_mfv = money_flow_volume.iloc[start_idx:end_idx].sum()
            period_volume = volume.iloc[start_idx:end_idx].sum()
            
            if period_volume > 0:
                cmf = period_mfv / period_volume
            else:
                cmf = 0
            
            cmf_values.append(cmf)
        
        return pd.Series(cmf_values, index=data.index)
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = self.cmf_period + self.momentum_period + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate CMF
        self.cmf = self.calculate_cmf(data)
        
        # Calculate CMF momentum
        self.cmf_momentum = self.cmf.diff(periods=self.momentum_period)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.cmf_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"CMF period: {self.cmf_period}, Buy/Sell thresholds: {self.buy_threshold}/{self.sell_threshold}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return self.cmf_period + self.momentum_period + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on CMF analysis."""
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
        min_required = self.cmf_period + self.momentum_period
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
        current_cmf = self.cmf.iloc[current_idx]
        current_cmf_momentum = self.cmf_momentum.iloc[current_idx]
        
        # Get previous CMF for crossover detection
        previous_cmf = self.cmf.iloc[current_idx - 1] if current_idx > 0 else current_cmf
        
        # Volume confirmation if enabled
        volume_confirmed = True
        if self.volume_confirmation:
            current_volume = current_data['volume']
            avg_volume = self.volume_ma.iloc[current_idx]
            volume_confirmed = current_volume >= avg_volume
        
        # Check for NaN values
        if pd.isna(current_cmf) or pd.isna(current_cmf_momentum):
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
            "cmf": current_cmf,
            "cmf_momentum": current_cmf_momentum,
            "previous_cmf": previous_cmf,
            "volume_confirmed": volume_confirmed
        }
        
        # Generate signals based on CMF analysis
        if volume_confirmed:
            # Bullish signals
            if current_cmf >= self.buy_threshold:
                action = "buy"
                # Strength based on how far above threshold
                strength = min((current_cmf - self.buy_threshold) / (0.5 - self.buy_threshold), 1.0)
                confidence = 0.7
                
                # Boost confidence if momentum is positive
                if current_cmf_momentum > 0:
                    confidence += 0.2
                    metadata["signal_type"] = "cmf_buy_with_momentum"
                else:
                    metadata["signal_type"] = "cmf_buy"
                    
            # Bearish signals
            elif current_cmf <= self.sell_threshold:
                action = "sell"
                # Strength based on how far below threshold
                strength = min((self.sell_threshold - current_cmf) / (abs(self.sell_threshold) + 0.5), 1.0)
                confidence = 0.7
                
                # Boost confidence if momentum is negative
                if current_cmf_momentum < 0:
                    confidence += 0.2
                    metadata["signal_type"] = "cmf_sell_with_momentum"
                else:
                    metadata["signal_type"] = "cmf_sell"
                    
            # Crossover signals (additional opportunities)
            elif (previous_cmf < 0 and current_cmf > 0 and current_cmf_momentum > 0):
                # CMF crosses above zero with positive momentum
                action = "buy"
                strength = min(abs(current_cmf_momentum) * 10, 0.8)  # Scale momentum
                confidence = 0.6
                metadata["signal_type"] = "cmf_zero_crossover_buy"
                
            elif (previous_cmf > 0 and current_cmf < 0 and current_cmf_momentum < 0):
                # CMF crosses below zero with negative momentum
                action = "sell"
                strength = min(abs(current_cmf_momentum) * 10, 0.8)  # Scale momentum
                confidence = 0.6
                metadata["signal_type"] = "cmf_zero_crossover_sell"
        
        # Adjust strength based on momentum
        if action != "hold" and abs(current_cmf_momentum) > 0.01:
            momentum_boost = min(abs(current_cmf_momentum) * 5, 1.2)
            strength = min(strength * momentum_boost, 1.0)
        
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
            'cmf_period': hp.choice('cmf_period', list(range(10, 51))),
            'buy_threshold': hp.uniform('buy_threshold', 0.05, 0.3),
            'sell_threshold': hp.uniform('sell_threshold', -0.3, -0.05),
            'momentum_period': hp.choice('momentum_period', list(range(3, 11))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.cmf is None:
            return {}
        
        return {
            'cmf': float(self.cmf.iloc[-1]) if not pd.isna(self.cmf.iloc[-1]) else 0.0,
            'cmf_momentum': float(self.cmf_momentum.iloc[-1]) if not pd.isna(self.cmf_momentum.iloc[-1]) else 0.0
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
        return f"CMF({self.cmf_period}, {self.buy_threshold}, {self.sell_threshold})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"CMFStrategy(cmf_period={self.cmf_period}, buy_threshold={self.buy_threshold}, sell_threshold={self.sell_threshold})" 