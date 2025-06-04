"""
Historical Volatility Strategy

This strategy analyzes historical volatility patterns to detect volatility regimes
and adapts trading approach: mean reversion in high volatility, trend following in low volatility.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from ..utils.logger import get_logger


class HistoricalVolatilityStrategy(BaseStrategy):
    """
    Historical Volatility Strategy.
    
    Analyzes rolling volatility to detect volatility regimes and adapts trading approach:
    - High volatility: Mean reversion trading
    - Low volatility: Trend following trading
    """
    
    def __init__(
        self,
        volatility_period: int = 20,
        regime_period: int = 60,
        high_vol_threshold: float = 0.75,
        low_vol_threshold: float = 0.25,
        trend_period: int = 50,
        momentum_period: int = 10,
        volume_confirmation: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Historical Volatility Strategy.
        
        Args:
            volatility_period: Period for volatility calculation (10-50)
            regime_period: Period for volatility regime detection (30-120)
            high_vol_threshold: Percentile threshold for high volatility (0.6-0.9)
            low_vol_threshold: Percentile threshold for low volatility (0.1-0.4)
            trend_period: Period for trend analysis (20-100)
            momentum_period: Period for momentum calculation (5-20)
            volume_confirmation: Whether to require volume confirmation
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'volatility_period': volatility_period,
            'regime_period': regime_period,
            'high_vol_threshold': high_vol_threshold,
            'low_vol_threshold': low_vol_threshold,
            'trend_period': trend_period,
            'momentum_period': momentum_period,
            'volume_confirmation': volume_confirmation
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"HistoricalVolatility_{volatility_period}_{regime_period}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.volatility_period = volatility_period
        self.regime_period = regime_period
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.trend_period = trend_period
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
        self.volatility = None
        self.volatility_regime = None
        self.trend_ma = None
        self.momentum = None
        self.volume_ma = None
        self.volatility_percentile = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (10 <= self.volatility_period <= 50):
            self.logger.error(f"Volatility period {self.volatility_period} outside valid range [10, 50]")
            return False
        
        if not (30 <= self.regime_period <= 120):
            self.logger.error(f"Regime period {self.regime_period} outside valid range [30, 120]")
            return False
        
        if not (0.6 <= self.high_vol_threshold <= 0.9):
            self.logger.error(f"High vol threshold {self.high_vol_threshold} outside valid range [0.6, 0.9]")
            return False
        
        if not (0.1 <= self.low_vol_threshold <= 0.4):
            self.logger.error(f"Low vol threshold {self.low_vol_threshold} outside valid range [0.1, 0.4]")
            return False
        
        if self.low_vol_threshold >= self.high_vol_threshold:
            self.logger.error("Low vol threshold must be less than high vol threshold")
            return False
        
        if not (20 <= self.trend_period <= 100):
            self.logger.error(f"Trend period {self.trend_period} outside valid range [20, 100]")
            return False
        
        if not (5 <= self.momentum_period <= 20):
            self.logger.error(f"Momentum period {self.momentum_period} outside valid range [5, 20]")
            return False
        
        return True
    
    def calculate_historical_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate rolling historical volatility."""
        close = data['close']
        
        # Calculate log returns
        log_returns = np.log(close / close.shift(1))
        
        # Calculate rolling standard deviation (volatility)
        volatility = log_returns.rolling(window=self.volatility_period).std()
        
        # Annualize volatility (assuming daily data)
        volatility = volatility * np.sqrt(252)
        
        return volatility
    
    def detect_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """Detect volatility regime based on percentiles."""
        # Calculate rolling percentiles
        vol_rolling = volatility.rolling(window=self.regime_period, min_periods=self.regime_period // 2)
        
        high_threshold = vol_rolling.quantile(self.high_vol_threshold)
        low_threshold = vol_rolling.quantile(self.low_vol_threshold)
        
        # Classify regime
        regime = pd.Series(index=volatility.index, dtype=str)
        regime[volatility >= high_threshold] = "high"
        regime[volatility <= low_threshold] = "low"
        regime[(volatility > low_threshold) & (volatility < high_threshold)] = "normal"
        
        return regime
    
    def calculate_momentum(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price momentum."""
        close = data['close']
        return close.pct_change(periods=self.momentum_period)
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.volatility_period, self.regime_period, self.trend_period) + 20
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate historical volatility
        self.volatility = self.calculate_historical_volatility(data)
        
        # Detect volatility regime
        self.volatility_regime = self.detect_volatility_regime(self.volatility)
        
        # Calculate volatility percentile for current analysis
        self.volatility_percentile = self.volatility.rolling(
            window=self.regime_period, min_periods=self.regime_period // 2
        ).rank(pct=True)
        
        # Calculate trend indicator
        self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_period)
        
        # Calculate momentum
        self.momentum = self.calculate_momentum(data)
        
        # Calculate volume moving average for confirmation
        if self.volume_confirmation:
            self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.volatility_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"Volatility period: {self.volatility_period}, Regime period: {self.regime_period}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.volatility_period, self.regime_period, self.trend_period) + 20
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on volatility regime analysis."""
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
        min_required = max(self.volatility_period, self.regime_period, self.trend_period)
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
        current_volatility = self.volatility.iloc[current_idx]
        current_regime = self.volatility_regime.iloc[current_idx]
        current_vol_percentile = self.volatility_percentile.iloc[current_idx]
        current_trend_ma = self.trend_ma.iloc[current_idx]
        current_momentum = self.momentum.iloc[current_idx]
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_confirmation:
            current_volume = current_data['volume']
            avg_volume = self.volume_ma.iloc[current_idx]
            volume_confirmed = current_volume >= avg_volume
        
        # Check for NaN values
        if (pd.isna(current_volatility) or pd.isna(current_trend_ma) or 
            pd.isna(current_momentum) or pd.isna(current_vol_percentile)):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Trend analysis
        trend_bullish = current_price > current_trend_ma
        trend_bearish = current_price < current_trend_ma
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "volatility": current_volatility,
            "volatility_regime": current_regime,
            "volatility_percentile": current_vol_percentile,
            "momentum": current_momentum,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish,
            "volume_confirmed": volume_confirmed
        }
        
        # Generate signals based on volatility regime
        if volume_confirmed:
            if current_regime == "high":
                # High volatility: Mean reversion strategy
                if abs(current_momentum) > 0.02:  # Significant price movement
                    if current_momentum < -0.02:  # Oversold
                        action = "buy"
                        strength = min(abs(current_momentum) * 20, 1.0)
                        confidence = 0.7
                        metadata["signal_type"] = "high_vol_mean_reversion_buy"
                        
                    elif current_momentum > 0.02:  # Overbought
                        action = "sell"
                        strength = min(abs(current_momentum) * 20, 1.0)
                        confidence = 0.7
                        metadata["signal_type"] = "high_vol_mean_reversion_sell"
                        
            elif current_regime == "low":
                # Low volatility: Trend following strategy
                if abs(current_momentum) > 0.005:  # Any momentum in low vol
                    if current_momentum > 0 and trend_bullish:
                        action = "buy"
                        strength = min(current_momentum * 50, 1.0)
                        confidence = 0.8
                        metadata["signal_type"] = "low_vol_trend_following_buy"
                        
                    elif current_momentum < 0 and trend_bearish:
                        action = "sell"
                        strength = min(abs(current_momentum) * 50, 1.0)
                        confidence = 0.8
                        metadata["signal_type"] = "low_vol_trend_following_sell"
                        
            else:  # Normal volatility regime
                # Balanced approach: moderate momentum with trend confirmation
                if abs(current_momentum) > 0.01:
                    if current_momentum > 0.01 and trend_bullish:
                        action = "buy"
                        strength = min(current_momentum * 30, 0.8)
                        confidence = 0.6
                        metadata["signal_type"] = "normal_vol_momentum_buy"
                        
                    elif current_momentum < -0.01 and trend_bearish:
                        action = "sell"
                        strength = min(abs(current_momentum) * 30, 0.8)
                        confidence = 0.6
                        metadata["signal_type"] = "normal_vol_momentum_sell"
        
        # Adjust strength based on volatility percentile
        if action != "hold":
            if current_vol_percentile > 0.8:  # Very high volatility
                strength = min(strength * 1.2, 1.0)
                confidence = min(confidence + 0.1, 1.0)
            elif current_vol_percentile < 0.2:  # Very low volatility
                strength = min(strength * 1.1, 1.0)
                confidence = min(confidence + 0.05, 1.0)
        
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
            'volatility_period': hp.choice('volatility_period', list(range(10, 51))),
            'regime_period': hp.choice('regime_period', list(range(30, 121, 10))),
            'high_vol_threshold': hp.uniform('high_vol_threshold', 0.6, 0.9),
            'low_vol_threshold': hp.uniform('low_vol_threshold', 0.1, 0.4),
            'trend_period': hp.choice('trend_period', list(range(20, 101, 10))),
            'momentum_period': hp.choice('momentum_period', list(range(5, 21))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.volatility is None:
            return {}
        
        return {
            'volatility': float(self.volatility.iloc[-1]) if not pd.isna(self.volatility.iloc[-1]) else 0.0,
            'volatility_regime': str(self.volatility_regime.iloc[-1]) if not pd.isna(self.volatility_regime.iloc[-1]) else "unknown",
            'volatility_percentile': float(self.volatility_percentile.iloc[-1]) if not pd.isna(self.volatility_percentile.iloc[-1]) else 0.0,
            'momentum': float(self.momentum.iloc[-1]) if not pd.isna(self.momentum.iloc[-1]) else 0.0
        }
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and volatility regime.
        
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
        
        # Volatility regime adjustments
        vol_regime = signal.metadata.get('volatility_regime', 'normal')
        vol_percentile = signal.metadata.get('volatility_percentile', 0.5)
        
        if vol_regime == "high":
            # Reduce size in high volatility
            adjusted_size = adjusted_size * 0.7
        elif vol_regime == "low":
            # Increase size in low volatility
            adjusted_size = adjusted_size * 1.2
        
        # Additional adjustment based on volatility percentile
        if vol_percentile > 0.9:  # Extreme high volatility
            adjusted_size = adjusted_size * 0.6
        elif vol_percentile < 0.1:  # Extreme low volatility
            adjusted_size = adjusted_size * 1.3
        
        # Standard volatility adjustment
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"HistoricalVolatility({self.volatility_period}, {self.regime_period})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HistoricalVolatilityStrategy(volatility_period={self.volatility_period}, regime_period={self.regime_period}, high_vol_threshold={self.high_vol_threshold})" 