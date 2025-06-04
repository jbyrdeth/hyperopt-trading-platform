"""
Williams %R Strategy

This strategy uses the Williams %R oscillator for momentum analysis with multiple
timeframe alignment, pattern recognition, failure swing detection, and volume confirmation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class WilliamsRStrategy(BaseStrategy):
    """
    Williams %R Strategy.
    
    Uses Williams %R oscillator for momentum reversal signals with pattern recognition,
    failure swing detection, and multi-timeframe alignment for enhanced accuracy.
    """
    
    def __init__(
        self,
        wr_period: int = 14,
        buy_level: float = -80.0,
        sell_level: float = -20.0,
        trend_filter_period: int = 50,
        volume_multiplier: float = 1.5,
        failure_swing_bars: int = 5,
        multi_timeframe: bool = True,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Williams %R Strategy.
        
        Args:
            wr_period: Williams %R calculation period (10-50)
            buy_level: Oversold threshold for buy signals (-85 to -70)
            sell_level: Overbought threshold for sell signals (-30 to -15)
            trend_filter_period: Period for trend confirmation (20-100)
            volume_multiplier: Volume confirmation requirement (1.0-3.0)
            failure_swing_bars: Pattern detection window (3-15)
            multi_timeframe: Whether to use multiple timeframe analysis
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'wr_period': wr_period,
            'buy_level': buy_level,
            'sell_level': sell_level,
            'trend_filter_period': trend_filter_period,
            'volume_multiplier': volume_multiplier,
            'failure_swing_bars': failure_swing_bars,
            'multi_timeframe': multi_timeframe
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"WilliamsR_{wr_period}_{buy_level}_{sell_level}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.wr_period = wr_period
        self.buy_level = buy_level
        self.sell_level = sell_level
        self.trend_filter_period = trend_filter_period
        self.volume_multiplier = volume_multiplier
        self.failure_swing_bars = failure_swing_bars
        self.multi_timeframe = multi_timeframe
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Indicator storage
        self.williams_r = None
        self.williams_r_fast = None  # Shorter period for multi-timeframe
        self.williams_r_slow = None  # Longer period for multi-timeframe
        self.trend_ma = None
        self.volume_ma = None
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if not (10 <= self.wr_period <= 50):
            self.logger.error(f"Williams %R period {self.wr_period} outside valid range [10, 50]")
            return False
        
        if not (-85.0 <= self.buy_level <= -70.0):
            self.logger.error(f"Buy level {self.buy_level} outside valid range [-85, -70]")
            return False
        
        if not (-30.0 <= self.sell_level <= -15.0):
            self.logger.error(f"Sell level {self.sell_level} outside valid range [-30, -15]")
            return False
        
        if self.buy_level >= self.sell_level:
            self.logger.error("Buy level must be less than sell level")
            return False
        
        if not (20 <= self.trend_filter_period <= 100):
            self.logger.error(f"Trend filter period {self.trend_filter_period} outside valid range [20, 100]")
            return False
        
        if not (1.0 <= self.volume_multiplier <= 3.0):
            self.logger.error(f"Volume multiplier {self.volume_multiplier} outside valid range [1.0, 3.0]")
            return False
        
        if not (3 <= self.failure_swing_bars <= 15):
            self.logger.error(f"Failure swing bars {self.failure_swing_bars} outside valid range [3, 15]")
            return False
        
        return True
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate highest high and lowest low over period
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def detect_failure_swing(self, williams_r: pd.Series, current_idx: int, swing_type: str) -> bool:
        """Detect failure swing patterns in Williams %R."""
        if current_idx < self.failure_swing_bars * 2:
            return False
        
        # Get lookback window
        start_idx = max(0, current_idx - self.failure_swing_bars * 2)
        wr_window = williams_r.iloc[start_idx:current_idx + 1]
        
        if len(wr_window) < self.failure_swing_bars:
            return False
        
        if swing_type == "bullish":
            # Bullish failure swing: %R fails to reach new low below -80
            # Look for pattern where %R goes below -80, rises, then fails to go below previous low
            
            # Find the lowest point in the window
            min_idx = wr_window.idxmin()
            min_value = wr_window.min()
            
            if min_value > -80:  # Must have been oversold
                return False
            
            # Check if there's a subsequent low that fails to break the previous low
            min_idx_pos = wr_window.index.get_loc(min_idx)
            if min_idx_pos < len(wr_window) - 3:  # Need some bars after the low
                subsequent_window = wr_window.iloc[min_idx_pos + 1:]
                if len(subsequent_window) >= 3:
                    subsequent_min = subsequent_window.min()
                    # Failure swing: subsequent low is higher than previous low
                    if subsequent_min > min_value and subsequent_min < -70:
                        return True
        
        elif swing_type == "bearish":
            # Bearish failure swing: %R fails to reach new high above -20
            # Look for pattern where %R goes above -20, falls, then fails to go above previous high
            
            # Find the highest point in the window
            max_idx = wr_window.idxmax()
            max_value = wr_window.max()
            
            if max_value < -20:  # Must have been overbought
                return False
            
            # Check if there's a subsequent high that fails to break the previous high
            max_idx_pos = wr_window.index.get_loc(max_idx)
            if max_idx_pos < len(wr_window) - 3:  # Need some bars after the high
                subsequent_window = wr_window.iloc[max_idx_pos + 1:]
                if len(subsequent_window) >= 3:
                    subsequent_max = subsequent_window.max()
                    # Failure swing: subsequent high is lower than previous high
                    if subsequent_max < max_value and subsequent_max > -30:
                        return True
        
        return False
    
    def check_multi_timeframe_alignment(self, current_idx: int, signal_type: str) -> bool:
        """Check if multiple timeframes are aligned for the signal."""
        if not self.multi_timeframe:
            return True
        
        if (self.williams_r_fast is None or self.williams_r_slow is None or
            current_idx >= len(self.williams_r_fast) or current_idx >= len(self.williams_r_slow)):
            return True
        
        current_fast = self.williams_r_fast.iloc[current_idx]
        current_slow = self.williams_r_slow.iloc[current_idx]
        
        if pd.isna(current_fast) or pd.isna(current_slow):
            return True
        
        if signal_type == "buy":
            # For buy signals, both timeframes should be oversold or recovering
            return current_fast < -50 and current_slow < -60
        elif signal_type == "sell":
            # For sell signals, both timeframes should be overbought or declining
            return current_fast > -50 and current_slow > -40
        
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the data
        self.data = data
        
        min_periods = max(self.wr_period, self.trend_filter_period) + self.failure_swing_bars * 2 + 10
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Calculate Williams %R
        self.williams_r = self.calculate_williams_r(data, self.wr_period)
        
        # Calculate multi-timeframe Williams %R if enabled
        if self.multi_timeframe:
            fast_period = max(5, self.wr_period // 2)
            slow_period = min(50, self.wr_period * 2)
            self.williams_r_fast = self.calculate_williams_r(data, fast_period)
            self.williams_r_slow = self.calculate_williams_r(data, slow_period)
        
        # Calculate trend filter
        self.trend_ma = ta.trend.sma_indicator(data['close'], window=self.trend_filter_period)
        
        # Calculate volume moving average for confirmation
        self.volume_ma = ta.trend.sma_indicator(data['volume'], window=self.wr_period)
        
        # Mark as initialized
        self.is_initialized = True
        
        self.logger.info(f"Initialized {self.name} with {len(data)} data points")
        self.logger.info(f"Williams %R period: {self.wr_period}, Levels: {self.buy_level} to {self.sell_level}")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.wr_period, self.trend_filter_period) + self.failure_swing_bars * 2 + 10
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on Williams %R analysis."""
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
        min_required = max(self.wr_period, self.trend_filter_period) + self.failure_swing_bars * 2
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
        current_wr = self.williams_r.iloc[current_idx]
        
        # Get previous values for momentum detection
        prev_wr = self.williams_r.iloc[current_idx - 1] if current_idx > 0 else current_wr
        
        # Trend filter
        current_trend_ma = self.trend_ma.iloc[current_idx]
        trend_bullish = current_price > current_trend_ma if not pd.isna(current_trend_ma) else True
        trend_bearish = current_price < current_trend_ma if not pd.isna(current_trend_ma) else True
        
        # Volume confirmation
        current_volume = current_data['volume']
        avg_volume = self.volume_ma.iloc[current_idx]
        volume_confirmed = current_volume >= (avg_volume * self.volume_multiplier)
        
        # Check for NaN values
        if pd.isna(current_wr):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={"reason": "NaN values in indicators"}
            )
        
        # Detect failure swings
        bullish_failure_swing = self.detect_failure_swing(self.williams_r, current_idx, "bullish")
        bearish_failure_swing = self.detect_failure_swing(self.williams_r, current_idx, "bearish")
        
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "williams_r": current_wr,
            "buy_level": self.buy_level,
            "sell_level": self.sell_level,
            "trend_bullish": trend_bullish,
            "trend_bearish": trend_bearish,
            "volume_confirmed": volume_confirmed,
            "bullish_failure_swing": bullish_failure_swing,
            "bearish_failure_swing": bearish_failure_swing
        }
        
        # Generate signals based on Williams %R analysis
        if volume_confirmed:
            # Bullish signals: %R rises from oversold levels
            if (current_wr > self.buy_level and prev_wr <= self.buy_level and 
                trend_bullish and self.check_multi_timeframe_alignment(current_idx, "buy")):
                action = "buy"
                strength = min((current_wr - self.buy_level) / abs(self.buy_level - self.sell_level), 1.0)
                confidence = 0.8
                metadata["signal_type"] = "williams_r_oversold_recovery_buy"
                
                # Boost for failure swing
                if bullish_failure_swing:
                    strength = min(strength * 1.4, 1.0)
                    confidence = min(confidence + 0.15, 1.0)
                    metadata["signal_type"] = "williams_r_failure_swing_buy"
            
            # Bearish signals: %R falls from overbought levels
            elif (current_wr < self.sell_level and prev_wr >= self.sell_level and 
                  trend_bearish and self.check_multi_timeframe_alignment(current_idx, "sell")):
                action = "sell"
                strength = min((self.sell_level - current_wr) / abs(self.buy_level - self.sell_level), 1.0)
                confidence = 0.8
                metadata["signal_type"] = "williams_r_overbought_decline_sell"
                
                # Boost for failure swing
                if bearish_failure_swing:
                    strength = min(strength * 1.4, 1.0)
                    confidence = min(confidence + 0.15, 1.0)
                    metadata["signal_type"] = "williams_r_failure_swing_sell"
            
            # Failure swing only signals (without level crossover)
            elif bullish_failure_swing and current_wr < -50 and trend_bullish:
                action = "buy"
                strength = 0.7
                confidence = 0.7
                metadata["signal_type"] = "williams_r_failure_swing_only_buy"
                
            elif bearish_failure_swing and current_wr > -50 and trend_bearish:
                action = "sell"
                strength = 0.7
                confidence = 0.7
                metadata["signal_type"] = "williams_r_failure_swing_only_sell"
            
            # Extreme level signals (additional confirmation)
            elif current_wr <= -90 and trend_bullish:
                action = "buy"
                strength = min((-90 - current_wr) / 10, 0.8)
                confidence = 0.6
                metadata["signal_type"] = "williams_r_extreme_oversold_buy"
                
            elif current_wr >= -10 and trend_bearish:
                action = "sell"
                strength = min((current_wr + 10) / 10, 0.8)
                confidence = 0.6
                metadata["signal_type"] = "williams_r_extreme_overbought_sell"
        
        # Adjust strength based on momentum
        if action != "hold":
            momentum = abs(current_wr - prev_wr)
            momentum_boost = min(momentum / 10, 0.3)
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
            'wr_period': hp.choice('wr_period', list(range(10, 51))),
            'buy_level': hp.uniform('buy_level', -85.0, -70.0),
            'sell_level': hp.uniform('sell_level', -30.0, -15.0),
            'trend_filter_period': hp.choice('trend_filter_period', list(range(20, 101, 10))),
            'volume_multiplier': hp.uniform('volume_multiplier', 1.0, 3.0),
            'failure_swing_bars': hp.choice('failure_swing_bars', list(range(3, 16))),
            'multi_timeframe': hp.choice('multi_timeframe', [True, False]),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized or self.williams_r is None:
            return {}
        
        indicators = {
            'williams_r': float(self.williams_r.iloc[-1]) if not pd.isna(self.williams_r.iloc[-1]) else 0.0,
            'buy_level': float(self.buy_level),
            'sell_level': float(self.sell_level)
        }
        
        if self.multi_timeframe and self.williams_r_fast is not None and self.williams_r_slow is not None:
            indicators.update({
                'williams_r_fast': float(self.williams_r_fast.iloc[-1]) if not pd.isna(self.williams_r_fast.iloc[-1]) else 0.0,
                'williams_r_slow': float(self.williams_r_slow.iloc[-1]) if not pd.isna(self.williams_r_slow.iloc[-1]) else 0.0
            })
        
        return indicators
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and Williams %R levels.
        
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
        
        # Williams %R specific adjustments
        williams_r = signal.metadata.get('williams_r', -50)
        
        # Increase size for extreme Williams %R levels
        if signal.action == 'buy' and williams_r < -85:
            adjusted_size = adjusted_size * 1.3
        elif signal.action == 'sell' and williams_r > -15:
            adjusted_size = adjusted_size * 1.3
        
        # Boost for failure swing patterns
        if "failure_swing" in signal.metadata.get("signal_type", ""):
            adjusted_size = adjusted_size * 1.2
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"WilliamsR({self.wr_period}, {self.buy_level}, {self.sell_level})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"WilliamsRStrategy(wr_period={self.wr_period}, buy_level={self.buy_level}, sell_level={self.sell_level})" 