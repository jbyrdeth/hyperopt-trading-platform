"""
Pivot Points Strategy

This strategy calculates various types of pivot points and generates trading signals
based on price interactions with these key levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy, Signal, Position


class PivotPointsStrategy(BaseStrategy):
    """
    Pivot Points trading strategy.
    
    This strategy:
    1. Calculates multiple types of pivot points (Standard, Fibonacci, Woodie, Camarilla)
    2. Identifies support and resistance levels from pivot calculations
    3. Generates buy signals near support pivots
    4. Generates sell signals near resistance pivots
    5. Uses multiple timeframes for confirmation
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Pivot Points strategy."""
        default_params = {
            'pivot_type': 'standard',           # 'standard', 'fibonacci', 'woodie', 'camarilla'
            'timeframe': 'daily',               # 'daily', 'weekly', 'monthly'
            'price_tolerance': 0.005,           # Price tolerance for pivot interaction (0.5%)
            'volume_confirmation': True,        # Use volume for signal confirmation
            'volume_threshold': 1.2,            # Volume multiplier for confirmation
            'trend_filter': True,               # Use trend filter for signals
            'trend_period': 20,                 # Period for trend calculation
            'signal_strength_threshold': 0.3,   # Minimum signal strength
            'max_signals_per_day': 3,           # Maximum signals per day
            'use_multiple_levels': True,        # Use S1/S2/R1/R2 levels
            'level_weight_decay': 0.8           # Weight decay for distant levels
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("PivotPoints", default_params)
        
        # Strategy state
        self.current_pivots: Dict[str, float] = {}
        self.pivot_calculation_time: Optional[datetime] = None
        self.daily_signal_count = 0
        self.last_signal_date: Optional[datetime] = None
        
    @property
    def required_periods(self) -> int:
        """Return the minimum number of periods required for the strategy."""
        return max(
            self.parameters['trend_period'] * 2,
            30  # Minimum for reliable pivot calculation
        )
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        valid_pivot_types = ['standard', 'fibonacci', 'woodie', 'camarilla']
        valid_timeframes = ['daily', 'weekly', 'monthly']
        
        if self.parameters['pivot_type'] not in valid_pivot_types:
            self.logger.error(f"Invalid pivot_type. Must be one of: {valid_pivot_types}")
            return False
            
        if self.parameters['timeframe'] not in valid_timeframes:
            self.logger.error(f"Invalid timeframe. Must be one of: {valid_timeframes}")
            return False
            
        if self.parameters['price_tolerance'] <= 0 or self.parameters['price_tolerance'] > 0.05:
            self.logger.error("price_tolerance must be between 0 and 0.05")
            return False
            
        if self.parameters['trend_period'] < 5:
            self.logger.error("trend_period must be at least 5")
            return False
            
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.data = data.copy()
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(data)
        
        # Calculate initial pivot points
        self._calculate_pivot_points(data.index[-1])
        
        self.is_initialized = True
        self.logger.info(f"Initialized {self.name} strategy with {len(data)} data points")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        indicators = {}
        
        # Volume moving average for confirmation
        if self.parameters['volume_confirmation']:
            indicators['volume_ma'] = data['volume'].rolling(
                window=self.parameters['trend_period']
            ).mean()
        
        # Trend indicators
        if self.parameters['trend_filter']:
            indicators['trend_ma'] = data['close'].rolling(
                window=self.parameters['trend_period']
            ).mean()
            
            # Price position relative to trend MA
            indicators['price_vs_trend'] = data['close'] / indicators['trend_ma'] - 1
        
        # Daily high/low for pivot calculation
        indicators['daily_high'] = data['high'].resample('D').max()
        indicators['daily_low'] = data['low'].resample('D').min()
        indicators['daily_close'] = data['close'].resample('D').last()
        
        return indicators
    
    def _get_pivot_period_data(self, current_time: datetime) -> Tuple[float, float, float]:
        """Get high, low, close for the pivot calculation period."""
        if self.parameters['timeframe'] == 'daily':
            # Use previous day's data
            prev_day = current_time - timedelta(days=1)
            day_data = self.data[self.data.index.date == prev_day.date()]
            
            if len(day_data) == 0:
                # Fallback to last available data
                day_data = self.data.iloc[-24:] if len(self.data) >= 24 else self.data
            
        elif self.parameters['timeframe'] == 'weekly':
            # Use previous week's data
            week_start = current_time - timedelta(days=7)
            day_data = self.data[self.data.index >= week_start]
            
        else:  # monthly
            # Use previous month's data
            month_start = current_time - timedelta(days=30)
            day_data = self.data[self.data.index >= month_start]
        
        if len(day_data) == 0:
            return 0.0, 0.0, 0.0
        
        high = day_data['high'].max()
        low = day_data['low'].min()
        close = day_data['close'].iloc[-1]
        
        return high, low, close
    
    def _calculate_standard_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate standard pivot points."""
        pivot = (high + low + close) / 3
        
        pivots = {
            'PP': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + (high - low),
            'R3': high + 2 * (pivot - low),
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low),
            'S3': low - 2 * (high - pivot)
        }
        
        return pivots
    
    def _calculate_fibonacci_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Fibonacci pivot points."""
        pivot = (high + low + close) / 3
        range_hl = high - low
        
        pivots = {
            'PP': pivot,
            'R1': pivot + 0.382 * range_hl,
            'R2': pivot + 0.618 * range_hl,
            'R3': pivot + range_hl,
            'S1': pivot - 0.382 * range_hl,
            'S2': pivot - 0.618 * range_hl,
            'S3': pivot - range_hl
        }
        
        return pivots
    
    def _calculate_woodie_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Woodie pivot points."""
        pivot = (high + low + 2 * close) / 4
        
        pivots = {
            'PP': pivot,
            'R1': 2 * pivot - low,
            'R2': pivot + (high - low),
            'R3': high + 2 * (pivot - low),
            'S1': 2 * pivot - high,
            'S2': pivot - (high - low),
            'S3': low - 2 * (high - pivot)
        }
        
        return pivots
    
    def _calculate_camarilla_pivots(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Camarilla pivot points."""
        range_hl = high - low
        
        pivots = {
            'PP': close,
            'R1': close + range_hl * 1.1 / 12,
            'R2': close + range_hl * 1.1 / 6,
            'R3': close + range_hl * 1.1 / 4,
            'R4': close + range_hl * 1.1 / 2,
            'S1': close - range_hl * 1.1 / 12,
            'S2': close - range_hl * 1.1 / 6,
            'S3': close - range_hl * 1.1 / 4,
            'S4': close - range_hl * 1.1 / 2
        }
        
        return pivots
    
    def _calculate_pivot_points(self, current_time: datetime) -> None:
        """Calculate pivot points based on the selected method."""
        high, low, close = self._get_pivot_period_data(current_time)
        
        if high == 0 or low == 0 or close == 0:
            return
        
        pivot_type = self.parameters['pivot_type']
        
        if pivot_type == 'standard':
            self.current_pivots = self._calculate_standard_pivots(high, low, close)
        elif pivot_type == 'fibonacci':
            self.current_pivots = self._calculate_fibonacci_pivots(high, low, close)
        elif pivot_type == 'woodie':
            self.current_pivots = self._calculate_woodie_pivots(high, low, close)
        elif pivot_type == 'camarilla':
            self.current_pivots = self._calculate_camarilla_pivots(high, low, close)
        
        self.pivot_calculation_time = current_time
        
        self.logger.debug(f"Calculated {pivot_type} pivots: {self.current_pivots}")
    
    def _should_recalculate_pivots(self, current_time: datetime) -> bool:
        """Check if pivots should be recalculated."""
        if not self.pivot_calculation_time:
            return True
        
        if self.parameters['timeframe'] == 'daily':
            # Recalculate daily at market open or if date changed
            return current_time.date() != self.pivot_calculation_time.date()
        elif self.parameters['timeframe'] == 'weekly':
            # Recalculate weekly on Monday
            return current_time.weekday() == 0 and current_time.date() != self.pivot_calculation_time.date()
        else:  # monthly
            # Recalculate monthly on first day of month
            return current_time.day == 1 and current_time.date() != self.pivot_calculation_time.date()
    
    def _get_nearest_pivot_level(self, current_price: float) -> Tuple[Optional[str], Optional[float], str]:
        """Get the nearest pivot level and its type (support/resistance)."""
        if not self.current_pivots:
            return None, None, 'none'
        
        min_distance = float('inf')
        nearest_level = None
        nearest_price = None
        level_type = 'none'
        
        for level_name, level_price in self.current_pivots.items():
            distance = abs(current_price - level_price) / current_price
            
            if distance < min_distance:
                min_distance = distance
                nearest_level = level_name
                nearest_price = level_price
                
                # Determine if it's support or resistance
                if current_price > level_price:
                    level_type = 'support'
                else:
                    level_type = 'resistance'
        
        return nearest_level, nearest_price, level_type
    
    def _get_signal_strength(
        self,
        current_price: float,
        pivot_price: float,
        level_name: str,
        current_volume: float
    ) -> float:
        """Calculate signal strength based on pivot level interaction."""
        # Distance factor (closer to pivot = stronger signal)
        price_distance = abs(current_price - pivot_price) / current_price
        distance_factor = max(0, 1 - (price_distance / self.parameters['price_tolerance']))
        
        # Level importance factor (PP > R1/S1 > R2/S2 > R3/S3)
        importance_weights = {
            'PP': 1.0,
            'R1': 0.9, 'S1': 0.9,
            'R2': 0.8, 'S2': 0.8,
            'R3': 0.7, 'S3': 0.7,
            'R4': 0.6, 'S4': 0.6
        }
        importance_factor = importance_weights.get(level_name, 0.5)
        
        # Volume confirmation
        volume_factor = 1.0
        if self.parameters['volume_confirmation'] and 'volume_ma' in self.indicators:
            avg_volume = self.indicators['volume_ma'].iloc[-1]
            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_factor = min(volume_ratio / self.parameters['volume_threshold'], 1.5)
        
        # Trend alignment factor
        trend_factor = 1.0
        if self.parameters['trend_filter'] and 'price_vs_trend' in self.indicators:
            price_vs_trend = self.indicators['price_vs_trend'].iloc[-1]
            if not pd.isna(price_vs_trend):
                # Positive trend bias for buy signals, negative for sell signals
                trend_factor = 1.0 + abs(price_vs_trend) * 0.5
        
        # Combine factors
        signal_strength = distance_factor * importance_factor * volume_factor * trend_factor
        
        return min(signal_strength, 1.0)
    
    def _reset_daily_signal_count(self, current_time: datetime) -> None:
        """Reset daily signal count if new day."""
        if (not self.last_signal_date or 
            current_time.date() != self.last_signal_date.date()):
            self.daily_signal_count = 0
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on pivot point levels."""
        if not self.is_initialized:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Recalculate pivots if needed
        if self._should_recalculate_pivots(current_time):
            self._calculate_pivot_points(current_time)
        
        # Reset daily signal count
        self._reset_daily_signal_count(current_time)
        
        # Check if we've reached max signals for the day
        if self.daily_signal_count >= self.parameters['max_signals_per_day']:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        current_price = current_data['close']
        current_volume = current_data['volume']
        
        # Get nearest pivot level
        nearest_level, pivot_price, level_type = self._get_nearest_pivot_level(current_price)
        
        if not nearest_level or not pivot_price:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        # Check if price is within tolerance of pivot level
        price_distance = abs(current_price - pivot_price) / current_price
        
        if price_distance > self.parameters['price_tolerance']:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        # Calculate signal strength
        signal_strength = self._get_signal_strength(
            current_price, pivot_price, nearest_level, current_volume
        )
        
        if signal_strength < self.parameters['signal_strength_threshold']:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        # Determine signal action based on level type and price position
        action = 'hold'
        confidence = 0.0
        
        if level_type == 'support' and current_price >= pivot_price:
            # Buy signal near support
            action = 'buy'
            confidence = min(signal_strength, 1.0)
            
        elif level_type == 'resistance' and current_price <= pivot_price:
            # Sell signal near resistance
            action = 'sell'
            confidence = min(signal_strength, 1.0)
        
        # Apply trend filter
        if self.parameters['trend_filter'] and 'price_vs_trend' in self.indicators:
            price_vs_trend = self.indicators['price_vs_trend'].iloc[-1]
            if not pd.isna(price_vs_trend):
                # Reduce confidence for counter-trend signals
                if (action == 'buy' and price_vs_trend < -0.02) or \
                   (action == 'sell' and price_vs_trend > 0.02):
                    confidence *= 0.7
        
        if action != 'hold':
            self.daily_signal_count += 1
            self.last_signal_date = current_time
            self.signals_generated += 1
            
            return Signal(
                timestamp=current_time,
                action=action,
                strength=signal_strength,
                price=current_price,
                confidence=confidence,
                metadata={
                    'signal_type': f'pivot_{level_type}',
                    'pivot_level': nearest_level,
                    'pivot_price': pivot_price,
                    'price_distance': price_distance,
                    'pivot_type': self.parameters['pivot_type'],
                    'timeframe': self.parameters['timeframe']
                }
            )
        
        return Signal(
            timestamp=current_time,
            action='hold',
            strength=0.0,
            price=current_price,
            confidence=0.0
        )
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """Calculate position size based on signal strength and risk management."""
        if signal.action == 'hold':
            return 0.0
        
        # Base position size as percentage of available capital
        base_position_pct = 0.08  # 8% base allocation
        
        # Adjust based on signal strength and confidence
        strength_multiplier = signal.strength * signal.confidence
        position_pct = base_position_pct * strength_multiplier
        
        # Adjust based on pivot level importance
        if signal.metadata and 'pivot_level' in signal.metadata:
            level_name = signal.metadata['pivot_level']
            if level_name == 'PP':
                position_pct *= 1.2  # Increase for main pivot
            elif level_name in ['R1', 'S1']:
                position_pct *= 1.1  # Slight increase for primary levels
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = 1.0 / (1.0 + current_volatility * 1.5)
        position_pct *= volatility_adjustment
        
        # Calculate position size
        position_value = available_capital * position_pct
        position_size = position_value / current_price
        
        # Apply direction
        if signal.action == 'sell':
            position_size = -position_size
        
        return position_size
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state."""
        base_info = super().get_strategy_info()
        
        strategy_info = {
            'pivot_type': self.parameters['pivot_type'],
            'timeframe': self.parameters['timeframe'],
            'current_pivots': self.current_pivots,
            'pivot_calculation_time': self.pivot_calculation_time.isoformat() if self.pivot_calculation_time else None,
            'daily_signal_count': self.daily_signal_count,
            'last_signal_date': self.last_signal_date.isoformat() if self.last_signal_date else None
        }
        
        base_info.update(strategy_info)
        return base_info 