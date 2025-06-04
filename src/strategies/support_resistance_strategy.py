"""
Support/Resistance Strategy

This strategy identifies dynamic support and resistance levels using local minima and maxima,
then generates trading signals based on price interactions with these levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from scipy.signal import argrelextrema

from .base_strategy import BaseStrategy, Signal, Position


class SupportResistanceStrategy(BaseStrategy):
    """
    Support/Resistance trading strategy.
    
    This strategy:
    1. Identifies support and resistance levels using local extrema
    2. Tracks level strength based on number of touches
    3. Generates buy signals near support levels
    4. Generates sell signals near resistance levels
    5. Uses volume confirmation for signal validation
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Support/Resistance strategy."""
        default_params = {
            'lookback_period': 20,          # Period for finding local extrema
            'min_distance': 5,              # Minimum distance between extrema
            'level_tolerance': 0.02,        # Price tolerance for level interaction (2%)
            'min_touches': 2,               # Minimum touches to confirm level
            'max_levels': 10,               # Maximum number of levels to track
            'volume_threshold': 1.2,        # Volume multiplier for confirmation
            'breakout_threshold': 0.005,    # Threshold for breakout confirmation (0.5%)
            'strength_decay': 0.95,         # Level strength decay factor
            'confirmation_bars': 2,         # Bars to confirm breakout/bounce
            'use_volume_confirmation': True  # Whether to use volume confirmation
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("SupportResistance", default_params)
        
        # Strategy state
        self.support_levels: List[Dict[str, Any]] = []
        self.resistance_levels: List[Dict[str, Any]] = []
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown = 5  # Minimum bars between signals
        
    @property
    def required_periods(self) -> int:
        """Return the minimum number of periods required for the strategy."""
        return max(
            self.parameters['lookback_period'] * 2,
            50  # Minimum for reliable level detection
        )
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        required_params = [
            'lookback_period', 'min_distance', 'level_tolerance',
            'min_touches', 'max_levels', 'volume_threshold'
        ]
        
        for param in required_params:
            if param not in self.parameters:
                self.logger.error(f"Missing required parameter: {param}")
                return False
        
        # Validate parameter ranges
        if self.parameters['lookback_period'] < 5:
            self.logger.error("lookback_period must be at least 5")
            return False
            
        if self.parameters['level_tolerance'] <= 0 or self.parameters['level_tolerance'] > 0.1:
            self.logger.error("level_tolerance must be between 0 and 0.1")
            return False
            
        if self.parameters['min_touches'] < 2:
            self.logger.error("min_touches must be at least 2")
            return False
            
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.data = data.copy()
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(data)
        
        # Initialize support/resistance levels
        self._initialize_levels()
        
        self.is_initialized = True
        self.logger.info(f"Initialized {self.name} strategy with {len(data)} data points")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        indicators = {}
        
        # Volume moving average for confirmation
        indicators['volume_ma'] = data['volume'].rolling(
            window=self.parameters['lookback_period']
        ).mean()
        
        # Price volatility for dynamic tolerance
        indicators['price_volatility'] = data['close'].rolling(
            window=self.parameters['lookback_period']
        ).std() / data['close'].rolling(window=self.parameters['lookback_period']).mean()
        
        # High/Low moving averages for trend context
        indicators['high_ma'] = data['high'].rolling(
            window=self.parameters['lookback_period']
        ).mean()
        indicators['low_ma'] = data['low'].rolling(
            window=self.parameters['lookback_period']
        ).mean()
        
        return indicators
    
    def _initialize_levels(self) -> None:
        """Initialize support and resistance levels from historical data."""
        if len(self.data) < self.required_periods:
            return
        
        # Find local extrema
        highs = self.data['high'].values
        lows = self.data['low'].values
        
        # Find local maxima (resistance candidates)
        resistance_indices = argrelextrema(
            highs,
            np.greater,
            order=self.parameters['min_distance']
        )[0]
        
        # Find local minima (support candidates)
        support_indices = argrelextrema(
            lows,
            np.less,
            order=self.parameters['min_distance']
        )[0]
        
        # Create resistance levels
        for idx in resistance_indices[-self.parameters['max_levels']:]:
            if idx < len(self.data):
                level = {
                    'price': highs[idx],
                    'timestamp': self.data.index[idx],
                    'touches': 1,
                    'strength': 1.0,
                    'last_touch': self.data.index[idx],
                    'volume_at_creation': self.data.iloc[idx]['volume']
                }
                self.resistance_levels.append(level)
        
        # Create support levels
        for idx in support_indices[-self.parameters['max_levels']:]:
            if idx < len(self.data):
                level = {
                    'price': lows[idx],
                    'timestamp': self.data.index[idx],
                    'touches': 1,
                    'strength': 1.0,
                    'last_touch': self.data.index[idx],
                    'volume_at_creation': self.data.iloc[idx]['volume']
                }
                self.support_levels.append(level)
        
        # Sort levels by strength
        self.resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
        self.support_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        self.logger.info(f"Initialized {len(self.resistance_levels)} resistance and {len(self.support_levels)} support levels")
    
    def _update_levels(self, current_time: datetime, current_data: pd.Series) -> None:
        """Update support and resistance levels."""
        current_price = current_data['close']
        current_volume = current_data['volume']
        tolerance = self.parameters['level_tolerance']
        
        # Update existing levels
        for level in self.resistance_levels + self.support_levels:
            # Check if price is touching the level
            price_diff = abs(current_price - level['price']) / level['price']
            
            if price_diff <= tolerance:
                level['touches'] += 1
                level['strength'] = min(level['strength'] * 1.1, 3.0)  # Cap at 3.0
                level['last_touch'] = current_time
            else:
                # Decay strength over time
                level['strength'] *= self.parameters['strength_decay']
        
        # Remove weak levels
        self.resistance_levels = [
            level for level in self.resistance_levels 
            if level['strength'] > 0.1 and level['touches'] >= self.parameters['min_touches']
        ]
        self.support_levels = [
            level for level in self.support_levels 
            if level['strength'] > 0.1 and level['touches'] >= self.parameters['min_touches']
        ]
        
        # Add new levels if price creates new extrema
        self._check_for_new_levels(current_time, current_data)
    
    def _check_for_new_levels(self, current_time: datetime, current_data: pd.Series) -> None:
        """Check for new support/resistance levels."""
        lookback = self.parameters['lookback_period']
        current_idx = self.data.index.get_loc(current_time)
        
        if current_idx < lookback:
            return
        
        # Get recent data
        recent_data = self.data.iloc[current_idx - lookback:current_idx + 1]
        
        # Check if current high is a local maximum
        if (current_data['high'] == recent_data['high'].max() and
            len(self.resistance_levels) < self.parameters['max_levels']):
            
            # Check if this level is significantly different from existing ones
            is_new_level = True
            for level in self.resistance_levels:
                price_diff = abs(current_data['high'] - level['price']) / level['price']
                if price_diff <= self.parameters['level_tolerance']:
                    is_new_level = False
                    break
            
            if is_new_level:
                new_level = {
                    'price': current_data['high'],
                    'timestamp': current_time,
                    'touches': 1,
                    'strength': 1.0,
                    'last_touch': current_time,
                    'volume_at_creation': current_data['volume']
                }
                self.resistance_levels.append(new_level)
        
        # Check if current low is a local minimum
        if (current_data['low'] == recent_data['low'].min() and
            len(self.support_levels) < self.parameters['max_levels']):
            
            # Check if this level is significantly different from existing ones
            is_new_level = True
            for level in self.support_levels:
                price_diff = abs(current_data['low'] - level['price']) / level['price']
                if price_diff <= self.parameters['level_tolerance']:
                    is_new_level = False
                    break
            
            if is_new_level:
                new_level = {
                    'price': current_data['low'],
                    'timestamp': current_time,
                    'touches': 1,
                    'strength': 1.0,
                    'last_touch': current_time,
                    'volume_at_creation': current_data['volume']
                }
                self.support_levels.append(new_level)
    
    def _get_signal_strength(
        self,
        level: Dict[str, Any],
        current_price: float,
        current_volume: float,
        signal_type: str
    ) -> float:
        """Calculate signal strength based on level properties."""
        # Base strength from level strength and touches
        base_strength = min(level['strength'] * level['touches'] / 10.0, 1.0)
        
        # Distance factor (closer to level = stronger signal)
        price_distance = abs(current_price - level['price']) / level['price']
        distance_factor = max(0, 1 - (price_distance / self.parameters['level_tolerance']))
        
        # Volume confirmation
        volume_factor = 1.0
        if self.parameters['use_volume_confirmation']:
            avg_volume = self.indicators['volume_ma'].iloc[-1]
            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_factor = min(volume_ratio / self.parameters['volume_threshold'], 1.5)
        
        # Combine factors
        signal_strength = base_strength * distance_factor * volume_factor
        
        return min(signal_strength, 1.0)
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on support/resistance levels."""
        if not self.is_initialized:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Update levels
        self._update_levels(current_time, current_data)
        
        current_price = current_data['close']
        current_volume = current_data['volume']
        
        # Check for signal cooldown
        if (self.last_signal_time and 
            (current_time - self.last_signal_time).total_seconds() < self.signal_cooldown * 3600):
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        best_signal = None
        best_strength = 0.0
        
        # Check for buy signals near support levels
        for level in self.support_levels:
            price_diff = (current_price - level['price']) / level['price']
            
            # Price is near or slightly above support
            if -self.parameters['level_tolerance'] <= price_diff <= self.parameters['breakout_threshold']:
                strength = self._get_signal_strength(level, current_price, current_volume, 'buy')
                
                if strength > best_strength:
                    confidence = min(level['touches'] / 5.0, 1.0)  # More touches = higher confidence
                    
                    best_signal = Signal(
                        timestamp=current_time,
                        action='buy',
                        strength=strength,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'signal_type': 'support_bounce',
                            'level_price': level['price'],
                            'level_strength': level['strength'],
                            'level_touches': level['touches'],
                            'price_distance': abs(price_diff)
                        }
                    )
                    best_strength = strength
        
        # Check for sell signals near resistance levels
        for level in self.resistance_levels:
            price_diff = (current_price - level['price']) / level['price']
            
            # Price is near or slightly below resistance
            if -self.parameters['breakout_threshold'] <= price_diff <= self.parameters['level_tolerance']:
                strength = self._get_signal_strength(level, current_price, current_volume, 'sell')
                
                if strength > best_strength:
                    confidence = min(level['touches'] / 5.0, 1.0)  # More touches = higher confidence
                    
                    best_signal = Signal(
                        timestamp=current_time,
                        action='sell',
                        strength=strength,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'signal_type': 'resistance_rejection',
                            'level_price': level['price'],
                            'level_strength': level['strength'],
                            'level_touches': level['touches'],
                            'price_distance': abs(price_diff)
                        }
                    )
                    best_strength = strength
        
        # Return best signal or hold
        if best_signal and best_strength > 0.3:  # Minimum strength threshold
            self.last_signal_time = current_time
            self.signals_generated += 1
            return best_signal
        
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
        base_position_pct = 0.1  # 10% base allocation
        
        # Adjust based on signal strength and confidence
        strength_multiplier = signal.strength * signal.confidence
        position_pct = base_position_pct * strength_multiplier
        
        # Adjust for volatility (reduce size in high volatility)
        volatility_adjustment = 1.0 / (1.0 + current_volatility * 2)
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
            'support_levels_count': len(self.support_levels),
            'resistance_levels_count': len(self.resistance_levels),
            'strongest_support': max(self.support_levels, key=lambda x: x['strength'])['price'] if self.support_levels else None,
            'strongest_resistance': max(self.resistance_levels, key=lambda x: x['strength'])['price'] if self.resistance_levels else None,
            'total_level_touches': sum(level['touches'] for level in self.support_levels + self.resistance_levels),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
        }
        
        base_info.update(strategy_info)
        return base_info 