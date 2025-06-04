"""
Fibonacci Retracement Strategy

This strategy identifies swing highs and lows, calculates Fibonacci retracement levels,
and generates trading signals based on price interactions with these key levels.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

from .base_strategy import BaseStrategy, Signal, Position


class FibonacciRetracementStrategy(BaseStrategy):
    """
    Fibonacci Retracement trading strategy.
    
    This strategy:
    1. Identifies significant swing highs and lows
    2. Calculates Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
    3. Generates buy signals at retracement levels in uptrends
    4. Generates sell signals at retracement levels in downtrends
    5. Uses volume and momentum confirmation
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Fibonacci Retracement strategy."""
        default_params = {
            'swing_lookback': 20,               # Period for identifying swing points
            'min_swing_size': 0.03,             # Minimum swing size (3%)
            'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.786],  # Fibonacci levels
            'price_tolerance': 0.008,           # Price tolerance for level interaction (0.8%)
            'volume_confirmation': True,        # Use volume for signal confirmation
            'volume_threshold': 1.3,            # Volume multiplier for confirmation
            'momentum_confirmation': True,      # Use momentum for signal confirmation
            'momentum_period': 14,              # Period for momentum calculation
            'trend_strength_threshold': 0.02,   # Minimum trend strength
            'max_retracement_age': 50,          # Maximum age of retracement in bars
            'signal_strength_threshold': 0.4,   # Minimum signal strength
            'use_extension_levels': False,      # Use Fibonacci extension levels
            'extension_levels': [1.272, 1.414, 1.618]  # Extension levels
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("FibonacciRetracement", default_params)
        
        # Strategy state
        self.current_swing_high: Optional[Dict[str, Any]] = None
        self.current_swing_low: Optional[Dict[str, Any]] = None
        self.fib_levels_cache: Dict[str, float] = {}
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown = 3  # Minimum bars between signals
        
    @property
    def required_periods(self) -> int:
        """Return the minimum number of periods required for the strategy."""
        return max(
            self.parameters['swing_lookback'] * 3,
            self.parameters['momentum_period'] * 2,
            60  # Minimum for reliable swing detection
        )
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.parameters['swing_lookback'] < 5:
            self.logger.error("swing_lookback must be at least 5")
            return False
            
        if self.parameters['min_swing_size'] <= 0 or self.parameters['min_swing_size'] > 0.2:
            self.logger.error("min_swing_size must be between 0 and 0.2")
            return False
            
        if self.parameters['price_tolerance'] <= 0 or self.parameters['price_tolerance'] > 0.05:
            self.logger.error("price_tolerance must be between 0 and 0.05")
            return False
            
        if not self.parameters['fib_levels'] or len(self.parameters['fib_levels']) == 0:
            self.logger.error("fib_levels cannot be empty")
            return False
            
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.data = data.copy()
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(data)
        
        # Initialize swing points
        self._initialize_swing_points()
        
        self.is_initialized = True
        self.logger.info(f"Initialized {self.name} strategy with {len(data)} data points")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        indicators = {}
        
        # Volume moving average for confirmation
        if self.parameters['volume_confirmation']:
            indicators['volume_ma'] = data['volume'].rolling(
                window=self.parameters['momentum_period']
            ).mean()
        
        # Momentum indicators
        if self.parameters['momentum_confirmation']:
            # Rate of Change for momentum
            indicators['roc'] = data['close'].pct_change(
                periods=self.parameters['momentum_period']
            ) * 100
            
            # RSI for momentum confirmation
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(
                window=self.parameters['momentum_period']
            ).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(
                window=self.parameters['momentum_period']
            ).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Trend strength indicator
        indicators['trend_strength'] = (
            data['close'].rolling(window=self.parameters['swing_lookback']).max() -
            data['close'].rolling(window=self.parameters['swing_lookback']).min()
        ) / data['close']
        
        # Price volatility
        indicators['volatility'] = data['close'].rolling(
            window=self.parameters['momentum_period']
        ).std() / data['close'].rolling(window=self.parameters['momentum_period']).mean()
        
        return indicators
    
    def _find_swing_points(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Find swing highs and lows in the data."""
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Find local maxima (swing highs)
        high_indices = argrelextrema(
            highs,
            np.greater,
            order=self.parameters['swing_lookback'] // 2
        )[0]
        
        # Find local minima (swing lows)
        low_indices = argrelextrema(
            lows,
            np.less,
            order=self.parameters['swing_lookback'] // 2
        )[0]
        
        swing_highs = []
        swing_lows = []
        
        # Create swing high objects
        for idx in high_indices:
            if idx < len(data):
                swing_high = {
                    'price': highs[idx],
                    'timestamp': data.index[idx],
                    'index': idx,
                    'volume': data.iloc[idx]['volume']
                }
                swing_highs.append(swing_high)
        
        # Create swing low objects
        for idx in low_indices:
            if idx < len(data):
                swing_low = {
                    'price': lows[idx],
                    'timestamp': data.index[idx],
                    'index': idx,
                    'volume': data.iloc[idx]['volume']
                }
                swing_lows.append(swing_low)
        
        return swing_highs, swing_lows
    
    def _initialize_swing_points(self) -> None:
        """Initialize swing points from historical data."""
        if len(self.data) < self.required_periods:
            return
        
        swing_highs, swing_lows = self._find_swing_points(self.data)
        
        # Find the most recent significant swing high and low
        current_price = self.data['close'].iloc[-1]
        
        # Find most recent swing high
        for swing_high in reversed(swing_highs):
            swing_size = abs(swing_high['price'] - current_price) / current_price
            if swing_size >= self.parameters['min_swing_size']:
                self.current_swing_high = swing_high
                break
        
        # Find most recent swing low
        for swing_low in reversed(swing_lows):
            swing_size = abs(current_price - swing_low['price']) / swing_low['price']
            if swing_size >= self.parameters['min_swing_size']:
                self.current_swing_low = swing_low
                break
        
        # Calculate Fibonacci levels
        self._calculate_fibonacci_levels()
        
        self.logger.info(f"Initialized swing points - High: {self.current_swing_high}, Low: {self.current_swing_low}")
    
    def _calculate_fibonacci_levels(self) -> None:
        """Calculate Fibonacci retracement levels."""
        self.fib_levels_cache = {}
        
        if not self.current_swing_high or not self.current_swing_low:
            return
        
        high_price = self.current_swing_high['price']
        low_price = self.current_swing_low['price']
        
        # Determine trend direction
        if self.current_swing_high['timestamp'] > self.current_swing_low['timestamp']:
            # Downtrend - retracing from high to low
            price_range = high_price - low_price
            for level in self.parameters['fib_levels']:
                self.fib_levels_cache[f'fib_{level:.3f}'] = high_price - (price_range * level)
        else:
            # Uptrend - retracing from low to high
            price_range = high_price - low_price
            for level in self.parameters['fib_levels']:
                self.fib_levels_cache[f'fib_{level:.3f}'] = low_price + (price_range * level)
        
        # Add extension levels if enabled
        if self.parameters['use_extension_levels']:
            for ext_level in self.parameters['extension_levels']:
                if self.current_swing_high['timestamp'] > self.current_swing_low['timestamp']:
                    # Downtrend extensions
                    self.fib_levels_cache[f'ext_{ext_level:.3f}'] = high_price - (price_range * ext_level)
                else:
                    # Uptrend extensions
                    self.fib_levels_cache[f'ext_{ext_level:.3f}'] = low_price + (price_range * ext_level)
    
    def _update_swing_points(self, current_time: datetime, current_data: pd.Series) -> None:
        """Update swing points if new significant swings are detected."""
        current_price = current_data['close']
        current_high = current_data['high']
        current_low = current_data['low']
        
        # Check for new swing high
        if (not self.current_swing_high or 
            current_high > self.current_swing_high['price']):
            
            # Verify it's a significant swing
            if self.current_swing_low:
                swing_size = (current_high - self.current_swing_low['price']) / self.current_swing_low['price']
                if swing_size >= self.parameters['min_swing_size']:
                    self.current_swing_high = {
                        'price': current_high,
                        'timestamp': current_time,
                        'index': len(self.data) - 1,
                        'volume': current_data['volume']
                    }
                    self._calculate_fibonacci_levels()
        
        # Check for new swing low
        if (not self.current_swing_low or 
            current_low < self.current_swing_low['price']):
            
            # Verify it's a significant swing
            if self.current_swing_high:
                swing_size = (self.current_swing_high['price'] - current_low) / current_low
                if swing_size >= self.parameters['min_swing_size']:
                    self.current_swing_low = {
                        'price': current_low,
                        'timestamp': current_time,
                        'index': len(self.data) - 1,
                        'volume': current_data['volume']
                    }
                    self._calculate_fibonacci_levels()
    
    def _get_trend_direction(self) -> str:
        """Determine current trend direction based on swing points."""
        if not self.current_swing_high or not self.current_swing_low:
            return 'sideways'
        
        if self.current_swing_high['timestamp'] > self.current_swing_low['timestamp']:
            return 'downtrend'
        else:
            return 'uptrend'
    
    def _get_nearest_fib_level(self, current_price: float) -> Tuple[Optional[str], Optional[float], float]:
        """Get the nearest Fibonacci level and distance."""
        if not self.fib_levels_cache:
            return None, None, float('inf')
        
        min_distance = float('inf')
        nearest_level = None
        nearest_price = None
        
        for level_name, level_price in self.fib_levels_cache.items():
            distance = abs(current_price - level_price) / current_price
            
            if distance < min_distance:
                min_distance = distance
                nearest_level = level_name
                nearest_price = level_price
        
        return nearest_level, nearest_price, min_distance
    
    def _get_signal_strength(
        self,
        current_price: float,
        fib_price: float,
        fib_level: str,
        current_volume: float,
        trend_direction: str
    ) -> float:
        """Calculate signal strength based on Fibonacci level interaction."""
        # Distance factor (closer to level = stronger signal)
        price_distance = abs(current_price - fib_price) / current_price
        distance_factor = max(0, 1 - (price_distance / self.parameters['price_tolerance']))
        
        # Fibonacci level importance (61.8% and 38.2% are most important)
        level_importance = {
            'fib_0.618': 1.0,
            'fib_0.382': 0.95,
            'fib_0.500': 0.9,
            'fib_0.236': 0.8,
            'fib_0.786': 0.85
        }
        importance_factor = level_importance.get(fib_level, 0.7)
        
        # Volume confirmation
        volume_factor = 1.0
        if self.parameters['volume_confirmation'] and 'volume_ma' in self.indicators:
            avg_volume = self.indicators['volume_ma'].iloc[-1]
            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_factor = min(volume_ratio / self.parameters['volume_threshold'], 1.5)
        
        # Momentum confirmation
        momentum_factor = 1.0
        if self.parameters['momentum_confirmation']:
            if 'rsi' in self.indicators:
                rsi = self.indicators['rsi'].iloc[-1]
                if not pd.isna(rsi):
                    # RSI divergence from extremes suggests reversal
                    if trend_direction == 'uptrend' and rsi < 70:
                        momentum_factor *= 1.2
                    elif trend_direction == 'downtrend' and rsi > 30:
                        momentum_factor *= 1.2
        
        # Trend strength factor
        trend_factor = 1.0
        if 'trend_strength' in self.indicators:
            trend_strength = self.indicators['trend_strength'].iloc[-1]
            if not pd.isna(trend_strength) and trend_strength > self.parameters['trend_strength_threshold']:
                trend_factor = 1.0 + trend_strength
        
        # Combine factors
        signal_strength = distance_factor * importance_factor * volume_factor * momentum_factor * trend_factor
        
        return min(signal_strength, 1.0)
    
    def _is_retracement_valid(self, current_time: datetime) -> bool:
        """Check if the current retracement is still valid (not too old)."""
        if not self.current_swing_high or not self.current_swing_low:
            return False
        
        # Check age of most recent swing point
        most_recent_swing = max(
            self.current_swing_high['timestamp'],
            self.current_swing_low['timestamp']
        )
        
        # Calculate age in bars (approximate)
        age_hours = (current_time - most_recent_swing).total_seconds() / 3600
        max_age_hours = self.parameters['max_retracement_age']
        
        return age_hours <= max_age_hours
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on Fibonacci retracement levels."""
        if not self.is_initialized:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Update swing points
        self._update_swing_points(current_time, current_data)
        
        # Check if retracement is still valid
        if not self._is_retracement_valid(current_time):
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Check for signal cooldown
        if (self.last_signal_time and 
            (current_time - self.last_signal_time).total_seconds() < self.signal_cooldown * 3600):
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        current_price = current_data['close']
        current_volume = current_data['volume']
        
        # Get nearest Fibonacci level
        nearest_level, fib_price, distance = self._get_nearest_fib_level(current_price)
        
        if not nearest_level or distance > self.parameters['price_tolerance']:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        # Determine trend direction
        trend_direction = self._get_trend_direction()
        
        if trend_direction == 'sideways':
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        # Calculate signal strength
        signal_strength = self._get_signal_strength(
            current_price, fib_price, nearest_level, current_volume, trend_direction
        )
        
        if signal_strength < self.parameters['signal_strength_threshold']:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_price,
                confidence=0.0
            )
        
        # Determine signal action based on trend and level
        action = 'hold'
        confidence = 0.0
        
        if trend_direction == 'uptrend' and current_price <= fib_price:
            # Buy signal at retracement level in uptrend
            action = 'buy'
            confidence = signal_strength
            
        elif trend_direction == 'downtrend' and current_price >= fib_price:
            # Sell signal at retracement level in downtrend
            action = 'sell'
            confidence = signal_strength
        
        if action != 'hold':
            self.last_signal_time = current_time
            self.signals_generated += 1
            
            return Signal(
                timestamp=current_time,
                action=action,
                strength=signal_strength,
                price=current_price,
                confidence=confidence,
                metadata={
                    'signal_type': f'fibonacci_{trend_direction}',
                    'fib_level': nearest_level,
                    'fib_price': fib_price,
                    'price_distance': distance,
                    'trend_direction': trend_direction,
                    'swing_high': self.current_swing_high['price'] if self.current_swing_high else None,
                    'swing_low': self.current_swing_low['price'] if self.current_swing_low else None
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
        base_position_pct = 0.12  # 12% base allocation
        
        # Adjust based on signal strength and confidence
        strength_multiplier = signal.strength * signal.confidence
        position_pct = base_position_pct * strength_multiplier
        
        # Adjust based on Fibonacci level importance
        if signal.metadata and 'fib_level' in signal.metadata:
            fib_level = signal.metadata['fib_level']
            if 'fib_0.618' in fib_level:
                position_pct *= 1.3  # Golden ratio is most important
            elif 'fib_0.382' in fib_level:
                position_pct *= 1.2  # Second most important
            elif 'fib_0.500' in fib_level:
                position_pct *= 1.1  # 50% retracement
        
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
            'current_swing_high': self.current_swing_high,
            'current_swing_low': self.current_swing_low,
            'fib_levels_count': len(self.fib_levels_cache),
            'fib_levels': self.fib_levels_cache,
            'trend_direction': self._get_trend_direction(),
            'retracement_valid': self._is_retracement_valid(datetime.now()) if self.is_initialized else False,
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None
        }
        
        base_info.update(strategy_info)
        return base_info 