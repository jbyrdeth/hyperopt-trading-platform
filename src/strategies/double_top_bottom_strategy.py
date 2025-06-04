"""
Double Top/Bottom Pattern Recognition Strategy

This strategy identifies classic double top and double bottom reversal patterns
and generates trading signals based on pattern completion and confirmation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

from .base_strategy import BaseStrategy, Signal, Position


class DoubleTopBottomStrategy(BaseStrategy):
    """
    Double Top/Bottom pattern recognition strategy.
    
    This strategy:
    1. Identifies potential double top and double bottom patterns
    2. Validates patterns using price action and volume confirmation
    3. Generates sell signals on double top pattern completion
    4. Generates buy signals on double bottom pattern completion
    5. Uses neckline breaks for pattern confirmation
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize Double Top/Bottom strategy."""
        default_params = {
            'lookback_period': 30,              # Period for pattern detection
            'peak_similarity_threshold': 0.02,  # Max difference between peaks (2%)
            'min_pattern_height': 0.05,         # Minimum pattern height (5%)
            'min_valley_depth': 0.02,           # Minimum valley depth between peaks (2%)
            'max_pattern_width': 100,           # Maximum pattern width in bars
            'min_pattern_width': 20,            # Minimum pattern width in bars
            'neckline_break_threshold': 0.005,  # Neckline break confirmation (0.5%)
            'volume_confirmation': True,        # Use volume for pattern confirmation
            'volume_decline_threshold': 0.8,    # Volume decline on second peak/trough
            'pattern_timeout': 50,              # Pattern timeout in bars
            'confirmation_bars': 2,             # Bars to confirm neckline break
            'trend_filter': True,               # Use trend filter
            'trend_period': 50,                 # Period for trend determination
            'signal_strength_threshold': 0.5    # Minimum signal strength
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("DoubleTopBottom", default_params)
        
        # Strategy state
        self.potential_patterns: List[Dict[str, Any]] = []
        self.confirmed_patterns: List[Dict[str, Any]] = []
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown = 5  # Minimum bars between signals
        
    @property
    def required_periods(self) -> int:
        """Return the minimum number of periods required for the strategy."""
        return max(
            self.parameters['max_pattern_width'] * 2,
            self.parameters['trend_period'] * 2,
            100  # Minimum for reliable pattern detection
        )
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.parameters['lookback_period'] < 10:
            self.logger.error("lookback_period must be at least 10")
            return False
            
        if self.parameters['peak_similarity_threshold'] <= 0 or self.parameters['peak_similarity_threshold'] > 0.1:
            self.logger.error("peak_similarity_threshold must be between 0 and 0.1")
            return False
            
        if self.parameters['min_pattern_height'] <= 0 or self.parameters['min_pattern_height'] > 0.2:
            self.logger.error("min_pattern_height must be between 0 and 0.2")
            return False
            
        if self.parameters['min_pattern_width'] >= self.parameters['max_pattern_width']:
            self.logger.error("min_pattern_width must be less than max_pattern_width")
            return False
            
        return True
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize the strategy with historical data."""
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.data = data.copy()
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(data)
        
        # Initialize pattern detection
        self._initialize_pattern_detection()
        
        self.is_initialized = True
        self.logger.info(f"Initialized {self.name} strategy with {len(data)} data points")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate technical indicators."""
        indicators = {}
        
        # Volume moving average for confirmation
        if self.parameters['volume_confirmation']:
            indicators['volume_ma'] = data['volume'].rolling(
                window=self.parameters['lookback_period']
            ).mean()
        
        # Trend indicators
        if self.parameters['trend_filter']:
            indicators['trend_ma'] = data['close'].rolling(
                window=self.parameters['trend_period']
            ).mean()
            
            # Trend direction
            indicators['trend_direction'] = np.where(
                data['close'] > indicators['trend_ma'], 1, -1
            )
        
        # Price volatility for pattern validation
        indicators['volatility'] = data['close'].rolling(
            window=self.parameters['lookback_period']
        ).std() / data['close'].rolling(window=self.parameters['lookback_period']).mean()
        
        # Support and resistance levels
        indicators['resistance'] = data['high'].rolling(
            window=self.parameters['lookback_period']
        ).max()
        indicators['support'] = data['low'].rolling(
            window=self.parameters['lookback_period']
        ).min()
        
        return indicators
    
    def _find_peaks_and_troughs(self, data: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Find peaks and troughs in the price data."""
        highs = data['high'].values
        lows = data['low'].values
        
        # Find local maxima (peaks)
        peak_indices = argrelextrema(
            highs,
            np.greater,
            order=self.parameters['lookback_period'] // 3
        )[0]
        
        # Find local minima (troughs)
        trough_indices = argrelextrema(
            lows,
            np.less,
            order=self.parameters['lookback_period'] // 3
        )[0]
        
        peaks = []
        troughs = []
        
        # Create peak objects
        for idx in peak_indices:
            if idx < len(data):
                peak = {
                    'price': highs[idx],
                    'timestamp': data.index[idx],
                    'index': idx,
                    'volume': data.iloc[idx]['volume'],
                    'type': 'peak'
                }
                peaks.append(peak)
        
        # Create trough objects
        for idx in trough_indices:
            if idx < len(data):
                trough = {
                    'price': lows[idx],
                    'timestamp': data.index[idx],
                    'index': idx,
                    'volume': data.iloc[idx]['volume'],
                    'type': 'trough'
                }
                troughs.append(trough)
        
        return peaks, troughs
    
    def _initialize_pattern_detection(self) -> None:
        """Initialize pattern detection from historical data."""
        if len(self.data) < self.required_periods:
            return
        
        # Find peaks and troughs
        peaks, troughs = self._find_peaks_and_troughs(self.data)
        
        # Look for potential double top patterns
        self._detect_double_top_patterns(peaks, troughs)
        
        # Look for potential double bottom patterns
        self._detect_double_bottom_patterns(peaks, troughs)
        
        self.logger.info(f"Initialized with {len(self.potential_patterns)} potential patterns")
    
    def _detect_double_top_patterns(self, peaks: List[Dict], troughs: List[Dict]) -> None:
        """Detect potential double top patterns."""
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1 = peaks[i]
                peak2 = peaks[j]
                
                # Check pattern width constraints
                pattern_width = peak2['index'] - peak1['index']
                if (pattern_width < self.parameters['min_pattern_width'] or 
                    pattern_width > self.parameters['max_pattern_width']):
                    continue
                
                # Check peak similarity
                price_diff = abs(peak1['price'] - peak2['price']) / peak1['price']
                if price_diff > self.parameters['peak_similarity_threshold']:
                    continue
                
                # Find valley between peaks
                valley = self._find_valley_between_points(peak1, peak2, troughs)
                if not valley:
                    continue
                
                # Validate pattern height
                pattern_height = min(peak1['price'], peak2['price']) - valley['price']
                height_ratio = pattern_height / valley['price']
                if height_ratio < self.parameters['min_pattern_height']:
                    continue
                
                # Validate valley depth
                valley_depth = (min(peak1['price'], peak2['price']) - valley['price']) / min(peak1['price'], peak2['price'])
                if valley_depth < self.parameters['min_valley_depth']:
                    continue
                
                # Create pattern object
                pattern = {
                    'type': 'double_top',
                    'peak1': peak1,
                    'peak2': peak2,
                    'valley': valley,
                    'neckline': valley['price'],
                    'pattern_height': pattern_height,
                    'pattern_width': pattern_width,
                    'created_time': peak2['timestamp'],
                    'confirmed': False,
                    'target_price': valley['price'] - pattern_height,
                    'stop_loss': max(peak1['price'], peak2['price'])
                }
                
                self.potential_patterns.append(pattern)
    
    def _detect_double_bottom_patterns(self, peaks: List[Dict], troughs: List[Dict]) -> None:
        """Detect potential double bottom patterns."""
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                trough1 = troughs[i]
                trough2 = troughs[j]
                
                # Check pattern width constraints
                pattern_width = trough2['index'] - trough1['index']
                if (pattern_width < self.parameters['min_pattern_width'] or 
                    pattern_width > self.parameters['max_pattern_width']):
                    continue
                
                # Check trough similarity
                price_diff = abs(trough1['price'] - trough2['price']) / trough1['price']
                if price_diff > self.parameters['peak_similarity_threshold']:
                    continue
                
                # Find peak between troughs
                peak = self._find_peak_between_points(trough1, trough2, peaks)
                if not peak:
                    continue
                
                # Validate pattern height
                pattern_height = peak['price'] - max(trough1['price'], trough2['price'])
                height_ratio = pattern_height / max(trough1['price'], trough2['price'])
                if height_ratio < self.parameters['min_pattern_height']:
                    continue
                
                # Validate peak height
                peak_height = (peak['price'] - max(trough1['price'], trough2['price'])) / max(trough1['price'], trough2['price'])
                if peak_height < self.parameters['min_valley_depth']:
                    continue
                
                # Create pattern object
                pattern = {
                    'type': 'double_bottom',
                    'trough1': trough1,
                    'trough2': trough2,
                    'peak': peak,
                    'neckline': peak['price'],
                    'pattern_height': pattern_height,
                    'pattern_width': pattern_width,
                    'created_time': trough2['timestamp'],
                    'confirmed': False,
                    'target_price': peak['price'] + pattern_height,
                    'stop_loss': min(trough1['price'], trough2['price'])
                }
                
                self.potential_patterns.append(pattern)
    
    def _find_valley_between_points(self, point1: Dict, point2: Dict, troughs: List[Dict]) -> Optional[Dict]:
        """Find the lowest trough between two peaks."""
        candidates = [
            trough for trough in troughs
            if point1['index'] < trough['index'] < point2['index']
        ]
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda x: x['price'])
    
    def _find_peak_between_points(self, point1: Dict, point2: Dict, peaks: List[Dict]) -> Optional[Dict]:
        """Find the highest peak between two troughs."""
        candidates = [
            peak for peak in peaks
            if point1['index'] < peak['index'] < point2['index']
        ]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x['price'])
    
    def _validate_pattern_volume(self, pattern: Dict[str, Any]) -> bool:
        """Validate pattern using volume analysis."""
        if not self.parameters['volume_confirmation']:
            return True
        
        if pattern['type'] == 'double_top':
            # Volume should decline on second peak
            vol1 = pattern['peak1']['volume']
            vol2 = pattern['peak2']['volume']
            return vol2 < vol1 * self.parameters['volume_decline_threshold']
        
        else:  # double_bottom
            # Volume should increase on second trough
            vol1 = pattern['trough1']['volume']
            vol2 = pattern['trough2']['volume']
            return vol2 > vol1 / self.parameters['volume_decline_threshold']
    
    def _check_neckline_break(self, pattern: Dict[str, Any], current_price: float) -> bool:
        """Check if neckline has been broken."""
        neckline = pattern['neckline']
        threshold = self.parameters['neckline_break_threshold']
        
        if pattern['type'] == 'double_top':
            # Price should break below neckline
            return current_price < neckline * (1 - threshold)
        else:  # double_bottom
            # Price should break above neckline
            return current_price > neckline * (1 + threshold)
    
    def _update_patterns(self, current_time: datetime, current_data: pd.Series) -> None:
        """Update pattern status and remove expired patterns."""
        current_price = current_data['close']
        
        # Remove expired patterns
        self.potential_patterns = [
            pattern for pattern in self.potential_patterns
            if (current_time - pattern['created_time']).total_seconds() / 3600 <= self.parameters['pattern_timeout']
        ]
        
        # Check for pattern confirmations
        for pattern in self.potential_patterns:
            if not pattern['confirmed']:
                # Check volume validation
                if self._validate_pattern_volume(pattern):
                    # Check neckline break
                    if self._check_neckline_break(pattern, current_price):
                        pattern['confirmed'] = True
                        pattern['confirmation_time'] = current_time
                        pattern['confirmation_price'] = current_price
                        self.confirmed_patterns.append(pattern)
                        
                        self.logger.info(f"Pattern confirmed: {pattern['type']} at {current_price}")
    
    def _get_signal_strength(
        self,
        pattern: Dict[str, Any],
        current_price: float,
        current_volume: float
    ) -> float:
        """Calculate signal strength based on pattern characteristics."""
        # Base strength from pattern height
        height_factor = min(pattern['pattern_height'] / current_price, 0.2) * 5  # Normalize to 0-1
        
        # Volume confirmation factor
        volume_factor = 1.0
        if self.parameters['volume_confirmation'] and 'volume_ma' in self.indicators:
            avg_volume = self.indicators['volume_ma'].iloc[-1]
            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_factor = min(volume_ratio / 1.5, 1.5)
        
        # Pattern width factor (optimal width gets higher score)
        optimal_width = (self.parameters['min_pattern_width'] + self.parameters['max_pattern_width']) / 2
        width_deviation = abs(pattern['pattern_width'] - optimal_width) / optimal_width
        width_factor = max(0.5, 1 - width_deviation)
        
        # Trend alignment factor
        trend_factor = 1.0
        if self.parameters['trend_filter'] and 'trend_direction' in self.indicators:
            trend_dir = self.indicators['trend_direction'].iloc[-1]
            if pattern['type'] == 'double_top' and trend_dir == -1:
                trend_factor = 1.3  # Bearish pattern in downtrend
            elif pattern['type'] == 'double_bottom' and trend_dir == 1:
                trend_factor = 1.3  # Bullish pattern in uptrend
        
        # Combine factors
        signal_strength = height_factor * volume_factor * width_factor * trend_factor
        
        return min(signal_strength, 1.0)
    
    def generate_signal(self, current_time: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on double top/bottom patterns."""
        if not self.is_initialized:
            return Signal(
                timestamp=current_time,
                action='hold',
                strength=0.0,
                price=current_data['close'],
                confidence=0.0
            )
        
        # Update patterns
        self._update_patterns(current_time, current_data)
        
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
        
        # Look for newly confirmed patterns
        for pattern in self.confirmed_patterns:
            # Check if this is a fresh confirmation
            if (hasattr(pattern, 'confirmation_time') and 
                pattern['confirmation_time'] == current_time):
                
                # Calculate signal strength
                signal_strength = self._get_signal_strength(pattern, current_price, current_volume)
                
                if signal_strength < self.parameters['signal_strength_threshold']:
                    continue
                
                # Determine signal action
                if pattern['type'] == 'double_top':
                    action = 'sell'
                else:  # double_bottom
                    action = 'buy'
                
                confidence = signal_strength
                
                self.last_signal_time = current_time
                self.signals_generated += 1
                
                return Signal(
                    timestamp=current_time,
                    action=action,
                    strength=signal_strength,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'signal_type': f'{pattern["type"]}_confirmation',
                        'pattern_type': pattern['type'],
                        'neckline': pattern['neckline'],
                        'target_price': pattern['target_price'],
                        'stop_loss': pattern['stop_loss'],
                        'pattern_height': pattern['pattern_height'],
                        'pattern_width': pattern['pattern_width'],
                        'confirmation_price': pattern['confirmation_price']
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
        base_position_pct = 0.15  # 15% base allocation for pattern signals
        
        # Adjust based on signal strength and confidence
        strength_multiplier = signal.strength * signal.confidence
        position_pct = base_position_pct * strength_multiplier
        
        # Adjust based on pattern characteristics
        if signal.metadata:
            pattern_height = signal.metadata.get('pattern_height', 0)
            if pattern_height > 0:
                # Larger patterns get larger positions
                height_multiplier = min(pattern_height / current_price * 10, 1.5)
                position_pct *= height_multiplier
        
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
            'potential_patterns_count': len(self.potential_patterns),
            'confirmed_patterns_count': len(self.confirmed_patterns),
            'double_top_patterns': len([p for p in self.potential_patterns if p['type'] == 'double_top']),
            'double_bottom_patterns': len([p for p in self.potential_patterns if p['type'] == 'double_bottom']),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'recent_patterns': [
                {
                    'type': p['type'],
                    'created_time': p['created_time'].isoformat(),
                    'confirmed': p['confirmed'],
                    'neckline': p['neckline']
                }
                for p in self.potential_patterns[-5:]  # Last 5 patterns
            ]
        }
        
        base_info.update(strategy_info)
        return base_info 