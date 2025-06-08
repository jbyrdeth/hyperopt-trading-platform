"""
Multi-Timeframe MACD Strategy

This strategy analyzes MACD signals across multiple timeframes to generate
high-quality signals with improved accuracy and reduced false positives.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class MTFMACDStrategy(BaseStrategy):
    """
    Multi-Timeframe MACD Strategy.
    
    This strategy:
    1. Analyzes MACD signals across multiple timeframes (1H, 4H, 1D)
    2. Generates signals when MACD conditions align across timeframes
    3. Uses MACD divergence detection for early signals
    4. Provides MACD strength scoring for position sizing
    5. Includes histogram analysis for momentum confirmation
    """
    
    def __init__(
        self,
        primary_timeframe: str = '1H',
        secondary_timeframe: str = '4H',
        tertiary_timeframe: str = '1D',
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        macd_alignment_threshold: float = 0.7,
        histogram_threshold: float = 0.0001,
        divergence_lookback: int = 20,
        signal_confirmation_bars: int = 2,
        volume_confirmation: bool = True,
        volume_threshold: float = 1.2,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Multi-Timeframe MACD Strategy.
        
        Args:
            primary_timeframe: Primary timeframe for signals ('1H', '4H', '1D')
            secondary_timeframe: Secondary timeframe for confirmation
            tertiary_timeframe: Tertiary timeframe for overall trend
            macd_fast: Fast EMA period (8-15)
            macd_slow: Slow EMA period (20-35)
            macd_signal: Signal line EMA period (7-12)
            macd_alignment_threshold: Minimum alignment score (0.5-1.0)
            histogram_threshold: Minimum histogram value for signals (0.0001-0.001)
            divergence_lookback: Periods to look back for divergence (10-30)
            signal_confirmation_bars: Bars to confirm signal (1-5)
            volume_confirmation: Whether to use volume confirmation
            volume_threshold: Volume multiplier for confirmation (1.0-3.0)
            position_size_pct: Percentage of capital to use per trade
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        # Prepare parameters and risk parameters for base class
        parameters = {
            'primary_timeframe': primary_timeframe,
            'secondary_timeframe': secondary_timeframe,
            'tertiary_timeframe': tertiary_timeframe,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'macd_alignment_threshold': macd_alignment_threshold,
            'histogram_threshold': histogram_threshold,
            'divergence_lookback': divergence_lookback,
            'signal_confirmation_bars': signal_confirmation_bars,
            'volume_confirmation': volume_confirmation,
            'volume_threshold': volume_threshold
        }
        
        risk_params = {
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        super().__init__(
            name=f"MTFMACD_{primary_timeframe}_{secondary_timeframe}_{tertiary_timeframe}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframe = secondary_timeframe
        self.tertiary_timeframe = tertiary_timeframe
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.macd_alignment_threshold = macd_alignment_threshold
        self.histogram_threshold = histogram_threshold
        self.divergence_lookback = divergence_lookback
        self.signal_confirmation_bars = signal_confirmation_bars
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Validate parameters
        if not self.validate_parameters():
            raise ValueError("Invalid strategy parameters")
        
        self.logger = get_logger(f"strategy.{self.name}")
        
        # Timeframe data storage
        self.timeframe_data: Dict[str, pd.DataFrame] = {}
        self.timeframe_indicators: Dict[str, Dict[str, pd.Series]] = {}
        
        # Timeframe mappings (in minutes)
        self.timeframe_minutes = {
            '1H': 60,
            '4H': 240,
            '1D': 1440
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        valid_timeframes = ['1H', '4H', '1D']
        
        if self.primary_timeframe not in valid_timeframes:
            self.logger.error(f"Invalid primary_timeframe: {self.primary_timeframe}")
            return False
        
        if self.secondary_timeframe not in valid_timeframes:
            self.logger.error(f"Invalid secondary_timeframe: {self.secondary_timeframe}")
            return False
        
        if self.tertiary_timeframe not in valid_timeframes:
            self.logger.error(f"Invalid tertiary_timeframe: {self.tertiary_timeframe}")
            return False
        
        if not (8 <= self.macd_fast <= 15):
            self.logger.error(f"MACD fast period {self.macd_fast} outside valid range [8, 15]")
            return False
        
        if not (20 <= self.macd_slow <= 35):
            self.logger.error(f"MACD slow period {self.macd_slow} outside valid range [20, 35]")
            return False
        
        if not (7 <= self.macd_signal <= 12):
            self.logger.error(f"MACD signal period {self.macd_signal} outside valid range [7, 12]")
            return False
        
        if self.macd_fast >= self.macd_slow:
            self.logger.error("MACD fast period must be less than slow period")
            return False
        
        if not (0.5 <= self.macd_alignment_threshold <= 1.0):
            self.logger.error(f"MACD alignment threshold {self.macd_alignment_threshold} outside valid range [0.5, 1.0]")
            return False
        
        if not (0.0001 <= self.histogram_threshold <= 0.001):
            self.logger.error(f"Histogram threshold {self.histogram_threshold} outside valid range [0.0001, 0.001]")
            return False
        
        if not (10 <= self.divergence_lookback <= 30):
            self.logger.error(f"Divergence lookback {self.divergence_lookback} outside valid range [10, 30]")
            return False
        
        if not (1 <= self.signal_confirmation_bars <= 5):
            self.logger.error(f"Signal confirmation bars {self.signal_confirmation_bars} outside valid range [1, 5]")
            return False
        
        if not (1.0 <= self.volume_threshold <= 3.0):
            self.logger.error(f"Volume threshold {self.volume_threshold} outside valid range [1.0, 3.0]")
            return False
        
        return True
    
    def _resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        freq_map = {
            '1H': '1h',
            '4H': '4h', 
            '1D': '1D'
        }
        
        freq = freq_map.get(timeframe, '1h')
        
        resampled = data.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def _calculate_timeframe_indicators(self, timeframe: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate indicators for a specific timeframe."""
        indicators = {}
        
        # MACD
        macd_data = ta.trend.MACD(
            close=data['close'],
            window_fast=self.macd_fast,
            window_slow=self.macd_slow,
            window_sign=self.macd_signal
        )
        
        indicators['macd_line'] = macd_data.macd()
        indicators['macd_signal'] = macd_data.macd_signal()
        indicators['macd_histogram'] = macd_data.macd_diff()
        
        # MACD conditions
        indicators['macd_bullish'] = indicators['macd_line'] > indicators['macd_signal']
        indicators['macd_bearish'] = indicators['macd_line'] < indicators['macd_signal']
        
        # MACD crossovers
        indicators['macd_bullish_crossover'] = (
            (indicators['macd_line'] > indicators['macd_signal']) &
            (indicators['macd_line'].shift(1) <= indicators['macd_signal'].shift(1))
        )
        indicators['macd_bearish_crossover'] = (
            (indicators['macd_line'] < indicators['macd_signal']) &
            (indicators['macd_line'].shift(1) >= indicators['macd_signal'].shift(1))
        )
        
        # Zero line crossovers
        indicators['macd_zero_bullish'] = (
            (indicators['macd_line'] > 0) &
            (indicators['macd_line'].shift(1) <= 0)
        )
        indicators['macd_zero_bearish'] = (
            (indicators['macd_line'] < 0) &
            (indicators['macd_line'].shift(1) >= 0)
        )
        
        # Histogram analysis
        indicators['histogram_increasing'] = indicators['macd_histogram'] > indicators['macd_histogram'].shift(1)
        indicators['histogram_decreasing'] = indicators['macd_histogram'] < indicators['macd_histogram'].shift(1)
        indicators['histogram_positive'] = indicators['macd_histogram'] > 0
        indicators['histogram_negative'] = indicators['macd_histogram'] < 0
        
        # MACD momentum
        indicators['macd_momentum'] = indicators['macd_line'].pct_change(periods=3)
        indicators['signal_momentum'] = indicators['macd_signal'].pct_change(periods=3)
        
        # MACD divergence detection
        indicators['price_highs'] = data['high'].rolling(window=self.divergence_lookback).max()
        indicators['price_lows'] = data['low'].rolling(window=self.divergence_lookback).min()
        indicators['macd_highs'] = indicators['macd_line'].rolling(window=self.divergence_lookback).max()
        indicators['macd_lows'] = indicators['macd_line'].rolling(window=self.divergence_lookback).min()
        
        # Bullish divergence: price makes lower lows, MACD makes higher lows
        indicators['bullish_divergence'] = (
            (data['low'] <= indicators['price_lows'].shift(1)) &
            (indicators['macd_line'] >= indicators['macd_lows'].shift(1)) &
            (indicators['macd_line'] < 0)
        )
        
        # Bearish divergence: price makes higher highs, MACD makes lower highs
        indicators['bearish_divergence'] = (
            (data['high'] >= indicators['price_highs'].shift(1)) &
            (indicators['macd_line'] <= indicators['macd_highs'].shift(1)) &
            (indicators['macd_line'] > 0)
        )
        
        # Volume indicators
        if self.volume_confirmation:
            indicators['volume_ma'] = ta.trend.sma_indicator(data['volume'], window=self.macd_slow)
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        
        return indicators
    
    def _get_macd_alignment_score(self, current_time: datetime) -> Tuple[float, str]:
        """Calculate MACD alignment score across timeframes."""
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        macd_conditions = []
        macd_strengths = []
        
        for tf in timeframes:
            if tf not in self.timeframe_indicators:
                continue
            
            indicators = self.timeframe_indicators[tf]
            
            try:
                # Find the closest timestamp in the timeframe data
                tf_data = self.timeframe_data[tf]
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(indicators['macd_line']):
                    macd_line = indicators['macd_line'].iloc[closest_idx]
                    macd_signal = indicators['macd_signal'].iloc[closest_idx]
                    macd_histogram = indicators['macd_histogram'].iloc[closest_idx]
                    
                    if not (pd.isna(macd_line) or pd.isna(macd_signal) or pd.isna(macd_histogram)):
                        # Determine MACD condition
                        if macd_line > macd_signal and abs(macd_histogram) > self.histogram_threshold:
                            macd_conditions.append('bullish')
                            macd_strengths.append(abs(macd_histogram))
                        elif macd_line < macd_signal and abs(macd_histogram) > self.histogram_threshold:
                            macd_conditions.append('bearish')
                            macd_strengths.append(abs(macd_histogram))
                        else:
                            macd_conditions.append('neutral')
                            macd_strengths.append(0.0)
                            
            except (KeyError, IndexError):
                continue
        
        if len(macd_conditions) < 2:
            return 0.0, 'insufficient_data'
        
        # Calculate alignment score
        # Check if MACD conditions are aligned
        unique_conditions = set(macd_conditions)
        
        if len(unique_conditions) == 1:
            # Perfect alignment
            alignment_score = 1.0
            consensus_condition = macd_conditions[0]
        elif len(unique_conditions) == 2:
            # Partial alignment
            condition_counts = {cond: macd_conditions.count(cond) for cond in unique_conditions}
            dominant_condition = max(condition_counts, key=condition_counts.get)
            alignment_score = condition_counts[dominant_condition] / len(macd_conditions)
            consensus_condition = dominant_condition
        else:
            # No alignment
            alignment_score = 0.0
            consensus_condition = 'mixed'
        
        # Weight alignment score by MACD strength
        if consensus_condition in ['bullish', 'bearish'] and macd_strengths:
            avg_strength = sum(macd_strengths) / len(macd_strengths)
            strength_multiplier = min(avg_strength / self.histogram_threshold, 3.0)
            alignment_score = min(alignment_score * strength_multiplier, 1.0)
        
        return alignment_score, consensus_condition
    
    def _detect_mtf_divergence(self, current_time: datetime) -> Tuple[bool, bool]:
        """Detect divergence across multiple timeframes."""
        bullish_divergences = []
        bearish_divergences = []
        
        timeframes = [self.primary_timeframe, self.secondary_timeframe]
        
        for tf in timeframes:
            if tf not in self.timeframe_indicators:
                continue
            
            indicators = self.timeframe_indicators[tf]
            
            try:
                tf_data = self.timeframe_data[tf]
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(indicators['bullish_divergence']):
                    bullish_div = indicators['bullish_divergence'].iloc[closest_idx]
                    bearish_div = indicators['bearish_divergence'].iloc[closest_idx]
                    
                    if not pd.isna(bullish_div):
                        bullish_divergences.append(bullish_div)
                    if not pd.isna(bearish_div):
                        bearish_divergences.append(bearish_div)
                        
            except (KeyError, IndexError):
                continue
        
        # Return True if divergence detected in any timeframe
        has_bullish_divergence = any(bullish_divergences)
        has_bearish_divergence = any(bearish_divergences)
        
        return has_bullish_divergence, has_bearish_divergence
    
    def _detect_mtf_crossovers(self, current_time: datetime) -> Tuple[bool, bool]:
        """Detect MACD crossovers across multiple timeframes."""
        bullish_crossovers = []
        bearish_crossovers = []
        
        timeframes = [self.primary_timeframe, self.secondary_timeframe]
        
        for tf in timeframes:
            if tf not in self.timeframe_indicators:
                continue
            
            indicators = self.timeframe_indicators[tf]
            
            try:
                tf_data = self.timeframe_data[tf]
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(indicators['macd_bullish_crossover']):
                    bullish_cross = indicators['macd_bullish_crossover'].iloc[closest_idx]
                    bearish_cross = indicators['macd_bearish_crossover'].iloc[closest_idx]
                    
                    if not pd.isna(bullish_cross):
                        bullish_crossovers.append(bullish_cross)
                    if not pd.isna(bearish_cross):
                        bearish_crossovers.append(bearish_cross)
                        
            except (KeyError, IndexError):
                continue
        
        # Return True if crossover detected in any timeframe
        has_bullish_crossover = any(bullish_crossovers)
        has_bearish_crossover = any(bearish_crossovers)
        
        return has_bullish_crossover, has_bearish_crossover
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the original data
        self.data = data
        
        min_periods = max(self.macd_slow + self.macd_signal + self.divergence_lookback, 100)
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Resample data for each timeframe
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        
        for tf in timeframes:
            try:
                resampled_data = self._resample_data(data, tf)
                
                if len(resampled_data) < self.macd_slow + self.macd_signal + self.divergence_lookback:
                    self.logger.warning(f"Insufficient data for {tf} timeframe")
                    continue
                
                self.timeframe_data[tf] = resampled_data
                self.timeframe_indicators[tf] = self._calculate_timeframe_indicators(tf, resampled_data)
                
                self.logger.debug(f"Initialized {tf} timeframe with {len(resampled_data)} periods")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {tf} timeframe: {e}")
                continue
        
        # Mark as initialized if we have at least 2 timeframes
        if len(self.timeframe_data) >= 2:
            self.is_initialized = True
            self.logger.info(f"Initialized {self.name} with {len(self.timeframe_data)} timeframes")
        else:
            self.is_initialized = False
            self.logger.error("Failed to initialize - need at least 2 timeframes")
    
    @property
    def required_periods(self) -> int:
        """Get the minimum number of periods required for strategy initialization."""
        return max(self.macd_slow + self.macd_signal + self.divergence_lookback, 100)
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on multi-timeframe MACD analysis."""
        if not self.is_initialized:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_data['close'],
                confidence=0.0,
                metadata={"reason": "Strategy not initialized"}
            )
        
        current_price = current_data['close']
        current_volume = current_data['volume']
        
        # Get MACD alignment score and consensus
        alignment_score, consensus_condition = self._get_macd_alignment_score(timestamp)
        
        # Detect divergences and crossovers
        bullish_divergence, bearish_divergence = self._detect_mtf_divergence(timestamp)
        bullish_crossover, bearish_crossover = self._detect_mtf_crossovers(timestamp)
        
        # Check if alignment meets threshold or we have strong signals
        strong_signals = bullish_divergence or bearish_divergence or bullish_crossover or bearish_crossover
        
        if alignment_score < self.macd_alignment_threshold and not strong_signals:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={
                    "reason": "Insufficient MACD alignment and no strong signals",
                    "alignment_score": alignment_score,
                    "consensus_condition": consensus_condition,
                    "bullish_divergence": bullish_divergence,
                    "bearish_divergence": bearish_divergence,
                    "bullish_crossover": bullish_crossover,
                    "bearish_crossover": bearish_crossover
                }
            )
        
        # Volume confirmation
        volume_confirmed = True
        if self.volume_confirmation:
            primary_indicators = self.timeframe_indicators.get(self.primary_timeframe, {})
            if 'volume_ratio' in primary_indicators:
                try:
                    # Get current volume ratio
                    primary_data = self.timeframe_data[self.primary_timeframe]
                    closest_idx = primary_data.index.get_indexer([timestamp], method='nearest')[0]
                    
                    if closest_idx >= 0 and closest_idx < len(primary_indicators['volume_ratio']):
                        volume_ratio = primary_indicators['volume_ratio'].iloc[closest_idx]
                        volume_confirmed = volume_ratio >= self.volume_threshold
                except (KeyError, IndexError):
                    volume_confirmed = True  # Default to confirmed if can't calculate
        
        # Generate signal based on MACD conditions, divergences, and crossovers
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "alignment_score": alignment_score,
            "consensus_condition": consensus_condition,
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
            "bullish_crossover": bullish_crossover,
            "bearish_crossover": bearish_crossover,
            "volume_confirmed": volume_confirmed
        }
        
        if volume_confirmed:
            # Priority 1: Divergence signals (highest confidence)
            if bullish_divergence and consensus_condition in ['bullish', 'neutral']:
                action = "buy"
                strength = 0.9
                confidence = 0.95
                metadata["signal_type"] = "mtf_macd_bullish_divergence"
                
            elif bearish_divergence and consensus_condition in ['bearish', 'neutral']:
                action = "sell"
                strength = 0.9
                confidence = 0.95
                metadata["signal_type"] = "mtf_macd_bearish_divergence"
                
            # Priority 2: Crossover signals (high confidence)
            elif bullish_crossover and consensus_condition in ['bullish', 'neutral']:
                action = "buy"
                strength = 0.8
                confidence = 0.85
                metadata["signal_type"] = "mtf_macd_bullish_crossover"
                
            elif bearish_crossover and consensus_condition in ['bearish', 'neutral']:
                action = "sell"
                strength = 0.8
                confidence = 0.85
                metadata["signal_type"] = "mtf_macd_bearish_crossover"
                
            # Priority 3: Aligned MACD conditions (medium confidence)
            elif alignment_score >= self.macd_alignment_threshold:
                if consensus_condition == 'bullish':
                    action = "buy"
                    strength = alignment_score * 0.7
                    confidence = alignment_score
                    metadata["signal_type"] = "mtf_macd_bullish_alignment"
                    
                elif consensus_condition == 'bearish':
                    action = "sell"
                    strength = alignment_score * 0.7
                    confidence = alignment_score
                    metadata["signal_type"] = "mtf_macd_bearish_alignment"
        
        # Get current MACD values for metadata
        try:
            primary_indicators = self.timeframe_indicators.get(self.primary_timeframe, {})
            if 'macd_line' in primary_indicators:
                primary_data = self.timeframe_data[self.primary_timeframe]
                closest_idx = primary_data.index.get_indexer([timestamp], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(primary_indicators['macd_line']):
                    current_macd = primary_indicators['macd_line'].iloc[closest_idx]
                    current_signal = primary_indicators['macd_signal'].iloc[closest_idx]
                    current_histogram = primary_indicators['macd_histogram'].iloc[closest_idx]
                    
                    metadata["current_macd"] = float(current_macd) if not pd.isna(current_macd) else None
                    metadata["current_signal"] = float(current_signal) if not pd.isna(current_signal) else None
                    metadata["current_histogram"] = float(current_histogram) if not pd.isna(current_histogram) else None
        except (KeyError, IndexError):
            pass
        
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
            'primary_timeframe': hp.choice('primary_timeframe', ['1H', '4H']),
            'secondary_timeframe': hp.choice('secondary_timeframe', ['4H', '1D']),
            'tertiary_timeframe': hp.choice('tertiary_timeframe', ['1D']),
            'macd_fast': hp.choice('macd_fast', list(range(8, 16))),
            'macd_slow': hp.choice('macd_slow', list(range(20, 36))),
            'macd_signal': hp.choice('macd_signal', list(range(7, 13))),
            'macd_alignment_threshold': hp.uniform('macd_alignment_threshold', 0.5, 1.0),
            'histogram_threshold': hp.uniform('histogram_threshold', 0.0001, 0.001),
            'divergence_lookback': hp.choice('divergence_lookback', list(range(10, 31))),
            'signal_confirmation_bars': hp.choice('signal_confirmation_bars', list(range(1, 6))),
            'volume_confirmation': hp.choice('volume_confirmation', [True, False]),
            'volume_threshold': hp.uniform('volume_threshold', 1.0, 3.0),
            'position_size_pct': hp.uniform('position_size_pct', 0.8, 1.0),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.08)
        }
    
    def get_current_indicators(self) -> Dict[str, float]:
        """Get current indicator values for analysis."""
        if not self.is_initialized:
            return {}
        
        indicators = {}
        
        for tf in [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]:
            if tf in self.timeframe_indicators:
                tf_indicators = self.timeframe_indicators[tf]
                
                for indicator_name in ['macd_line', 'macd_signal', 'macd_histogram', 'macd_momentum']:
                    if indicator_name in tf_indicators and len(tf_indicators[indicator_name]) > 0:
                        value = tf_indicators[indicator_name].iloc[-1]
                        indicators[f'{tf}_{indicator_name}'] = float(value) if not pd.isna(value) else 0.0
        
        return indicators
    
    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        available_capital: float,
        current_volatility: float
    ) -> float:
        """
        Calculate position size based on signal strength and MACD analysis.
        
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
        
        # MACD-specific adjustments
        signal_type = signal.metadata.get('signal_type', '')
        alignment_score = signal.metadata.get('alignment_score', 0.5)
        
        # Increase size for divergence signals (highest confidence)
        if 'divergence' in signal_type:
            adjusted_size = adjusted_size * 1.5
        # Increase size for crossover signals (high confidence)
        elif 'crossover' in signal_type:
            adjusted_size = adjusted_size * 1.3
        
        # Increase size for high alignment
        if alignment_score > 0.8:
            adjusted_size = adjusted_size * 1.2
        elif alignment_score > 0.6:
            adjusted_size = adjusted_size * 1.1
        
        # Adjust based on current MACD histogram strength
        current_histogram = signal.metadata.get('current_histogram')
        if current_histogram is not None:
            histogram_strength = abs(current_histogram) / self.histogram_threshold
            if histogram_strength > 3.0:  # Very strong histogram
                adjusted_size = adjusted_size * 1.3
            elif histogram_strength > 2.0:  # Strong histogram
                adjusted_size = adjusted_size * 1.1
        
        # Adjust based on volatility (reduce size in high volatility)
        volatility_adjustment = max(0.5, 1.0 - (current_volatility * 2))
        final_size = adjusted_size * volatility_adjustment
        
        # Apply direction
        if signal.action == 'sell':
            final_size = -final_size
        
        return final_size
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        return f"MTFMACD({self.primary_timeframe}, {self.secondary_timeframe}, {self.tertiary_timeframe})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MTFMACDStrategy(primary={self.primary_timeframe}, secondary={self.secondary_timeframe}, tertiary={self.tertiary_timeframe})" 