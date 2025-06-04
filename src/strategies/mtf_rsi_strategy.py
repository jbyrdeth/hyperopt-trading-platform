"""
Multi-Timeframe RSI Strategy

This strategy analyzes RSI values across multiple timeframes to generate
high-quality signals with improved accuracy and reduced false positives.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import ta

from .base_strategy import BaseStrategy, Signal
from utils.logger import get_logger


class MTFRSIStrategy(BaseStrategy):
    """
    Multi-Timeframe RSI Strategy.
    
    This strategy:
    1. Analyzes RSI values across multiple timeframes (1H, 4H, 1D)
    2. Generates signals when RSI conditions align across timeframes
    3. Uses RSI divergence detection for early signals
    4. Provides RSI strength scoring for position sizing
    5. Includes overbought/oversold confirmation across timeframes
    """
    
    def __init__(
        self,
        primary_timeframe: str = '1H',
        secondary_timeframe: str = '4H',
        tertiary_timeframe: str = '1D',
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        rsi_alignment_threshold: float = 0.7,
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
        Initialize Multi-Timeframe RSI Strategy.
        
        Args:
            primary_timeframe: Primary timeframe for signals ('1H', '4H', '1D')
            secondary_timeframe: Secondary timeframe for confirmation
            tertiary_timeframe: Tertiary timeframe for overall trend
            rsi_period: RSI calculation period (10-25)
            rsi_overbought: RSI overbought level (65-85)
            rsi_oversold: RSI oversold level (15-35)
            rsi_alignment_threshold: Minimum alignment score (0.5-1.0)
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
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'rsi_alignment_threshold': rsi_alignment_threshold,
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
            name=f"MTFRSI_{primary_timeframe}_{secondary_timeframe}_{tertiary_timeframe}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframe = secondary_timeframe
        self.tertiary_timeframe = tertiary_timeframe
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.rsi_alignment_threshold = rsi_alignment_threshold
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
        
        if not (10 <= self.rsi_period <= 25):
            self.logger.error(f"RSI period {self.rsi_period} outside valid range [10, 25]")
            return False
        
        if not (65 <= self.rsi_overbought <= 85):
            self.logger.error(f"RSI overbought {self.rsi_overbought} outside valid range [65, 85]")
            return False
        
        if not (15 <= self.rsi_oversold <= 35):
            self.logger.error(f"RSI oversold {self.rsi_oversold} outside valid range [15, 35]")
            return False
        
        if self.rsi_oversold >= self.rsi_overbought:
            self.logger.error("RSI oversold level must be less than overbought level")
            return False
        
        if not (0.5 <= self.rsi_alignment_threshold <= 1.0):
            self.logger.error(f"RSI alignment threshold {self.rsi_alignment_threshold} outside valid range [0.5, 1.0]")
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
        
        # RSI
        indicators['rsi'] = ta.momentum.rsi(data['close'], window=self.rsi_period)
        
        # RSI conditions
        indicators['rsi_overbought'] = indicators['rsi'] >= self.rsi_overbought
        indicators['rsi_oversold'] = indicators['rsi'] <= self.rsi_oversold
        indicators['rsi_neutral'] = (indicators['rsi'] > self.rsi_oversold) & (indicators['rsi'] < self.rsi_overbought)
        
        # RSI momentum (rate of change)
        indicators['rsi_momentum'] = indicators['rsi'].pct_change(periods=3)
        
        # RSI moving average for smoothing
        indicators['rsi_ma'] = ta.trend.sma_indicator(indicators['rsi'], window=5)
        
        # RSI divergence detection
        indicators['price_highs'] = data['high'].rolling(window=self.divergence_lookback).max()
        indicators['price_lows'] = data['low'].rolling(window=self.divergence_lookback).min()
        indicators['rsi_highs'] = indicators['rsi'].rolling(window=self.divergence_lookback).max()
        indicators['rsi_lows'] = indicators['rsi'].rolling(window=self.divergence_lookback).min()
        
        # Bullish divergence: price makes lower lows, RSI makes higher lows
        indicators['bullish_divergence'] = (
            (data['low'] <= indicators['price_lows'].shift(1)) &
            (indicators['rsi'] >= indicators['rsi_lows'].shift(1)) &
            (indicators['rsi'] < 50)
        )
        
        # Bearish divergence: price makes higher highs, RSI makes lower highs
        indicators['bearish_divergence'] = (
            (data['high'] >= indicators['price_highs'].shift(1)) &
            (indicators['rsi'] <= indicators['rsi_highs'].shift(1)) &
            (indicators['rsi'] > 50)
        )
        
        # Volume indicators
        if self.volume_confirmation:
            indicators['volume_ma'] = ta.trend.sma_indicator(data['volume'], window=self.rsi_period)
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        
        return indicators
    
    def _get_rsi_alignment_score(self, current_time: datetime) -> Tuple[float, str]:
        """Calculate RSI alignment score across timeframes."""
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        rsi_values = []
        rsi_conditions = []
        
        for tf in timeframes:
            if tf not in self.timeframe_indicators:
                continue
            
            indicators = self.timeframe_indicators[tf]
            
            try:
                # Find the closest timestamp in the timeframe data
                tf_data = self.timeframe_data[tf]
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(indicators['rsi']):
                    rsi_value = indicators['rsi'].iloc[closest_idx]
                    
                    if not pd.isna(rsi_value):
                        rsi_values.append(rsi_value)
                        
                        # Determine RSI condition
                        if rsi_value >= self.rsi_overbought:
                            rsi_conditions.append('overbought')
                        elif rsi_value <= self.rsi_oversold:
                            rsi_conditions.append('oversold')
                        else:
                            rsi_conditions.append('neutral')
                            
            except (KeyError, IndexError):
                continue
        
        if len(rsi_values) < 2:
            return 0.0, 'insufficient_data'
        
        # Calculate alignment score
        # Check if RSI conditions are aligned
        unique_conditions = set(rsi_conditions)
        
        if len(unique_conditions) == 1:
            # Perfect alignment
            alignment_score = 1.0
            consensus_condition = rsi_conditions[0]
        elif len(unique_conditions) == 2:
            # Partial alignment
            condition_counts = {cond: rsi_conditions.count(cond) for cond in unique_conditions}
            dominant_condition = max(condition_counts, key=condition_counts.get)
            alignment_score = condition_counts[dominant_condition] / len(rsi_conditions)
            consensus_condition = dominant_condition
        else:
            # No alignment
            alignment_score = 0.0
            consensus_condition = 'mixed'
        
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
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the original data
        self.data = data
        
        min_periods = max(self.rsi_period + self.divergence_lookback, 100)
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Resample data for each timeframe
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        
        for tf in timeframes:
            try:
                resampled_data = self._resample_data(data, tf)
                
                if len(resampled_data) < self.rsi_period + self.divergence_lookback:
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
        return max(self.rsi_period + self.divergence_lookback, 100)
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on multi-timeframe RSI analysis."""
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
        
        # Get RSI alignment score and consensus
        alignment_score, consensus_condition = self._get_rsi_alignment_score(timestamp)
        
        # Detect divergences
        bullish_divergence, bearish_divergence = self._detect_mtf_divergence(timestamp)
        
        # Check if alignment meets threshold
        if alignment_score < self.rsi_alignment_threshold and not (bullish_divergence or bearish_divergence):
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={
                    "reason": "Insufficient RSI alignment and no divergence",
                    "alignment_score": alignment_score,
                    "consensus_condition": consensus_condition,
                    "bullish_divergence": bullish_divergence,
                    "bearish_divergence": bearish_divergence
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
        
        # Generate signal based on RSI conditions and divergences
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "alignment_score": alignment_score,
            "consensus_condition": consensus_condition,
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
            "volume_confirmed": volume_confirmed
        }
        
        if volume_confirmed:
            # Priority 1: Divergence signals (stronger)
            if bullish_divergence and consensus_condition in ['oversold', 'neutral']:
                action = "buy"
                strength = 0.8
                confidence = 0.9
                metadata["signal_type"] = "mtf_rsi_bullish_divergence"
                
            elif bearish_divergence and consensus_condition in ['overbought', 'neutral']:
                action = "sell"
                strength = 0.8
                confidence = 0.9
                metadata["signal_type"] = "mtf_rsi_bearish_divergence"
                
            # Priority 2: Aligned RSI conditions
            elif alignment_score >= self.rsi_alignment_threshold:
                if consensus_condition == 'oversold':
                    action = "buy"
                    strength = alignment_score * 0.7
                    confidence = alignment_score
                    metadata["signal_type"] = "mtf_rsi_oversold"
                    
                elif consensus_condition == 'overbought':
                    action = "sell"
                    strength = alignment_score * 0.7
                    confidence = alignment_score
                    metadata["signal_type"] = "mtf_rsi_overbought"
        
        # Get current RSI values for metadata
        try:
            primary_indicators = self.timeframe_indicators.get(self.primary_timeframe, {})
            if 'rsi' in primary_indicators:
                primary_data = self.timeframe_data[self.primary_timeframe]
                closest_idx = primary_data.index.get_indexer([timestamp], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(primary_indicators['rsi']):
                    current_rsi = primary_indicators['rsi'].iloc[closest_idx]
                    metadata["current_rsi"] = float(current_rsi) if not pd.isna(current_rsi) else None
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
            'rsi_period': hp.choice('rsi_period', list(range(10, 26))),
            'rsi_overbought': hp.uniform('rsi_overbought', 65, 85),
            'rsi_oversold': hp.uniform('rsi_oversold', 15, 35),
            'rsi_alignment_threshold': hp.uniform('rsi_alignment_threshold', 0.5, 1.0),
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
                
                for indicator_name in ['rsi', 'rsi_momentum']:
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
        Calculate position size based on signal strength and RSI analysis.
        
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
        
        # RSI-specific adjustments
        signal_type = signal.metadata.get('signal_type', '')
        alignment_score = signal.metadata.get('alignment_score', 0.5)
        
        # Increase size for divergence signals (higher confidence)
        if 'divergence' in signal_type:
            adjusted_size = adjusted_size * 1.4
        
        # Increase size for high alignment
        if alignment_score > 0.8:
            adjusted_size = adjusted_size * 1.2
        elif alignment_score > 0.6:
            adjusted_size = adjusted_size * 1.1
        
        # Adjust based on current RSI extremes
        current_rsi = signal.metadata.get('current_rsi')
        if current_rsi is not None:
            if current_rsi <= 20 or current_rsi >= 80:  # Extreme levels
                adjusted_size = adjusted_size * 1.3
            elif current_rsi <= 25 or current_rsi >= 75:  # Strong levels
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
        return f"MTFRSI({self.primary_timeframe}, {self.secondary_timeframe}, {self.tertiary_timeframe})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MTFRSIStrategy(primary={self.primary_timeframe}, secondary={self.secondary_timeframe}, tertiary={self.tertiary_timeframe})" 