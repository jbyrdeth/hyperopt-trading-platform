"""
Multi-Timeframe Trend Analysis Strategy

This strategy analyzes trend direction across multiple timeframes using moving averages
to generate high-quality signals with reduced false positives.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import ta

from .base_strategy import BaseStrategy, Signal
from src.utils.logger import get_logger


class MTFTrendAnalysisStrategy(BaseStrategy):
    """
    Multi-Timeframe Trend Analysis Strategy.
    
    This strategy:
    1. Analyzes trend direction across multiple timeframes (1H, 4H, 1D)
    2. Uses moving averages to determine trend strength and direction
    3. Generates signals only when trends align across timeframes
    4. Provides trend strength scoring for position sizing
    5. Includes trend change detection for early signals
    """
    
    def __init__(
        self,
        primary_timeframe: str = '1H',
        secondary_timeframe: str = '4H', 
        tertiary_timeframe: str = '1D',
        ma_period_short: int = 20,
        ma_period_long: int = 50,
        ma_type: str = 'EMA',
        trend_alignment_threshold: float = 0.7,
        trend_strength_threshold: float = 0.02,
        signal_confirmation_bars: int = 2,
        volume_confirmation: bool = True,
        volume_threshold: float = 1.2,
        position_size_pct: float = 0.95,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        **kwargs
    ):
        """
        Initialize Multi-Timeframe Trend Analysis Strategy.
        
        Args:
            primary_timeframe: Primary timeframe for signals ('1H', '4H', '1D')
            secondary_timeframe: Secondary timeframe for confirmation
            tertiary_timeframe: Tertiary timeframe for overall trend
            ma_period_short: Short moving average period (10-30)
            ma_period_long: Long moving average period (30-100)
            ma_type: Moving average type ('SMA', 'EMA', 'WMA')
            trend_alignment_threshold: Minimum alignment score (0.5-1.0)
            trend_strength_threshold: Minimum trend strength (0.01-0.05)
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
            'ma_period_short': ma_period_short,
            'ma_period_long': ma_period_long,
            'ma_type': ma_type,
            'trend_alignment_threshold': trend_alignment_threshold,
            'trend_strength_threshold': trend_strength_threshold,
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
            name=f"MTFTrend_{primary_timeframe}_{secondary_timeframe}_{tertiary_timeframe}",
            parameters=parameters,
            risk_params=risk_params
        )
        
        # Set instance attributes for easy access
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframe = secondary_timeframe
        self.tertiary_timeframe = tertiary_timeframe
        self.ma_period_short = ma_period_short
        self.ma_period_long = ma_period_long
        self.ma_type = ma_type
        self.trend_alignment_threshold = trend_alignment_threshold
        self.trend_strength_threshold = trend_strength_threshold
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
        valid_ma_types = ['SMA', 'EMA', 'WMA']
        
        if self.primary_timeframe not in valid_timeframes:
            self.logger.error(f"Invalid primary_timeframe: {self.primary_timeframe}")
            return False
        
        if self.secondary_timeframe not in valid_timeframes:
            self.logger.error(f"Invalid secondary_timeframe: {self.secondary_timeframe}")
            return False
        
        if self.tertiary_timeframe not in valid_timeframes:
            self.logger.error(f"Invalid tertiary_timeframe: {self.tertiary_timeframe}")
            return False
        
        if not (10 <= self.ma_period_short <= 30):
            self.logger.error(f"Short MA period {self.ma_period_short} outside valid range [10, 30]")
            return False
        
        if not (30 <= self.ma_period_long <= 100):
            self.logger.error(f"Long MA period {self.ma_period_long} outside valid range [30, 100]")
            return False
        
        if self.ma_period_short >= self.ma_period_long:
            self.logger.error("Short MA period must be less than long MA period")
            return False
        
        if self.ma_type not in valid_ma_types:
            self.logger.error(f"Invalid MA type: {self.ma_type}")
            return False
        
        if not (0.5 <= self.trend_alignment_threshold <= 1.0):
            self.logger.error(f"Trend alignment threshold {self.trend_alignment_threshold} outside valid range [0.5, 1.0]")
            return False
        
        if not (0.01 <= self.trend_strength_threshold <= 0.05):
            self.logger.error(f"Trend strength threshold {self.trend_strength_threshold} outside valid range [0.01, 0.05]")
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
    
    def _calculate_moving_average(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate moving average based on type."""
        if self.ma_type == 'SMA':
            return ta.trend.sma_indicator(data, window=period)
        elif self.ma_type == 'EMA':
            return ta.trend.ema_indicator(data, window=period)
        elif self.ma_type == 'WMA':
            return ta.trend.wma_indicator(data, window=period)
        else:
            return ta.trend.sma_indicator(data, window=period)
    
    def _calculate_timeframe_indicators(self, timeframe: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate indicators for a specific timeframe."""
        indicators = {}
        
        # Moving averages
        indicators['ma_short'] = self._calculate_moving_average(data['close'], self.ma_period_short)
        indicators['ma_long'] = self._calculate_moving_average(data['close'], self.ma_period_long)
        
        # Trend direction (1 = bullish, -1 = bearish, 0 = neutral)
        indicators['trend_direction'] = pd.Series(index=data.index, dtype=float)
        indicators['trend_direction'][indicators['ma_short'] > indicators['ma_long']] = 1
        indicators['trend_direction'][indicators['ma_short'] < indicators['ma_long']] = -1
        indicators['trend_direction'][indicators['ma_short'] == indicators['ma_long']] = 0
        
        # Trend strength (distance between MAs relative to price)
        ma_distance = abs(indicators['ma_short'] - indicators['ma_long'])
        indicators['trend_strength'] = ma_distance / data['close']
        
        # Price position relative to MAs
        indicators['price_vs_ma_short'] = (data['close'] - indicators['ma_short']) / indicators['ma_short']
        indicators['price_vs_ma_long'] = (data['close'] - indicators['ma_long']) / indicators['ma_long']
        
        # MA slope (trend momentum)
        indicators['ma_short_slope'] = indicators['ma_short'].pct_change(periods=3)
        indicators['ma_long_slope'] = indicators['ma_long'].pct_change(periods=3)
        
        # Volume indicators
        if self.volume_confirmation:
            indicators['volume_ma'] = ta.trend.sma_indicator(data['volume'], window=self.ma_period_short)
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        
        return indicators
    
    def _get_trend_alignment_score(self, current_time: datetime) -> float:
        """Calculate trend alignment score across timeframes."""
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        trend_directions = []
        trend_strengths = []
        
        for tf in timeframes:
            if tf not in self.timeframe_indicators:
                continue
            
            indicators = self.timeframe_indicators[tf]
            
            # Get current trend direction and strength
            try:
                # Find the closest timestamp in the timeframe data
                tf_data = self.timeframe_data[tf]
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(indicators['trend_direction']):
                    trend_dir = indicators['trend_direction'].iloc[closest_idx]
                    trend_str = indicators['trend_strength'].iloc[closest_idx]
                    
                    if not pd.isna(trend_dir) and not pd.isna(trend_str):
                        trend_directions.append(trend_dir)
                        trend_strengths.append(trend_str)
            except (KeyError, IndexError):
                continue
        
        if len(trend_directions) < 2:
            return 0.0
        
        # Calculate alignment score
        # Perfect alignment = all trends in same direction
        # Weighted by trend strength
        
        total_weight = sum(trend_strengths)
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted average direction
        weighted_direction = sum(d * s for d, s in zip(trend_directions, trend_strengths)) / total_weight
        
        # Calculate alignment score (0 to 1)
        # 1.0 = perfect alignment, 0.0 = no alignment
        max_possible_alignment = max(trend_strengths) * len(trend_directions)
        actual_alignment = abs(weighted_direction) * sum(trend_strengths)
        
        alignment_score = actual_alignment / max_possible_alignment if max_possible_alignment > 0 else 0.0
        
        return min(alignment_score, 1.0)
    
    def _get_trend_direction_consensus(self, current_time: datetime) -> int:
        """Get consensus trend direction across timeframes."""
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        directions = []
        
        for tf in timeframes:
            if tf not in self.timeframe_indicators:
                continue
            
            indicators = self.timeframe_indicators[tf]
            
            try:
                tf_data = self.timeframe_data[tf]
                closest_idx = tf_data.index.get_indexer([current_time], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(indicators['trend_direction']):
                    trend_dir = indicators['trend_direction'].iloc[closest_idx]
                    if not pd.isna(trend_dir):
                        directions.append(trend_dir)
            except (KeyError, IndexError):
                continue
        
        if len(directions) == 0:
            return 0
        
        # Return consensus direction
        avg_direction = sum(directions) / len(directions)
        
        if avg_direction > 0.5:
            return 1  # Bullish
        elif avg_direction < -0.5:
            return -1  # Bearish
        else:
            return 0  # Neutral
    
    def initialize(self, data: pd.DataFrame) -> None:
        """Initialize strategy with historical data."""
        # Store the original data
        self.data = data
        
        min_periods = max(self.ma_period_long, 100)
        if len(data) < min_periods:
            self.logger.warning(f"Insufficient data for strategy initialization. Need at least {min_periods} periods")
            self.is_initialized = False
            return
        
        # Resample data for each timeframe
        timeframes = [self.primary_timeframe, self.secondary_timeframe, self.tertiary_timeframe]
        
        for tf in timeframes:
            try:
                resampled_data = self._resample_data(data, tf)
                
                if len(resampled_data) < self.ma_period_long:
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
        return max(self.ma_period_long, 100)
    
    def generate_signal(self, timestamp: datetime, current_data: pd.Series) -> Signal:
        """Generate trading signal based on multi-timeframe trend analysis."""
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
        
        # Get trend alignment score
        alignment_score = self._get_trend_alignment_score(timestamp)
        
        # Get trend direction consensus
        trend_consensus = self._get_trend_direction_consensus(timestamp)
        
        # Check if alignment meets threshold
        if alignment_score < self.trend_alignment_threshold:
            return Signal(
                timestamp=timestamp,
                action="hold",
                strength=0.0,
                price=current_price,
                confidence=0.0,
                metadata={
                    "reason": "Insufficient trend alignment",
                    "alignment_score": alignment_score,
                    "trend_consensus": trend_consensus
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
        
        # Generate signal based on trend consensus
        action = "hold"
        strength = 0.0
        confidence = 0.0
        metadata = {
            "alignment_score": alignment_score,
            "trend_consensus": trend_consensus,
            "volume_confirmed": volume_confirmed
        }
        
        if volume_confirmed and abs(trend_consensus) > 0:
            # Calculate signal strength based on alignment and trend strength
            primary_indicators = self.timeframe_indicators.get(self.primary_timeframe, {})
            
            try:
                primary_data = self.timeframe_data[self.primary_timeframe]
                closest_idx = primary_data.index.get_indexer([timestamp], method='nearest')[0]
                
                if closest_idx >= 0 and closest_idx < len(primary_indicators.get('trend_strength', [])):
                    trend_strength = primary_indicators['trend_strength'].iloc[closest_idx]
                    
                    if not pd.isna(trend_strength) and trend_strength >= self.trend_strength_threshold:
                        if trend_consensus > 0:
                            action = "buy"
                            metadata["signal_type"] = "mtf_trend_bullish"
                        else:
                            action = "sell"
                            metadata["signal_type"] = "mtf_trend_bearish"
                        
                        # Calculate strength and confidence
                        strength = min(alignment_score * (trend_strength / self.trend_strength_threshold), 1.0)
                        confidence = alignment_score
                        
                        metadata["trend_strength"] = trend_strength
                        
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
            'ma_period_short': hp.choice('ma_period_short', list(range(10, 31))),
            'ma_period_long': hp.choice('ma_period_long', list(range(30, 101, 10))),
            'ma_type': hp.choice('ma_type', ['SMA', 'EMA', 'WMA']),
            'trend_alignment_threshold': hp.uniform('trend_alignment_threshold', 0.5, 1.0),
            'trend_strength_threshold': hp.uniform('trend_strength_threshold', 0.01, 0.05),
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
                
                for indicator_name in ['trend_direction', 'trend_strength', 'ma_short', 'ma_long']:
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
        Calculate position size based on signal strength and multi-timeframe analysis.
        
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
        
        # Multi-timeframe specific adjustments
        alignment_score = signal.metadata.get('alignment_score', 0.5)
        trend_strength = signal.metadata.get('trend_strength', 0.01)
        
        # Increase size for high alignment and strong trends
        if alignment_score > 0.8:
            adjusted_size = adjusted_size * 1.3
        elif alignment_score > 0.6:
            adjusted_size = adjusted_size * 1.1
        
        # Increase size for strong trends
        if trend_strength > self.trend_strength_threshold * 2:
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
        return f"MTFTrend({self.primary_timeframe}, {self.secondary_timeframe}, {self.tertiary_timeframe})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"MTFTrendAnalysisStrategy(primary={self.primary_timeframe}, secondary={self.secondary_timeframe}, tertiary={self.tertiary_timeframe})" 