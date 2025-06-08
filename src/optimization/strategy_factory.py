"""
Strategy Factory

This module provides a centralized factory for creating and managing trading strategies
and their hyperparameter spaces for optimization.
"""

from typing import Dict, List, Type, Any, Tuple, Optional
from abc import ABC, abstractmethod

# Import strategy implementations and utilities.  Using absolute imports here
# avoids issues when the package is imported either as ``optimization`` or
# ``src.optimization``.  Both layouts occur in this repository (for example in
# tests we import ``optimization.strategy_factory`` directly).  Attempting
# relative imports would fail when ``optimization`` is a top-level package.

try:
    from src.strategies.base_strategy import BaseStrategy
except ImportError:
    from strategies.base_strategy import BaseStrategy
try:
    from src.strategies.moving_average_crossover import MovingAverageCrossoverStrategy
    from src.strategies.macd_strategy import MACDStrategy
    from src.strategies.rsi_strategy import RSIStrategy
    from src.strategies.bollinger_bands_strategy import BollingerBandsStrategy
    from src.strategies.momentum_strategy import MomentumStrategy
    
    # Volume-based strategies
    from src.strategies.vwap_strategy import VWAPStrategy
    from src.strategies.obv_strategy import OBVStrategy
    from src.strategies.ad_strategy import ADStrategy
    from src.strategies.cmf_strategy import CMFStrategy
    
    # Volatility-based strategies
    from src.strategies.atr_strategy import ATRStrategy
    from src.strategies.bollinger_squeeze_strategy import BollingerSqueezeStrategy
    from src.strategies.keltner_channel_strategy import KeltnerChannelStrategy
    from src.strategies.historical_volatility_strategy import HistoricalVolatilityStrategy
    
    # Advanced momentum strategies
    from src.strategies.roc_strategy import ROCStrategy
    from src.strategies.stochastic_strategy import StochasticStrategy
    from src.strategies.williams_r_strategy import WilliamsRStrategy
    from src.strategies.ultimate_oscillator_strategy import UltimateOscillatorStrategy
    
    # Pattern recognition strategies
    from src.strategies.support_resistance_strategy import SupportResistanceStrategy
    from src.strategies.pivot_points_strategy import PivotPointsStrategy
    from src.strategies.fibonacci_retracement_strategy import FibonacciRetracementStrategy
    from src.strategies.double_top_bottom_strategy import DoubleTopBottomStrategy
    
    # Multi-timeframe strategies
    from src.strategies.mtf_trend_analysis_strategy import MTFTrendAnalysisStrategy
    from src.strategies.mtf_rsi_strategy import MTFRSIStrategy
    from src.strategies.mtf_macd_strategy import MTFMACDStrategy
    
    from src.utils.logger import get_logger
except ImportError:
    from strategies.moving_average_crossover import MovingAverageCrossoverStrategy
    from strategies.macd_strategy import MACDStrategy
    from strategies.rsi_strategy import RSIStrategy
    from strategies.bollinger_bands_strategy import BollingerBandsStrategy
    from strategies.momentum_strategy import MomentumStrategy
    
    # Volume-based strategies
    from strategies.vwap_strategy import VWAPStrategy
    from strategies.obv_strategy import OBVStrategy
    from strategies.ad_strategy import ADStrategy
    from strategies.cmf_strategy import CMFStrategy
    
    # Volatility-based strategies
    from strategies.atr_strategy import ATRStrategy
    from strategies.bollinger_squeeze_strategy import BollingerSqueezeStrategy
    from strategies.keltner_channel_strategy import KeltnerChannelStrategy
    from strategies.historical_volatility_strategy import HistoricalVolatilityStrategy
    
    # Advanced momentum strategies
    from strategies.roc_strategy import ROCStrategy
    from strategies.stochastic_strategy import StochasticStrategy
    from strategies.williams_r_strategy import WilliamsRStrategy
    from strategies.ultimate_oscillator_strategy import UltimateOscillatorStrategy
    
    # Pattern recognition strategies
    from strategies.support_resistance_strategy import SupportResistanceStrategy
    from strategies.pivot_points_strategy import PivotPointsStrategy
    from strategies.fibonacci_retracement_strategy import FibonacciRetracementStrategy
    from strategies.double_top_bottom_strategy import DoubleTopBottomStrategy
    
    # Multi-timeframe strategies
    from strategies.mtf_trend_analysis_strategy import MTFTrendAnalysisStrategy
    from strategies.mtf_rsi_strategy import MTFRSIStrategy
    from strategies.mtf_macd_strategy import MTFMACDStrategy
    
    from src.utils.logger import get_logger


class StrategyRegistry:
    """Registry for all available trading strategies."""
    
    def __init__(self):
        """Initialize the strategy registry."""
        self.logger = get_logger("strategy_registry")
        self._strategies: Dict[str, Type[BaseStrategy]] = {}
        self._parameter_spaces: Dict[str, Dict[str, Any]] = {}
        self._categories: Dict[str, List[str]] = {}
        
        # Register all available strategies
        self._register_core_strategies()
        
        self.logger.info(f"Initialized strategy registry with {len(self._strategies)} strategies")
    
    def _register_core_strategies(self):
        """Register all core trading strategies."""
        
        # Trend Following Strategies
        self.register_strategy(
            "MovingAverageCrossover",
            MovingAverageCrossoverStrategy,
            category="trend_following"
        )
        
        self.register_strategy(
            "MACD",
            MACDStrategy,
            category="trend_following"
        )
        
        # Mean Reversion Strategies
        self.register_strategy(
            "RSI",
            RSIStrategy,
            category="mean_reversion"
        )
        
        self.register_strategy(
            "BollingerBands",
            BollingerBandsStrategy,
            category="volatility"
        )
        
        # Momentum Strategies
        self.register_strategy(
            "Momentum",
            MomentumStrategy,
            category="momentum"
        )
        
        # Advanced Momentum Strategies
        self.register_strategy(
            "ROC",
            ROCStrategy,
            category="momentum"
        )
        
        self.register_strategy(
            "Stochastic",
            StochasticStrategy,
            category="momentum"
        )
        
        self.register_strategy(
            "WilliamsR",
            WilliamsRStrategy,
            category="momentum"
        )
        
        self.register_strategy(
            "UltimateOscillator",
            UltimateOscillatorStrategy,
            category="momentum"
        )
        
        # Volume-Based Strategies
        self.register_strategy(
            "VWAP",
            VWAPStrategy,
            category="volume"
        )
        
        self.register_strategy(
            "OBV",
            OBVStrategy,
            category="volume"
        )
        
        self.register_strategy(
            "AD",
            ADStrategy,
            category="volume"
        )
        
        self.register_strategy(
            "CMF",
            CMFStrategy,
            category="volume"
        )
        
        # Volatility-Based Strategies
        self.register_strategy(
            "ATR",
            ATRStrategy,
            category="volatility"
        )
        
        self.register_strategy(
            "BollingerSqueeze",
            BollingerSqueezeStrategy,
            category="volatility"
        )
        
        self.register_strategy(
            "KeltnerChannel",
            KeltnerChannelStrategy,
            category="volatility"
        )
        
        self.register_strategy(
            "HistoricalVolatility",
            HistoricalVolatilityStrategy,
            category="volatility"
        )
        
        # Pattern Recognition Strategies
        self.register_strategy(
            "SupportResistance",
            SupportResistanceStrategy,
            category="pattern_recognition"
        )
        
        self.register_strategy(
            "PivotPoints",
            PivotPointsStrategy,
            category="pattern_recognition"
        )
        
        self.register_strategy(
            "FibonacciRetracement",
            FibonacciRetracementStrategy,
            category="pattern_recognition"
        )
        
        self.register_strategy(
            "DoubleTopBottom",
            DoubleTopBottomStrategy,
            category="pattern_recognition"
        )
        
        # Multi-Timeframe Strategies
        self.register_strategy(
            "MTFTrendAnalysis",
            MTFTrendAnalysisStrategy,
            category="multi_timeframe"
        )
        
        self.register_strategy(
            "MTFRSI",
            MTFRSIStrategy,
            category="multi_timeframe"
        )
        
        self.register_strategy(
            "MTFMACD",
            MTFMACDStrategy,
            category="multi_timeframe"
        )
    
    def register_strategy(
        self,
        name: str,
        strategy_class: Type[BaseStrategy],
        category: str = "other"
    ):
        """
        Register a new strategy.
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
            category: Strategy category
        """
        self._strategies[name] = strategy_class
        
        # Get parameter space from strategy if available, otherwise create default
        param_space = {}
        try:
            # Try to create a dummy instance to get parameter space
            dummy_instance = strategy_class()
            if hasattr(dummy_instance, 'get_parameter_space'):
                param_space = dummy_instance.get_parameter_space()
            else:
                # Create default parameter space based on strategy type
                param_space = self._create_default_parameter_space(name, strategy_class)
        except Exception as e:
            self.logger.warning(f"Could not get parameter space for {name}: {e}")
            # Create default parameter space
            param_space = self._create_default_parameter_space(name, strategy_class)
        
        self._parameter_spaces[name] = param_space
        
        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        self.logger.debug(f"Registered strategy: {name} ({category}) with {len(param_space)} parameters")
    
    def _create_default_parameter_space(self, name: str, strategy_class: Type[BaseStrategy]) -> Dict[str, Any]:
        """Create default parameter space for a strategy."""
        from hyperopt import hp
        
        # Common parameters for all strategies
        param_space = {
            'position_size_pct': hp.uniform('position_size_pct', 0.01, 0.1),
            'stop_loss_pct': hp.uniform('stop_loss_pct', 0.01, 0.05),
            'take_profit_pct': hp.uniform('take_profit_pct', 0.02, 0.1)
        }
        
        # Strategy-specific parameters
        if name == "MovingAverageCrossover":
            param_space.update({
                'fast_period': hp.choice('fast_period', [5, 8, 10, 12, 15, 20]),
                'slow_period': hp.choice('slow_period', [20, 25, 30, 40, 50, 60]),
                'ma_type': hp.choice('ma_type', ['SMA', 'EMA', 'WMA']),
                'signal_threshold': hp.uniform('signal_threshold', 0.001, 0.01)
            })
        
        elif name == "MACD":
            param_space.update({
                'fast_period': hp.choice('fast_period', [8, 10, 12, 15]),
                'slow_period': hp.choice('slow_period', [20, 24, 26, 30]),
                'signal_period': hp.choice('signal_period', [7, 9, 12, 15]),
                'histogram_threshold': hp.uniform('histogram_threshold', 0.0, 0.5)
            })
        
        elif name == "RSI":
            param_space.update({
                'rsi_period': hp.choice('rsi_period', [10, 12, 14, 16, 20, 25]),
                'oversold': hp.choice('oversold', [20, 25, 30, 35]),
                'overbought': hp.choice('overbought', [65, 70, 75, 80]),
                'exit_signal': hp.choice('exit_signal', ['opposite', 'middle', 'trailing'])
            })
        
        elif name == "BollingerBands":
            param_space.update({
                'period': hp.choice('period', [15, 20, 25, 30]),
                'std_dev': hp.uniform('std_dev', 1.5, 2.5),
                'squeeze_threshold': hp.uniform('squeeze_threshold', 0.1, 0.3),
                'entry_method': hp.choice('entry_method', ['breakout', 'mean_reversion'])
            })
        
        elif name == "Momentum":
            param_space.update({
                'period': hp.choice('period', [10, 12, 15, 20, 25]),
                'threshold': hp.uniform('threshold', 0.01, 0.05)
            })
        
        return param_space
    
    def get_strategy_class(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class by name."""
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found. Available: {list(self._strategies.keys())}")
        return self._strategies[name]
    
    def get_parameter_space(self, name: str) -> Dict[str, Any]:
        """Get parameter space for a strategy."""
        if name not in self._parameter_spaces:
            raise ValueError(f"Parameter space for '{name}' not found")
        return self._parameter_spaces[name]
    
    def get_strategies_by_category(self, category: str) -> List[str]:
        """Get all strategies in a category."""
        return self._categories.get(category, [])
    
    def get_all_strategies(self) -> List[str]:
        """Get all registered strategy names."""
        return list(self._strategies.keys())
    
    def get_all_categories(self) -> List[str]:
        """Get all strategy categories."""
        return list(self._categories.keys())
    
    def create_strategy(self, name: str, **kwargs) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            name: Strategy name
            **kwargs: Strategy parameters
            
        Returns:
            Strategy instance
        """
        strategy_class = self.get_strategy_class(name)
        return strategy_class(**kwargs)
    
    def get_strategy_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive information about a strategy."""
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found")
        
        strategy_class = self._strategies[name]
        
        # Find category
        category = "other"
        for cat, strategies in self._categories.items():
            if name in strategies:
                category = cat
                break
        
        # Get parameter space
        param_space = self._parameter_spaces.get(name, {})
        
        return {
            "name": name,
            "class": strategy_class.__name__,
            "category": category,
            "module": strategy_class.__module__,
            "docstring": strategy_class.__doc__,
            "parameter_count": len(param_space),
            "parameters": list(param_space.keys())
        }


class StrategyFactory:
    """
    Factory for creating and managing trading strategies.
    
    Provides high-level interface for strategy creation, optimization,
    and batch operations.
    """
    
    def __init__(self):
        """Initialize the strategy factory."""
        self.registry = StrategyRegistry()
        self.logger = get_logger("strategy_factory")
    
    def create_strategy(self, name: str, **kwargs) -> BaseStrategy:
        """Create a strategy instance with given parameters."""
        return self.registry.create_strategy(name, **kwargs)
    
    def get_strategy_class(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class by name."""
        return self.registry.get_strategy_class(name)
    
    def get_parameter_space(self, name: str) -> Dict[str, Any]:
        """Get parameter space for a strategy."""
        return self.registry.get_parameter_space(name)
    
    def get_strategies_by_category(self, category: str) -> Dict[str, Type[BaseStrategy]]:
        """Get all strategies in a category."""
        strategy_names = self.registry.get_strategies_by_category(category)
        return {name: self.registry.get_strategy_class(name) for name in strategy_names}
    
    def get_all_strategies(self) -> List[str]:
        """Get all registered strategy names."""
        return self.registry.get_all_strategies()
    
    def get_all_categories(self) -> List[str]:
        """Get all strategy categories."""
        return self.registry.get_all_categories()
    
    def get_optimization_candidates(
        self,
        categories: Optional[List[str]] = None,
        strategy_names: Optional[List[str]] = None
    ) -> List[Tuple[Type[BaseStrategy], Dict[str, Any]]]:
        """
        Get strategies and their parameter spaces for optimization.
        
        Args:
            categories: List of categories to include (None for all)
            strategy_names: Specific strategy names to include (None for all)
            
        Returns:
            List of (strategy_class, parameter_space) tuples
        """
        candidates = []
        
        if strategy_names:
            # Use specific strategies
            for name in strategy_names:
                try:
                    strategy_class = self.registry.get_strategy_class(name)
                    param_space = self.registry.get_parameter_space(name)
                    candidates.append((strategy_class, param_space))
                except ValueError as e:
                    self.logger.warning(f"Skipping strategy {name}: {e}")
        
        elif categories:
            # Use strategies from specific categories
            for category in categories:
                strategy_names = self.registry.get_strategies_by_category(category)
                for name in strategy_names:
                    try:
                        strategy_class = self.registry.get_strategy_class(name)
                        param_space = self.registry.get_parameter_space(name)
                        candidates.append((strategy_class, param_space))
                    except ValueError as e:
                        self.logger.warning(f"Skipping strategy {name}: {e}")
        
        else:
            # Use all strategies
            for name in self.registry.get_all_strategies():
                try:
                    strategy_class = self.registry.get_strategy_class(name)
                    param_space = self.registry.get_parameter_space(name)
                    candidates.append((strategy_class, param_space))
                except ValueError as e:
                    self.logger.warning(f"Skipping strategy {name}: {e}")
        
        self.logger.info(f"Selected {len(candidates)} strategies for optimization")
        return candidates
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all available strategies."""
        summary = {
            "total_strategies": len(self.registry.get_all_strategies()),
            "categories": {},
            "strategies": {}
        }
        
        # Category breakdown
        for category in self.registry.get_all_categories():
            strategies = self.registry.get_strategies_by_category(category)
            summary["categories"][category] = {
                "count": len(strategies),
                "strategies": strategies
            }
        
        # Individual strategy info
        for name in self.registry.get_all_strategies():
            summary["strategies"][name] = self.registry.get_strategy_info(name)
        
        return summary
    
    def validate_strategy_parameters(self, name: str, params: Dict[str, Any]) -> bool:
        """
        Validate parameters for a strategy.
        
        Args:
            name: Strategy name
            params: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        try:
            strategy = self.create_strategy(name, **params)
            return strategy.validate_parameters()
        except Exception as e:
            self.logger.error(f"Parameter validation failed for {name}: {e}")
            return False
    
    def get_default_parameters(self, name: str) -> Dict[str, Any]:
        """
        Get default parameters for a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Dictionary of default parameters
        """
        try:
            # Create strategy with no parameters to get defaults
            strategy_class = self.registry.get_strategy_class(name)
            strategy = strategy_class()
            
            # Extract parameters from strategy instance
            default_params = {}
            if hasattr(strategy, 'parameters'):
                default_params.update(strategy.parameters)
            
            # Add common parameters
            for attr in ['fast_period', 'slow_period', 'period', 'rsi_period', 
                        'threshold', 'position_size_pct', 'stop_loss_pct', 'take_profit_pct']:
                if hasattr(strategy, attr):
                    default_params[attr] = getattr(strategy, attr)
            
            return default_params
            
        except Exception as e:
            self.logger.error(f"Could not get default parameters for {name}: {e}")
            return {}


# Global strategy factory instance
strategy_factory = StrategyFactory() 