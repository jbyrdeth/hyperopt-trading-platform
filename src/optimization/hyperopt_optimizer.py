"""
Hyperparameter Optimization Engine

This module implements a comprehensive hyperparameter optimization system using
Hyperopt's TPE (Tree-structured Parzen Estimator) algorithm with multi-objective
optimization capabilities for trading strategy optimization.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss

# NumPy compatibility fix for hyperopt
class CompatibleRandomState(np.random.RandomState):
    """
    Enhanced RandomState with 'integers' method for hyperopt compatibility.
    
    The hyperopt library expects RandomState objects to have an 'integers' method,
    but this is only available in numpy.random.Generator (from np.random.default_rng).
    This wrapper adds a compatible 'integers' method to RandomState.
    """
    
    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        """
        Return random integers from low (inclusive) to high (exclusive).
        
        This is a compatibility wrapper around randint to match the signature
        of numpy.random.Generator.integers().
        
        Args:
            low: Lowest (signed) integers to be drawn from the distribution
            high: If provided, one above the largest (signed) integer to be drawn
            size: Output shape
            dtype: Desired dtype of the result
            endpoint: If True, sample from the interval [low, high] instead of [low, high)
            
        Returns:
            Array of random integers or scalar integer
        """
        if high is None:
            high = low
            low = 0
        
        # Convert endpoint behavior: integers() is exclusive by default, 
        # randint() is exclusive by default too, so we're compatible
        if endpoint:
            high = high + 1
        
        result = self.randint(low, high, size=size)
        
        # Handle scalar vs array results
        if np.isscalar(result):
            return dtype(result)
        else:
            return result.astype(dtype)


def create_compatible_random_state(seed=None):
    """
    Create a RandomState object with integers() method for hyperopt compatibility.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        CompatibleRandomState object with integers() method
    """
    return CompatibleRandomState(seed)

# Use absolute imports
try:
    from src.strategies.base_strategy import BaseStrategy
    from src.strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from src.utils.logger import get_logger, log_performance
except ImportError:
    # Fallback for relative imports when running as module
    from ..strategies.base_strategy import BaseStrategy
    from ..strategies.backtesting_engine import BacktestingEngine, BacktestResults
    from ..utils.logger import get_logger, log_performance


@dataclass
class OptimizationObjective:
    """Optimization objective configuration."""
    name: str
    weight: float
    maximize: bool = True
    target_value: Optional[float] = None
    
    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    max_evals: int = 100
    timeout_minutes: Optional[int] = None
    early_stop_rounds: int = 20
    n_jobs: int = 1
    random_state: Optional[int] = None
    cache_results: bool = True
    cache_dir: str = "optimization_cache"
    
    # Multi-objective settings
    objectives: List[OptimizationObjective] = None
    
    # Validation settings
    validation_split: float = 0.3
    walk_forward_periods: Optional[int] = None
    
    def __post_init__(self):
        if self.objectives is None:
            # Default objectives
            self.objectives = [
                OptimizationObjective("sharpe_ratio", 0.4, maximize=True),
                OptimizationObjective("annual_return", 0.3, maximize=True),
                OptimizationObjective("max_drawdown", 0.3, maximize=False)
            ]
        
        # Validate objectives
        total_weight = sum(obj.weight for obj in self.objectives)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Objective weights must sum to 1.0, got {total_weight}")


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_metrics: Dict[str, float]
    all_trials: List[Dict[str, Any]]
    optimization_time: float
    total_evaluations: int
    cache_hits: int
    validation_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class OptimizationCache:
    """Cache for optimization results to avoid re-running identical optimizations."""
    
    def __init__(self, cache_dir: str = "optimization_cache"):
        """Initialize optimization cache."""
        self.cache_dir = cache_dir
        self.logger = get_logger("optimization_cache")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache index
        self.index_file = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, str]:
        """Load cache index from disk."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_key(
        self,
        strategy_class: type,
        params: Dict[str, Any],
        data_hash: str,
        config_hash: str
    ) -> str:
        """Generate cache key for optimization parameters."""
        key_data = {
            'strategy': strategy_class.__name__,
            'params': sorted(params.items()),
            'data_hash': data_hash,
            'config_hash': config_hash
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result."""
        if cache_key in self.cache_index:
            cache_file = self.cache_index[cache_key]
            cache_path = os.path.join(self.cache_dir, cache_file)
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load cached result: {e}")
                    # Remove invalid cache entry
                    del self.cache_index[cache_key]
                    self._save_cache_index()
        
        return None
    
    def put(self, cache_key: str, result: Dict[str, Any]):
        """Store result in cache."""
        try:
            cache_file = f"{cache_key}.pkl"
            cache_path = os.path.join(self.cache_dir, cache_file)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            self.cache_index[cache_key] = cache_file
            self._save_cache_index()
            
        except Exception as e:
            self.logger.error(f"Failed to cache result: {e}")
    
    def clear(self):
        """Clear all cached results."""
        try:
            for cache_file in self.cache_index.values():
                cache_path = os.path.join(self.cache_dir, cache_file)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            
            self.cache_index.clear()
            self._save_cache_index()
            
            self.logger.info("Cache cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")


class ParameterSpaceSerializer:
    """
    Utility class to serialize and deserialize hyperopt parameter spaces.
    
    Hyperopt objects like hp.choice() and hp.uniform() are not directly serializable,
    so we need to convert them to a serializable format for multiprocessing.
    """
    
    @staticmethod
    def serialize_parameter_space(param_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert hyperopt parameter space to serializable format.
        
        Args:
            param_space: Dictionary containing hyperopt parameter definitions
            
        Returns:
            Serializable dictionary representation
        """
        serialized = {}
        
        for key, value in param_space.items():
            if hasattr(value, 'name') and hasattr(value, 'pos_args'):
                # This is a hyperopt distribution object
                # Map internal hyperopt names to more recognizable names
                name_mapping = {
                    'switch': 'choice',  # hp.choice uses 'switch' internally
                    'float': 'uniform',  # hp.uniform uses 'float' internally
                    'hyperopt_param': 'randint'  # hp.randint uses 'hyperopt_param' internally
                }
                
                mapped_name = name_mapping.get(value.name, value.name)
                
                serialized[key] = {
                    '_hyperopt_type': mapped_name,
                    '_hyperopt_original_name': value.name,  # Keep original for debugging
                    '_hyperopt_args': getattr(value, 'pos_args', []),
                    '_hyperopt_kwargs': getattr(value, 'kwargs', {})
                }
            else:
                # Regular value, keep as-is
                serialized[key] = value
        
        return serialized
    
    @staticmethod
    def deserialize_parameter_space(serialized_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert serialized parameter space back to hyperopt format.
        
        Args:
            serialized_space: Serialized parameter space dictionary
            
        Returns:
            Hyperopt parameter space
        """
        from src.utils.logger import get_logger
        logger = get_logger("parameter_deserializer")
        
        param_space = {}
        logger.debug(f"Deserializing parameter space: {serialized_space}")
        
        for key, value in serialized_space.items():
            if isinstance(value, dict) and '_hyperopt_type' in value:
                # Reconstruct hyperopt object
                dist_type = value['_hyperopt_type']
                args = value.get('_hyperopt_args', [])
                kwargs = value.get('_hyperopt_kwargs', {})
                
                # Handle special cases for hyperopt arguments
                if dist_type == 'choice':
                    # hp.choice expects (name, options)
                    if len(args) >= 1:
                        param_space[key] = hp.choice(key, args[1:] if len(args) > 1 else args[0])
                    else:
                        raise ValueError(f"Invalid choice args for {key}: {args}")
                elif dist_type == 'uniform':
                    # hp.uniform expects (name, low, high)
                    if len(args) >= 3:
                        # Extract low and high bounds from original args
                        low, high = args[1], args[2]
                        param_space[key] = hp.uniform(key, low, high)
                        logger.debug(f"Reconstructed uniform parameter {key}: low={low}, high={high}")
                    elif len(args) == 2:
                        # Some cases might have just low/high without name
                        low, high = args[0], args[1]
                        param_space[key] = hp.uniform(key, low, high)
                        logger.debug(f"Reconstructed uniform parameter {key}: low={low}, high={high}")
                    else:
                        raise ValueError(f"Invalid uniform args for {key}: {args}. Expected at least 2 arguments (low, high)")
                elif dist_type == 'randint':
                    # hp.randint expects (name, upper)
                    if len(args) >= 2:
                        param_space[key] = hp.randint(key, args[1])
                    elif len(args) == 1:
                        param_space[key] = hp.randint(key, args[0])
                    else:
                        raise ValueError(f"Invalid randint args for {key}: {args}. Expected at least 1 argument (upper)")
                elif dist_type == 'normal':
                    # hp.normal expects (name, mu, sigma)
                    if len(args) >= 3:
                        param_space[key] = hp.normal(key, args[1], args[2])
                    elif len(args) == 2:
                        param_space[key] = hp.normal(key, args[0], args[1])
                    else:
                        param_space[key] = hp.normal(key, 0, 1)  # fallback
                elif dist_type == 'lognormal':
                    # hp.lognormal expects (name, mu, sigma)
                    if len(args) >= 3:
                        param_space[key] = hp.lognormal(key, args[1], args[2])
                    elif len(args) == 2:
                        param_space[key] = hp.lognormal(key, args[0], args[1])
                    else:
                        param_space[key] = hp.lognormal(key, 0, 1)  # fallback
                elif dist_type == 'quniform':
                    # hp.quniform expects (name, low, high, q)
                    if len(args) >= 4:
                        param_space[key] = hp.quniform(key, args[1], args[2], args[3])
                    elif len(args) == 3:
                        param_space[key] = hp.quniform(key, args[0], args[1], args[2])
                    else:
                        param_space[key] = hp.quniform(key, 0, 1, 1)  # fallback
                elif dist_type == 'qnormal':
                    # hp.qnormal expects (name, mu, sigma, q)
                    if len(args) >= 4:
                        param_space[key] = hp.qnormal(key, args[1], args[2], args[3])
                    elif len(args) == 3:
                        param_space[key] = hp.qnormal(key, args[0], args[1], args[2])
                    else:
                        param_space[key] = hp.qnormal(key, 0, 1, 1)  # fallback
                elif dist_type == 'qlognormal':
                    # hp.qlognormal expects (name, mu, sigma, q)
                    if len(args) >= 4:
                        param_space[key] = hp.qlognormal(key, args[1], args[2], args[3])
                    elif len(args) == 3:
                        param_space[key] = hp.qlognormal(key, args[0], args[1], args[2])
                    else:
                        param_space[key] = hp.qlognormal(key, 0, 1, 1)  # fallback
                else:
                    # Fallback: try to get the function from hp module
                    try:
                        hp_func = getattr(hp, dist_type)
                        param_space[key] = hp_func(key, 0, 1)  # Default args
                    except AttributeError:
                        raise ValueError(f"Unknown hyperopt distribution type: {dist_type}")
            else:
                # Regular value
                param_space[key] = value
        
        logger.debug(f"Successfully deserialized parameter space with {len(param_space)} parameters")
        return param_space
    
    @staticmethod
    def is_serializable(param_space: Dict[str, Any]) -> bool:
        """
        Check if a parameter space is already serializable.
        
        Args:
            param_space: Parameter space to check
            
        Returns:
            True if serializable, False otherwise
        """
        try:
            pickle.dumps(param_space)
            return True
        except (pickle.PicklingError, TypeError):
            return False


class HyperoptOptimizer:
    """
    Hyperparameter optimization engine using Hyperopt TPE algorithm.
    
    Features:
    - Multi-objective optimization with configurable weights
    - Parallel processing support
    - Result caching to avoid re-computation
    - Early stopping for efficiency
    - Walk-forward validation
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """
        Initialize the optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = get_logger("hyperopt_optimizer")
        
        # Initialize cache
        self.cache = OptimizationCache(self.config.cache_dir) if self.config.cache_results else None
        
        # Optimization state
        self.current_data = None
        self.current_strategy_class = None
        self.current_symbol = None
        self.cache_hits = 0
        
        self.logger.info(f"Initialized HyperoptOptimizer with config: {self.config}")
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of the data for caching."""
        # Use a subset of data characteristics for hash
        data_summary = {
            'length': len(data),
            'start_date': data.index[0].isoformat(),
            'end_date': data.index[-1].isoformat(),
            'columns': list(data.columns),
            'close_sum': float(data['close'].sum()),
            'volume_sum': float(data['volume'].sum()) if 'volume' in data.columns else 0
        }
        return hashlib.md5(json.dumps(data_summary, sort_keys=True).encode()).hexdigest()
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of the optimization configuration."""
        config_dict = {
            'objectives': [(obj.name, obj.weight, obj.maximize) for obj in self.config.objectives],
            'validation_split': self.config.validation_split,
            'walk_forward_periods': self.config.walk_forward_periods
        }
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets."""
        if self.config.validation_split <= 0:
            return data, data
        
        split_idx = int(len(data) * (1 - self.config.validation_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        self.logger.debug(f"Split data: train={len(train_data)}, validation={len(val_data)}")
        return train_data, val_data
    
    def _calculate_composite_score(self, results: BacktestResults) -> float:
        """Calculate composite score based on multiple objectives."""
        scores = []
        for objective in self.config.objectives:
            metric_value = getattr(results, objective.name, 0.0)
            
            if objective.maximize:
                score = metric_value
            else:
                # For minimization objectives (like drawdown), invert the score
                score = -metric_value
            
            # Apply weight
            weighted_score = score * objective.weight
            scores.append(weighted_score)
        
        return sum(scores)
    
    def _convert_choice_indices_to_values(self, params: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert hyperopt choice indices back to actual choice values.
        
        Args:
            params: Parameters with potential choice indices
            parameter_space: Original hyperopt parameter space
            
        Returns:
            Parameters with actual choice values
        """
        converted_params = params.copy()
        
        for param_name, param_value in params.items():
            if param_name in parameter_space:
                param_space_obj = parameter_space[param_name]
                
                # Check if it's a hyperopt choice object
                if hasattr(param_space_obj, 'name') and hasattr(param_space_obj, 'options'):
                    if param_space_obj.name == 'hyperopt_choice':
                        # Convert index to actual choice value
                        try:
                            if isinstance(param_value, (int, float)) and 0 <= param_value < len(param_space_obj.options):
                                converted_params[param_name] = param_space_obj.options[int(param_value)]
                        except (IndexError, TypeError) as e:
                            self.logger.warning(f"Could not convert choice index {param_value} for parameter {param_name}: {e}")
                            # Keep original value if conversion fails
                            pass
       
        return converted_params
    
    def _objective_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Objective function for hyperopt optimization.
        
        Args:
            params: Parameters from hyperopt
            
        Returns:
            Dictionary with loss and status
        """
        try:
            # Debug logging to see what hyperopt is passing
            self.logger.info(f"Objective function received params: {params}")
            
            # Validate required parameters
            if not params:
                raise ValueError("No parameters provided")
            
            # Create strategy instance
            strategy = self.current_strategy_class(**params)
            
            # Validate parameters
            if not strategy.validate_parameters():
                return {'loss': 1e6, 'status': STATUS_FAIL}
            
            # Initialize strategy
            strategy.initialize(self.current_data)
            if not strategy.is_initialized:
                return {'loss': 1e6, 'status': STATUS_FAIL}
            
            # Run backtest
            engine = BacktestingEngine(initial_capital=100000)
            results = engine.backtest_strategy(strategy, self.current_data, self.current_symbol)
            
            # Calculate composite score
            score = self._calculate_composite_score(results)
            
            # Validation on out-of-sample data
            validation_score = None
            if len(self.current_data) > 0 and self.config.validation_split > 0:
                strategy.reset()
                strategy.initialize(self.current_data)
                if strategy.is_initialized:
                    val_results = engine.backtest_strategy(strategy, self.current_data, self.current_symbol)
                    validation_score = self._calculate_composite_score(val_results)
            
            # Prepare result
            result = {
                'loss': -score,  # Hyperopt minimizes, so negate for maximization
                'status': STATUS_OK,
                'eval_time': time.time(),
                'train_score': score,
                'validation_score': validation_score,
                'metrics': results.to_dict()
            }
            
            # Cache result
            if self.cache:
                data_hash = self._calculate_data_hash(self.current_data)
                config_hash = self._calculate_config_hash()
                cache_key = self.cache._get_cache_key(
                    self.current_strategy_class, params, data_hash, config_hash
                )
                self.cache.put(cache_key, result)
            
            self.logger.debug(f"Evaluated params {params}: score={score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return {'loss': 1e6, 'status': STATUS_FAIL}
    
    @log_performance("optimize_strategy")
    def optimize_strategy(
        self,
        strategy_class: type,
        parameter_space: Dict[str, Any],
        data: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> OptimizationResult:
        """
        Optimize strategy hyperparameters.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Hyperopt parameter space
            data: Historical data for optimization
            symbol: Trading symbol
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        self.logger.info(f"Starting optimization for {strategy_class.__name__}")
        start_time = time.time()
        
        # Set current optimization context
        self.current_data = data
        self.current_strategy_class = strategy_class
        self.current_symbol = symbol
        self.cache_hits = 0
        
        # Create trials object
        trials = Trials()
        
        # Set up early stopping
        early_stop_fn = no_progress_loss(self.config.early_stop_rounds)
        
        # Run optimization
        try:
            best = fmin(
                fn=self._objective_function,
                space=parameter_space,
                algo=tpe.suggest,
                max_evals=self.config.max_evals,
                trials=trials,
                early_stop_fn=early_stop_fn,
                rstate=create_compatible_random_state(self.config.random_state),
                verbose=False
            )
            
            # Get best trial
            best_trial = min(trials.trials, key=lambda x: x['result']['loss'])
            best_score = -best_trial['result']['loss']  # Convert back from loss
            best_metrics = best_trial['result']['metrics']
            
            # Extract actual parameter values from best trial
            # The 'vals' dict contains lists, so we need to extract the single values
            best_params = {}
            for key, value_list in best_trial['misc']['vals'].items():
                if len(value_list) == 1:
                    best_params[key] = value_list[0]
                else:
                    # This shouldn't happen in single evaluations, but handle gracefully
                    best_params[key] = value_list[0] if value_list else None
            
            # Debug logging to understand parameter extraction
            self.logger.debug(f"Raw best_trial['misc']['vals']: {best_trial['misc']['vals']}")
            self.logger.debug(f"Extracted best_params: {best_params}")
            
            # For hp.choice parameters, we need to convert indices back to actual choices
            # This requires knowledge of the parameter space structure
            best_params = self._convert_choice_indices_to_values(best_params, parameter_space)
            
            self.logger.debug(f"Final best_params after choice conversion: {best_params}")
            
            # Validation results
            validation_results = None
            if 'validation_score' in best_trial['result']:
                validation_results = {
                    'validation_score': best_trial['result']['validation_score'],
                    'train_score': best_trial['result']['train_score']
                }
            
            # Compile all trial results
            all_trials = []
            for trial in trials.trials:
                trial_data = {
                    'params': trial['misc']['vals'],
                    'score': -trial['result']['loss'],
                    'status': trial['result']['status'],
                    'metrics': trial['result'].get('metrics', {})
                }
                all_trials.append(trial_data)
            
            optimization_time = time.time() - start_time
            
            result = OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                best_metrics=best_metrics,
                all_trials=all_trials,
                optimization_time=optimization_time,
                total_evaluations=len(trials.trials),
                cache_hits=self.cache_hits,
                validation_results=validation_results
            )
            
            self.logger.info(
                f"Optimization completed: {len(trials.trials)} evaluations, "
                f"{self.cache_hits} cache hits, {optimization_time:.2f}s, "
                f"best score: {best_score:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
    
    def optimize_multiple_strategies(
        self,
        strategies: List[Tuple[type, Dict[str, Any]]],
        data: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> Dict[str, OptimizationResult]:
        """
        Optimize multiple strategies in parallel.
        
        Args:
            strategies: List of (strategy_class, parameter_space) tuples
            data: Historical data for optimization
            symbol: Trading symbol
            
        Returns:
            Dictionary mapping strategy names to optimization results
        """
        self.logger.info(f"Starting multi-strategy optimization for {len(strategies)} strategies")
        
        results = {}
        
        if self.config.n_jobs == 1:
            # Sequential optimization
            for strategy_class, param_space in strategies:
                result = self.optimize_strategy(strategy_class, param_space, data, symbol)
                results[strategy_class.__name__] = result
        else:
            # Parallel optimization with serialization handling
            serializer = ParameterSpaceSerializer()
            
            # Prepare serializable strategies
            serializable_strategies = []
            for strategy_class, param_space in strategies:
                if not serializer.is_serializable(param_space):
                    # Serialize the parameter space
                    serialized_space = serializer.serialize_parameter_space(param_space)
                    self.logger.debug(f"Serialized parameter space for {strategy_class.__name__}")
                else:
                    serialized_space = param_space
                
                serializable_strategies.append((strategy_class, serialized_space))
            
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                future_to_strategy = {
                    executor.submit(
                        self._optimize_strategy_with_deserialization, 
                        strategy_class, 
                        param_space, 
                        data, 
                        symbol
                    ): strategy_class.__name__
                    for strategy_class, param_space in serializable_strategies
                }
                
                for future in as_completed(future_to_strategy):
                    strategy_name = future_to_strategy[future]
                    try:
                        result = future.result()
                        results[strategy_name] = result
                        self.logger.info(f"Completed optimization for {strategy_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to optimize {strategy_name}: {e}")
        
        self.logger.info(f"Multi-strategy optimization completed: {len(results)} strategies")
        return results
    
    def _optimize_strategy_with_deserialization(
        self,
        strategy_class: type,
        param_space: Dict[str, Any],
        data: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> OptimizationResult:
        """
        Helper method for parallel optimization that handles parameter space deserialization.
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space (potentially serialized)
            data: Historical data for optimization
            symbol: Trading symbol
            
        Returns:
            OptimizationResult with best parameters and metrics
        """
        serializer = ParameterSpaceSerializer()
        
        # Check if parameter space needs deserialization
        needs_deserialization = any(
            isinstance(v, dict) and '_hyperopt_type' in v 
            for v in param_space.values()
        )
        
        if needs_deserialization:
            # Deserialize the parameter space
            param_space = serializer.deserialize_parameter_space(param_space)
        
        # Run the optimization
        return self.optimize_strategy(strategy_class, param_space, data, symbol)
    
    def walk_forward_optimization(
        self,
        strategy_class: type,
        parameter_space: Dict[str, Any],
        data: pd.DataFrame,
        window_size: int,
        step_size: int = None,
        symbol: str = "UNKNOWN"
    ) -> List[OptimizationResult]:
        """
        Perform walk-forward optimization.
        
        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Hyperopt parameter space
            data: Historical data
            window_size: Size of optimization window
            step_size: Step size for walking forward (default: window_size // 4)
            symbol: Trading symbol
            
        Returns:
            List of optimization results for each window
        """
        if step_size is None:
            step_size = window_size // 4
        
        self.logger.info(
            f"Starting walk-forward optimization: window={window_size}, step={step_size}"
        )
        
        results = []
        start_idx = 0
        
        while start_idx + window_size <= len(data):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]
            
            self.logger.info(
                f"Optimizing window {len(results) + 1}: "
                f"{window_data.index[0].date()} to {window_data.index[-1].date()}"
            )
            
            result = self.optimize_strategy(strategy_class, parameter_space, window_data, symbol)
            results.append(result)
            
            start_idx += step_size
        
        self.logger.info(f"Walk-forward optimization completed: {len(results)} windows")
        return results 