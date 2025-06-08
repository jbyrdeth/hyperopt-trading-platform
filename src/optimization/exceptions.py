"""
Custom Exceptions for Optimization System

This module defines custom exception classes for better error handling
and reporting throughout the optimization pipeline.
"""

from typing import Optional, Dict, Any, List


class OptimizationError(Exception):
    """Base exception for optimization-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "OPTIMIZATION_ERROR"
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


class InvalidStrategyError(OptimizationError):
    """Raised when an invalid or unknown strategy is specified."""
    
    def __init__(self, strategy_name: str, available_strategies: Optional[List[str]] = None):
        self.strategy_name = strategy_name
        self.available_strategies = available_strategies or []
        
        if self.available_strategies:
            message = f"Strategy '{strategy_name}' not found. Available strategies: {', '.join(self.available_strategies)}"
        else:
            message = f"Strategy '{strategy_name}' not found"
            
        super().__init__(
            message=message,
            error_code="INVALID_STRATEGY",
            context={
                "strategy_name": strategy_name,
                "available_strategies": self.available_strategies
            }
        )


class OptimizationTimeoutError(OptimizationError):
    """Raised when optimization exceeds timeout limits."""
    
    def __init__(self, strategy_name: str, timeout_minutes: int, elapsed_minutes: Optional[float] = None):
        self.strategy_name = strategy_name
        self.timeout_minutes = timeout_minutes
        self.elapsed_minutes = elapsed_minutes
        
        elapsed_str = f" (elapsed: {elapsed_minutes:.1f}min)" if elapsed_minutes else ""
        message = f"Optimization for '{strategy_name}' exceeded {timeout_minutes}min timeout{elapsed_str}"
        
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_TIMEOUT",
            context={
                "strategy_name": strategy_name,
                "timeout_minutes": timeout_minutes,
                "elapsed_minutes": elapsed_minutes
            }
        )


class BatchOptimizationError(OptimizationError):
    """Raised when batch optimization fails."""
    
    def __init__(self, batch_id: str, failed_strategies: List[str], total_strategies: int, underlying_errors: Optional[List[Dict[str, Any]]] = None):
        self.batch_id = batch_id
        self.failed_strategies = failed_strategies
        self.total_strategies = total_strategies
        self.underlying_errors = underlying_errors or []
        
        success_count = total_strategies - len(failed_strategies)
        message = f"Batch optimization {batch_id} partially failed: {success_count}/{total_strategies} successful"
        
        super().__init__(
            message=message,
            error_code="BATCH_OPTIMIZATION_FAILURE",
            context={
                "batch_id": batch_id,
                "failed_strategies": failed_strategies,
                "success_count": success_count,
                "total_strategies": total_strategies,
                "underlying_errors": self.underlying_errors
            }
        )


class ParameterValidationError(OptimizationError):
    """Raised when optimization parameters are invalid."""
    
    def __init__(self, parameter_name: str, value: Any, expected_type: str, constraints: Optional[str] = None):
        self.parameter_name = parameter_name
        self.value = value
        self.expected_type = expected_type
        self.constraints = constraints
        
        message = f"Invalid parameter '{parameter_name}': {value} (expected {expected_type}"
        if constraints:
            message += f", {constraints}"
        message += ")"
        
        super().__init__(
            message=message,
            error_code="PARAMETER_VALIDATION_ERROR",
            context={
                "parameter_name": parameter_name,
                "value": str(value),
                "expected_type": expected_type,
                "constraints": constraints
            }
        )


class DataValidationError(OptimizationError):
    """Raised when market data is invalid or insufficient."""
    
    def __init__(self, data_issue: str, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        self.data_issue = data_issue
        self.symbol = symbol
        self.timeframe = timeframe
        
        context_str = f" for {symbol}" if symbol else ""
        context_str += f" ({timeframe})" if timeframe else ""
        message = f"Data validation failed{context_str}: {data_issue}"
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context={
                "data_issue": data_issue,
                "symbol": symbol,
                "timeframe": timeframe
            }
        )


class ResourceExhaustionError(OptimizationError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, resource_type: str, current_usage: float, limit: float, unit: str = ""):
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        self.unit = unit
        
        unit_str = f" {unit}" if unit else ""
        message = f"{resource_type} exhausted: {current_usage:.1f}{unit_str} / {limit:.1f}{unit_str} limit"
        
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTION",
            context={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "unit": unit
            }
        )


class SerializationError(OptimizationError):
    """Raised when strategy or parameter serialization fails."""
    
    def __init__(self, object_type: str, serialization_method: str, underlying_error: Optional[str] = None):
        self.object_type = object_type
        self.serialization_method = serialization_method
        self.underlying_error = underlying_error
        
        message = f"Failed to serialize {object_type} using {serialization_method}"
        if underlying_error:
            message += f": {underlying_error}"
            
        super().__init__(
            message=message,
            error_code="SERIALIZATION_ERROR",
            context={
                "object_type": object_type,
                "serialization_method": serialization_method,
                "underlying_error": underlying_error
            }
        )


class ConcurrencyError(OptimizationError):
    """Raised when concurrent execution encounters issues."""
    
    def __init__(self, worker_id: Optional[str] = None, max_workers: Optional[int] = None, active_workers: Optional[int] = None):
        self.worker_id = worker_id
        self.max_workers = max_workers
        self.active_workers = active_workers
        
        message = "Concurrency error in parallel optimization"
        if worker_id:
            message += f" (worker: {worker_id})"
        if max_workers and active_workers:
            message += f" (workers: {active_workers}/{max_workers})"
            
        super().__init__(
            message=message,
            error_code="CONCURRENCY_ERROR",
            context={
                "worker_id": worker_id,
                "max_workers": max_workers,
                "active_workers": active_workers
            }
        ) 