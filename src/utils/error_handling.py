"""
Production Error Handling Framework

Comprehensive error handling for the trading strategy platform.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Optional, Union, Dict
from datetime import datetime
import json

class ProductionErrorHandler:
    """Production-grade error handling and recovery system."""
    
    def __init__(self):
        self.logger = logging.getLogger('error_handler')
        self.error_counts = {}
        self.circuit_breakers = {}
        
    def handle_strategy_error(self, strategy_name: str, error: Exception, context: Dict[str, Any] = None):
        """Handle strategy-specific errors with context."""
        error_info = {
            'strategy': strategy_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Strategy error in {strategy_name}: {error}", extra=error_info)
        
        # Track error frequency
        error_key = f"{strategy_name}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Circuit breaker logic
        if self.error_counts[error_key] > 5:  # More than 5 errors
            self.logger.critical(f"Circuit breaker triggered for {strategy_name}")
            self.circuit_breakers[strategy_name] = datetime.now()
            
    def handle_backtesting_error(self, error: Exception, strategy_name: str = None, data_info: Dict = None):
        """Handle backtesting engine errors."""
        error_info = {
            'component': 'backtesting_engine',
            'strategy': strategy_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'data_info': data_info or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Backtesting error: {error}", extra=error_info)
        
    def handle_api_error(self, error: Exception, endpoint: str, request_data: Dict = None):
        """Handle API-specific errors."""
        error_info = {
            'component': 'api',
            'endpoint': endpoint,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'request_data': request_data or {},
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"API error at {endpoint}: {error}", extra=error_info)
        
    def is_circuit_open(self, strategy_name: str) -> bool:
        """Check if circuit breaker is open for a strategy."""
        if strategy_name in self.circuit_breakers:
            # Auto-reset after 5 minutes
            if (datetime.now() - self.circuit_breakers[strategy_name]).seconds > 300:
                del self.circuit_breakers[strategy_name]
                self.logger.info(f"Circuit breaker reset for {strategy_name}")
                return False
            return True
        return False

def with_error_handling(component: str = "general"):
    """Decorator for adding error handling to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = ProductionErrorHandler()
                
                if component == "strategy":
                    strategy_name = kwargs.get('strategy_name', 'unknown')
                    handler.handle_strategy_error(strategy_name, e, {'args': args, 'kwargs': kwargs})
                elif component == "backtesting":
                    strategy_name = kwargs.get('strategy_name', None)
                    handler.handle_backtesting_error(e, strategy_name, kwargs)
                elif component == "api":
                    endpoint = kwargs.get('endpoint', func.__name__)
                    handler.handle_api_error(e, endpoint, kwargs)
                else:
                    logging.getLogger(component).error(f"Error in {func.__name__}: {e}", exc_info=True)
                
                # Re-raise for proper error propagation
                raise
                
        return wrapper
    return decorator

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    wait_time = backoff_factor * (2 ** attempt)
                    logging.getLogger('retry').warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    
        return wrapper
    return decorator

# Global error handler instance
error_handler = ProductionErrorHandler()
