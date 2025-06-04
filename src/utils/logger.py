"""
Logging utility for the trading optimization system.
Provides structured logging with file rotation and console output.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        return super().format(record)


class TradingLogger:
    """
    Enhanced logger for trading optimization system.
    
    Features:
    - File rotation
    - Colored console output
    - Structured formatting
    - Performance tracking
    - Error aggregation
    """
    
    def __init__(
        self,
        name: str = "trading_optimizer",
        level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        file_rotation: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize the trading logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            console_output: Whether to output to console
            file_rotation: Whether to rotate log files
            max_file_size: Maximum file size before rotation
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_file = log_file
        self.console_output = console_output
        self.file_rotation = file_rotation
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.error_count = 0
        self.warning_count = 0
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            
            # Colored formatter for console
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            if self.file_rotation:
                # Rotating file handler
                file_handler = logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count
                )
            else:
                # Regular file handler
                file_handler = logging.FileHandler(self.log_file)
            
            file_handler.setLevel(self.level)
            
            # Detailed formatter for file
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.warning_count += 1
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.error_count += 1
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.error_count += 1
        self.logger.critical(message, **kwargs)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            **metrics: Additional metrics to log
        """
        metric_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        self.info(f"PERFORMANCE | {operation} | Duration: {duration:.2f}s | {metric_str}")
    
    def log_strategy_result(self, strategy_name: str, metrics: dict):
        """
        Log strategy optimization result.
        
        Args:
            strategy_name: Name of the strategy
            metrics: Performance metrics dictionary
        """
        self.info(
            f"STRATEGY | {strategy_name} | "
            f"Return: {metrics.get('annual_return', 0):.2%} | "
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | "
            f"Drawdown: {metrics.get('max_drawdown', 0):.2%}"
        )
    
    def log_optimization_progress(self, current: int, total: int, strategy: str = None):
        """
        Log optimization progress.
        
        Args:
            current: Current iteration
            total: Total iterations
            strategy: Current strategy being optimized
        """
        progress = (current / total) * 100
        strategy_info = f" | Strategy: {strategy}" if strategy else ""
        self.info(f"PROGRESS | {current}/{total} ({progress:.1f}%){strategy_info}")
    
    def get_stats(self) -> dict:
        """Get logging statistics."""
        runtime = (datetime.now() - self.start_time).total_seconds()
        return {
            'runtime_seconds': runtime,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'start_time': self.start_time.isoformat(),
            'log_file': self.log_file
        }
    
    def log_system_info(self):
        """Log system information."""
        import platform
        import psutil
        
        self.info("="*60)
        self.info("TRADING OPTIMIZATION SYSTEM STARTUP")
        self.info("="*60)
        self.info(f"Python Version: {platform.python_version()}")
        self.info(f"Platform: {platform.platform()}")
        self.info(f"CPU Cores: {psutil.cpu_count()}")
        self.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        self.info(f"Log Level: {logging.getLevelName(self.level)}")
        if self.log_file:
            self.info(f"Log File: {self.log_file}")
        self.info("="*60)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    name: str = "trading_optimizer"
) -> logging.Logger:
    """
    Setup logging for the trading optimization system.
    
    Args:
        level: Logging level
        log_file: Path to log file
        console_output: Whether to output to console
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    # Create trading logger
    trading_logger = TradingLogger(
        name=name,
        level=level,
        log_file=log_file,
        console_output=console_output
    )
    
    # Log system information
    trading_logger.log_system_info()
    
    return trading_logger.logger


def get_logger(name: str = "trading_optimizer") -> logging.Logger:
    """Get existing logger instance."""
    return logging.getLogger(name)


# Context manager for performance logging
class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: logging.Logger = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation_name} after {duration:.2f}s: {exc_val}")


# Decorator for automatic function timing
def log_performance(operation_name: str = None):
    """Decorator to automatically log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger()
            
            with PerformanceTimer(name, logger):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator 