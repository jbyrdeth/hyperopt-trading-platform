"""
Performance Optimization Utilities

This module provides comprehensive performance optimization tools for the
trading strategy optimization platform, including caching, database optimization,
async processing, memory management, and performance monitoring.
"""

import asyncio
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable, Union
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
import redis
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import gc
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    cache_hit_rate: float
    database_query_time: float
    api_response_time: float
    throughput: float
    error_rate: float

class MemoryTracker:
    """Track memory usage and detect memory leaks."""
    
    def __init__(self):
        self.snapshots = []
        self.baseline = None
        
    def start_tracking(self):
        """Start memory tracking."""
        tracemalloc.start()
        self.baseline = tracemalloc.take_snapshot()
        
    def take_snapshot(self, label: str = None):
        """Take a memory snapshot."""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append({
                'timestamp': datetime.utcnow(),
                'label': label or f'snapshot_{len(self.snapshots)}',
                'snapshot': snapshot
            })
            return snapshot
        return None
        
    def get_memory_diff(self, snapshot1=None, snapshot2=None):
        """Get memory difference between snapshots."""
        if not snapshot1:
            snapshot1 = self.baseline
        if not snapshot2:
            snapshot2 = self.snapshots[-1]['snapshot'] if self.snapshots else None
            
        if snapshot1 and snapshot2:
            top_stats = snapshot2.compare_to(snapshot1, 'lineno')
            return top_stats[:10]  # Top 10 memory differences
        return []

class CacheManager:
    """Advanced caching system with Redis and in-memory caching."""
    
    def __init__(self, redis_url: str = None):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'sets': 0}
        
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments."""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats['hits'] += 1
                    return pickle.loads(value)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Fall back to memory cache
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[key]
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        self.cache_stats['sets'] += 1
        
        # Try Redis first
        if self.redis_client:
            try:
                serialized = pickle.dumps(value)
                self.redis_client.setex(key, ttl, serialized)
                return
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # Fall back to memory cache
        self.memory_cache[key] = value
        
        # Implement simple TTL for memory cache
        if len(self.memory_cache) > 1000:  # Simple eviction
            # Remove 20% of oldest entries
            keys_to_remove = list(self.memory_cache.keys())[:200]
            for k in keys_to_remove:
                del self.memory_cache[k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'sets': self.cache_stats['sets'],
            'hit_rate': hit_rate,
            'cache_size': len(self.memory_cache)
        }

class DatabaseOptimizer:
    """Database query optimization and connection pooling."""
    
    def __init__(self, engine):
        self.engine = engine
        self.query_stats = {}
        self.slow_queries = []
        
    def log_query(self, query: str, execution_time: float):
        """Log query execution statistics."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                'query': query[:100] + '...' if len(query) > 100 else query,
                'executions': 0,
                'total_time': 0,
                'avg_time': 0,
                'max_time': 0
            }
        
        stats = self.query_stats[query_hash]
        stats['executions'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['executions']
        stats['max_time'] = max(stats['max_time'], execution_time)
        
        # Track slow queries
        if execution_time > 1.0:  # Queries taking more than 1 second
            self.slow_queries.append({
                'query': query,
                'execution_time': execution_time,
                'timestamp': datetime.utcnow()
            })
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict]:
        """Get slowest queries."""
        return sorted(self.slow_queries, 
                     key=lambda x: x['execution_time'], 
                     reverse=True)[:limit]
    
    def optimize_connection_pool(self):
        """Optimize database connection pool settings."""
        if hasattr(self.engine.pool, 'size'):
            current_size = self.engine.pool.size()
            recommended_size = min(20, max(5, psutil.cpu_count() * 2))
            
            if current_size != recommended_size:
                logger.info(f"Recommended pool size: {recommended_size} (current: {current_size})")

class AsyncProcessor:
    """Asynchronous processing for heavy computational tasks."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or psutil.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, self.max_workers))
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        
    async def submit_cpu_intensive_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit CPU-intensive task to process pool."""
        task_id = hashlib.md5(f"{func.__name__}:{time.time()}".encode()).hexdigest()
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self.process_executor, func, *args, **kwargs)
        
        self.active_tasks[task_id] = {
            'future': future,
            'started_at': datetime.utcnow(),
            'type': 'cpu_intensive'
        }
        
        return task_id
    
    async def submit_io_task(self, func: Callable, *args, **kwargs) -> str:
        """Submit I/O task to thread pool."""
        task_id = hashlib.md5(f"{func.__name__}:{time.time()}".encode()).hexdigest()
        
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self.thread_executor, func, *args, **kwargs)
        
        self.active_tasks[task_id] = {
            'future': future,
            'started_at': datetime.utcnow(),
            'type': 'io_task'
        }
        
        return task_id
    
    async def get_task_result(self, task_id: str) -> Any:
        """Get result of submitted task."""
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]['future']
            try:
                result = await future
                del self.active_tasks[task_id]
                return result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                del self.active_tasks[task_id]
                raise
        else:
            raise ValueError(f"Task {task_id} not found")
    
    def get_active_tasks(self) -> Dict[str, Dict]:
        """Get information about active tasks."""
        return {
            task_id: {
                'type': info['type'],
                'duration': (datetime.utcnow() - info['started_at']).total_seconds(),
                'status': 'running' if not info['future'].done() else 'completed'
            }
            for task_id, info in self.active_tasks.items()
        }

class PerformanceMonitor:
    """Monitor system and application performance."""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'memory_usage': 85.0,  # Percentage
            'cpu_usage': 80.0,     # Percentage
            'response_time': 2.0,   # Seconds
            'error_rate': 5.0       # Percentage
        }
        
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        network_io = psutil.net_io_counters()
        
        return {
            'memory_usage_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_usage_percent': cpu_percent,
            'disk_read_mb_per_sec': disk_io.read_bytes / (1024**2) if disk_io else 0,
            'disk_write_mb_per_sec': disk_io.write_bytes / (1024**2) if disk_io else 0,
            'network_bytes_sent': network_io.bytes_sent if network_io else 0,
            'network_bytes_recv': network_io.bytes_recv if network_io else 0,
            'timestamp': time.time()
        }
    
    def log_metrics(self, custom_metrics: Dict[str, float] = None):
        """Log performance metrics."""
        system_metrics = self.collect_system_metrics()
        
        if custom_metrics:
            system_metrics.update(custom_metrics)
        
        self.metrics_history.append(system_metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Check for alerts
        self._check_alerts(system_metrics)
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        if metrics['memory_usage_percent'] > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {metrics['memory_usage_percent']:.1f}%")
        
        if metrics['cpu_usage_percent'] > self.alert_thresholds['cpu_usage']:
            alerts.append(f"High CPU usage: {metrics['cpu_usage_percent']:.1f}%")
        
        if alerts:
            logger.warning(f"Performance alerts: {', '.join(alerts)}")
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        avg_memory = np.mean([m['memory_usage_percent'] for m in recent_metrics])
        avg_cpu = np.mean([m['cpu_usage_percent'] for m in recent_metrics])
        max_memory = max([m['memory_usage_percent'] for m in recent_metrics])
        max_cpu = max([m['cpu_usage_percent'] for m in recent_metrics])
        
        return {
            'window_minutes': window_minutes,
            'avg_memory_usage': avg_memory,
            'avg_cpu_usage': avg_cpu,
            'max_memory_usage': max_memory,
            'max_cpu_usage': max_cpu,
            'samples_count': len(recent_metrics)
        }

def performance_benchmark(func: Callable) -> Callable:
    """Decorator to benchmark function performance."""
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            logger.info(f"Performance benchmark - {func.__name__}: "
                       f"Time: {execution_time:.3f}s, Memory: {memory_used:.2f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {e}")
            raise
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = await func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            logger.info(f"Performance benchmark - {func.__name__}: "
                       f"Time: {execution_time:.3f}s, Memory: {memory_used:.2f}MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Async function {func.__name__} failed: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def memory_efficient_dataframe_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    # Convert object columns to category where appropriate
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def optimize_numpy_arrays(*arrays) -> List[np.ndarray]:
    """Optimize NumPy arrays for memory efficiency."""
    optimized = []
    
    for arr in arrays:
        if arr.dtype == np.float64:
            # Check if we can use float32 without significant precision loss
            if np.allclose(arr.astype(np.float32), arr, rtol=1e-6):
                optimized.append(arr.astype(np.float32))
            else:
                optimized.append(arr)
        elif arr.dtype == np.int64:
            # Check if we can use a smaller integer type
            if arr.min() >= np.iinfo(np.int32).min and arr.max() <= np.iinfo(np.int32).max:
                optimized.append(arr.astype(np.int32))
            else:
                optimized.append(arr)
        else:
            optimized.append(arr)
    
    return optimized

class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, redis_url: str = None):
        self.cache_manager = CacheManager(redis_url)
        self.async_processor = AsyncProcessor()
        self.performance_monitor = PerformanceMonitor()
        self.memory_tracker = MemoryTracker()
        self.db_optimizer = None
        
        # Start memory tracking
        self.memory_tracker.start_tracking()
        
    def set_database_optimizer(self, engine):
        """Set database optimizer with engine."""
        self.db_optimizer = DatabaseOptimizer(engine)
    
    def cached_function(self, ttl: int = 3600):
        """Decorator for caching function results."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = self.cache_manager._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                result = self.cache_manager.get(key)
                if result is not None:
                    return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.cache_manager.set(key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    async def optimize_strategy_calculation(self, strategy_func: Callable, 
                                          data: pd.DataFrame, 
                                          parameters: Dict[str, Any]) -> Any:
        """Optimize strategy calculation using multiple techniques."""
        
        # 1. Memory optimization
        optimized_data = memory_efficient_dataframe_processing(data.copy())
        
        # 2. Check cache first
        cache_key = self.cache_manager._generate_key(
            f"{strategy_func.__name__}_calculation",
            (hash(str(optimized_data.values.tobytes())), ),
            parameters
        )
        
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for strategy calculation: {strategy_func.__name__}")
            return cached_result
        
        # 3. Submit to async processor for CPU-intensive calculation
        task_id = await self.async_processor.submit_cpu_intensive_task(
            strategy_func, optimized_data, parameters
        )
        
        # 4. Get result and cache it
        result = await self.async_processor.get_task_result(task_id)
        self.cache_manager.set(cache_key, result, ttl=1800)  # 30 minutes
        
        return result
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        cache_stats = self.cache_manager.get_stats()
        performance_summary = self.performance_monitor.get_performance_summary()
        active_tasks = self.async_processor.get_active_tasks()
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'cache_performance': cache_stats,
            'system_performance': performance_summary,
            'active_background_tasks': len(active_tasks),
            'memory_snapshots': len(self.memory_tracker.snapshots),
            'optimization_recommendations': []
        }
        
        # Generate recommendations
        if cache_stats['hit_rate'] < 0.5:
            report['optimization_recommendations'].append(
                "Low cache hit rate - consider increasing cache TTL or adjusting cache strategy"
            )
        
        if performance_summary.get('avg_memory_usage', 0) > 80:
            report['optimization_recommendations'].append(
                "High memory usage detected - consider memory optimization techniques"
            )
        
        if performance_summary.get('avg_cpu_usage', 0) > 70:
            report['optimization_recommendations'].append(
                "High CPU usage detected - consider async processing or load balancing"
            )
        
        return report
    
    def cleanup_resources(self):
        """Clean up resources and perform garbage collection."""
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Clear memory cache if it's too large
        if len(self.cache_manager.memory_cache) > 500:
            keys_to_remove = list(self.cache_manager.memory_cache.keys())[:100]
            for key in keys_to_remove:
                del self.cache_manager.memory_cache[key]
            logger.info(f"Cleared {len(keys_to_remove)} cache entries")
        
        # Log current memory usage
        memory = psutil.virtual_memory()
        logger.info(f"Current memory usage: {memory.percent}%")

# Global optimizer instance
_global_optimizer = None

def get_performance_optimizer(redis_url: str = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(redis_url)
    return _global_optimizer 