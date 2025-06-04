"""
Performance Middleware for FastAPI

This middleware integrates performance optimization tools with the FastAPI application,
providing request/response monitoring, caching, and performance analytics.
"""

import time
import asyncio
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import json
import hashlib

from ..utils.performance_optimizer import get_performance_optimizer, performance_benchmark

logger = logging.getLogger(__name__)

class PerformanceMiddleware:
    """FastAPI middleware for performance optimization and monitoring."""
    
    def __init__(self, app, redis_url: str = None):
        self.app = app
        self.optimizer = get_performance_optimizer(redis_url)
        self.request_stats = {
            'total_requests': 0,
            'total_response_time': 0.0,
            'error_count': 0,
            'cached_responses': 0
        }
        
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance monitoring."""
        start_time = time.time()
        
        # Track request
        self.request_stats['total_requests'] += 1
        
        # Check if this is a cacheable GET request
        cache_key = None
        if request.method == "GET" and self._is_cacheable_endpoint(request.url.path):
            cache_key = self._generate_cache_key(request)
            cached_response = self.optimizer.cache_manager.get(cache_key)
            
            if cached_response:
                self.request_stats['cached_responses'] += 1
                
                # Create response from cache
                response = JSONResponse(
                    content=cached_response['content'],
                    status_code=cached_response['status_code'],
                    headers=cached_response.get('headers', {})
                )
                response.headers["X-Cache-Status"] = "HIT"
                
                # Log cache hit
                execution_time = time.time() - start_time
                logger.info(f"Cache HIT for {request.url.path} - {execution_time:.3f}s")
                
                return response
        
        # Process request normally
        try:
            response = await call_next(request)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            self.request_stats['total_response_time'] += execution_time
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{execution_time:.3f}s"
            response.headers["X-Cache-Status"] = "MISS"
            
            # Cache successful GET responses
            if (cache_key and response.status_code == 200 and 
                request.method == "GET" and hasattr(response, 'body')):
                
                try:
                    # For JSONResponse, get the content
                    if isinstance(response, JSONResponse):
                        content = response.body.decode() if response.body else "{}"
                        cache_data = {
                            'content': json.loads(content),
                            'status_code': response.status_code,
                            'headers': dict(response.headers)
                        }
                        
                        # Cache for appropriate TTL based on endpoint
                        ttl = self._get_cache_ttl(request.url.path)
                        self.optimizer.cache_manager.set(cache_key, cache_data, ttl)
                        
                        logger.info(f"Cached response for {request.url.path} (TTL: {ttl}s)")
                        
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            # Log performance metrics
            self.optimizer.performance_monitor.log_metrics({
                'api_response_time': execution_time,
                'endpoint': request.url.path,
                'method': request.method,
                'status_code': response.status_code
            })
            
            # Log slow requests
            if execution_time > 2.0:
                logger.warning(f"Slow request: {request.method} {request.url.path} - {execution_time:.3f}s")
            
            return response
            
        except Exception as e:
            self.request_stats['error_count'] += 1
            execution_time = time.time() - start_time
            
            logger.error(f"Request failed: {request.method} {request.url.path} - {execution_time:.3f}s - {str(e)}")
            
            # Log error metrics
            self.optimizer.performance_monitor.log_metrics({
                'api_response_time': execution_time,
                'endpoint': request.url.path,
                'method': request.method,
                'error': True
            })
            
            raise
    
    def _is_cacheable_endpoint(self, path: str) -> bool:
        """Determine if an endpoint should be cached."""
        cacheable_patterns = [
            '/api/v1/strategies',
            '/api/v1/health',
            '/api/v1/data',
            '/api/v1/business/dashboard',
            '/api/v1/business/metrics',
            '/api/v1/business/portfolio',
            '/api/v1/monitoring'
        ]
        
        # Don't cache endpoints with dynamic parameters unless specifically listed
        non_cacheable_patterns = [
            '/api/v1/optimize',
            '/api/v1/validate',
            '/api/v1/export'
        ]
        
        for pattern in non_cacheable_patterns:
            if path.startswith(pattern):
                return False
        
        for pattern in cacheable_patterns:
            if path.startswith(pattern):
                return True
        
        return False
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        # Include path, query parameters, and relevant headers
        key_components = [
            request.url.path,
            str(sorted(request.query_params.items())),
            request.headers.get('X-API-Key', '')[:10]  # Include partial API key for user-specific caching
        ]
        
        key_string = ':'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cache_ttl(self, path: str) -> int:
        """Get cache TTL based on endpoint type."""
        if '/health' in path:
            return 30  # 30 seconds for health checks
        elif '/strategies' in path:
            return 300  # 5 minutes for strategy list
        elif '/dashboard' in path:
            return 60   # 1 minute for dashboard data
        elif '/metrics' in path:
            return 30   # 30 seconds for metrics
        elif '/portfolio' in path:
            return 120  # 2 minutes for portfolio data
        else:
            return 60   # Default 1 minute
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        total_requests = self.request_stats['total_requests']
        avg_response_time = (
            self.request_stats['total_response_time'] / total_requests 
            if total_requests > 0 else 0
        )
        
        cache_hit_rate = (
            self.request_stats['cached_responses'] / total_requests 
            if total_requests > 0 else 0
        )
        
        error_rate = (
            self.request_stats['error_count'] / total_requests 
            if total_requests > 0 else 0
        )
        
        return {
            'total_requests': total_requests,
            'average_response_time': avg_response_time,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
            'cached_responses': self.request_stats['cached_responses'],
            'error_count': self.request_stats['error_count']
        }

class AsyncRequestProcessor:
    """Process heavy requests asynchronously."""
    
    def __init__(self):
        self.optimizer = get_performance_optimizer()
        self.pending_tasks = {}
    
    async def submit_heavy_request(self, request_id: str, func: Callable, *args, **kwargs) -> str:
        """Submit a heavy request for background processing."""
        task_id = await self.optimizer.async_processor.submit_cpu_intensive_task(
            func, *args, **kwargs
        )
        
        self.pending_tasks[request_id] = task_id
        return task_id
    
    async def get_request_result(self, request_id: str) -> Any:
        """Get result of background request."""
        if request_id in self.pending_tasks:
            task_id = self.pending_tasks[request_id]
            result = await self.optimizer.async_processor.get_task_result(task_id)
            del self.pending_tasks[request_id]
            return result
        else:
            raise ValueError(f"Request {request_id} not found")
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get status of background request."""
        if request_id in self.pending_tasks:
            task_id = self.pending_tasks[request_id]
            active_tasks = self.optimizer.async_processor.get_active_tasks()
            
            if task_id in active_tasks:
                return {
                    'status': active_tasks[task_id]['status'],
                    'duration': active_tasks[task_id]['duration'],
                    'type': active_tasks[task_id]['type']
                }
        
        return {'status': 'not_found'}

# Global instances
_performance_middleware = None
_async_processor = None

def get_performance_middleware(redis_url: str = None) -> PerformanceMiddleware:
    """Get global performance middleware instance."""
    global _performance_middleware
    if _performance_middleware is None:
        _performance_middleware = PerformanceMiddleware(None, redis_url)
    return _performance_middleware

def get_async_processor() -> AsyncRequestProcessor:
    """Get global async processor instance."""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncRequestProcessor()
    return _async_processor 