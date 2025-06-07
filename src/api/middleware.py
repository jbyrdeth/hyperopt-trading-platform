"""
Middleware Components

Custom middleware for request processing, rate limiting, logging, and error handling.
"""

import time
import logging
import json
import uuid
from typing import Callable, Optional, Dict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse, Response
from .auth import (
    verify_api_key, get_api_key_info, check_rate_limit, 
    get_client_ip, add_security_headers, RequestLogger
)
from .models import ErrorResponse

logger = logging.getLogger(__name__)

# Request timing tracker
class RequestTimer:
    """Track request timing for performance monitoring."""
    
    def __init__(self):
        self.request_times = deque(maxlen=10000)  # Keep last 10k requests
        
    def record_request(self, duration: float, path: str):
        """Record a request's timing."""
        self.request_times.append({
            "duration": duration,
            "path": path,
            "timestamp": datetime.utcnow()
        })
        
    def get_average_response_time(self) -> float:
        """Get average response time in seconds."""
        if not self.request_times:
            return 0.0
        return sum(req["duration"] for req in self.request_times) / len(self.request_times)
        
    def get_slow_requests(self, threshold_seconds: float = 1.0) -> int:
        """Get count of slow requests above threshold."""
        return len([req for req in self.request_times if req["duration"] > threshold_seconds])

# Global request timer
request_timer = RequestTimer()

# Request logger instance
request_logger = RequestLogger()


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware.
    
    Applies rate limits based on API key and endpoint category.
    """
    # Skip rate limiting for health checks
    if request.url.path in ["/api/v1/health", "/api/v1/ping", "/metrics"]:
        return await call_next(request)
    
    # Get API key from header
    api_key = request.headers.get("X-API-Key")
    
    if api_key:
        # Get API key info
        key_info = get_api_key_info(api_key)
        
        if key_info:
            try:
                # Check rate limits
                rate_limit_info = check_rate_limit(request, api_key, key_info)
                
                # Add rate limit headers to response
                response = await call_next(request)
                response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(rate_limit_info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info["reset_time"]))
                
                return response
                
            except HTTPException as e:
                # Rate limit exceeded
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.detail
                )
    
    # No API key or invalid key - let auth middleware handle it
    return await call_next(request)


async def logging_middleware(request: Request, call_next):
    """
    Request logging middleware.
    
    Logs all requests with timing and response information.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Get API key for logging
    api_key = request.headers.get("X-API-Key")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Record timing
        request_timer.record_request(response_time, request.url.path)
        
        # Log request
        request_logger.log_request(
            request=request,
            api_key=api_key,
            response_status=response.status_code,
            response_time=response_time
        )
        
        # Add timing headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Request-ID"] = request_id
        
        # Add security headers
        add_security_headers(response, request)
        
        return response
        
    except Exception as e:
        # Calculate response time even for errors
        response_time = time.time() - start_time
        
        # Log error
        logger.error(f"Request {request_id} failed: {str(e)}")
        request_logger.log_request(
            request=request,
            api_key=api_key,
            response_status=500,
            response_time=response_time
        )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "error_message": "An internal server error occurred",
                "request_id": request_id
            }
        )


async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global error handler for unhandled exceptions.
    
    Provides consistent error responses and logging.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    # Log the error
    logger.error(f"Unhandled exception in request {request_id}: {str(exc)}", exc_info=True)
    
    # Determine error type and response
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": "HTTP_EXCEPTION",
                "error_message": exc.detail,
                "request_id": request_id
            }
        )
    elif isinstance(exc, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "error_code": "VALIDATION_ERROR", 
                "error_message": str(exc),
                "request_id": request_id
            }
        )
    else:
        # Generic server error
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_SERVER_ERROR",
                "error_message": "An unexpected error occurred",
                "request_id": request_id
            }
        )


# Request size limiting middleware
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB


async def request_size_middleware(request: Request, call_next: Callable) -> Response:
    """
    Request size limiting middleware.
    
    Prevents processing of overly large requests to protect server resources.
    """
    # Check content length header
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_SIZE:
            error_response = ErrorResponse(
                error_code="REQUEST_TOO_LARGE",
                error_message=f"Request size {content_length} bytes exceeds limit of {MAX_REQUEST_SIZE} bytes",
                request_id=getattr(request.state, "request_id", str(uuid.uuid4()))
            )
            
            return JSONResponse(
                status_code=413,
                content=error_response.dict()
            )
    
    return await call_next(request)


# CORS preflight handling middleware
async def cors_preflight_middleware(request: Request, call_next: Callable) -> Response:
    """
    Handle CORS preflight requests.
    """
    if request.method == "OPTIONS":
        response = Response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
        return response
    
    return await call_next(request)


async def performance_monitoring_middleware(request: Request, call_next: Callable) -> Response:
    """
    Performance monitoring middleware.
    
    Tracks request processing times and identifies performance issues.
    """
    start_time = time.time()
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # Record timing
        request_timer.record_request(processing_time, request.url.path)
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Record timing even for failed requests
        request_timer.record_request(processing_time, request.url.path)
        
        raise


# Health check bypass middleware
async def health_check_bypass_middleware(request: Request, call_next: Callable) -> Response:
    """
    Bypass certain middleware for health check endpoints.
    """
    if request.url.path == "/api/v1/health":
        # Minimal processing for health checks
        return JSONResponse(
            content={
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return await call_next(request) 