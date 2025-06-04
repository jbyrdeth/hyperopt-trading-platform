"""
Authentication and Security Module

Handles API key authentication, rate limiting, and security measures for the 
trading strategy optimization API.
"""

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import APIKeyHeader
from typing import Dict, Optional
import time
import hashlib
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
import os

# Make Redis optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# API Key configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting configuration
class RateLimiter:
    """
    Comprehensive rate limiting system.
    """
    
    def __init__(self):
        # In-memory rate limiting (for development)
        self.requests = defaultdict(lambda: deque())
        
        # Try to use Redis for production (if available)
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()  # Test connection
                logger.info("✅ Redis connected for rate limiting")
            except Exception as e:
                logger.warning(f"⚠️  Redis not available, using in-memory rate limiting: {e}")
        else:
            logger.info("📝 Redis not installed, using in-memory rate limiting")
    
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, dict]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            key: Unique identifier (API key + endpoint)
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        
        if self.redis_client:
            return self._redis_rate_limit(key, limit, window_seconds, current_time)
        else:
            return self._memory_rate_limit(key, limit, window_seconds, current_time)
    
    def _redis_rate_limit(self, key: str, limit: int, window_seconds: int, current_time: float) -> tuple[bool, dict]:
        """Redis-based rate limiting."""
        try:
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, current_time - window_seconds)
            pipe.zcard(key)
            pipe.zadd(key, {str(current_time): current_time})
            pipe.expire(key, window_seconds)
            results = pipe.execute()
            
            request_count = results[1]
            
            rate_limit_info = {
                "limit": limit,
                "remaining": max(0, limit - request_count),
                "reset_time": current_time + window_seconds,
                "window_seconds": window_seconds
            }
            
            return request_count <= limit, rate_limit_info
            
        except Exception as e:
            logger.error(f"Redis rate limiting error: {e}")
            # Fallback to memory-based rate limiting
            return self._memory_rate_limit(key, limit, window_seconds, current_time)
    
    def _memory_rate_limit(self, key: str, limit: int, window_seconds: int, current_time: float) -> tuple[bool, dict]:
        """Memory-based rate limiting."""
        requests = self.requests[key]
        
        # Remove old requests outside the window
        while requests and requests[0] <= current_time - window_seconds:
            requests.popleft()
        
        # Add current request
        requests.append(current_time)
        
        rate_limit_info = {
            "limit": limit,
            "remaining": max(0, limit - len(requests)),
            "reset_time": current_time + window_seconds,
            "window_seconds": window_seconds
        }
        
        return len(requests) <= limit, rate_limit_info


# Global rate limiter instance
rate_limiter = RateLimiter()

# Rate limit configurations per endpoint category
RATE_LIMITS = {
    "general": {"limit": 100, "window": 3600},        # 100 requests/hour
    "optimization": {"limit": 10, "window": 3600},    # 10 requests/hour
    "data": {"limit": 1000, "window": 3600},          # 1000 requests/hour
    "export": {"limit": 50, "window": 3600},          # 50 requests/hour
    "health": {"limit": 1000, "window": 60},          # 1000 requests/minute
}

# Valid API keys (in production, these would be in a secure database)
VALID_API_KEYS = {
    # Development keys
    "dev_key_123": {
        "name": "Development Key",
        "permissions": ["read", "write", "admin"],
        "rate_limit_multiplier": 1.0,
        "created_at": "2024-01-01T00:00:00Z"
    },
    "test_key_456": {
        "name": "Test Key", 
        "permissions": ["read"],
        "rate_limit_multiplier": 0.5,
        "created_at": "2024-01-01T00:00:00Z"
    },
    # Production keys would be loaded from environment or database
}

# Load API keys from environment in production
def load_api_keys_from_env():
    """Load API keys from environment variables."""
    env_keys = os.getenv("API_KEYS", "")
    if env_keys:
        try:
            import json
            keys = json.loads(env_keys)
            VALID_API_KEYS.update(keys)
            logger.info(f"Loaded {len(keys)} API keys from environment")
        except Exception as e:
            logger.error(f"Failed to load API keys from environment: {e}")

# Load keys on module import
load_api_keys_from_env()


def get_api_key_info(api_key: str) -> Optional[Dict]:
    """Get API key information."""
    return VALID_API_KEYS.get(api_key)


def verify_api_key(api_key: Optional[str] = Depends(API_KEY_HEADER)) -> Dict:
    """
    Verify API key authentication.
    
    Args:
        api_key: API key from header
        
    Returns:
        API key information
        
    Raises:
        HTTPException: If authentication fails
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "MISSING_API_KEY",
                "error_message": "API key is required. Include X-API-Key header."
            },
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    key_info = get_api_key_info(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "INVALID_API_KEY",
                "error_message": "Invalid API key provided."
            },
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    return key_info


def require_permission(permission: str):
    """
    Dependency to require specific permission.
    
    Args:
        permission: Required permission (read, write, admin)
    """
    def check_permission(key_info: Dict = Depends(verify_api_key)):
        if permission not in key_info.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "INSUFFICIENT_PERMISSIONS",
                    "error_message": f"Permission '{permission}' required."
                }
            )
        return key_info
    
    return check_permission


def get_rate_limit_category(path: str) -> str:
    """Determine rate limit category based on request path."""
    if "/optimize" in path:
        return "optimization"
    elif "/data" in path:
        return "data"
    elif "/export" in path:
        return "export"
    elif "/health" in path:
        return "health"
    else:
        return "general"


def check_rate_limit(request: Request, api_key: str, key_info: Dict) -> Dict:
    """
    Check rate limiting for the request.
    
    Args:
        request: FastAPI request object
        api_key: API key
        key_info: API key information
        
    Returns:
        Rate limit information
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Determine rate limit category
    category = get_rate_limit_category(request.url.path)
    rate_config = RATE_LIMITS[category]
    
    # Apply rate limit multiplier for the API key
    multiplier = key_info.get("rate_limit_multiplier", 1.0)
    effective_limit = int(rate_config["limit"] * multiplier)
    
    # Create unique key for this API key + endpoint category
    rate_key = f"rate_limit:{api_key}:{category}"
    
    # Check rate limit
    is_allowed, rate_info = rate_limiter.is_allowed(
        rate_key, 
        effective_limit, 
        rate_config["window"]
    )
    
    if not is_allowed:
        reset_time = datetime.fromtimestamp(rate_info["reset_time"])
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "error_message": f"Rate limit exceeded for {category} endpoints.",
                "rate_limit": {
                    "limit": effective_limit,
                    "remaining": 0,
                    "reset_time": reset_time.isoformat(),
                    "category": category
                }
            },
            headers={
                "X-RateLimit-Limit": str(effective_limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(rate_info["reset_time"])),
                "Retry-After": str(rate_config["window"])
            }
        )
    
    return rate_info


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded headers (when behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct connection
    return request.client.host if request.client else "unknown"


def hash_api_key(api_key: str) -> str:
    """Hash API key for logging (to avoid exposing keys in logs)."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


# Security headers middleware
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self'",
}


def add_security_headers(response, request: Request):
    """Add security headers to response."""
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    
    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit_info"):
        rate_info = request.state.rate_limit_info
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(int(rate_info["reset_time"]))


# Request logging
class RequestLogger:
    """Request logging utility."""
    
    def __init__(self):
        self.logger = logging.getLogger("api.requests")
    
    def log_request(self, request: Request, api_key: Optional[str] = None, 
                   response_status: Optional[int] = None, 
                   response_time: Optional[float] = None):
        """Log API request."""
        client_ip = get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if api_key:
            log_data["api_key_hash"] = hash_api_key(api_key)
        
        if response_status:
            log_data["status_code"] = response_status
        
        if response_time:
            log_data["response_time_ms"] = round(response_time * 1000, 2)
        
        self.logger.info(f"API Request: {log_data}")


# Global request logger
request_logger = RequestLogger() 