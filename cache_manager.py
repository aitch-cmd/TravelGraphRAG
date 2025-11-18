import json
import redis
from typing import Any, Optional, Callable
from functools import wraps
import hashlib
import config
from logger import get_logger

logger = get_logger(__name__)

class CacheManager:
    """Redis cache manager for efficient data caching."""
    
    def __init__(
        self,
        host: str = config.REDIS_HOST,
        port: int = config.REDIS_PORT,
        db: int = config.REDIS_DB,
        password: Optional[str] = getattr(config, 'REDIS_PASSWORD', None),
        default_ttl: int = 3600
    ):
        """
        Initialize Redis cache manager.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.default_ttl = default_ttl
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self.client is not None
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = f"{prefix}:{str(args)}:{str(sorted(kwargs.items()))}"
        return f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.is_available():
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_available():
            return False
        
        try:
            result = self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "embeddings:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.is_available():
            return 0
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    def flush_all(self) -> bool:
        """Clear all cache data."""
        if not self.is_available():
            return False
        
        try:
            self.client.flushdb()
            logger.warning("Flushed all cache data")
            return True
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False


def cached(prefix: str, ttl: Optional[int] = None):
    """
    Decorator for caching function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        
    Usage:
        @cached(prefix="embeddings", ttl=3600)
        def get_embedding(text: str):
            return model.encode(text)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheManager()
            
            # Generate cache key
            cache_key = cache._generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Returning cached result for {func.__name__}")
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# Global cache instance
cache_manager = CacheManager()