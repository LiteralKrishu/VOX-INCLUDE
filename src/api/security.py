"""
VOX-INCLUDE: API Security Module

Implements authentication, rate limiting, and security features
for the REST API.
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict


class APIKeyManager:
    """
    Manages API key authentication.
    
    For production, integrate with a proper secrets manager
    or database-backed key storage.
    """
    
    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._setup_default_keys()
    
    def _setup_default_keys(self):
        """Setup default API keys from environment or generate demo keys."""
        # Check for environment-provided keys
        env_key = os.environ.get("VOX_API_KEY")
        if env_key:
            self.register_key(env_key, "env_user", ["read", "write"])
        
        # Demo key for development (disabled in production)
        if os.environ.get("VOX_ENV", "development") == "development":
            self.register_key("demo-key-for-dev-only", "demo", ["read"])
    
    def generate_key(self) -> str:
        """Generate a new secure API key."""
        return secrets.token_urlsafe(32)
    
    def register_key(
        self,
        key: str,
        user_id: str,
        permissions: list,
        expires_at: Optional[datetime] = None
    ) -> None:
        """Register a new API key."""
        key_hash = self._hash_key(key)
        self._keys[key_hash] = {
            "user_id": user_id,
            "permissions": permissions,
            "expires_at": expires_at,
            "created_at": datetime.now(),
            "last_used": None,
        }
    
    def validate_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return associated metadata.
        
        Returns None if invalid or expired.
        """
        key_hash = self._hash_key(key)
        
        if key_hash not in self._keys:
            return None
        
        key_data = self._keys[key_hash]
        
        # Check expiration
        if key_data["expires_at"] and datetime.now() > key_data["expires_at"]:
            return None
        
        # Update last used
        key_data["last_used"] = datetime.now()
        
        return key_data
    
    def has_permission(self, key: str, permission: str) -> bool:
        """Check if a key has a specific permission."""
        key_data = self.validate_key(key)
        if not key_data:
            return False
        return permission in key_data["permissions"]
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(key)
        if key_hash in self._keys:
            del self._keys[key_hash]
            return True
        return False
    
    def _hash_key(self, key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()


class RateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Limits requests per time window per client.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self._minute_buckets: Dict[str, list] = defaultdict(list)
        self._hour_buckets: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request is allowed for the given client.
        
        Returns True if within rate limits.
        """
        now = time.time()
        
        # Clean old entries
        self._cleanup(client_id, now)
        
        # Check minute limit
        if len(self._minute_buckets[client_id]) >= self.rpm:
            return False
        
        # Check hour limit
        if len(self._hour_buckets[client_id]) >= self.rph:
            return False
        
        # Record request
        self._minute_buckets[client_id].append(now)
        self._hour_buckets[client_id].append(now)
        
        return True
    
    def _cleanup(self, client_id: str, now: float) -> None:
        """Remove expired entries from buckets."""
        minute_ago = now - 60
        hour_ago = now - 3600
        
        self._minute_buckets[client_id] = [
            t for t in self._minute_buckets[client_id] if t > minute_ago
        ]
        self._hour_buckets[client_id] = [
            t for t in self._hour_buckets[client_id] if t > hour_ago
        ]
    
    def get_remaining(self, client_id: str) -> Dict[str, int]:
        """Get remaining requests for a client."""
        now = time.time()
        self._cleanup(client_id, now)
        
        return {
            "rpm_remaining": max(0, self.rpm - len(self._minute_buckets[client_id])),
            "rph_remaining": max(0, self.rph - len(self._hour_buckets[client_id])),
        }


class SecurityMiddleware:
    """
    Security utilities for API hardening.
    """
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """Sanitize user input text."""
        if not text:
            return ""
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        return text
    
    @staticmethod
    def validate_audio_size(audio_bytes: bytes, max_mb: float = 10.0) -> bool:
        """Validate audio file size."""
        max_bytes = max_mb * 1024 * 1024
        return len(audio_bytes) <= max_bytes
    
    @staticmethod
    def get_client_info(request) -> Dict[str, str]:
        """Extract client information from request."""
        return {
            "ip": getattr(request, "client", {}).get("host", "unknown"),
            "user_agent": request.headers.get("User-Agent", "unknown"),
        }


# FastAPI dependency functions
def get_api_key_or_none(request) -> Optional[str]:
    """Extract API key from request headers."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return request.headers.get("X-API-Key")


# Global instances
api_key_manager = APIKeyManager()
rate_limiter = RateLimiter()
