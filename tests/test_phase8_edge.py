"""
Phase 8 Edge Deployment Tests - VOX-INCLUDE

Tests for model optimization and edge deployment components:
- TFLite conversion utilities
- Offline inference engine
- Cultural adaptability
- API security
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from src.edge import (
    TFLiteConverter, ModelBenchmark,
    OfflineInferenceEngine, CulturalAdaptability
)
from src.api.security import APIKeyManager, RateLimiter, SecurityMiddleware


class TestTFLiteConverter:
    """Tests for TFLite model conversion."""

    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "models" / "tflite"
        converter = TFLiteConverter(str(output_dir))
        
        assert output_dir.exists()

    def test_get_model_info_missing(self):
        """Test handling of missing model file."""
        converter = TFLiteConverter()
        info = converter.get_model_info("nonexistent.tflite")
        
        assert "error" in info


class TestOfflineInferenceEngine:
    """Tests for offline inference."""

    def test_offline_mode_toggle(self):
        """Test offline mode can be toggled."""
        engine = OfflineInferenceEngine()
        
        assert not engine.is_offline()
        
        engine.set_offline_mode(True)
        assert engine.is_offline()
        
        engine.set_offline_mode(False)
        assert not engine.is_offline()

    def test_simple_emotion_fallback(self):
        """Test fallback emotion prediction works."""
        engine = OfflineInferenceEngine()
        
        # High energy → angry
        high_features = np.array([0.9, 0.8, 0.7])
        result = engine._simple_emotion_fallback(high_features)
        assert result["emotion"] == "angry"
        assert result["is_fallback"] == True
        
        # Low energy → sad
        low_features = np.array([0.1, 0.2, 0.15])
        result = engine._simple_emotion_fallback(low_features)
        assert result["emotion"] == "sad"

    def test_loaded_models_tracking(self):
        """Test loaded models list."""
        engine = OfflineInferenceEngine()
        
        assert len(engine.get_loaded_models()) == 0
        
        engine.unload_all()
        assert len(engine.get_loaded_models()) == 0


class TestCulturalAdaptability:
    """Tests for cultural adaptability."""

    def test_available_profiles(self):
        """Test all profiles are available."""
        adapter = CulturalAdaptability()
        profiles = adapter.get_available_profiles()
        
        assert "western" in profiles
        assert "eastern_asian" in profiles
        assert "neutral" in profiles

    def test_set_profile(self):
        """Test profile can be set."""
        adapter = CulturalAdaptability()
        
        assert adapter.set_cultural_profile("eastern_asian")
        assert adapter.current_profile == "eastern_asian"
        
        assert not adapter.set_cultural_profile("invalid")
        assert adapter.current_profile == "eastern_asian"  # Unchanged

    def test_confidence_adjustment(self):
        """Test emotion confidence is adjusted for culture."""
        adapter = CulturalAdaptability()
        
        raw_confidence = 0.6
        
        # Eastern Asian cultures express more subtly
        adapter.set_cultural_profile("eastern_asian")
        adjusted = adapter.adjust_emotion_confidence("joy", raw_confidence)
        assert adjusted > raw_confidence  # Boosted
        
        # Western is more direct
        adapter.set_cultural_profile("western")
        adjusted = adapter.adjust_emotion_confidence("joy", raw_confidence)
        assert adjusted == raw_confidence  # No change

    def test_intent_cultural_hints(self):
        """Test cultural hints are added for indirect cultures."""
        adapter = CulturalAdaptability()
        
        # Eastern Asian cultures are more indirect
        adapter.set_cultural_profile("eastern_asian")
        result = adapter.adjust_intent_interpretation("question", 0.8)
        
        assert len(result["cultural_hints"]) > 0
        assert "indirect request" in result["cultural_hints"][0].lower()


class TestAPIKeyManager:
    """Tests for API key management."""

    def test_generate_key(self):
        """Test key generation."""
        manager = APIKeyManager()
        key = manager.generate_key()
        
        assert len(key) > 20
        assert key != manager.generate_key()  # Unique

    def test_register_and_validate(self):
        """Test key registration and validation."""
        manager = APIKeyManager()
        key = "test-secret-key-123"
        
        manager.register_key(key, "test_user", ["read", "write"])
        
        result = manager.validate_key(key)
        assert result is not None
        assert result["user_id"] == "test_user"
        assert "read" in result["permissions"]

    def test_invalid_key(self):
        """Test invalid key returns None."""
        manager = APIKeyManager()
        
        assert manager.validate_key("nonexistent") is None

    def test_permission_check(self):
        """Test permission checking."""
        manager = APIKeyManager()
        key = "perm-test-key"
        
        manager.register_key(key, "user", ["read"])
        
        assert manager.has_permission(key, "read")
        assert not manager.has_permission(key, "write")

    def test_key_revocation(self):
        """Test key revocation."""
        manager = APIKeyManager()
        key = "revoke-test-key"
        
        manager.register_key(key, "user", ["read"])
        assert manager.validate_key(key) is not None
        
        manager.revoke_key(key)
        assert manager.validate_key(key) is None


class TestRateLimiter:
    """Tests for rate limiting."""

    def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        limiter = RateLimiter(requests_per_minute=10)
        
        for _ in range(10):
            assert limiter.is_allowed("client1")

    def test_blocks_over_limit(self):
        """Test requests over limit are blocked."""
        limiter = RateLimiter(requests_per_minute=5)
        
        for _ in range(5):
            limiter.is_allowed("client1")
        
        assert not limiter.is_allowed("client1")

    def test_separate_clients(self):
        """Test clients are tracked separately."""
        limiter = RateLimiter(requests_per_minute=2)
        
        limiter.is_allowed("client1")
        limiter.is_allowed("client1")
        
        assert not limiter.is_allowed("client1")
        assert limiter.is_allowed("client2")  # Different client

    def test_remaining_count(self):
        """Test remaining requests count."""
        limiter = RateLimiter(requests_per_minute=10)
        
        remaining = limiter.get_remaining("new_client")
        assert remaining["rpm_remaining"] == 10
        
        limiter.is_allowed("new_client")
        remaining = limiter.get_remaining("new_client")
        assert remaining["rpm_remaining"] == 9


class TestSecurityMiddleware:
    """Tests for security utilities."""

    def test_input_sanitization(self):
        """Test input is sanitized."""
        # Null byte removal
        text = "hello\x00world"
        sanitized = SecurityMiddleware.sanitize_input(text)
        assert "\x00" not in sanitized
        
        # Length truncation
        long_text = "a" * 20000
        sanitized = SecurityMiddleware.sanitize_input(long_text, max_length=100)
        assert len(sanitized) == 100

    def test_audio_size_validation(self):
        """Test audio size validation."""
        # 1MB should pass
        small_audio = b"x" * (1 * 1024 * 1024)
        assert SecurityMiddleware.validate_audio_size(small_audio, max_mb=10)
        
        # 15MB should fail
        large_audio = b"x" * (15 * 1024 * 1024)
        assert not SecurityMiddleware.validate_audio_size(large_audio, max_mb=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
