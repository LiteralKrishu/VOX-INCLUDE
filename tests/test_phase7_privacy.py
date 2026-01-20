"""
Phase 7 Privacy Tests - VOX-INCLUDE

Tests for privacy and ethical architecture components:
- AudioAnonymizer (differential privacy, secure deletion)
- ConsentManager (permission levels, audit logging)
- TransparencyDashboard (explainability)
- API endpoints for privacy management
"""

import pytest
import numpy as np
from datetime import datetime

from src.privacy import (
    AudioAnonymizer, DataMinimizer,
    ConsentManager, TransparencyDashboard,
    PermissionLevel, DataCategory
)


class TestAudioAnonymizer:
    """Tests for audio anonymization."""

    def test_extract_features_only(self):
        """Test that features are extracted without retaining raw audio."""
        anonymizer = AudioAnonymizer(epsilon=1.0)
        
        # Create dummy audio
        audio = np.random.randn(16000).astype(np.float32)  # 1 second
        
        features = anonymizer.extract_features_only(audio, sample_rate=16000)
        
        assert "mfcc_mean" in features
        assert "mfcc_std" in features
        assert "energy" in features
        assert len(features["mfcc_mean"]) == 13  # 13 MFCCs

    def test_differential_privacy_adds_noise(self):
        """Test that differential privacy adds noise to features."""
        anonymizer = AudioAnonymizer(epsilon=0.1)  # Low epsilon = more noise
        
        original = {"value": 1.0, "list": [1.0, 2.0, 3.0]}
        noisy = anonymizer._add_differential_privacy_noise(original)
        
        # Values should be different due to noise
        assert noisy["value"] != original["value"]
        assert noisy["list"] != original["list"]

    def test_speaker_id_anonymization(self):
        """Test speaker ID is properly anonymized."""
        anonymizer = AudioAnonymizer()
        
        speaker_id = "john_doe_123"
        anon_id = anonymizer.anonymize_speaker_id(speaker_id)
        
        # Should be irreversible hash
        assert anon_id != speaker_id
        assert len(anon_id) == 16  # Truncated hash
        
        # Same input with same salt should give same output
        anon_id2 = anonymizer.anonymize_speaker_id(speaker_id, session_salt="fixed")
        anon_id3 = anonymizer.anonymize_speaker_id(speaker_id, session_salt="fixed")
        assert anon_id2 == anon_id3

    def test_pii_redaction(self):
        """Test PII is redacted from text."""
        text = "Contact me at john@example.com or 555-123-4567"
        redacted = DataMinimizer.redact_pii(text)
        
        assert "[EMAIL]" in redacted
        assert "[PHONE]" in redacted
        assert "john@example.com" not in redacted


class TestConsentManager:
    """Tests for consent management."""

    def test_default_no_consent(self):
        """Test that default state is no consent."""
        manager = ConsentManager("user1", "session1")
        
        assert manager.current_level == PermissionLevel.NONE
        assert not manager.is_allowed(DataCategory.AUDIO)

    def test_set_permission_level(self):
        """Test setting permission levels."""
        manager = ConsentManager("user1", "session1")
        
        manager.set_permission_level(PermissionLevel.VOICE_ONLY)
        
        assert manager.current_level == PermissionLevel.VOICE_ONLY
        assert manager.is_allowed(DataCategory.AUDIO)
        assert manager.is_allowed(DataCategory.EMOTION)
        assert not manager.is_allowed(DataCategory.FACIAL)

    def test_opaque_mode(self):
        """Test right to opaqueness."""
        manager = ConsentManager("user1", "session1")
        
        manager.enable_opaque_mode()
        assert manager.is_opaque()
        
        manager.disable_opaque_mode()
        assert not manager.is_opaque()

    def test_audit_logging(self):
        """Test that actions are logged."""
        manager = ConsentManager("user1", "session1")
        
        manager.set_permission_level(PermissionLevel.VOICE_ONLY)
        manager.log_data_access(DataCategory.AUDIO, "Testing")
        
        assert len(manager.audit_log) >= 2  # At least consent + access

    def test_data_export(self):
        """Test data export functionality."""
        manager = ConsentManager("user1", "session1")
        manager.set_permission_level(PermissionLevel.VOICE_BEHAVIORAL)
        
        export = manager.export_user_data()
        
        assert export["user_id"] == "user1"
        assert export["current_permission_level"] == "VOICE_BEHAVIORAL"
        assert "consent_history" in export
        assert "audit_log" in export

    def test_data_deletion(self):
        """Test right to be forgotten."""
        manager = ConsentManager("user1", "session1")
        manager.set_permission_level(PermissionLevel.FULL_MULTIMODAL)
        
        result = manager.request_deletion()
        
        assert result["user_id"] == "user1"
        assert manager.current_level == PermissionLevel.NONE
        assert len(manager.consent_history) == 0


class TestTransparencyDashboard:
    """Tests for AI explainability."""

    def test_explain_decision(self):
        """Test decision explanation generation."""
        emotion = {"emotion": "joy", "confidence": 0.85}
        cognitive = {"state": "engaged", "confidence": 0.75}
        
        explanation = TransparencyDashboard.explain_decision(emotion, cognitive)
        
        assert "joy" in explanation["summary"]
        assert explanation["confidence_breakdown"]["emotion"] == 0.85
        assert len(explanation["factors"]) >= 2
        assert "audio_features" in explanation["data_used"]

    def test_processing_summary(self):
        """Test processing summary generation."""
        manager = ConsentManager("user1", "session1")
        manager.set_permission_level(PermissionLevel.VOICE_ONLY)
        
        summary = TransparencyDashboard.get_processing_summary(manager)
        
        assert summary["permission_level"] == "VOICE_ONLY"
        assert "audio" in summary["data_categories_active"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
