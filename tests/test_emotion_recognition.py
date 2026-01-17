"""Tests for emotion recognition module."""

import numpy as np
import pytest


class TestEmotionResult:
    """Test EmotionResult dataclass."""
    
    def test_emotion_result_creation(self):
        """Test creating EmotionResult."""
        from src.emotion_recognition import EmotionResult
        
        result = EmotionResult(
            emotion="happy",
            confidence=0.85,
            probabilities={"happy": 0.85, "neutral": 0.1, "sad": 0.05}
        )
        
        assert result.emotion == "happy"
        assert result.confidence == 0.85
        assert result.is_confident == True  # Default threshold is 0.6
    
    def test_emotion_result_to_dict(self):
        """Test converting EmotionResult to dictionary."""
        from src.emotion_recognition import EmotionResult
        
        result = EmotionResult(
            emotion="sad",
            confidence=0.7,
            probabilities={"sad": 0.7},
            arousal=-0.4,
            valence=-0.6
        )
        
        d = result.to_dict()
        assert d["emotion"] == "sad"
        assert d["valence"] == -0.6
        assert d["arousal"] == -0.4


class TestEmotionRecognizer:
    """Test EmotionRecognizer class."""
    
    def test_recognizer_initialization(self):
        """Test recognizer initializes without loading model."""
        from src.emotion_recognition import EmotionRecognizer
        
        recognizer = EmotionRecognizer()
        
        assert recognizer.is_loaded == False
        assert recognizer.sample_rate == 16000
    
    def test_supported_emotions(self):
        """Test getting supported emotions before loading."""
        from src.emotion_recognition import EmotionRecognizer
        
        recognizer = EmotionRecognizer()
        emotions = recognizer.supported_emotions
        
        assert "neutral" in emotions
        assert "happy" in emotions
        assert "angry" in emotions
    
    @pytest.mark.skipif(
        True,  # Skip by default - requires model download
        reason="Requires model download and GPU/CPU resources"
    )
    def test_prediction(self):
        """Test emotion prediction (requires model)."""
        from src.emotion_recognition import EmotionRecognizer
        
        recognizer = EmotionRecognizer()
        
        # Create test audio
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        
        result = recognizer.predict(audio)
        
        assert result.emotion in recognizer.supported_emotions
        assert 0 <= result.confidence <= 1


class TestSimplisticRecognizer:
    """Test fallback SimplisticEmotionRecognizer."""
    
    def test_simplistic_prediction(self):
        """Test simple feature-based prediction."""
        from src.emotion_recognition.models import SimplisticEmotionRecognizer
        
        recognizer = SimplisticEmotionRecognizer()
        
        # Create quiet audio (should predict sad/neutral)
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        
        result = recognizer.predict(audio)
        
        assert result.emotion in ["neutral", "sad", "happy", "angry"]
        assert 0 <= result.confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
