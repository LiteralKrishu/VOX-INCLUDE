"""Tests for audio processing module."""

import numpy as np
import pytest


class TestFeatureExtractor:
    """Test feature extraction functionality."""
    
    def test_mfcc_extraction(self):
        """Test MFCC extraction from dummy audio."""
        from src.audio_processing import FeatureExtractor
        
        # Create dummy audio (1 second of sine wave)
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        extractor = FeatureExtractor()
        features = extractor.extract(audio)
        
        # Check MFCC shape
        assert features.mfcc is not None
        assert features.mfcc.shape[0] == 40  # n_mfcc coefficients
        assert features.mfcc.shape[1] > 0  # frames
    
    def test_energy_extraction(self):
        """Test energy feature extraction."""
        from src.audio_processing import FeatureExtractor
        
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.5
        
        extractor = FeatureExtractor()
        features = extractor.extract(audio)
        
        assert features.energy is not None
        assert len(features.energy) > 0
    
    def test_aggregated_features(self):
        """Test aggregated feature computation."""
        from src.audio_processing import FeatureExtractor
        
        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.5
        
        extractor = FeatureExtractor()
        features = extractor.extract(audio)
        aggregated = features.get_aggregated()
        
        assert "mfcc_mean" in aggregated
        assert "mfcc_std" in aggregated
        assert len(aggregated["mfcc_mean"]) == 40


class TestAudioCapture:
    """Test audio capture functionality."""
    
    def test_buffer_initialization(self):
        """Test audio buffer is properly initialized."""
        from src.audio_processing import AudioCapture
        
        capture = AudioCapture(buffer_duration=2.0)
        
        assert capture.sample_rate == 16000
        assert capture.buffer_size == 32000  # 2 seconds * 16000 Hz
        assert len(capture.audio_buffer) == 32000
    
    def test_get_recent_audio(self):
        """Test getting recent audio from buffer."""
        from src.audio_processing import AudioCapture
        
        capture = AudioCapture(buffer_duration=5.0)
        
        # Manually fill buffer with test data
        capture.audio_buffer[:16000] = np.ones(16000, dtype=np.float32)
        capture.buffer_index = 16000
        
        recent = capture.get_recent_audio(1.0)
        assert len(recent) == 16000
        assert np.all(recent == 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
