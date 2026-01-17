"""Audio processing module for VOX-INCLUDE."""

from .audio_capture import AudioCapture
from .feature_extraction import FeatureExtractor

__all__ = ["AudioCapture", "FeatureExtractor"]
