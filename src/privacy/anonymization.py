"""
VOX-INCLUDE: Data Anonymization Module

Implements privacy-preserving techniques:
- Feature extraction only (discard raw audio)
- Speaker identity removal
- Differential privacy for aggregate stats
- Secure deletion of temporary buffers
"""

import numpy as np
from typing import Dict, Any, Optional
import hashlib
import secrets


class AudioAnonymizer:
    """
    Anonymizes audio data by removing speaker-identifying characteristics
    while preserving emotion-relevant features.
    """

    def __init__(self, epsilon: float = 1.0):
        """
        Args:
            epsilon: Differential privacy parameter. Lower = more privacy, less utility.
        """
        self.epsilon = epsilon
        self._temp_buffers: Dict[str, np.ndarray] = {}

    def extract_features_only(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Extract privacy-safe features from audio, discarding raw waveform.
        
        Only extracts aggregate statistics (MFCC means, spectral centroids)
        that don't allow voice reconstruction.
        
        Args:
            audio_data: Raw audio samples
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary of anonymized features
        """
        try:
            import librosa
            
            # Extract high-level features only (no raw spectrograms)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            
            # Aggregate to prevent reconstruction
            features = {
                "mfcc_mean": mfccs.mean(axis=1).tolist(),
                "mfcc_std": mfccs.std(axis=1).tolist(),
                "energy": float(np.sqrt(np.mean(audio_data ** 2))),
                "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(audio_data))),
            }
            
            # Apply differential privacy noise
            features = self._add_differential_privacy_noise(features)
            
            return features
            
        finally:
            # Secure deletion: overwrite audio buffer
            self._secure_delete(audio_data)

    def _add_differential_privacy_noise(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add calibrated Laplacian noise for differential privacy.
        """
        noisy_features = {}
        
        for key, value in features.items():
            if isinstance(value, list):
                noise = np.random.laplace(0, 1.0 / self.epsilon, len(value))
                noisy_features[key] = (np.array(value) + noise).tolist()
            elif isinstance(value, (int, float)):
                noise = np.random.laplace(0, 1.0 / self.epsilon)
                noisy_features[key] = value + noise
            else:
                noisy_features[key] = value
                
        return noisy_features

    def _secure_delete(self, data: np.ndarray) -> None:
        """
        Overwrite memory buffer with zeros before Python garbage collection.
        """
        if data is not None and hasattr(data, 'fill'):
            try:
                data.fill(0)
            except (ValueError, TypeError):
                pass  # Array may be read-only or non-writable

    def anonymize_speaker_id(self, speaker_id: str, session_salt: Optional[str] = None) -> str:
        """
        Create a pseudonymous identifier that cannot be reversed to original.
        
        Args:
            speaker_id: Original identifier
            session_salt: Optional per-session salt for unlinkability across sessions
            
        Returns:
            Anonymized, non-reversible identifier
        """
        if session_salt is None:
            session_salt = secrets.token_hex(16)
            
        combined = f"{speaker_id}:{session_salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def clear_all_buffers(self) -> None:
        """
        Securely clear all temporary audio buffers.
        """
        for key in list(self._temp_buffers.keys()):
            self._secure_delete(self._temp_buffers[key])
            del self._temp_buffers[key]


class DataMinimizer:
    """
    Ensures data minimization principles are followed.
    """

    @staticmethod
    def filter_to_essential(data: Dict[str, Any], allowed_fields: set) -> Dict[str, Any]:
        """
        Filter data to only essential fields.
        """
        return {k: v for k, v in data.items() if k in allowed_fields}

    @staticmethod
    def redact_pii(text: str) -> str:
        """
        Basic PII redaction for text transcripts.
        
        Replaces potential identifiers with [REDACTED].
        """
        import re
        
        # Email pattern
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers (various formats)
        text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
        
        # Credit card-like numbers
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        
        return text
