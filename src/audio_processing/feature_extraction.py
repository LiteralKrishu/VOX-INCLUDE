"""Feature extraction module for audio processing."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from ..utils.config import config


@dataclass
class AudioFeatures:
    """Container for extracted audio features."""
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    mfcc_delta: Optional[np.ndarray] = None  # First derivative
    mfcc_delta2: Optional[np.ndarray] = None  # Second derivative
    pitch: Optional[np.ndarray] = None  # Fundamental frequency
    energy: Optional[np.ndarray] = None  # Frame energy/RMS
    spectral_centroid: Optional[np.ndarray] = None
    spectral_rolloff: Optional[np.ndarray] = None
    zero_crossing_rate: Optional[np.ndarray] = None
    snr: Optional[float] = None  # Signal-to-Noise Ratio in dB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary."""
        return {
            "mfcc": self.mfcc.tolist() if self.mfcc is not None else None,
            "pitch": self.pitch.tolist() if self.pitch is not None else None,
            "energy": self.energy.tolist() if self.energy is not None else None,
            "snr": self.snr
        }
    
    def get_aggregated(self) -> Dict[str, float]:
        """Get mean/std aggregated features for classification."""
        features = {}
        
        if self.mfcc is not None:
            features["mfcc_mean"] = np.mean(self.mfcc, axis=1).tolist()
            features["mfcc_std"] = np.std(self.mfcc, axis=1).tolist()
        
        if self.pitch is not None:
            valid_pitch = self.pitch[self.pitch > 0]
            if len(valid_pitch) > 0:
                features["pitch_mean"] = float(np.mean(valid_pitch))
                features["pitch_std"] = float(np.std(valid_pitch))
        
        if self.energy is not None:
            features["energy_mean"] = float(np.mean(self.energy))
            features["energy_std"] = float(np.std(self.energy))
            
        if self.snr is not None:
            features["snr"] = self.snr
        
        return features


class FeatureExtractor:
    """Extract acoustic features from audio for emotion recognition."""
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        n_mfcc: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None
    ):
        """Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length between frames
        """
        self._check_librosa()
        
        feature_config = config.features
        self.sample_rate = sample_rate or config.audio.get("sample_rate", 16000)
        self.n_mfcc = n_mfcc or feature_config.get("mfcc_coefficients", 40)
        self.n_fft = n_fft or feature_config.get("n_fft", 2048)
        self.hop_length = hop_length or feature_config.get("hop_length", 512)
    
    def _check_librosa(self) -> None:
        """Check if librosa is available."""
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for feature extraction. "
                "Install it with: pip install librosa"
            )
    
    def extract(
        self,
        audio: np.ndarray,
        compute_deltas: bool = True,
        compute_pitch: bool = True,
        compute_spectral: bool = True
    ) -> AudioFeatures:
        """Extract features from audio.
        
        Args:
            audio: Audio signal as numpy array
            compute_deltas: Compute MFCC derivatives
            compute_pitch: Extract pitch/F0
            compute_spectral: Extract spectral features
            
        Returns:
            AudioFeatures object containing extracted features
        """
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Extract features
        return self._extract_all(audio, compute_deltas, compute_pitch, compute_spectral)

    def _extract_all(self, audio, compute_deltas, compute_pitch, compute_spectral):
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Initialize features
        features = AudioFeatures(mfcc=mfcc)
        
        # Calculate SNR (Signal-to-Noise Ratio) proxy
        # using dynamic range of RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)[0]
        if len(rms) > 0:
            signal_power = np.percentile(rms, 95)
            noise_floor = np.percentile(rms, 5)
            if noise_floor > 0:
                features.snr = 20 * np.log10(signal_power / noise_floor)
            else:
                features.snr = 0.0  # Or max value if perfectly clean, but 0 indicates silence/issue
        
        features.energy = rms
        
        # Compute MFCC deltas
        if compute_deltas:
            features.mfcc_delta = librosa.feature.delta(mfcc)
            features.mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Extract pitch using pyin
        if compute_pitch:
            try:
                f0, voiced_flag, voiced_prob = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                features.pitch = f0
            except Exception:
                features.pitch = None
        
        # Extract spectral features
        if compute_spectral:
            features.spectral_centroid = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features.spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            
            features.zero_crossing_rate = librosa.feature.zero_crossing_rate(
                audio,
                frame_length=self.n_fft,
                hop_length=self.hop_length
            )[0]
        
        return features
    
    def extract_for_model(self, audio: np.ndarray) -> np.ndarray:
        """Extract features formatted for model input.
        
        Args:
            audio: Audio signal
            
        Returns:
            Feature matrix shaped for model input [features x frames]
        """
        features = self.extract(audio, compute_deltas=True)
        
        # Stack MFCC and deltas
        feature_matrix = np.vstack([
            features.mfcc,
            features.mfcc_delta if features.mfcc_delta is not None else np.zeros_like(features.mfcc),
            features.mfcc_delta2 if features.mfcc_delta2 is not None else np.zeros_like(features.mfcc)
        ])
        
        return feature_matrix
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Preprocess audio signal.
        
        Args:
            audio: Raw audio signal
            
        Returns:
            Preprocessed audio
        """
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        return audio
