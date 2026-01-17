"""Emotion recognition models using pre-trained transformers."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

from ..utils.config import config


class Emotion(Enum):
    """Supported emotion categories."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"


@dataclass
class EmotionResult:
    """Result of emotion recognition."""
    emotion: str  # Primary emotion
    confidence: float  # Confidence score 0-1
    probabilities: Dict[str, float]  # All emotion probabilities
    arousal: Optional[float] = None  # Emotional intensity (-1 to 1)
    valence: Optional[float] = None  # Positive/negative (-1 to 1)
    features_used: List[str] = field(default_factory=list)
    momentum: Optional[float] = None  # Emotional momentum
    trend: Optional[str] = None  # Emotional trend
    
    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "arousal": self.arousal,
            "valence": self.valence,
            "momentum": self.momentum,
            "trend": self.trend
        }
    
    @property
    def is_confident(self) -> bool:
        """Check if prediction meets confidence threshold."""
        threshold = config.emotion.get("confidence_threshold", 0.6)
        return self.confidence >= threshold


class EmotionRecognizer:
    """Speech emotion recognition using pre-trained models."""
    
    # Valence-Arousal mapping for emotions
    EMOTION_VA_MAP = {
        "neutral": (0.0, 0.0),
        "happy": (0.8, 0.6),
        "sad": (-0.6, -0.4),
        "angry": (-0.5, 0.8),
        "fearful": (-0.7, 0.5),
        "disgusted": (-0.6, 0.3),
        "surprised": (0.3, 0.7)
    }


@dataclass
class TrajectoryPoint:
    """A point in the emotional trajectory."""
    emotion: str
    intensity: float  # Arousal or other intensity metric
    confidence: float
    timestamp: float  # Relative timestamp


class EmotionTrajectory:
    """Tracks emotional state over time to detect trends and momentum."""
    
    def __init__(self, history_size: int = 10, decay_factor: float = 0.9):
        self.history: List[TrajectoryPoint] = []
        self.history_size = history_size
        self.decay_factor = decay_factor
    
    def add(self, emotion: str, arousal: float, confidence: float):
        """Add a new data point."""
        import time
        self.history.append(TrajectoryPoint(
            emotion=emotion,
            intensity=arousal,
            confidence=confidence,
            timestamp=time.time()
        ))
        if len(self.history) > self.history_size:
            self.history.pop(0)
    
    def get_momentum(self) -> float:
        """Calculate emotional momentum (rate of change in arousal)."""
        if len(self.history) < 3:
            return 0.0
        
        # Simple linear regression of arousal over time
        times = np.array([p.timestamp for p in self.history])
        intensities = np.array([p.intensity for p in self.history])
        
        # Normalize time
        times = times - times[0]
        if times[-1] <= 0: return 0.0
        
        slope, _ = np.polyfit(times, intensities, 1)
        return float(slope)
    
    def get_dominant_trend(self) -> str:
        """Get the dominant emotional trend (rising/falling/stable)."""
        momentum = self.get_momentum()
        if momentum > 0.1: return "rising_intensity"
        if momentum < -0.1: return "falling_intensity"
        return "stable"


class EmotionRecognizer:
    """Speech emotion recognition using pre-trained models."""
    
    # Valence-Arousal mapping for emotions
    EMOTION_VA_MAP = {
        "neutral": (0.0, 0.0),
        "happy": (0.8, 0.6),
        "sad": (-0.6, -0.4),
        "angry": (-0.5, 0.8),
        "fearful": (-0.7, 0.5),
        "disgusted": (-0.6, 0.3),
        "surprised": (0.3, 0.7)
    }    
    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        """Initialize emotion recognizer.
        
        Args:
            model_name: HuggingFace model name (default: from config)
            device: Device to run model on ("cpu" or "cuda")
        """
        emotion_config = config.emotion
        self.model_name = model_name or emotion_config.get(
            "model_name",
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        self.device = device
        self.sample_rate = config.audio.get("sample_rate", 16000)
        
        # Lazy load model
        self._model = None
        self._processor = None
        self._is_loaded = False
        self._use_fallback = False
        
        # Emotion labels from model
        self._labels: List[str] = []
        
        # Trajectory tracking
        self._trajectory = EmotionTrajectory()
    
    def load_model(self) -> None:
        """Load the pre-trained model."""
        if self._is_loaded:
            return
        
        try:
            import torch
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            
            print(f"Loading emotion model: {self.model_name}")
            
            # Try loading with Auto classes first
            try:
                self._processor = AutoFeatureExtractor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self._model = AutoModelForAudioClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"AutoFeatureExtractor failed: {e}")
                # Fallback: Try Wav2Vec2 specific classes
                from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
                self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.model_name
                )
                self._model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    self.model_name
                )
            
            self._model.to(self.device)
            self._model.eval()
            
            # Get emotion labels
            if hasattr(self._model.config, 'id2label') and self._model.config.id2label:
                self._labels = list(self._model.config.id2label.values())
            else:
                # Default labels if not found in model config
                self._labels = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
            
            self._is_loaded = True
            print(f"Model loaded. Labels: {self._labels}")
            
        except ImportError as e:
            raise ImportError(
                f"transformers and torch are required. Install with: "
                f"pip install transformers torch. Error: {e}"
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using SimplisticEmotionRecognizer as fallback")
            self._use_fallback = True
            self._is_loaded = True
    
    def predict(self, audio: np.ndarray) -> EmotionResult:
        """Predict emotion from audio.
        
        Args:
            audio: Audio signal as numpy array (float32, mono, 16kHz)
            
        Returns:
            EmotionResult with emotion prediction
        """
        if not self._is_loaded:
            self.load_model()
        
        # Use fallback if model failed to load
        if self._use_fallback:
            fallback = SimplisticEmotionRecognizer()
            return fallback.predict(audio)
        
        import torch
        
        # Ensure correct format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Process audio
        inputs = self._processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get results
        probs_np = probs.cpu().numpy()[0]
        predicted_idx = np.argmax(probs_np)
        predicted_emotion = self._labels[predicted_idx].lower()
        confidence = float(probs_np[predicted_idx])
        
        # Build probability dict
        probabilities = {
            label.lower(): float(prob) 
            for label, prob in zip(self._labels, probs_np)
        }
        
        # Get valence/arousal
        valence, arousal = self._compute_va(probabilities)
        
        # Update trajectory
        self._trajectory.add(predicted_emotion, arousal, confidence)
        momentum = self._trajectory.get_momentum()
        trend = self._trajectory.get_dominant_trend()
        
        return EmotionResult(
            emotion=predicted_emotion,
            confidence=confidence,
            probabilities=probabilities,
            valence=valence,
            arousal=arousal,
            momentum=momentum,
            trend=trend,
            features_used=["wav2vec2_embeddings", "trajectory"]
        )
    
    def _compute_va(self, probabilities: Dict[str, float]) -> Tuple[float, float]:
        """Compute weighted valence and arousal from emotion probabilities."""
        valence = 0.0
        arousal = 0.0
        
        for emotion, prob in probabilities.items():
            if emotion in self.EMOTION_VA_MAP:
                v, a = self.EMOTION_VA_MAP[emotion]
                valence += v * prob
                arousal += a * prob
        
        return valence, arousal
    
    def predict_batch(self, audio_list: List[np.ndarray]) -> List[EmotionResult]:
        """Predict emotion for multiple audio samples.
        
        Args:
            audio_list: List of audio arrays
            
        Returns:
            List of EmotionResults
        """
        return [self.predict(audio) for audio in audio_list]
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        if self._is_loaded:
            return [l.lower() for l in self._labels]
        return list(self.EMOTION_VA_MAP.keys())


class SimplisticEmotionRecognizer:
    """Fallback emotion recognizer using simple audio features.
    
    Used when pre-trained models are not available.
    """
    
    def __init__(self):
        from ..audio_processing import FeatureExtractor
        self.feature_extractor = FeatureExtractor()
    
    def predict(self, audio: np.ndarray) -> EmotionResult:
        """Simple rule-based emotion estimation from audio features."""
        features = self.feature_extractor.extract(audio)
        
        # Simple heuristics based on audio characteristics
        energy_mean = float(np.mean(features.energy)) if features.energy is not None else 0.5
        
        # Get pitch statistics
        pitch = features.pitch
        pitch_mean = 0.0
        pitch_var = 0.0
        if pitch is not None:
            valid_pitch = pitch[~np.isnan(pitch) & (pitch > 0)]
            if len(valid_pitch) > 0:
                pitch_mean = float(np.mean(valid_pitch))
                pitch_var = float(np.var(valid_pitch))
        
        # Simple rules (placeholder - needs proper training)
        if energy_mean > 0.1 and pitch_var > 1000:
            emotion = "angry"
            confidence = 0.5
        elif energy_mean < 0.02:
            emotion = "sad"
            confidence = 0.4
        elif pitch_mean > 200 and energy_mean > 0.05:
            emotion = "happy"
            confidence = 0.45
        else:
            emotion = "neutral"
            confidence = 0.6
        
        return EmotionResult(
            emotion=emotion,
            confidence=confidence,
            probabilities={emotion: confidence},
            features_used=["mfcc", "energy", "pitch"]
        )
