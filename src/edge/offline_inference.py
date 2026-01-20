"""
VOX-INCLUDE: Offline Inference Engine

Provides offline-capable inference for critical functionality
when network connectivity is unavailable.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np


class OfflineInferenceEngine:
    """
    Offline-capable inference engine using TFLite models.
    
    Prioritizes on-device processing with cloud fallback.
    """
    
    def __init__(self, models_dir: str = "models/tflite"):
        self.models_dir = Path(models_dir)
        self._interpreters: Dict[str, Any] = {}
        self._is_offline = False
        
    def set_offline_mode(self, offline: bool) -> None:
        """Enable or disable offline mode."""
        self._is_offline = offline
        
    def is_offline(self) -> bool:
        """Check if running in offline mode."""
        return self._is_offline
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a TFLite model for inference.
        
        Args:
            model_name: Name of model (e.g., 'emotion', 'intent')
            
        Returns:
            True if loaded successfully
        """
        try:
            import tensorflow as tf
            
            model_path = self.models_dir / f"{model_name}.tflite"
            
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                return False
            
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            self._interpreters[model_name] = {
                "interpreter": interpreter,
                "input_details": interpreter.get_input_details(),
                "output_details": interpreter.get_output_details(),
            }
            
            return True
            
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return False
    
    def predict(
        self,
        model_name: str,
        input_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Run inference using a loaded TFLite model.
        
        Args:
            model_name: Name of the loaded model
            input_data: Input tensor data
            
        Returns:
            Output tensor or None on failure
        """
        if model_name not in self._interpreters:
            if not self.load_model(model_name):
                return None
        
        try:
            model_info = self._interpreters[model_name]
            interpreter = model_info["interpreter"]
            input_details = model_info["input_details"]
            output_details = model_info["output_details"]
            
            # Ensure correct dtype
            input_dtype = input_details[0]['dtype']
            input_data = input_data.astype(input_dtype)
            
            # Reshape if needed
            expected_shape = input_details[0]['shape']
            if input_data.shape != tuple(expected_shape):
                # Try to reshape or pad
                input_data = np.reshape(input_data, expected_shape)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            return output_data
            
        except Exception as e:
            print(f"Inference failed for {model_name}: {e}")
            return None
    
    def predict_emotion_offline(
        self,
        audio_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Offline emotion prediction from audio features.
        
        Uses simplified rule-based fallback if TFLite model unavailable.
        """
        # Try TFLite model first
        output = self.predict("emotion", audio_features)
        
        if output is not None:
            emotions = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
            idx = np.argmax(output)
            return {
                "emotion": emotions[idx] if idx < len(emotions) else "neutral",
                "confidence": float(np.max(output)),
                "probabilities": {e: float(output[0][i]) for i, e in enumerate(emotions)},
                "is_offline": True
            }
        
        # Fallback: Simple energy-based heuristic
        return self._simple_emotion_fallback(audio_features)
    
    def _simple_emotion_fallback(self, features: np.ndarray) -> Dict[str, Any]:
        """Simple rule-based emotion fallback when model unavailable."""
        energy = np.mean(np.abs(features)) if features.size > 0 else 0.5
        
        if energy < 0.3:
            emotion = "sad"
        elif energy > 0.7:
            emotion = "angry"
        else:
            emotion = "neutral"
        
        return {
            "emotion": emotion,
            "confidence": 0.5,  # Low confidence for rule-based
            "probabilities": {},
            "is_offline": True,
            "is_fallback": True
        }
    
    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self._interpreters.keys())
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        self._interpreters.clear()


class CulturalAdaptability:
    """
    Cultural adaptability checks for emotion and intent recognition.
    
    Adjusts interpretation based on cultural context.
    """
    
    # Cultural expression intensity mappings
    CULTURAL_PROFILES = {
        "western": {
            "expression_intensity": 1.0,
            "directness": 0.8,
            "emotional_display": 0.9,
        },
        "eastern_asian": {
            "expression_intensity": 0.6,
            "directness": 0.5,
            "emotional_display": 0.5,
        },
        "south_asian": {
            "expression_intensity": 0.9,
            "directness": 0.6,
            "emotional_display": 0.8,
        },
        "middle_eastern": {
            "expression_intensity": 0.95,
            "directness": 0.7,
            "emotional_display": 0.85,
        },
        "latin": {
            "expression_intensity": 1.1,
            "directness": 0.75,
            "emotional_display": 1.0,
        },
        "neutral": {
            "expression_intensity": 1.0,
            "directness": 1.0,
            "emotional_display": 1.0,
        }
    }
    
    def __init__(self, default_profile: str = "neutral"):
        self.current_profile = default_profile
        
    def set_cultural_profile(self, profile: str) -> bool:
        """Set the cultural profile for interpretation."""
        if profile in self.CULTURAL_PROFILES:
            self.current_profile = profile
            return True
        return False
    
    def adjust_emotion_confidence(
        self,
        emotion: str,
        raw_confidence: float
    ) -> float:
        """
        Adjust emotion confidence based on cultural expression norms.
        
        Some cultures express emotions more subtly, so raw confidence
        may underestimate intensity.
        """
        profile = self.CULTURAL_PROFILES.get(self.current_profile, self.CULTURAL_PROFILES["neutral"])
        
        intensity_factor = profile["expression_intensity"]
        
        # For low-expression cultures, boost the confidence of detected emotions
        if intensity_factor < 1.0:
            adjusted = min(1.0, raw_confidence / intensity_factor)
        else:
            adjusted = raw_confidence
        
        return adjusted
    
    def adjust_intent_interpretation(
        self,
        intent: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Adjust intent interpretation for cultural directness.
        
        In indirect cultures, a 'request' might be phrased as a 'question'.
        """
        profile = self.CULTURAL_PROFILES.get(self.current_profile, self.CULTURAL_PROFILES["neutral"])
        
        directness = profile["directness"]
        
        # In indirect cultures, statements might be requests
        cultural_hints = []
        
        if directness < 0.6:
            if intent == "question":
                cultural_hints.append("May be indirect request")
            if intent == "statement":
                cultural_hints.append("May be implicit agreement/disagreement")
        
        return {
            "adjusted_intent": intent,
            "cultural_hints": cultural_hints,
            "directness_factor": directness,
            "profile": self.current_profile
        }
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available cultural profiles."""
        return list(self.CULTURAL_PROFILES.keys())
