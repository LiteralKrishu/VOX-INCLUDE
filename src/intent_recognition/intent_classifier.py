"""Intent classifier using pre-trained transformer models."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np

from ..utils.config import config


class Intent(Enum):
    """Supported intent categories."""
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    CLARIFICATION = "clarification"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    GREETING = "greeting"
    FAREWELL = "farewell"
    EXPRESSION = "expression"  # Emotional expression
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str  # Primary intent
    confidence: float  # Confidence score 0-1
    probabilities: Dict[str, float]  # All intent probabilities
    sub_intent: Optional[str] = None  # More specific intent category
    entities: Dict[str, str] = field(default_factory=dict)  # Extracted entities
    
    def to_dict(self) -> Dict:
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "sub_intent": self.sub_intent,
            "entities": self.entities
        }
    
    @property
    def is_confident(self) -> bool:
        """Check if prediction meets confidence threshold."""
        return self.confidence >= 0.5


class IntentClassifier:
    """Intent classification using pre-trained transformer models.
    
    Uses HuggingFace transformers for text-based intent recognition.
    Can be combined with prosodic features for enhanced accuracy.
    """
    
    # Intent patterns for rule-based fallback
    INTENT_PATTERNS = {
        "question": ["?", "what", "why", "how", "when", "where", "who", "which", "can you", "could you", "would you"],
        "request": ["please", "can you", "could you", "would you", "i need", "i want", "help me"],
        "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"],
        "farewell": ["bye", "goodbye", "see you", "take care", "goodnight"],
        "agreement": ["yes", "yeah", "sure", "okay", "ok", "agreed", "right", "correct"],
        "disagreement": ["no", "nope", "wrong", "incorrect", "disagree", "i don't think"],
        "clarification": ["what do you mean", "can you explain", "i don't understand", "pardon", "sorry"],
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        use_transformer: bool = True
    ):
        """Initialize intent classifier.
        
        Args:
            model_name: HuggingFace model for text classification
            device: Device to run model on
            use_transformer: Whether to use transformer model or rule-based
        """
        self.model_name = model_name or "facebook/bart-large-mnli"
        self.device = device
        self.use_transformer = use_transformer
        
        # Lazy load model
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        
        # Intent labels for zero-shot classification
        self._intent_labels = [
            "asking a question",
            "making a statement",
            "making a request",
            "asking for clarification",
            "expressing agreement",
            "expressing disagreement",
            "greeting someone",
            "saying goodbye",
            "expressing emotion"
        ]
        
        self._label_to_intent = {
            "asking a question": "question",
            "making a statement": "statement",
            "making a request": "request",
            "asking for clarification": "clarification",
            "expressing agreement": "agreement",
            "expressing disagreement": "disagreement",
            "greeting someone": "greeting",
            "saying goodbye": "farewell",
            "expressing emotion": "expression"
        }
    
    def load_model(self) -> None:
        """Load the pre-trained model for zero-shot classification."""
        if self._is_loaded or not self.use_transformer:
            return
        
        try:
            from transformers import pipeline
            
            print(f"Loading intent model: {self.model_name}")
            
            self._model = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            self._is_loaded = True
            
            print("Intent model loaded successfully")
            
        except ImportError as e:
            print(f"Transformers not available: {e}")
            print("Using rule-based intent classification")
            self.use_transformer = False
        except Exception as e:
            print(f"Failed to load intent model: {e}")
            print("Using rule-based intent classification")
            self.use_transformer = False
    
    def classify(
        self,
        text: str,
        prosodic_features: Optional[Dict[str, float]] = None
    ) -> IntentResult:
        """Classify the intent of input text.
        
        Args:
            text: Input text to classify
            prosodic_features: Optional prosodic features (pitch, energy, etc.)
            
        Returns:
            IntentResult with classification
        """
        if not text or not text.strip():
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                probabilities={"unknown": 1.0}
            )
        
        text = text.strip().lower()
        
        if self.use_transformer and not self._is_loaded:
            self.load_model()
        
        if self.use_transformer and self._model is not None:
            return self._classify_transformer(text, prosodic_features)
        else:
            return self._classify_rules(text, prosodic_features)
    
    def _classify_transformer(
        self,
        text: str,
        prosodic_features: Optional[Dict[str, float]] = None
    ) -> IntentResult:
        """Classify using transformer model."""
        result = self._model(text, self._intent_labels, multi_label=False)
        
        # Get top label
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        # Map to intent
        intent = self._label_to_intent.get(top_label, "unknown")
        
        # Build probabilities
        probabilities = {
            self._label_to_intent.get(label, "unknown"): score
            for label, score in zip(result["labels"], result["scores"])
        }
        
        # Adjust based on prosodic features if available
        if prosodic_features:
            intent, top_score = self._adjust_with_prosody(
                intent, top_score, probabilities, prosodic_features
            )
        
        return IntentResult(
            intent=intent,
            confidence=top_score,
            probabilities=probabilities
        )
    
    def _classify_rules(
        self,
        text: str,
        prosodic_features: Optional[Dict[str, float]] = None
    ) -> IntentResult:
        """Rule-based intent classification fallback."""
        text_lower = text.lower()
        
        # Check patterns
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = sum(1 for p in patterns if p in text_lower)
            if score > 0:
                scores[intent] = score
        
        # Determine primary intent
        if scores:
            max_intent = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[max_intent] / max(total, 1)
            probabilities = {k: v / total for k, v in scores.items()}
        else:
            # Default to statement
            max_intent = "statement"
            confidence = 0.5
            probabilities = {"statement": 0.5}
        
        # Quick question check
        if "?" in text:
            max_intent = "question"
            confidence = max(confidence, 0.7)
            probabilities["question"] = confidence
        
        return IntentResult(
            intent=max_intent,
            confidence=confidence,
            probabilities=probabilities
        )
    
    def _adjust_with_prosody(
        self,
        intent: str,
        confidence: float,
        probabilities: Dict[str, float],
        prosodic_features: Dict[str, float]
    ) -> Tuple[str, float]:
        """Adjust intent based on prosodic features.
        
        Rising pitch at end -> more likely question
        High energy -> more likely emphasis/request
        Low energy, slow speech -> possible confusion
        """
        pitch_slope = prosodic_features.get("pitch_slope", 0)
        energy_mean = prosodic_features.get("energy_mean", 0.5)
        
        # Rising intonation suggests question
        if pitch_slope > 0.1 and probabilities.get("question", 0) > 0.2:
            if intent != "question":
                intent = "question"
                confidence = max(confidence, 0.6)
        
        # High energy with request words suggests urgency
        if energy_mean > 0.7 and intent == "request":
            confidence = min(confidence + 0.1, 1.0)
        
        return intent, confidence
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
