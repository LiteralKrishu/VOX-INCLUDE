"""Cognitive state estimation from multimodal signals."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from enum import Enum
import numpy as np

from ..utils.bayesian import BayesianUpdater


class CognitiveState(Enum):
    """Cognitive states that can be detected."""
    ENGAGED = "engaged"  # Active, focused, participating
    COGNITIVE_OVERLOAD = "cognitive_overload"  # Too much information, confused
    PRODUCTIVE_STRUGGLE = "productive_struggle"  # Confused but trying
    PASSIVE_DISENGAGEMENT = "passive_disengagement"  # Checked out, bored
    SOCIAL_ANXIETY = "social_anxiety"  # Nervous, hesitant
    FRUSTRATION = "frustration"  # Rising frustration
    CONFIDENCE = "confidence"  # Feeling confident
    NEUTRAL = "neutral"  # Baseline state


@dataclass
class CognitiveStateResult:
    """Result of cognitive state estimation."""
    state: str  # Primary cognitive state
    confidence: float  # Confidence score 0-1
    probabilities: Dict[str, float]  # All state probabilities
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    bayesian_confidence: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "state": self.state,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "contributing_factors": self.contributing_factors,
            "recommendations": self.recommendations,
            "bayesian_confidence": self.bayesian_confidence
        }


class CognitiveStateEstimator:
    """Estimates cognitive state from multimodal signals.
    
    Fuses:
    - Emotion recognition results (arousal, valence)
    - Intent patterns (questions, clarifications)
    - Behavioral signals (response time, speech rate)
    - Conversation metrics (misunderstandings, repetitions)
    """
    
    # Weight factors for different signal sources
    SIGNAL_WEIGHTS = {
        "emotion": 0.35,
        "intent": 0.25,
        "behavioral": 0.20,
        "conversation": 0.20
    }
    
    # State detection rules based on signal combinations
    STATE_RULES = {
        CognitiveState.COGNITIVE_OVERLOAD: {
            "emotion_signals": ["confused", "fearful"],
            "high_arousal": True,
            "fast_speech": True,
            "high_repetition": True,
        },
        CognitiveState.PRODUCTIVE_STRUGGLE: {
            "emotion_signals": ["confused", "neutral"],
            "moderate_arousal": True,
            "questions_high": True,
            "engagement_maintained": True,
        },
        CognitiveState.PASSIVE_DISENGAGEMENT: {
            "emotion_signals": ["sad", "neutral"],
            "low_arousal": True,
            "low_energy": True,
            "long_pauses": True,
        },
        CognitiveState.SOCIAL_ANXIETY: {
            "emotion_signals": ["fearful", "sad"],
            "low_volume": True,
            "hesitation": True,
            "short_responses": True,
        },
        CognitiveState.FRUSTRATION: {
            "emotion_signals": ["angry", "disgusted"],
            "high_arousal": True,
            "rising_energy": True,
            "repetition": True,
        },
        CognitiveState.ENGAGED: {
            "emotion_signals": ["happy", "neutral"],
            "moderate_arousal": True,
            "balanced_turn_taking": True,
        },
        CognitiveState.CONFIDENCE: {
            "emotion_signals": ["happy", "neutral"],
            "high_energy": True,
            "clear_speech": True,
        }
    }
    
    # Recommendations for each state
    STATE_RECOMMENDATIONS = {
        CognitiveState.COGNITIVE_OVERLOAD: [
            "Simplify content and reduce information density",
            "Break down complex concepts into smaller parts",
            "Provide visual aids or summaries",
            "Suggest a brief pause"
        ],
        CognitiveState.PRODUCTIVE_STRUGGLE: [
            "Provide encouragement without giving answers",
            "Offer hints or scaffolding",
            "Allow more time for processing"
        ],
        CognitiveState.PASSIVE_DISENGAGEMENT: [
            "Increase interactivity",
            "Change approach or topic",
            "Ask engaging questions",
            "Suggest a break if session is long"
        ],
        CognitiveState.SOCIAL_ANXIETY: [
            "Reduce spotlight on individual",
            "Provide private communication channels",
            "Use encouraging and supportive language",
            "Allow written responses as alternative"
        ],
        CognitiveState.FRUSTRATION: [
            "Acknowledge the difficulty",
            "Offer alternative explanations",
            "Slow down the pace",
            "Validate feelings before problem-solving"
        ],
        CognitiveState.ENGAGED: [
            "Maintain current approach",
            "Consider increasing challenge level",
            "Offer extension materials if appropriate"
        ],
        CognitiveState.CONFIDENCE: [
            "Encourage continued participation",
            "Offer leadership opportunities",
            "Provide more challenging content"
        ],
        CognitiveState.NEUTRAL: [
            "Continue monitoring",
            "Maintain current engagement level"
        ]
    }
    
    def __init__(self):
        """Initialize cognitive state estimator."""
        self._history: List[CognitiveStateResult] = []
        
        # Initialize Bayesian updater with uniform priors
        initial_priors = {s.value: 1.0/len(CognitiveState) for s in CognitiveState}
        self.bayesian = BayesianUpdater(initial_priors)
    
    def estimate(
        self,
        emotion_data: Optional[Dict[str, Any]] = None,
        intent_data: Optional[Dict[str, Any]] = None,
        behavioral_data: Optional[Dict[str, Any]] = None,
        conversation_metrics: Optional[Dict[str, Any]] = None
    ) -> CognitiveStateResult:
        """Estimate cognitive state from multimodal signals.
        
        Args:
            emotion_data: Emotion recognition results
                - emotion: str, primary emotion
                - arousal: float, emotional intensity
                - valence: float, positive/negative
                - confidence: float
            intent_data: Intent classification results
                - intent: str, detected intent
                - confidence: float
            behavioral_data: Behavioral signals
                - speech_rate: float, words per second
                - pause_duration: float, average pause
                - volume: float, normalized volume
                - response_latency: float, seconds
            conversation_metrics: From MemoryGraph
                - clarification_rate: float
                - misunderstanding_score: int
                - question_rate: float
                - repeated_questions: int
                
        Returns:
            CognitiveStateResult with estimated state
        """
        # Compute feature signals
        signals = self._compute_signals(
            emotion_data or {},
            intent_data or {},
            behavioral_data or {},
            conversation_metrics or {}
        )
        
        # Score each cognitive state (Instantaneous evidence)
        state_scores = self._score_states(signals)
        
        # Normalize to probabilities (P(Evidence|State))
        total = sum(state_scores.values()) or 1
        evidence_probs = {k.value: v / total for k, v in state_scores.items()}
        
        # Determine reliability based on signal quality
        # Lower reliability if low SNR (if available) or low confidence in inputs
        reliability = 1.0
        if emotion_data:
            reliability *= emotion_data.get("confidence", 0.8)
        if intent_data:
            reliability *= intent_data.get("confidence", 0.8)
            
        # Bayesian Update (Temporal smoothing + Priors)
        bayesian_probs = self.bayesian.update(evidence_probs, reliability)
        
        # Get primary state from Bayesian result
        primary_state_str = max(bayesian_probs, key=bayesian_probs.get)
        confidence = bayesian_probs[primary_state_str]
        
        # Get recommendations
        primary_state = CognitiveState(primary_state_str)
        recommendations = self.STATE_RECOMMENDATIONS.get(primary_state, [])
        
        result = CognitiveStateResult(
            state=primary_state_str,
            confidence=confidence,
            probabilities=bayesian_probs,
            contributing_factors=signals,
            recommendations=recommendations,
            bayesian_confidence=confidence
        )
        
        # Track history
        self._history.append(result)
        if len(self._history) > 10:
            self._history.pop(0)
        
        return result
    
    def _compute_signals(
        self,
        emotion_data: Dict,
        intent_data: Dict,
        behavioral_data: Dict,
        conversation_metrics: Dict
    ) -> Dict[str, float]:
        """Compute normalized signals from input data."""
        signals = {}
        
        # Emotion signals
        if emotion_data:
            signals["arousal"] = emotion_data.get("arousal", 0)
            signals["valence"] = emotion_data.get("valence", 0)
            signals["emotion_confidence"] = emotion_data.get("confidence", 0)
            
            emotion = emotion_data.get("emotion", "neutral")
            signals["is_negative_emotion"] = 1.0 if emotion in [
                "angry", "sad", "fearful", "disgusted"
            ] else 0.0
            signals["is_confused"] = 1.0 if emotion in ["confused", "fearful"] else 0.0
        
        # Intent signals
        if intent_data:
            intent = intent_data.get("intent", "")
            signals["is_question"] = 1.0 if intent == "question" else 0.0
            signals["is_clarification"] = 1.0 if intent == "clarification" else 0.0
        
        # Behavioral signals
        if behavioral_data:
            speech_rate = behavioral_data.get("speech_rate", 2.5)  # words/sec
            signals["fast_speech"] = max(0, (speech_rate - 3.5) / 2)  # > 3.5 wps is fast
            signals["slow_speech"] = max(0, (1.5 - speech_rate) / 1.5)  # < 1.5 wps is slow
            
            volume = behavioral_data.get("volume", 0.5)
            signals["low_volume"] = max(0, 0.3 - volume) / 0.3 if volume < 0.3 else 0
            signals["high_volume"] = max(0, volume - 0.7) / 0.3 if volume > 0.7 else 0
            
            pause = behavioral_data.get("pause_duration", 0.5)
            signals["long_pauses"] = max(0, (pause - 2.0) / 2.0)  # > 2s is long
        
        # Conversation signals
        if conversation_metrics:
            signals["clarification_rate"] = conversation_metrics.get("clarification_rate", 0)
            signals["misunderstanding"] = min(
                conversation_metrics.get("misunderstanding_score", 0) / 5, 1.0
            )
            signals["question_rate"] = conversation_metrics.get("question_rate", 0)
            signals["repeated_questions"] = min(
                conversation_metrics.get("repeated_questions", 0) / 3, 1.0
            )
        
        return signals
    
    def _score_states(self, signals: Dict[str, float]) -> Dict[CognitiveState, float]:
        """Score each cognitive state based on signals."""
        scores = {state: 0.0 for state in CognitiveState}
        
        # Cognitive Overload: high arousal + confusion + fast speech + repetition
        scores[CognitiveState.COGNITIVE_OVERLOAD] = (
            signals.get("is_confused", 0) * 0.3 +
            max(0, signals.get("arousal", 0)) * 0.2 +
            signals.get("fast_speech", 0) * 0.2 +
            signals.get("misunderstanding", 0) * 0.3
        )
        
        # Productive Struggle: confusion but still engaged
        scores[CognitiveState.PRODUCTIVE_STRUGGLE] = (
            signals.get("is_question", 0) * 0.4 +
            signals.get("is_confused", 0) * 0.3 +
            (1 - signals.get("slow_speech", 0)) * 0.3
        ) * (1 - signals.get("misunderstanding", 0) * 0.5)
        
        # Passive Disengagement: low arousal + slow + long pauses
        scores[CognitiveState.PASSIVE_DISENGAGEMENT] = (
            max(0, -signals.get("arousal", 0)) * 0.3 +
            signals.get("slow_speech", 0) * 0.3 +
            signals.get("long_pauses", 0) * 0.4
        )
        
        # Social Anxiety: low volume + hesitation + negative valence
        scores[CognitiveState.SOCIAL_ANXIETY] = (
            signals.get("low_volume", 0) * 0.4 +
            signals.get("long_pauses", 0) * 0.3 +
            max(0, -signals.get("valence", 0)) * 0.3
        )
        
        # Frustration: high arousal + negative valence + repetition
        scores[CognitiveState.FRUSTRATION] = (
            max(0, signals.get("arousal", 0)) * 0.3 +
            signals.get("is_negative_emotion", 0) * 0.3 +
            signals.get("repeated_questions", 0) * 0.4
        )
        
        # Engaged: balanced signals, moderate arousal, positive/neutral
        engagement_base = 1.0 - max(
            scores[CognitiveState.COGNITIVE_OVERLOAD],
            scores[CognitiveState.PASSIVE_DISENGAGEMENT],
            scores[CognitiveState.FRUSTRATION]
        )
        scores[CognitiveState.ENGAGED] = engagement_base * 0.5 + (
            signals.get("question_rate", 0) * 0.25 +
            (1 - signals.get("is_negative_emotion", 0)) * 0.25
        )
        
        # Confidence: high energy, positive valence
        scores[CognitiveState.CONFIDENCE] = (
            max(0, signals.get("valence", 0)) * 0.4 +
            signals.get("high_volume", 0) * 0.3 +
            (1 - signals.get("long_pauses", 0)) * 0.3
        )
        
        # Neutral: baseline when nothing else is strong
        max_other = max(v for k, v in scores.items() if k != CognitiveState.NEUTRAL)
        scores[CognitiveState.NEUTRAL] = max(0, 0.5 - max_other)
        
        return scores
    
    def get_trend(self) -> Optional[str]:
        """Get trend in cognitive state (improving/declining/stable)."""
        if len(self._history) < 3:
            return None
        
        # Simple trend: compare engagement/positive states over time
        positive_states = {CognitiveState.ENGAGED.value, CognitiveState.CONFIDENCE.value}
        
        recent = self._history[-3:]
        scores = [
            1.0 if r.state in positive_states else 
            -1.0 if r.state in {CognitiveState.FRUSTRATION.value, 
                               CognitiveState.COGNITIVE_OVERLOAD.value} else 0.0
            for r in recent
        ]
        
        trend = scores[-1] - scores[0]
        if trend > 0.5:
            return "improving"
        elif trend < -0.5:
            return "declining"
        return "stable"
    
    def clear_history(self) -> None:
        """Clear state history."""
        self._history.clear()
