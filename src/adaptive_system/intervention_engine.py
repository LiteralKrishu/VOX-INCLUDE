"""Intervention engine for adaptive system responses."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum


class InterventionType(Enum):
    """Types of interventions the system can perform."""
    SIMPLIFY_CONTENT = "simplify_content"
    PROVIDE_EXAMPLES = "provide_examples"
    SUGGEST_BREAK = "suggest_break"
    SLOW_PACING = "slow_pacing"
    INCREASE_CHALLENGE = "increase_challenge"
    OFFER_ENCOURAGEMENT = "offer_encouragement"
    ENABLE_PRIVATE_MODE = "enable_private_mode"
    PROVIDE_ALTERNATIVES = "provide_alternatives"
    VISUAL_SUMMARY = "visual_summary"
    REDUCE_INFORMATION = "reduce_information"
    AUTO_SIMPLIFY = "auto_simplify"
    MICRO_BREAK = "micro_break"
    REMOVE_SPOTLIGHT = "remove_spotlight"
    ENABLE_PRIVATE_CHANNELS = "enable_private_channels"
    INCREASE_CHALLENGE_DEPTH = "increase_challenge_depth"
    OFFER_EXTENSION = "offer_extension"
    ACTIVATE_ALTERNATIVE_MODALITY = "activate_alternative_modality"
    NO_INTERVENTION = "no_intervention"


@dataclass
class Intervention:
    """An intervention action to take."""
    type: InterventionType
    priority: int  # 1-10, higher = more urgent
    message: Optional[str] = None  # User-facing message
    action_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "type": self.type.value,
            "priority": self.priority,
            "message": self.message,
            "action_data": self.action_data
        }


class InterventionEngine:
    """Decides on interventions based on cognitive state and context.
    
    Uses a rule-based system with the following priorities:
    1. User safety and wellbeing (e.g., suggest breaks during fatigue)
    2. Comprehension support (e.g., simplify during overload)
    3. Engagement maintenance (e.g., increase challenge when bored)
    4. Emotional support (e.g., encouragement during frustration)
    """
    
    # Intervention strategies for each cognitive state
    STATE_INTERVENTIONS = {
        "cognitive_overload": [
            (InterventionType.AUTO_SIMPLIFY, 9, "I'm simplifying the content for you."),
            (InterventionType.PROVIDE_EXAMPLES, 8, "Here is a foundational example."),
            (InterventionType.VISUAL_SUMMARY, 7, "Here's a visual summary map."),
        ],
        "productive_struggle": [
            (InterventionType.OFFER_ENCOURAGEMENT, 5, "You're getting close, keep going!"),
            (InterventionType.PROVIDE_EXAMPLES, 4, "Let's look at a similar case."),
        ],
        "passive_disengagement": [ # Maps to Cognitive Fatigue
            (InterventionType.MICRO_BREAK, 8, "Let's take a quick micro-break."),
            (InterventionType.SLOW_PACING, 6, None),
            (InterventionType.INCREASE_CHALLENGE, 5, None),
        ],
        "social_anxiety": [
            (InterventionType.REMOVE_SPOTLIGHT, 9, None),
            (InterventionType.ENABLE_PRIVATE_CHANNELS, 8, "You can reply privately here."),
            (InterventionType.OFFER_ENCOURAGEMENT, 6, "Take your time, no rush."),
        ],
        "frustration": [
            (InterventionType.ACTIVATE_ALTERNATIVE_MODALITY, 9, "Let's try a different way of learning this."),
            (InterventionType.OFFER_ENCOURAGEMENT, 7, "It's okay to find this difficult."),
            (InterventionType.SLOW_PACING, 6, None),
        ],
        "engaged": [ # Maps to High Engagement
            (InterventionType.INCREASE_CHALLENGE_DEPTH, 4, "Want to dive deeper into this?"),
            (InterventionType.OFFER_EXTENSION, 3, "Here are some extra resources."),
        ],
        "confidence": [
            (InterventionType.INCREASE_CHALLENGE, 4, "Ready for next level?"),
        ],
        "neutral": [
            (InterventionType.NO_INTERVENTION, 1, None),
        ],
    }
    
    # Minimum confidence to trigger intervention
    CONFIDENCE_THRESHOLD = 0.5
    
    # Cooldown settings (prevent intervention spam)
    INTERVENTION_COOLDOWN_SECONDS = 30
    
    def __init__(self):
        """Initialize intervention engine."""
        self._last_interventions: Dict[str, float] = {}  # type -> timestamp
        self._intervention_history: List[Intervention] = []
    
    def decide(
        self,
        cognitive_state: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Intervention]:
        """Decide on interventions based on cognitive state.
        
        Args:
            cognitive_state: The detected cognitive state
            confidence: Confidence in the state detection
            context: Additional context (conversation metrics, etc.)
            
        Returns:
            List of interventions to perform, ordered by priority
        """
        interventions = []
        
        # Check confidence threshold
        if confidence < self.CONFIDENCE_THRESHOLD:
            return [Intervention(
                type=InterventionType.NO_INTERVENTION,
                priority=0
            )]
        
        # Get interventions for this state
        state_key = cognitive_state.lower()
        if state_key not in self.STATE_INTERVENTIONS:
            state_key = "neutral"
            
        # Special logic for "Rising Confusion" (derived state)
        if context and context.get("trend") == "declining" and state_key in ["cognitive_overload", "frustration"]:
             # Boost priority of simplification
             interventions.append(Intervention(
                 type=InterventionType.AUTO_SIMPLIFY,
                 priority=10,
                 message="I noticed this is getting tricky. I've simplified the next part.",
                 action_data=self._get_action_data(InterventionType.AUTO_SIMPLIFY, context)
             ))
        
        for int_type, priority, message in self.STATE_INTERVENTIONS[state_key]:
            # Adjust priority based on confidence
            adjusted_priority = int(priority * confidence)
            
            # Create intervention
            intervention = Intervention(
                type=int_type,
                priority=adjusted_priority,
                message=message,
                action_data=self._get_action_data(int_type, context)
            )
            interventions.append(intervention)
        
        # Sort by priority (highest first)
        interventions.sort(key=lambda x: x.priority, reverse=True)
        
        # Record history
        if interventions and interventions[0].type != InterventionType.NO_INTERVENTION:
            self._intervention_history.append(interventions[0])
            if len(self._intervention_history) > 20:
                self._intervention_history.pop(0)
        
        return interventions
    
    def _get_action_data(
        self,
        intervention_type: InterventionType,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get specific action data for an intervention type."""
        action_data = {}
        
        if intervention_type in [InterventionType.SIMPLIFY_CONTENT, InterventionType.AUTO_SIMPLIFY]:
            action_data["simplification_level"] = "high"
            action_data["strategies"] = ["shorter_sentences", "basic_vocabulary", "active_voice"]
            
        elif intervention_type == InterventionType.MICRO_BREAK:
             action_data["duration_seconds"] = 120
             action_data["activity"] = "breathing_exercise"
        
        elif intervention_type == InterventionType.SLOW_PACING:
            action_data["pacing_reduction"] = 0.3  # 30% slower
        
        elif intervention_type in [InterventionType.INCREASE_CHALLENGE, InterventionType.INCREASE_CHALLENGE_DEPTH]:
            action_data["difficulty_increase"] = 0.2  # 20% harder
            action_data["depth_level"] = "advanced"
        
        elif intervention_type == InterventionType.SUGGEST_BREAK:
            # Get session duration from context
            session_duration = 0
            if context:
                session_duration = context.get("session_duration_seconds", 0)
            action_data["session_duration"] = session_duration
            action_data["recommended_break_minutes"] = 5 if session_duration > 1800 else 2
        
        elif intervention_type in [InterventionType.REMOVE_SPOTLIGHT, InterventionType.ENABLE_PRIVATE_CHANNELS]:
            action_data["disable_spotlight"] = True
            action_data["allow_text_responses"] = True
            action_data["hide_peer_video"] = True
            
        elif intervention_type == InterventionType.ACTIVATE_ALTERNATIVE_MODALITY:
            action_data["modality"] = "visual_interactive"
            action_data["diagram_type"] = "flowchart"
        
        return action_data
    
    def get_primary_intervention(
        self,
        cognitive_state: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Intervention:
        """Get the single most important intervention."""
        interventions = self.decide(cognitive_state, confidence, context)
        return interventions[0] if interventions else Intervention(
            type=InterventionType.NO_INTERVENTION,
            priority=0
        )
    
    def get_intervention_history(self) -> List[Dict]:
        """Get recent intervention history."""
        return [i.to_dict() for i in self._intervention_history]
    
    def should_intervene(self, cognitive_state: str, confidence: float) -> bool:
        """Quick check if intervention is needed."""
        if confidence < self.CONFIDENCE_THRESHOLD:
            return False
        
        no_intervention_states = {"engaged", "neutral", "confidence"}
        return cognitive_state.lower() not in no_intervention_states
