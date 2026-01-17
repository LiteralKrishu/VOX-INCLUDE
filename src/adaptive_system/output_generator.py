"""Adaptive output generator for tailoring content to cognitive state."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import random

@dataclass
class AdaptedContent:
    """adapted content output."""
    original_text: str
    adapted_text: str
    adaptation_type: str
    visual_aids: List[Dict[str, Any]]
    difficulty_score: float

class OutputGenerator:
    """Generates and adapts content based on intervention needs."""

    def __init__(self):
        """Initialize output generator."""
        self._simplification_rules = {
            "high": {
                "max_sentence_length": 10,
                "vocabulary": "basic"
            },
            "moderate": {
                "max_sentence_length": 15,
                "vocabulary": "standard"
            }
        }

    def adapt_content(
        self,
        text: str,
        intervention_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> AdaptedContent:
        """Adapt content based on intervention parameters.
        
        Args:
            text: Original content text
            intervention_type: Type of intervention (e.g. 'auto_simplify')
            parameters: Specific parameters from the intervention engine
            
        Returns:
            AdaptedContent object
        """
        if not parameters:
            parameters = {}
            
        adapted_text = text
        visual_aids = []
        difficulty = 0.5
        
        # 1. Simplification Logic
        if intervention_type in ["auto_simplify", "simplify_content"]:
            level = parameters.get("simplification_level", "moderate")
            adapted_text = self._simplify_text(text, level)
            difficulty = 0.3
            
        # 2. Paraphrasing Logic (for variety or explanation)
        elif intervention_type in ["provide_examples", "provide_alternatives"]:
            adapted_text = self._generate_paraphrase(text)
            difficulty = 0.5
            
        # 3. Visual Summary Logic
        elif intervention_type in ["visual_summary", "activate_alternative_modality"]:
            visual_aids = self._generate_visual_aids(text, parameters.get("diagram_type"))
            # heavily reduce text when showing visual summary
            adapted_text = self._summarize_text(text)
            difficulty = 0.4
            
        # 4. Challenge Increase
        elif intervention_type in ["increase_challenge", "increase_challenge_depth"]:
            adapted_text = self._increase_complexity(text)
            difficulty = 0.8

        return AdaptedContent(
            original_text=text,
            adapted_text=adapted_text,
            adaptation_type=intervention_type,
            visual_aids=visual_aids,
            difficulty_score=difficulty
        )

    def _simplify_text(self, text: str, level: str) -> str:
        """Mock simplification logic. REPLACE with NLP model."""
        # Simple rule-based mock
        if level == "high":
            return f"[SIMPLIFIED-HIGH] {text}"
        return f"[SIMPLIFIED-MODERATE] {text}"

    def _generate_paraphrase(self, text: str) -> str:
        """Mock paraphrase logic."""
        prefix = random.choice([
            "In other words, ",
            "To put it simply, ",
            "Another way to look at this is, "
        ])
        return f"{prefix}{text}"

    def _summarize_text(self, text: str) -> str:
        """Mock summarization."""
        # Just take first sentence or return truncated
        return text.split('.')[0] + "." if '.' in text else text

    def _increase_complexity(self, text: str) -> str:
        """Mock complexity increase."""
        return f"[ADVANCED INSIGHT] Regarding '{text}', consider the deeper implications..."

    def _generate_visual_aids(self, text: str, diagram_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Mock visual aid generation."""
        dtype = diagram_type or "concept_map"
        return [{
            "type": dtype,
            "title": "Visual Summary",
            "data_points": ["Point A", "Point B", "Point C"],
            "description": f"A {dtype} representing the key concepts."
        }]
