
import sys
import os
import unittest
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adaptive_system import InterventionEngine, InterventionType, OutputGenerator

class TestPhase5(unittest.TestCase):
    def setUp(self):
        self.engine = InterventionEngine()
        self.generator = OutputGenerator()
        
    def test_rising_confusion_logic(self):
        print("\n--- Testing Rising Confusion Logic ---")
        # Simulate Cognitive Overload with declining trend (Rising Confusion)
        context = {"trend": "declining"}
        interventions = self.engine.decide("cognitive_overload", 0.8, context)
        
        # Expect AUTO_SIMPLIFY to be top priority
        top_intervention = interventions[0]
        print(f"Top Intervention: {top_intervention.type}")
        
        self.assertEqual(top_intervention.type, InterventionType.AUTO_SIMPLIFY)
        self.assertEqual(top_intervention.priority, 10) # Highest priority boosted
        
    def test_cognitive_fatigue_logic(self):
        print("\n--- Testing Cognitive Fatigue Logic ---")
        # Simulate Passive Disengagement (Cognitive Fatigue)
        context = {"session_duration_seconds": 2000}
        interventions = self.engine.decide("passive_disengagement", 0.7, context)
        
        # Expect MICRO_BREAK
        top_intervention = interventions[0]
        print(f"Top Intervention: {top_intervention.type}")
        self.assertIn(top_intervention.type, [InterventionType.MICRO_BREAK, InterventionType.SLOW_PACING])
        
    def test_output_adaptation(self):
        print("\n--- Testing Output Adaptation ---")
        original_text = "The physiological manifestations of stress are multifaceted and complex."
        
        # Test Simplification
        adapted = self.generator.adapt_content(
            text=original_text,
            intervention_type="auto_simplify",
            parameters={"simplification_level": "high"}
        )
        print(f"Original: {original_text}")
        print(f"Adapted (Simplify): {adapted.adapted_text}")
        self.assertIn("[SIMPLIFIED", adapted.adapted_text)
        
        # Test Visual Summary
        adapted_visual = self.generator.adapt_content(
            text=original_text,
            intervention_type="visual_summary",
            parameters={"diagram_type": "mind_map"}
        )
        print(f"Adapted (Visual): {adapted_visual.visual_aids}")
        self.assertTrue(len(adapted_visual.visual_aids) > 0)
        self.assertEqual(adapted_visual.visual_aids[0]["type"], "mind_map")

if __name__ == "__main__":
    unittest.main()
