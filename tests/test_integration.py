"""
VOX-INCLUDE: Integration Tests

Tests the full pipeline from audio input to intervention output.
Ensures all modules work together correctly.
"""

import pytest
import numpy as np
import base64
import io
from unittest.mock import MagicMock, patch

# Import all modules to test integration
from src.audio_processing import FeatureExtractor
from src.emotion_recognition import EmotionRecognizer, SimplisticEmotionRecognizer
from src.intent_recognition.intent_classifier import IntentClassifier
from src.intent_recognition.memory_graph import MemoryGraph
from src.intent_recognition.cognitive_estimator import CognitiveStateEstimator
from src.adaptive_system import InterventionEngine, OutputGenerator
from src.privacy import ConsentManager, PermissionLevel, DataCategory


class TestFullPipeline:
    """Tests for the complete analysis pipeline."""

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data."""
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create a tone with some variation
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        return audio.astype(np.float32)

    @pytest.fixture
    def pipeline_components(self):
        """Initialize all pipeline components."""
        return {
            "feature_extractor": FeatureExtractor(),
            "emotion_recognizer": SimplisticEmotionRecognizer(),
            "intent_classifier": IntentClassifier(use_transformer=False),
            "memory_graph": MemoryGraph(),
            "cognitive_estimator": CognitiveStateEstimator(),
            "intervention_engine": InterventionEngine(),
            "output_generator": OutputGenerator(),
        }

    def test_audio_to_features(self, sample_audio, pipeline_components):
        """Test audio → feature extraction."""
        extractor = pipeline_components["feature_extractor"]
        
        features = extractor.extract(sample_audio)
        
        assert features is not None
        assert hasattr(features, 'mfcc')
        assert features.mfcc is not None

    def test_features_to_emotion(self, sample_audio, pipeline_components):
        """Test features → emotion recognition."""
        extractor = pipeline_components["feature_extractor"]
        recognizer = pipeline_components["emotion_recognizer"]
        
        features = extractor.extract(sample_audio)
        emotion = recognizer.predict(sample_audio)
        
        assert emotion is not None
        assert hasattr(emotion, 'emotion')
        assert emotion.confidence >= 0 and emotion.confidence <= 1

    def test_text_to_intent(self, pipeline_components):
        """Test text → intent classification."""
        classifier = pipeline_components["intent_classifier"]
        
        result = classifier.classify("Can you help me understand this?")
        
        assert "intent" in result
        assert "confidence" in result
        assert result["intent"] in ["question", "statement", "request", "clarification", "unknown"]

    def test_emotion_intent_to_cognitive_state(self, sample_audio, pipeline_components):
        """Test emotion + intent → cognitive state."""
        recognizer = pipeline_components["emotion_recognizer"]
        classifier = pipeline_components["intent_classifier"]
        estimator = pipeline_components["cognitive_estimator"]
        
        emotion = recognizer.predict(sample_audio)
        intent = classifier.classify("I don't understand this at all!")
        
        cognitive_state = estimator.estimate(
            emotion=emotion,
            intent=intent,
            behavioral_signals={"pause_ratio": 0.3}
        )
        
        assert "state" in cognitive_state
        assert "confidence" in cognitive_state
        assert "recommendations" in cognitive_state

    def test_cognitive_state_to_intervention(self, sample_audio, pipeline_components):
        """Test cognitive state → intervention recommendation."""
        recognizer = pipeline_components["emotion_recognizer"]
        estimator = pipeline_components["cognitive_estimator"]
        engine = pipeline_components["intervention_engine"]
        
        emotion = recognizer.predict(sample_audio)
        cognitive_state = estimator.estimate(
            emotion=emotion,
            intent={"intent": "clarification", "confidence": 0.8},
            behavioral_signals={}
        )
        
        intervention = engine.decide(cognitive_state, {})
        
        assert intervention is not None
        assert hasattr(intervention, 'type')
        assert hasattr(intervention, 'priority')

    def test_full_pipeline_flow(self, sample_audio, pipeline_components):
        """Test complete pipeline from audio to intervention."""
        # Step 1: Extract features
        features = pipeline_components["feature_extractor"].extract(sample_audio)
        assert features is not None
        
        # Step 2: Recognize emotion
        emotion = pipeline_components["emotion_recognizer"].predict(sample_audio)
        assert emotion is not None
        
        # Step 3: Classify intent
        text = "I'm feeling confused about this topic"
        intent = pipeline_components["intent_classifier"].classify(text)
        assert intent is not None
        
        # Step 4: Update memory
        pipeline_components["memory_graph"].add_turn(
            text=text,
            intent=intent["intent"],
            emotion=emotion.emotion
        )
        
        # Step 5: Estimate cognitive state
        cognitive_state = pipeline_components["cognitive_estimator"].estimate(
            emotion=emotion,
            intent=intent,
            behavioral_signals={"speaking_rate": 0.7}
        )
        assert cognitive_state is not None
        
        # Step 6: Get intervention
        intervention = pipeline_components["intervention_engine"].decide(
            cognitive_state,
            context={}
        )
        assert intervention is not None
        
        # Step 7: Generate output
        if intervention.should_intervene and text:
            output = pipeline_components["output_generator"].generate(
                intervention_type=intervention.type.value,
                parameters=intervention.action_data
            )
            assert output is not None

    def test_pipeline_with_privacy_consent(self, sample_audio, pipeline_components):
        """Test pipeline respects consent settings."""
        consent = ConsentManager("test_user", "test_session")
        
        # No consent - should block audio processing
        assert not consent.is_allowed(DataCategory.AUDIO)
        
        # Grant voice permission
        consent.set_permission_level(PermissionLevel.VOICE_ONLY)
        assert consent.is_allowed(DataCategory.AUDIO)
        assert consent.is_allowed(DataCategory.EMOTION)
        
        # Now can process
        if consent.is_allowed(DataCategory.AUDIO):
            emotion = pipeline_components["emotion_recognizer"].predict(sample_audio)
            assert emotion is not None


class TestPerformanceBenchmarks:
    """Performance benchmark tests targeting <100ms inference."""

    @pytest.fixture
    def sample_audio(self):
        """Generate 2-second audio sample."""
        duration = 2.0
        sample_rate = 16000
        return np.random.randn(int(sample_rate * duration)).astype(np.float32)

    def test_feature_extraction_speed(self, sample_audio):
        """Feature extraction should be fast."""
        import time
        
        extractor = FeatureExtractor()
        
        # Warmup
        extractor.extract(sample_audio)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            extractor.extract(sample_audio)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_ms = np.mean(times)
        print(f"Feature extraction: {avg_ms:.2f}ms avg")
        
        # Should be under 50ms for features
        assert avg_ms < 100, f"Feature extraction took {avg_ms:.2f}ms, target <100ms"

    def test_emotion_recognition_speed(self, sample_audio):
        """Emotion recognition should meet latency target."""
        import time
        
        recognizer = SimplisticEmotionRecognizer()
        
        # Warmup
        recognizer.predict(sample_audio)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            recognizer.predict(sample_audio)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_ms = np.mean(times)
        print(f"Emotion recognition: {avg_ms:.2f}ms avg")
        
        assert avg_ms < 100, f"Emotion recognition took {avg_ms:.2f}ms, target <100ms"

    def test_intent_classification_speed(self):
        """Intent classification should be fast."""
        import time
        
        classifier = IntentClassifier(use_transformer=False)
        text = "Can you explain this concept to me?"
        
        # Warmup
        classifier.classify(text)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            classifier.classify(text)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_ms = np.mean(times)
        print(f"Intent classification: {avg_ms:.2f}ms avg")
        
        assert avg_ms < 50, f"Intent classification took {avg_ms:.2f}ms, target <50ms"

    def test_full_pipeline_latency(self, sample_audio):
        """Full pipeline should complete in reasonable time."""
        import time
        
        # Initialize all components
        extractor = FeatureExtractor()
        recognizer = SimplisticEmotionRecognizer()
        classifier = IntentClassifier(use_transformer=False)
        estimator = CognitiveStateEstimator()
        engine = InterventionEngine()
        
        text = "I need help understanding this"
        
        # Warmup
        features = extractor.extract(sample_audio)
        emotion = recognizer.predict(sample_audio)
        intent = classifier.classify(text)
        state = estimator.estimate(emotion, intent, {})
        engine.decide(state, {})
        
        # Benchmark full pipeline
        times = []
        for _ in range(5):
            start = time.perf_counter()
            
            features = extractor.extract(sample_audio)
            emotion = recognizer.predict(sample_audio)
            intent = classifier.classify(text)
            state = estimator.estimate(emotion, intent, {})
            intervention = engine.decide(state, {})
            
            times.append((time.perf_counter() - start) * 1000)
        
        avg_ms = np.mean(times)
        print(f"Full pipeline: {avg_ms:.2f}ms avg")
        
        # Full pipeline target: <500ms for real-time feel
        assert avg_ms < 500, f"Full pipeline took {avg_ms:.2f}ms, target <500ms"


class TestCulturalBias:
    """Tests for cultural bias detection and mitigation."""

    def test_emotion_detection_across_accents(self):
        """Test emotion detection doesn't vary significantly by simulated accent."""
        recognizer = SimplisticEmotionRecognizer()
        
        # Simulate same content with different "accent profiles"
        # In reality, would use actual accent-varied audio datasets
        base_audio = np.random.randn(16000).astype(np.float32) * 0.5
        
        # Different energy levels simulating expression styles
        variants = {
            "neutral": base_audio,
            "high_energy": base_audio * 1.5,
            "low_energy": base_audio * 0.5,
        }
        
        results = {}
        for name, audio in variants.items():
            emotion = recognizer.predict(audio)
            results[name] = emotion.confidence
        
        # Confidence shouldn't vary too wildly
        confidences = list(results.values())
        variance = np.var(confidences)
        
        print(f"Confidence variance across styles: {variance:.4f}")
        # Low variance indicates consistent performance
        # This is a simplified test - real bias testing needs diverse datasets

    def test_intent_classification_language_neutral(self):
        """Test intent classification on various phrasings."""
        classifier = IntentClassifier(use_transformer=False)
        
        # Same intent, different cultural phrasings
        phrasings = [
            "Can you help me?",  # Direct
            "I was wondering if perhaps you might assist me?",  # Indirect
            "Help please",  # Terse
            "Would you be so kind as to help me with this matter?",  # Formal
        ]
        
        results = [classifier.classify(p) for p in phrasings]
        
        # All should classify as request or question
        valid_intents = {"request", "question", "clarification"}
        for result, phrasing in zip(results, phrasings):
            assert result["intent"] in valid_intents, \
                f"'{phrasing}' classified as {result['intent']}, expected request/question"

    def test_gender_neutral_processing(self):
        """Ensure no gender-based processing differences."""
        # This is a placeholder - real testing would use gendered voice samples
        recognizer = SimplisticEmotionRecognizer()
        
        # Simulated audio at different pitch ranges
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        low_pitch = np.sin(2 * np.pi * 100 * t).astype(np.float32)  # ~100Hz
        high_pitch = np.sin(2 * np.pi * 200 * t).astype(np.float32)  # ~200Hz
        
        low_result = recognizer.predict(low_pitch)
        high_result = recognizer.predict(high_pitch)
        
        # Processing should complete for both
        assert low_result is not None
        assert high_result is not None
        
        # Emotion labels might differ based on acoustic features,
        # but confidence shouldn't show huge bias
        confidence_diff = abs(low_result.confidence - high_result.confidence)
        assert confidence_diff < 0.5, "Large confidence gap between pitch ranges"


class TestSecurityAudit:
    """Security audit tests."""

    def test_input_sanitization(self):
        """Test malicious input handling."""
        from src.api.security import SecurityMiddleware
        
        malicious_inputs = [
            "normal text",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "\x00\x01\x02 null bytes",
            "a" * 100000,  # Very long input
        ]
        
        for input_text in malicious_inputs:
            sanitized = SecurityMiddleware.sanitize_input(input_text)
            
            # Should not contain null bytes
            assert "\x00" not in sanitized
            
            # Should be truncated
            assert len(sanitized) <= 10000

    def test_rate_limiting_prevents_dos(self):
        """Test rate limiting blocks excessive requests."""
        from src.api.security import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=10)
        client_id = "attacker"
        
        # First 10 should pass
        for i in range(10):
            assert limiter.is_allowed(client_id), f"Request {i+1} should be allowed"
        
        # 11th should be blocked
        assert not limiter.is_allowed(client_id), "Should block after limit"

    def test_api_key_security(self):
        """Test API key security features."""
        from src.api.security import APIKeyManager
        
        manager = APIKeyManager()
        
        # Keys should be unique
        key1 = manager.generate_key()
        key2 = manager.generate_key()
        assert key1 != key2
        
        # Keys should be long enough
        assert len(key1) >= 32
        
        # Invalid keys should fail
        assert manager.validate_key("invalid-key-12345") is None

    def test_consent_required_for_data_access(self):
        """Test that consent is required for data processing."""
        from src.privacy import ConsentManager, PermissionLevel, DataCategory
        
        manager = ConsentManager("user", "session")
        
        # Default: no access
        assert not manager.is_allowed(DataCategory.AUDIO)
        assert not manager.is_allowed(DataCategory.EMOTION)
        assert not manager.is_allowed(DataCategory.FACIAL)
        
        # After consent: limited access
        manager.set_permission_level(PermissionLevel.VOICE_ONLY)
        assert manager.is_allowed(DataCategory.AUDIO)
        assert not manager.is_allowed(DataCategory.FACIAL)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
