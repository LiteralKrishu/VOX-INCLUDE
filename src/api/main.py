"""FastAPI backend for VOX-INCLUDE."""

import io
import base64
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..utils.config import config
from ..audio_processing import FeatureExtractor
from ..emotion_recognition import EmotionRecognizer, EmotionResult
from src.intent_recognition.intent_classifier import IntentClassifier
from src.intent_recognition.memory_graph import MemoryGraph
from src.intent_recognition.cognitive_estimator import CognitiveStateEstimator
from src.adaptive_system import InterventionEngine, OutputGenerator
from src.privacy import (
    ConsentManager, TransparencyDashboard, PermissionLevel, DataCategory,
    AudioAnonymizer, DataMinimizer
)


# Global instances
emotion_recognizer: Optional[EmotionRecognizer] = None
feature_extractor: Optional[FeatureExtractor] = None
intent_classifier: Optional[IntentClassifier] = None
memory_graph: Optional[MemoryGraph] = None
cognitive_estimator: Optional[CognitiveStateEstimator] = None
intervention_engine: Optional[InterventionEngine] = None
output_generator: Optional[OutputGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global emotion_recognizer, feature_extractor
    global intent_classifier, memory_graph, cognitive_estimator, intervention_engine, output_generator
    
    print("Initializing VOX-INCLUDE API...")
    
    # Core modules
    feature_extractor = FeatureExtractor()
    
    # Use fast mode (SimplisticEmotionRecognizer) for real-time responsiveness
    # Set FAST_MODE=false in environment to use full HuggingFace model
    import os
    fast_mode = os.environ.get("VOX_FAST_MODE", "true").lower() == "true"
    
    if fast_mode:
        print("Using FAST MODE (SimplisticEmotionRecognizer) for real-time performance")
        from ..emotion_recognition import SimplisticEmotionRecognizer
        emotion_recognizer = SimplisticEmotionRecognizer()
    else:
        emotion_recognizer = EmotionRecognizer()
    
    # Milestone 2 modules
    intent_classifier = IntentClassifier(use_transformer=False)  # Start with rule-based
    memory_graph = MemoryGraph()
    cognitive_estimator = CognitiveStateEstimator()
    intervention_engine = InterventionEngine()
    output_generator = OutputGenerator()
    
    print("Initialization complete.")
    
    # Pre-load emotion model (only for non-fast mode)
    if not fast_mode:
        try:
            emotion_recognizer.load_model()
            print("Emotion model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not pre-load emotion model: {e}")
            print("Model will be loaded on first request")
    
    yield
    
    print("Shutting down VOX-INCLUDE API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="VOX-INCLUDE API",
        description="Voice-Oriented eXpressive INterpretation for Communication",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ============= Pydantic Models =============

class AudioBase64Request(BaseModel):
    """Request with base64 encoded audio."""
    audio_base64: str = Field(..., description="Base64 encoded audio data (WAV/PCM)")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")


class EmotionResponse(BaseModel):
    """Emotion analysis response."""
    emotion: str
    confidence: float
    probabilities: dict
    arousal: Optional[float] = None
    valence: Optional[float] = None
    is_confident: bool
    momentum: Optional[float] = None
    trend: Optional[str] = None


class FeatureResponse(BaseModel):
    """Audio features response."""
    mfcc_mean: List[float]
    mfcc_std: List[float]
    pitch_mean: Optional[float] = None
    pitch_std: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_std: Optional[float] = None
    snr: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool


class AnalysisResponse(BaseModel):
    """Full analysis response."""
    emotion: EmotionResponse
    features: FeatureResponse
    session_id: Optional[str] = None


class IntentRequest(BaseModel):
    """Request for intent classification."""
    text: str = Field(..., description="Text to classify intent")
    include_context: bool = Field(default=True, description="Include conversation context")


class IntentResponse(BaseModel):
    """Intent classification response."""
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    sub_intent: Optional[str] = None


class CognitiveStateResponse(BaseModel):
    """Cognitive state estimation response."""
    state: str
    confidence: float
    probabilities: Dict[str, float]
    contributing_factors: Dict[str, float]
    recommendations: List[str]
    trend: Optional[str] = None
    bayesian_confidence: Optional[float] = None


class InterventionResponse(BaseModel):
    """Intervention recommendation response."""
    intervention_type: str
    priority: int
    message: Optional[str] = None
    action_data: Dict[str, Any]
    should_intervene: bool
    adapted_content: Optional[str] = None
    visual_aids: Optional[List[Dict[str, Any]]] = None


class ComprehensiveAnalysisRequest(BaseModel):
    """Request for comprehensive multimodal analysis."""
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    text: Optional[str] = Field(None, description="Text transcription")
    sample_rate: int = Field(default=16000)
    session_id: Optional[str] = None


class ComprehensiveAnalysisResponse(BaseModel):
    """Comprehensive analysis combining all modalities."""
    emotion: Optional[EmotionResponse] = None
    intent: Optional[IntentResponse] = None
    cognitive_state: CognitiveStateResponse
    intervention: InterventionResponse
    conversation_metrics: Dict[str, Any]
    features: Optional[FeatureResponse] = None


class ConsentUpdateRequest(BaseModel):
    """Request to update consent settings."""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    permission_level: str = Field(..., description="Permission level: NONE, VOICE_ONLY, VOICE_BEHAVIORAL, FULL_MULTIMODAL")
    opaque_mode: bool = Field(default=False, description="Enable right to opaqueness")


class ConsentStatusResponse(BaseModel):
    """Consent status response."""
    user_id: str
    permission_level: str
    opaque_mode: bool
    allowed_categories: List[str]


class ExplainabilityResponse(BaseModel):
    """Explainable AI decision breakdown."""
    summary: str
    factors: List[Dict[str, Any]]
    confidence_breakdown: Dict[str, float]
    data_used: List[str]


class DataExportResponse(BaseModel):
    """User data export response."""
    user_id: str
    current_permission_level: str
    opaque_mode: bool
    consent_history: List[Dict[str, Any]]
    audit_log: List[Dict[str, Any]]


# ============= Helper Functions =============

def decode_audio(audio_base64: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode base64 audio to numpy array."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        
        # Try to load as WAV first
        try:
            import soundfile as sf
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            return audio.astype(np.float32)
        except Exception:
            pass
        
        # Fallback: treat as raw PCM float32
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        return audio
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}")


async def load_audio_file(file: UploadFile) -> np.ndarray:
    """Load audio from uploaded file."""
    try:
        import soundfile as sf
        import librosa
        
        contents = await file.read()
        audio, sr = sf.read(io.BytesIO(contents))
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to target rate
        target_sr = config.audio.get("sample_rate", 16000)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio.astype(np.float32)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load audio file: {e}")


# ============= API Endpoints =============

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=emotion_recognizer.is_loaded if emotion_recognizer else False
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        model_loaded=emotion_recognizer.is_loaded if emotion_recognizer else False
    )


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_speech(request: AudioBase64Request):
    """Analyze speech for emotion and extract features.
    
    Accepts base64 encoded audio and returns comprehensive analysis.
    """
    if not emotion_recognizer or not feature_extractor:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    # Decode audio
    audio = decode_audio(request.audio_base64, request.sample_rate)
    
    if len(audio) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short for analysis")
    
    # Run emotion recognition
    emotion_result = emotion_recognizer.predict(audio)
    
    # Extract features
    features = feature_extractor.extract(audio)
    aggregated = features.get_aggregated()
    
    return AnalysisResponse(
        emotion=EmotionResponse(
            emotion=emotion_result.emotion,
            confidence=emotion_result.confidence,
            probabilities=emotion_result.probabilities,
            arousal=emotion_result.arousal,
            valence=emotion_result.valence,
            is_confident=emotion_result.is_confident,
            momentum=emotion_result.momentum,
            trend=emotion_result.trend
        ),
        features=FeatureResponse(
            mfcc_mean=aggregated.get("mfcc_mean", []),
            mfcc_std=aggregated.get("mfcc_std", []),
            pitch_mean=aggregated.get("pitch_mean"),
            pitch_std=aggregated.get("pitch_std"),
            energy_mean=aggregated.get("energy_mean"),
            energy_std=aggregated.get("energy_std"),
            snr=aggregated.get("snr")
        )
    )


@app.post("/api/v1/emotion", response_model=EmotionResponse)
async def analyze_emotion(request: AudioBase64Request):
    """Analyze emotion from base64 encoded audio."""
    if not emotion_recognizer:
        raise HTTPException(status_code=503, detail="Emotion model not initialized")
    
    audio = decode_audio(request.audio_base64, request.sample_rate)
    
    if len(audio) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short")
    
    result = emotion_recognizer.predict(audio)
    
    return EmotionResponse(
        emotion=result.emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        arousal=result.arousal,
        valence=result.valence,
        is_confident=result.is_confident,
        momentum=result.momentum,
        trend=result.trend
    )


@app.post("/api/v1/emotion/file", response_model=EmotionResponse)
async def analyze_emotion_file(file: UploadFile = File(...)):
    """Analyze emotion from uploaded audio file."""
    if not emotion_recognizer:
        raise HTTPException(status_code=503, detail="Emotion model not initialized")
    
    audio = await load_audio_file(file)
    
    if len(audio) < 1000:
        raise HTTPException(status_code=400, detail="Audio too short")
    
    result = emotion_recognizer.predict(audio)
    
    return EmotionResponse(
        emotion=result.emotion,
        confidence=result.confidence,
        probabilities=result.probabilities,
        arousal=result.arousal,
        valence=result.valence,
        is_confident=result.is_confident,
        momentum=result.momentum,
        trend=result.trend
    )


@app.post("/api/v1/features", response_model=FeatureResponse)
async def extract_features(request: AudioBase64Request):
    """Extract audio features from base64 encoded audio."""
    if not feature_extractor:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")
    
    audio = decode_audio(request.audio_base64, request.sample_rate)
    features = feature_extractor.extract(audio)
    aggregated = features.get_aggregated()
    
    return FeatureResponse(
        mfcc_mean=aggregated.get("mfcc_mean", []),
        mfcc_std=aggregated.get("mfcc_std", []),
        pitch_mean=aggregated.get("pitch_mean"),
        pitch_std=aggregated.get("pitch_std"),
        energy_mean=aggregated.get("energy_mean"),
        energy_std=aggregated.get("energy_std"),
        snr=aggregated.get("snr")
    )


@app.get("/api/v1/emotions")
async def list_emotions():
    """List supported emotions."""
    if emotion_recognizer:
        return {"emotions": emotion_recognizer.supported_emotions}
    return {"emotions": ["neutral", "happy", "sad", "angry", "fearful"]}


# ============= Milestone 2: Intent & Cognitive State Endpoints =============

@app.post("/api/v1/intent", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """Classify intent from text input."""
    if not intent_classifier:
        raise HTTPException(status_code=503, detail="Intent classifier not initialized")
    
    result = intent_classifier.classify(request.text)
    
    # Add to memory graph
    if memory_graph and request.include_context:
        memory_graph.add_turn(
            text=request.text,
            speaker="user",
            intent=result.intent,
            confidence=result.confidence
        )
    
    return IntentResponse(
        intent=result.intent,
        confidence=result.confidence,
        probabilities=result.probabilities,
        sub_intent=result.sub_intent
    )


@app.get("/api/v1/cognitive-state", response_model=CognitiveStateResponse)
async def get_cognitive_state():
    """Get current estimated cognitive state based on conversation history."""
    if not cognitive_estimator or not memory_graph:
        raise HTTPException(status_code=503, detail="Cognitive estimator not initialized")
    
    # Get conversation metrics
    metrics = memory_graph.get_conversation_metrics()
    
    # Estimate cognitive state
    result = cognitive_estimator.estimate(
        conversation_metrics=metrics
    )
    
    return CognitiveStateResponse(
        state=result.state,
        confidence=result.confidence,
        probabilities=result.probabilities,
        contributing_factors=result.contributing_factors,
        recommendations=result.recommendations,
        trend=cognitive_estimator.get_trend(),
        bayesian_confidence=result.bayesian_confidence
    )


@app.get("/api/v1/intervention", response_model=InterventionResponse)
async def get_intervention():
    """Get recommended intervention based on current cognitive state."""
    if not cognitive_estimator or not intervention_engine or not memory_graph:
        raise HTTPException(status_code=503, detail="Intervention engine not initialized")
    
    # Get current cognitive state
    metrics = memory_graph.get_conversation_metrics()
    state_result = cognitive_estimator.estimate(conversation_metrics=metrics)
    
    # Get intervention
    intervention = intervention_engine.get_primary_intervention(
        cognitive_state=state_result.state,
        confidence=state_result.confidence,
        context=metrics
    )
    
    return InterventionResponse(
        intervention_type=intervention.type.value,
        priority=intervention.priority,
        message=intervention.message,
        action_data=intervention.action_data,
        should_intervene=intervention_engine.should_intervene(
            state_result.state, state_result.confidence
        ),
        adapted_content=None, # Populate if input text available? 
        # Actually endpoint /intervention is GET, so no input text to adapt usually. 
        # But we can add a dummy check or leave None.
        visual_aids=None
    )


@app.post("/api/v1/comprehensive-analysis", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """Perform comprehensive multimodal analysis.
    
    Combines emotion recognition (from audio) with intent classification (from text)
    to estimate cognitive state and recommend interventions.
    """
    emotion_response = None
    intent_response = None
    emotion_data = {}
    audio = None  # Initialize audio to None for later use
    
    # Timing for diagnostics
    start_time = time.time()
    
    # Process audio if provided
    if request.audio_base64 and emotion_recognizer:
        try:
            decode_start = time.time()
            audio = decode_audio(request.audio_base64, request.sample_rate)
            print(f"[TIMING] Audio decode: {(time.time() - decode_start)*1000:.1f}ms, samples: {len(audio)}")
            
            # Skip processing if audio is too short (empty/silent chunk)
            if len(audio) < 2048:
                print(f"Skipping analysis: Audio too short ({len(audio)} samples)")
                audio = None
                
            if audio is not None and len(audio) >= 1000:
                predict_start = time.time()
                emotion_result = emotion_recognizer.predict(audio)
                print(f"[TIMING] Emotion prediction: {(time.time() - predict_start)*1000:.1f}ms")
                
                emotion_response = EmotionResponse(
                    emotion=emotion_result.emotion,
                    confidence=emotion_result.confidence,
                    probabilities=emotion_result.probabilities,
                    arousal=emotion_result.arousal,
                    valence=emotion_result.valence,
                    is_confident=emotion_result.is_confident,
                    momentum=emotion_result.momentum,
                    trend=emotion_result.trend
                )
                emotion_data = {
                    "emotion": emotion_result.emotion,
                    "arousal": emotion_result.arousal,
                    "valence": emotion_result.valence,
                    "confidence": emotion_result.confidence
                }
        except Exception as e:
            print(f"Audio processing error: {e}")
    
    # Process text if provided
    intent_data = {}
    if request.text and intent_classifier:
        intent_result = intent_classifier.classify(request.text)
        intent_response = IntentResponse(
            intent=intent_result.intent,
            confidence=intent_result.confidence,
            probabilities=intent_result.probabilities,
            sub_intent=intent_result.sub_intent
        )
        intent_data = {
            "intent": intent_result.intent,
            "confidence": intent_result.confidence
        }
        
        # Add to memory graph
        if memory_graph:
            memory_graph.add_turn(
                text=request.text,
                speaker="user",
                intent=intent_result.intent,
                emotion=emotion_data.get("emotion"),
                confidence=intent_result.confidence
            )
    
    # Get conversation metrics
    metrics = memory_graph.get_conversation_metrics() if memory_graph else {}
    
    # Estimate cognitive state
    state_result = cognitive_estimator.estimate(
        emotion_data=emotion_data,
        intent_data=intent_data,
        conversation_metrics=metrics
    )
    
    # Get intervention
    intervention = intervention_engine.get_primary_intervention(
        cognitive_state=state_result.state,
        confidence=state_result.confidence,
        context=metrics
    )
    
    # Extract features (SNR) if audio present and successfully decoded
    features_response = None
    if audio is not None and feature_extractor:
        try:
            features = feature_extractor.extract(audio)
            aggregated = features.get_aggregated()
            features_response = FeatureResponse(
                mfcc_mean=aggregated.get("mfcc_mean", []),
                mfcc_std=aggregated.get("mfcc_std", []),
                energy_mean=aggregated.get("energy_mean"),
                energy_std=aggregated.get("energy_std"),
                pitch_mean=aggregated.get("pitch_mean"),
                pitch_std=aggregated.get("pitch_std"),
                snr=aggregated.get("snr")
            )
        except Exception as e:
            print(f"Feature extraction error: {e}")

    return ComprehensiveAnalysisResponse(
        emotion=emotion_response,
        intent=intent_response,
        cognitive_state=CognitiveStateResponse(
            state=state_result.state,
            confidence=state_result.confidence,
            probabilities=state_result.probabilities,
            contributing_factors=state_result.contributing_factors,
            recommendations=state_result.recommendations,
            trend=cognitive_estimator.get_trend(),
            bayesian_confidence=state_result.bayesian_confidence
        ),
        intervention=InterventionResponse(
            intervention_type=intervention.type.value,
            priority=intervention.priority,
            message=intervention.message,
            action_data=intervention.action_data,
            should_intervene=intervention_engine.should_intervene(
                state_result.state, state_result.confidence
            ),
            adapted_content=output_generator.adapt_content(
                text=request.text or "", 
                intervention_type=intervention.type.value,
                parameters=intervention.action_data
            ).adapted_text if request.text and output_generator else None,
            visual_aids=output_generator.adapt_content(
                text=request.text or "",
                intervention_type=intervention.type.value,
                parameters=intervention.action_data
             ).visual_aids if request.text and output_generator else None
        ),
        conversation_metrics=metrics,
        features=features_response
    )


@app.get("/api/v1/conversation/context")
async def get_conversation_context():
    """Get recent conversation context."""
    if not memory_graph:
        raise HTTPException(status_code=503, detail="Memory graph not initialized")
    
    return {
        "context": [t.to_dict() for t in memory_graph.get_context()],
        "metrics": memory_graph.get_conversation_metrics(),
        "active_topics": [t.name for t in memory_graph.get_active_topics()]
    }


@app.post("/api/v1/conversation/clear")
async def clear_conversation():
    """Clear conversation history and start fresh."""
    if memory_graph:
        memory_graph.clear()
    if cognitive_estimator:
        cognitive_estimator.clear_history()
    
    return {"status": "cleared", "message": "Conversation history cleared"}


# ============= Privacy & Consent Endpoints =============

# In-memory consent store (would be database in production)
_consent_managers: Dict[str, ConsentManager] = {}
_audio_anonymizer = AudioAnonymizer(epsilon=1.0)


def _get_consent_manager(user_id: str, session_id: str) -> ConsentManager:
    """Get or create consent manager for user."""
    key = f"{user_id}:{session_id}"
    if key not in _consent_managers:
        _consent_managers[key] = ConsentManager(user_id, session_id)
    return _consent_managers[key]


@app.post("/api/v1/privacy/consent", response_model=ConsentStatusResponse)
async def update_consent(request: ConsentUpdateRequest):
    """Update user consent settings."""
    try:
        level = PermissionLevel[request.permission_level.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid permission level: {request.permission_level}")
    
    manager = _get_consent_manager(request.user_id, request.session_id)
    manager.set_permission_level(level)
    
    if request.opaque_mode:
        manager.enable_opaque_mode()
    else:
        manager.disable_opaque_mode()
    
    allowed = manager._get_allowed_categories(manager.current_level)
    
    return ConsentStatusResponse(
        user_id=request.user_id,
        permission_level=manager.current_level.name,
        opaque_mode=manager.is_opaque(),
        allowed_categories=[c.value for c in allowed]
    )


@app.get("/api/v1/privacy/consent/{user_id}/{session_id}", response_model=ConsentStatusResponse)
async def get_consent_status(user_id: str, session_id: str):
    """Get current consent status for a user."""
    manager = _get_consent_manager(user_id, session_id)
    allowed = manager._get_allowed_categories(manager.current_level)
    
    return ConsentStatusResponse(
        user_id=user_id,
        permission_level=manager.current_level.name,
        opaque_mode=manager.is_opaque(),
        allowed_categories=[c.value for c in allowed]
    )


@app.get("/api/v1/privacy/export/{user_id}/{session_id}", response_model=DataExportResponse)
async def export_user_data(user_id: str, session_id: str):
    """Export all user data (GDPR compliance)."""
    manager = _get_consent_manager(user_id, session_id)
    data = manager.export_user_data()
    
    return DataExportResponse(**data)


@app.delete("/api/v1/privacy/delete/{user_id}/{session_id}")
async def delete_user_data(user_id: str, session_id: str):
    """Delete all user data (Right to be Forgotten)."""
    manager = _get_consent_manager(user_id, session_id)
    result = manager.request_deletion()
    
    # Remove from in-memory store
    key = f"{user_id}:{session_id}"
    if key in _consent_managers:
        del _consent_managers[key]
    
    return {"status": "deleted", "details": result}


@app.post("/api/v1/privacy/explain", response_model=ExplainabilityResponse)
async def explain_decision(
    emotion_result: Optional[Dict[str, Any]] = None,
    cognitive_result: Optional[Dict[str, Any]] = None,
    intervention_result: Optional[Dict[str, Any]] = None
):
    """
    Get explainable AI breakdown of a decision.
    
    This endpoint provides transparency about how the AI made its decisions.
    """
    explanation = TransparencyDashboard.explain_decision(
        emotion_result or {},
        cognitive_result or {},
        intervention_result
    )
    
    return ExplainabilityResponse(**explanation)


@app.get("/api/v1/privacy/transparency/{user_id}/{session_id}")
async def get_transparency_summary(user_id: str, session_id: str):
    """Get transparency summary showing what data is being processed."""
    manager = _get_consent_manager(user_id, session_id)
    summary = TransparencyDashboard.get_processing_summary(manager)
    return summary


# ============= Run Server =============

if __name__ == "__main__":
    import uvicorn
    
    host = config.api.get("host", "0.0.0.0")
    port = config.api.get("port", 8000)
    
    uvicorn.run(app, host=host, port=port)

