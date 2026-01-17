# VOX-INCLUDE: Walkthrough

**Milestones Completed:** 1 (Core Pipeline) + 2 (Intent Recognition & Cognitive State)

---

## Project Structure

```
VOX-INCLUDE/src/
├── audio_processing/      # Mic capture, MFCC/pitch extraction
├── emotion_recognition/   # Pre-trained wav2vec2 emotion model
├── intent_recognition/    # Intent classifier, memory graph, cognitive estimator
├── adaptive_system/       # Intervention engine
├── api/                   # FastAPI backend
└── utils/                 # Configuration
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/emotion` | POST | Emotion from audio |
| `/api/v1/intent` | POST | Intent from text |
| `/api/v1/cognitive-state` | GET | Current cognitive state |
| `/api/v1/intervention` | GET | Recommended intervention |
| `/api/v1/comprehensive-analysis` | POST | Full multimodal analysis |
| `/api/v1/conversation/context` | GET | Conversation history |
| `/api/v1/conversation/clear` | POST | Reset session |

### Milestone 2: Intent, Cognitive State & Advanced Features
- **Intent Recognition:** Implemented Zero-Shot Classification (BART) with fallback rules.
- **Conversational Memory:** Created `MemoryGraph` to track context, topics, and metrics.
- **Cognitive State Estimation:**
  - **Fusion:** Emotion + Intent + Behavior + Conversation
  - **Bayesian Calibration:** Implemented `BayesianUpdater` for temporally smoothed confidence. (NEW)
- **Advanced Emotion Analysis:**
  - **Trajectory Tracking:** Implemented `EmotionTrajectory` to detect emotional momentum and trends (rising/falling intensity). (NEW)
  - **Signal-to-Noise Ratio (SNR):** Added dynamic SNR calculation in `FeatureExtractor` to gauge audio quality. (NEW)
- **Adaptive Intervention:** Priority-based intervention engine.
- **API:**
  - `POST /api/v1/intent`: Classify intent
  - `GET /api/v1/cognitive-state`: Get estimated state (with Bayesian confidence)
  - `GET /api/v1/intervention`: Get recommendation
  - `POST /api/v1/comprehensive-analysis`: Full multimodal analysis
  - Updated standard endpoints (`/api/v1/analyze`, etc.) to include momentum, trend, and SNR.

### Milestone 3: Advanced Adaptive System (Phase 5)
- **Closed-Loop Intervention Engine:**
  - Implemented specific logic for **Cognitive Overload** (Auto-Simplify), **Cognitive Fatigue** (Micro-Break), and **Social Anxiety** (Private Channel).
  - Derived states like **Rising Confusion** trigger high-priority interventions.
- **Output Generator:**
  - **Simplification:** Reduces sentence complexity and vocabulary.
  - **Paraphrasing:** Generates alternative explanations.
  - **Visual Summaries:** Creates mock visual data structures for frontend rendering.
- **Integration:** Updated API models to return `adapted_content` and `visual_aids` with interventions.

### Testing Phase 5
Verified via `tests/test_phase5_interventions.py`:
1.  **Rising Confusion:** Confirmed system prioritizes `AUTO_SIMPLIFY` when trend is declining.
2.  **Cognitive Fatigue:** Confirmed system suggests `MICRO_BREAK`.
3.  **Adaptation:** Verified text simplification labels and visual summary structure generation.

---

## Quick Test

```bash
# Start server
python -m src.api.main

# Test intent classification
curl -X POST http://localhost:8000/api/v1/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "Can you help me understand this?"}'

# Get cognitive state
curl http://localhost:8000/api/v1/cognitive-state

# Full analysis with text
curl -X POST http://localhost:8000/api/v1/comprehensive-analysis \
  -H "Content-Type: application/json" \
  -d '{"text": "I don'\''t understand, can you explain again?"}'
```

---

## Next Steps

- [ ] **Phase 5**: Adaptive output generator (content simplification)
- [ ] **Phase 6**: Flutter mobile app
