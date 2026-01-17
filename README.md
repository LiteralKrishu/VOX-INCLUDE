# VOX-INCLUDE

**Voice-Oriented eXpressive INterpretation for Communication, Learning & Universal Design Ecosystems**

An emotion-aware, intent-interpreting voice intelligence platform for universal understanding.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m src.api.main
```

## Project Structure

```
VOX-INCLUDE/
├── src/
│   ├── audio_processing/    # Audio capture and feature extraction
│   ├── emotion_recognition/ # SER models (pre-trained)
│   ├── intent_recognition/  # Intent and cognitive state
│   ├── adaptive_system/     # Intervention engine
│   ├── privacy/             # Anonymization and consent
│   ├── api/                 # FastAPI backend
│   └── utils/               # Shared utilities
├── models/                  # Pre-trained model files
├── mobile_app/              # Flutter application
└── tests/                   # Unit and integration tests
```

## Current Phase

**Milestone 1: Core Pipeline**
- Audio processing with feature extraction
- Emotion recognition using pre-trained models
- FastAPI backend for mobile integration

## Domain Focus

Public Services - General population accessibility features

## License

MIT
