
import sys
import os
import numpy as np
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_processing.feature_extraction import FeatureExtractor
from src.emotion_recognition.models import EmotionRecognizer, EmotionTrajectory
from src.intent_recognition.cognitive_estimator import CognitiveStateEstimator, CognitiveState
from src.utils.bayesian import BayesianUpdater

def test_snr_calculation():
    print("\n--- Testing SNR Calculation ---")
    extractor = FeatureExtractor(sample_rate=16000)
    
    # 1. Silence (should have low/zero SNR or handle gracefully)
    silence = np.zeros(16000, dtype=np.float32)
    features_silence = extractor.extract(silence)
    print(f"Silence SNR: {features_silence.snr} (Expected: ~0 or handled)")
    
    # 2. Pure Sine Wave (High SNR)
    t = np.linspace(0, 1, 16000)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    features_sine = extractor.extract(sine)
    print(f"Sine Wave SNR: {features_sine.snr} (Expected: > 20dB)")
    
    # 3. Noisy Sine Wave (Lower SNR)
    noise = np.random.normal(0, 0.1, 16000).astype(np.float32)
    noisy_sine = (sine + noise).astype(np.float32)
    features_noisy = extractor.extract(noisy_sine)
    print(f"Noisy Sine SNR: {features_noisy.snr} (Expected: < Sine SNR)")

    if features_sine.snr > features_noisy.snr:
        print("SUCCESS: Sine SNR > Noisy SNR")
    else:
        print("FAILURE: SNR calculation detection failed")

def test_emotion_trajectory():
    print("\n--- Testing Emotion Trajectory ---")
    # minimal mock of EmotionRecognizer to access trajectory directly
    traj = EmotionTrajectory(history_size=5)
    
    # Simulate rising intensity (e.g. rising anger)
    print("Simulating rising intensity...")
    traj.add("angry", 0.5, 0.8)
    time.sleep(0.01) # ensure timestamp diff
    traj.add("angry", 0.6, 0.8)
    time.sleep(0.01)
    traj.add("angry", 0.7, 0.8)
    time.sleep(0.01)
    traj.add("angry", 0.8, 0.9)
    
    momentum = traj.get_momentum()
    trend = traj.get_dominant_trend()
    print(f"Momentum: {momentum:.4f}")
    print(f"Trend: {trend}")
    
    if trend == "rising_intensity" and momentum > 0:
        print("SUCCESS: Detected rising intensity")
    else:
        print("FAILURE: Did not detect rising intensity")
        
    # Simulate falling
    traj = EmotionTrajectory(history_size=5)
    traj.add("happy", 0.9, 0.9)
    time.sleep(0.01)
    traj.add("happy", 0.5, 0.9)
    time.sleep(0.01)
    traj.add("happy", 0.2, 0.9)
    
    print(f"Falling Momentum: {traj.get_momentum():.4f}")
    print(f"Falling Trend: {traj.get_dominant_trend()}")

def test_bayesian_fusion():
    print("\n--- Testing Bayesian Fusion ---")
    # Initialize uniform priors
    priors = {"engaged": 0.5, "bored": 0.5}
    updater = BayesianUpdater(priors)
    
    print(f"Initial Priors: {updater.get_priors()}")
    
    # Evidence strongly favoring 'engaged'
    evidence = {"engaged": 0.9, "bored": 0.1}
    
    # Update 1
    posterior1 = updater.update(evidence, reliability=1.0)
    print(f"Posterior 1 (Engaged): {posterior1['engaged']:.4f}")
    
    # Update 2 (Reinforcement)
    posterior2 = updater.update(evidence, reliability=1.0)
    print(f"Posterior 2 (Engaged): {posterior2['engaged']:.4f}")
    
    if posterior2['engaged'] > posterior1['engaged']:
        print("SUCCESS: Belief in 'engaged' increased with reinforced evidence")
    else:
        print("FAILURE: Bayesian update did not reinforce belief")
        
    # Test reliability
    evidence_weak = {"engaged": 0.1, "bored": 0.9} # contradicting
    posterior3 = updater.update(evidence_weak, reliability=0.1) # very unreliable source
    print(f"Posterior 3 (Ignored Contradiction): {posterior3['engaged']:.4f}")
    
    if posterior3['engaged'] > 0.5:
        print("SUCCESS: High prior maintained despite contradicting but unreliable evidence")
    else:
        print("FAILURE: Unreliable evidence swayed belief too much")

if __name__ == "__main__":
    test_snr_calculation()
    test_emotion_trajectory()
    test_bayesian_fusion()
