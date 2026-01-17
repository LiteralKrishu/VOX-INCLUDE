"""Audio capture module for real-time voice input."""

import threading
from typing import Callable, Optional
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from ..utils.config import config


class AudioCapture:
    """Real-time audio capture with buffering support."""
    
    def __init__(
        self,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        chunk_size: Optional[int] = None,
        buffer_duration: Optional[float] = None
    ):
        """Initialize audio capture.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: from config)
            channels: Number of audio channels (default: from config)
            chunk_size: Size of audio chunks to capture (default: from config)
            buffer_duration: Duration of circular buffer in seconds
        """
        audio_config = config.audio
        self.sample_rate = sample_rate or audio_config.get("sample_rate", 16000)
        self.channels = channels or audio_config.get("channels", 1)
        self.chunk_size = chunk_size or audio_config.get("chunk_size", 1024)
        self.buffer_duration = buffer_duration or audio_config.get("buffer_duration_seconds", 5.0)
        
        # Calculate buffer size
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        
        # Recording state
        self._recording = False
        self._stream = None
        self._pyaudio = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callback for real-time processing
        self._on_audio_callback: Optional[Callable[[np.ndarray], None]] = None
    
    def _check_pyaudio(self) -> None:
        """Check if PyAudio is available."""
        if not PYAUDIO_AVAILABLE:
            raise ImportError(
                "PyAudio is required for audio capture. "
                "Install it with: pip install pyaudio"
            )
    
    def start(self, on_audio: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """Start audio capture.
        
        Args:
            on_audio: Optional callback called with each audio chunk
        """
        self._check_pyaudio()
        
        if self._recording:
            return
        
        self._on_audio_callback = on_audio
        self._recording = True
        self._pyaudio = pyaudio.PyAudio()
        
        self._stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self._stream.start_stream()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback."""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        with self._lock:
            # Add to circular buffer
            chunk_len = len(audio_data)
            if self.buffer_index + chunk_len > self.buffer_size:
                # Wrap around
                overflow = (self.buffer_index + chunk_len) - self.buffer_size
                self.audio_buffer[self.buffer_index:] = audio_data[:-overflow]
                self.audio_buffer[:overflow] = audio_data[-overflow:]
                self.buffer_index = overflow
            else:
                self.audio_buffer[self.buffer_index:self.buffer_index + chunk_len] = audio_data
                self.buffer_index += chunk_len
        
        # Call user callback if provided
        if self._on_audio_callback:
            self._on_audio_callback(audio_data.copy())
        
        return (None, pyaudio.paContinue)
    
    def stop(self) -> None:
        """Stop audio capture."""
        self._recording = False
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None
    
    def get_buffer(self) -> np.ndarray:
        """Get the current audio buffer contents.
        
        Returns:
            Audio buffer as numpy array (ordered chronologically)
        """
        with self._lock:
            # Return buffer in chronological order
            return np.concatenate([
                self.audio_buffer[self.buffer_index:],
                self.audio_buffer[:self.buffer_index]
            ])
    
    def get_recent_audio(self, duration: float) -> np.ndarray:
        """Get the most recent audio of specified duration.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio array of specified duration
        """
        samples = int(min(duration * self.sample_rate, self.buffer_size))
        buffer = self.get_buffer()
        return buffer[-samples:]
    
    def clear_buffer(self) -> None:
        """Clear the audio buffer."""
        with self._lock:
            self.audio_buffer.fill(0)
            self.buffer_index = 0
    
    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def load_audio_file(filepath: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio from file.
    
    Args:
        filepath: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Audio data as numpy array
    """
    try:
        import librosa
        audio, _ = librosa.load(filepath, sr=sample_rate, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        raise ImportError("librosa is required for loading audio files")
