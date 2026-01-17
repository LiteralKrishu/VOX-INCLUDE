"""Configuration loader for VOX-INCLUDE."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager for VOX-INCLUDE."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "app": {
                "name": "VOX-INCLUDE",
                "version": "0.1.0",
                "debug": True
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "buffer_duration_seconds": 5.0
            },
            "features": {
                "mfcc_coefficients": 40,
                "n_fft": 2048,
                "hop_length": 512
            },
            "emotion": {
                "model_type": "pretrained",
                "confidence_threshold": 0.6
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @property
    def audio(self) -> Dict[str, Any]:
        return self._config.get("audio", {})
    
    @property
    def features(self) -> Dict[str, Any]:
        return self._config.get("features", {})
    
    @property
    def emotion(self) -> Dict[str, Any]:
        return self._config.get("emotion", {})
    
    @property
    def api(self) -> Dict[str, Any]:
        return self._config.get("api", {})


# Singleton instance
config = Config()
