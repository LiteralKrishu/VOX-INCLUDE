"""
VOX-INCLUDE: TensorFlow Lite Model Conversion Utilities

Provides tools for converting PyTorch/TensorFlow models to TFLite format
for efficient on-device inference.
"""

import os
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


class TFLiteConverter:
    """
    Converts emotion and intent models to TensorFlow Lite format.
    
    Supports:
    - PyTorch → ONNX → TFLite conversion
    - TensorFlow SavedModel → TFLite
    - Quantization for smaller model size
    """
    
    def __init__(self, output_dir: str = "models/tflite"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_pytorch_to_tflite(
        self,
        model_path: str,
        input_shape: Tuple[int, ...],
        output_name: str,
        quantize: bool = True
    ) -> Optional[str]:
        """
        Convert PyTorch model to TFLite via ONNX.
        
        Args:
            model_path: Path to .pt or .pth file
            input_shape: Model input shape (batch, channels, ...)
            output_name: Output filename without extension
            quantize: Apply dynamic quantization
            
        Returns:
            Path to converted .tflite file or None on failure
        """
        try:
            import torch
            import torch.onnx
            
            # Load PyTorch model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            
            # Export to ONNX
            onnx_path = self.output_dir / f"{output_name}.onnx"
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
            )
            
            # Convert ONNX to TFLite
            return self._onnx_to_tflite(str(onnx_path), output_name, quantize)
            
        except ImportError as e:
            print(f"Missing dependency for PyTorch conversion: {e}")
            return None
        except Exception as e:
            print(f"Conversion failed: {e}")
            return None
    
    def _onnx_to_tflite(
        self,
        onnx_path: str,
        output_name: str,
        quantize: bool = True
    ) -> Optional[str]:
        """Convert ONNX model to TFLite."""
        try:
            import onnx
            from onnx_tf.backend import prepare
            import tensorflow as tf
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            tf_path = self.output_dir / f"{output_name}_tf"
            tf_rep.export_graph(str(tf_path))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = self.output_dir / f"{output_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            return str(tflite_path)
            
        except ImportError as e:
            print(f"Missing dependency for ONNX/TF conversion: {e}")
            return None
        except Exception as e:
            print(f"ONNX to TFLite conversion failed: {e}")
            return None
    
    def convert_tensorflow_to_tflite(
        self,
        saved_model_dir: str,
        output_name: str,
        quantize: bool = True,
        representative_dataset: Optional[callable] = None
    ) -> Optional[str]:
        """
        Convert TensorFlow SavedModel to TFLite.
        
        Args:
            saved_model_dir: Path to SavedModel directory
            output_name: Output filename without extension
            quantize: Apply optimization
            representative_dataset: Callable for full integer quantization
            
        Returns:
            Path to converted .tflite file or None on failure
        """
        try:
            import tensorflow as tf
            
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                if representative_dataset:
                    converter.representative_dataset = representative_dataset
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
            
            tflite_model = converter.convert()
            
            tflite_path = self.output_dir / f"{output_name}.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            return str(tflite_path)
            
        except Exception as e:
            print(f"TF to TFLite conversion failed: {e}")
            return None
    
    def get_model_info(self, tflite_path: str) -> Dict[str, Any]:
        """Get information about a TFLite model."""
        try:
            import tensorflow as tf
            
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            file_size = os.path.getsize(tflite_path)
            
            return {
                "file_size_mb": file_size / (1024 * 1024),
                "inputs": [
                    {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
                    for d in input_details
                ],
                "outputs": [
                    {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
                    for d in output_details
                ],
            }
            
        except Exception as e:
            return {"error": str(e)}


class ModelBenchmark:
    """Benchmark model inference performance."""
    
    @staticmethod
    def benchmark_tflite(
        tflite_path: str,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark TFLite model inference time.
        
        Returns:
            Dict with avg_ms, min_ms, max_ms, std_ms
        """
        try:
            import tensorflow as tf
            import numpy as np
            import time
            
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create dummy input
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']
            dummy_input = np.random.randn(*input_shape).astype(input_dtype)
            
            # Warmup
            for _ in range(warmup_runs):
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], dummy_input)
                interpreter.invoke()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            return {
                "avg_ms": np.mean(times),
                "min_ms": np.min(times),
                "max_ms": np.max(times),
                "std_ms": np.std(times),
                "target_met": np.mean(times) < 100  # <100ms target
            }
            
        except Exception as e:
            return {"error": str(e)}
