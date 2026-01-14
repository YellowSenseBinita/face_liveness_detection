"""
Passive Liveness Model Module
Loads pretrained MobileNet+LSTM model and runs inference
Supports: Keras (.h5), ONNX (.onnx), PyTorch (.pth)
"""

import numpy as np
import os
from typing import Optional


class PassiveLivenessModel:
    """
    Loads and runs pretrained MobileNet+LSTM liveness model
    
    Model Architecture (expected):
    - Input: (batch, T, H, W, C) where T=16, H=W=112, C=3
    - MobileNet backbone extracts per-frame features
    - LSTM processes temporal sequence
    - Output: Single score ∈ [0, 1] (probability of LIVE)
    """
    
    def __init__(self, model_path: str, model_format: str = 'keras'):
        """
        Initialize model
        
        Args:
            model_path: Path to model file (.h5, .onnx, .pth)
            model_format: 'keras', 'onnx', or 'pytorch'
        """
        self.model_path = model_path
        self.model_format = model_format.lower()
        self.model = None
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found: {model_path}")
            print("→ Using dummy model for POC demonstration")
            self.use_dummy = True
        else:
            self.use_dummy = False
            self._load_model()
        
        print(f"✓ Passive model initialized: {model_format}")
    
    def _load_model(self):
        """Load the pretrained model based on format"""
        
        if self.model_format == 'keras':
            self._load_keras()
        elif self.model_format == 'onnx':
            self._load_onnx()
        elif self.model_format == 'pytorch':
            self._load_pytorch()
        else:
            raise ValueError(f"Unsupported format: {self.model_format}")
    
    def _load_keras(self):
        """Load Keras .h5 model"""
        try:
            from tensorflow import keras
            self.model = keras.models.load_model(self.model_path, compile=False)
            print(f"✓ Loaded Keras model from {self.model_path}")
        except ImportError:
            raise ImportError("Install TensorFlow: pip install tensorflow")
        except Exception as e:
            print(f"Error loading Keras model: {e}")
            self.use_dummy = True
    
    def _load_onnx(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            self.model = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            print(f"✓ Loaded ONNX model from {self.model_path}")
        except ImportError:
            raise ImportError("Install ONNX Runtime: pip install onnxruntime")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            self.use_dummy = True
    
    def _load_pytorch(self):
        """Load PyTorch model"""
        try:
            import torch
            self.model = torch.load(self.model_path, map_location='cpu')
            self.model.eval()
            print(f"✓ Loaded PyTorch model from {self.model_path}")
        except ImportError:
            raise ImportError("Install PyTorch: pip install torch")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            self.use_dummy = True
    
    def predict(self, frames: np.ndarray) -> float:
        """
        Run inference on frame sequence
        
        Args:
            frames: Shape (T, H, W, C) - preprocessed frames
            
        Returns:
            Score ∈ [0, 1] - probability of LIVE
        """
        if self.use_dummy:
            return self._predict_dummy(frames)
        
        # Add batch dimension: (1, T, H, W, C)
        batch_frames = np.expand_dims(frames, axis=0)
        
        if self.model_format == 'keras':
            return self._predict_keras(batch_frames)
        elif self.model_format == 'onnx':
            return self._predict_onnx(batch_frames)
        elif self.model_format == 'pytorch':
            return self._predict_pytorch(batch_frames)
    
    def _predict_keras(self, batch_frames: np.ndarray) -> float:
        """Keras inference"""
        prediction = self.model.predict(batch_frames, verbose=0)
        score = float(prediction[0][0])
        return np.clip(score, 0.0, 1.0)
    
    def _predict_onnx(self, batch_frames: np.ndarray) -> float:
        """ONNX inference"""
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        result = self.model.run(
            [output_name],
            {input_name: batch_frames.astype(np.float32)}
        )
        
        score = float(result[0][0][0])
        return np.clip(score, 0.0, 1.0)
    
    def _predict_pytorch(self, batch_frames: np.ndarray) -> float:
        """PyTorch inference"""
        import torch
        
        # Convert to PyTorch tensor
        # PyTorch expects (B, C, T, H, W) or (B, T, C, H, W) depending on model
        # Adjust based on your model architecture
        tensor = torch.from_numpy(batch_frames).float()
        
        with torch.no_grad():
            output = self.model(tensor)
        
        score = float(output[0].item())
        return np.clip(score, 0.0, 1.0)
    
    def _predict_dummy(self, frames: np.ndarray) -> float:
        """
        Dummy model for POC when pretrained model not available
        
        Uses heuristics:
        - Texture analysis (variance)
        - Motion detection (frame differences)
        - Returns score ∈ [0, 1]
        """
        T, H, W, C = frames.shape
        
        # Denormalize for analysis
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames_denorm = (frames * std + mean) * 255.0
        frames_denorm = np.clip(frames_denorm, 0, 255).astype(np.uint8)
        
        # 1. Texture analysis (Laplacian variance)
        texture_scores = []
        for i in range(T):
            gray = frames_denorm[i].mean(axis=2).astype(np.uint8)
            laplacian = np.abs(np.gradient(gray)[0] + np.gradient(gray)[1])
            texture_scores.append(laplacian.var())
        
        avg_texture = np.mean(texture_scores)
        texture_score = min(1.0, avg_texture / 1000.0)  # Normalize
        
        # 2. Motion analysis (frame-to-frame differences)
        motion_scores = []
        for i in range(1, T):
            diff = np.abs(frames_denorm[i].astype(float) - frames_denorm[i-1].astype(float))
            motion_scores.append(diff.mean())
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        motion_score = min(1.0, avg_motion / 10.0)  # Normalize
        
        # 3. Combine scores
        # Real faces have moderate texture and subtle motion
        combined_score = (texture_score * 0.6 + motion_score * 0.4)
        
        # Add some randomness for demonstration
        noise = np.random.uniform(-0.1, 0.1)
        final_score = np.clip(combined_score + noise, 0.0, 1.0)
        
        return float(final_score)