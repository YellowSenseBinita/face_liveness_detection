"""
Frame Processor Module
Handles preprocessing and stacking frames for CNN+LSTM model
"""

import cv2
import numpy as np
from typing import List, Tuple


class FrameProcessor:
    """Preprocesses and stacks frames into (T, H, W, C) format"""
    
    def __init__(self, target_size: Tuple[int, int] = (112, 112), 
                 num_frames: int = 16,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize processor
        
        Args:
            target_size: (H, W) for each frame
            num_frames: Temporal dimension
            mean: ImageNet mean for normalization
            std: ImageNet std for normalization
        """
        self.target_size = target_size
        self.num_frames = num_frames
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
        print(f"✓ Frame processor: {num_frames} frames @ {target_size}")
    
    def process_sequence(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """
        Process face crops into model input
        
        Args:
            face_crops: List of face crops (BGR images)
            
        Returns:
            Tensor of shape (T, H, W, C) normalized and ready for model
        """
        # Sample frames uniformly
        sampled = self._sample_uniform(face_crops)
        
        # Process each frame
        processed = []
        for crop in sampled:
            # Resize
            resized = cv2.resize(crop, (self.target_size[1], self.target_size[0]))
            
            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            normalized = (normalized - self.mean) / self.std
            
            processed.append(normalized)
        
        # Stack into (T, H, W, C)
        stacked = np.stack(processed, axis=0).astype(np.float32)
        
        return stacked
    
    def _sample_uniform(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Uniformly sample num_frames from sequence"""
        n = len(frames)
        
        if n >= self.num_frames:
            # Sample uniformly
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        else:
            # Repeat if not enough frames
            indices = np.linspace(0, n-1, self.num_frames, dtype=int)
        
        return [frames[i] for i in indices]
    
    def denormalize(self, frame: np.ndarray) -> np.ndarray:
        """Denormalize frame for visualization"""
        denorm = (frame * self.std + self.mean) * 255.0
        denorm = np.clip(denorm, 0, 255).astype(np.uint8)
        return denorm