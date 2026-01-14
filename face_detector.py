"""
Face Detector Module
Supports MediaPipe, MTCNN, and OpenCV backends
"""

import cv2
import numpy as np
from typing import List, Dict


class FaceDetector:
    """Unified face detection interface"""
    
    def __init__(self, detector_type: str = 'mediapipe', confidence: float = 0.7):
        """
        Initialize face detector
        
        Args:
            detector_type: 'mediapipe', 'mtcnn', or 'opencv'
            confidence: Minimum detection confidence
        """
        self.detector_type = detector_type.lower()
        self.confidence = confidence
        self.detector = None
        
        if self.detector_type == 'mediapipe':
            self._init_mediapipe()
        elif self.detector_type == 'mtcnn':
            self._init_mtcnn()
        elif self.detector_type == 'opencv':
            self._init_opencv()
        else:
            raise ValueError(f"Unknown detector: {detector_type}")
        
        print(f"✓ Face detector initialized: {self.detector_type}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Detection"""
        try:
            import mediapipe as mp
            if not hasattr(mp, 'solutions'):
                raise ImportError("MediaPipe version incompatible")
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=self.confidence
            )
        except (ImportError, AttributeError) as e:
            raise ImportError(f"MediaPipe error: {e}. Install: pip install mediapipe==0.10.9")
    
    def _init_mtcnn(self):
        """Initialize MTCNN"""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
        except ImportError:
            raise ImportError("Install MTCNN: pip install mtcnn tensorflow")
    
    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar cascade")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame
        
        Returns:
            List of dicts: [{'box': (x, y, w, h), 'confidence': float}, ...]
        """
        if self.detector_type == 'mediapipe':
            return self._detect_mediapipe(frame)
        elif self.detector_type == 'mtcnn':
            return self._detect_mtcnn(frame)
        elif self.detector_type == 'opencv':
            return self._detect_opencv(frame)
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """MediaPipe detection"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)
        
        faces = []
        if results.detections:
            h, w = frame.shape[:2]
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)
                
                faces.append({
                    'box': (x, y, width, height),
                    'confidence': detection.score[0]
                })
        return faces
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Dict]:
        """MTCNN detection"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        
        faces = []
        for result in results:
            if result['confidence'] >= self.confidence:
                x, y, w, h = result['box']
                faces.append({
                    'box': (max(0, x), max(0, y), w, h),
                    'confidence': result['confidence']
                })
        return faces
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Dict]:
        """OpenCV Haar Cascade detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_raw = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        return [{'box': tuple(box), 'confidence': 1.0} for box in faces_raw]
    
    def __del__(self):
        if self.detector_type == 'mediapipe' and self.detector:
            self.detector.close()