"""
Configuration for Face Liveness Detection Pipeline
All hyperparameters and model paths
"""


class Config:
    """Configuration class for liveness detection pipeline"""
    
    # ========== VIDEO CAPTURE ==========
    CAPTURE_DURATION = 2.0  # seconds
    TARGET_FPS = 30
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # ========== FACE DETECTION ==========
    FACE_DETECTOR_TYPE = 'mediapipe'  # Options: 'mediapipe', 'mtcnn', 'opencv'
    FACE_DETECTION_CONFIDENCE = 0.7
    MIN_DETECTION_RATE = 70.0  # Minimum % of frames with face detected
    
    # ========== FRAME PREPROCESSING ==========
    FRAME_SIZE = (112, 112)  # (H, W) - Standard for MobileNet
    NUM_FRAMES = 16  # Temporal dimension for LSTM
    SAMPLING_STRATEGY = 'uniform'  # Options: 'uniform', 'random', 'all'
    
    # ImageNet normalization (standard for MobileNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # ========== PASSIVE LIVENESS MODEL ==========
    # Path to pretrained MobileNet+LSTM model
    MODEL_PATH = './models/liveness_mobilenet_lstm.h5'  # or .onnx
    MODEL_FORMAT = 'keras'  # Options: 'keras', 'onnx', 'pytorch'
    
    # Decision thresholds
    PASSIVE_THRESHOLD = 0.55  # If score >= 0.55, classify as LIVE
    HIGH_CONFIDENCE_THRESHOLD = 0.75  # High confidence, no active needed
    
    # ========== ACTIVE LIVENESS (BLINK DETECTION) ==========
    BLINKS_REQUIRED = 2  # Number of blinks needed
    BLINK_EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold
    ACTIVE_TIMEOUT = 10.0  # seconds
    EAR_CONSEC_FRAMES = 3  # Consecutive frames below threshold = blink
    
    # ========== DEPLOYMENT ==========
    DEVICE = 'cpu'  # Options: 'cpu', 'cuda', 'mps'
    BATCH_SIZE = 1
    
    # ========== DISPLAY ==========
    SHOW_PREVIEW = True
    VERBOSE = True
    
    def __repr__(self):
        """String representation"""
        return f"Config(model={self.MODEL_PATH}, frames={self.NUM_FRAMES}, threshold={self.PASSIVE_THRESHOLD})"