"""
Configuration for Face Liveness Detection Pipeline
All hyperparameters and model paths
"""


class Config:
    """Configuration class for liveness detection pipeline"""
    
    # ========== VIDEO CAPTURE ==========
    CAPTURE_DURATION = 10.0  # seconds
    TARGET_FPS = 30
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    
    # ========== UI TIMING ==========
    WARMUP_DURATION = 3.0       # Time to align face
    ANALYSIS_DURATION = 2.0     # Time to gather passive data
    PREPARATION_DELAY = 1.5     # Pause before active challenges
    
    # ========== FACE DETECTION ==========
    FACE_DETECTOR_TYPE = 'mediapipe'  # Options: 'mediapipe', 'mtcnn', 'opencv'
    FACE_DETECTION_CONFIDENCE = 0.7
    MIN_DETECTION_RATE = 70.0  # Minimum % of frames with face detected
    
    # ========== FRAME PREPROCESSING ==========
    FRAME_SIZE = (224, 224)  # (H, W) - Standard for TSM
    NUM_FRAMES = 8  # Temporal dimension for TSM
    SAMPLING_STRATEGY = 'uniform'  # Options: 'uniform', 'random', 'all'
    
    # ImageNet normalization (standard for MobileNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # ========== PASSIVE LIVENESS MODEL ==========
    # Path to pretrained TSM model
    MODEL_PATH = './models/tsm_mobilenetv3_kinetics.pth' 
    MODEL_FORMAT = 'pytorch'  # Options: 'keras', 'onnx', 'pytorch'
    
    # Decision thresholds
    PASSIVE_THRESHOLD = 0.55  # If score >= 0.55, classify as LIVE
    HIGH_CONFIDENCE_THRESHOLD = 0.75  # High confidence, no active needed
    
    # ========== ACTIVE LIVENESS (CHALLENGES) ==========
    BLINKS_REQUIRED = 2  # Number of blinks needed
    MOUTH_OPEN_REQUIRED = 1  # Number of mouth opens needed
    BLINK_EAR_THRESHOLD = 0.21  # Eye Aspect Ratio threshold
    MOUTH_MAR_THRESHOLD = 0.35  # Mouth Aspect Ratio threshold
    ACTIVE_TIMEOUT = 10.0  # seconds
    EAR_CONSEC_FRAMES = 3  # Consecutive frames below threshold = blink
    
    # MediaPipe Specific
    MP_DETECTION_CONFIDENCE = 0.5
    MP_TRACKING_CONFIDENCE = 0.5
    
    # ========== DEPLOYMENT ==========
    DEVICE = 'cpu'  # Options: 'cpu', 'cuda', 'mps'
    BATCH_SIZE = 1
    
    # ========== DISPLAY ==========
    SHOW_PREVIEW = True
    VERBOSE = True
    
    def __repr__(self):
        """String representation"""
        return f"Config(model={self.MODEL_PATH}, frames={self.NUM_FRAMES}, threshold={self.PASSIVE_THRESHOLD})"