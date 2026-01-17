"""
Active Liveness Detector Module
Blink detection using Eye Aspect Ratio (EAR) with facial landmarks
Supports multiple backends: MediaPipe (preferred), dlib, OpenCV (fallback)
"""

import cv2
import numpy as np
import time
import logging
import sys
from typing import Tuple, List

# Configure logging
logger = logging.getLogger("ActiveDetector")


class ActiveLivenessDetector:
    """
    Active liveness detection using blink detection
    Uses Eye Aspect Ratio (EAR) method with facial landmarks
    """
    
    def __init__(self, ear_threshold: float = 0.21, mar_threshold: float = 0.35, consec_frames: int = 3):
        """
        Initialize active detector
        
        Args:
            ear_threshold: EAR value below which eye is considered closed
            mar_threshold: MAR value above which mouth is considered open
            consec_frames: Number of consecutive frames for valid blink/action
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self.use_mediapipe = False
        self.use_dlib = False
        self.use_opencv = False
        
        # Try MediaPipe first (most accurate)
        try:
            import mediapipe as mp
            # Check if solutions module exists
            if hasattr(mp, 'solutions'):
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.use_mediapipe = True
                logger.info("✓ Active liveness initialized: Blink detection (MediaPipe)")
                return
            else:
                raise AttributeError("MediaPipe solutions not available")
        except (ImportError, AttributeError) as e:
            logger.warning(f"MediaPipe not available: {e}")
        
        # Try dlib as fallback (good accuracy)
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            # Check if landmark predictor file exists
            try:
                self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
                self.use_dlib = True
                logger.info("✓ Active liveness initialized: Blink detection (dlib)")
                return
            except:
                logger.warning("dlib landmark file not found (need shape_predictor_68_face_landmarks.dat)")
        except ImportError:
            logger.warning("dlib not available")
        
        # Use OpenCV as final fallback (basic but works)
        try:
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            if self.eye_cascade.empty() or self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar cascades")
            
            self.use_opencv = True
            logger.info("✓ Active liveness initialized: Blink detection (OpenCV fallback)")
        except Exception as e:
            logger.error(f"Failed to initialize any blink detector: {e}")
            raise RuntimeError("No blink detection method available")
    
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            eye_landmarks: Array of 6 (x,y) points defining eye contour
            
        Returns:
            EAR value
        """
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C + 1e-6)  # Add epsilon to avoid division by zero
        return ear
    
    def calculate_mar(self, mouth_landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR)
        
        Args:
            mouth_landmarks: Array of (x,y) points for mouth
            Expects [top_lip, bottom_lip, left_corner, right_corner]
            
        Returns:
            MAR value
        """
        # Vertical distance between lips
        A = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[1])
        # Horizontal distance between corners
        B = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[3])
        
        mar = A / (B + 1e-6)
        return mar
    
    def get_eye_landmarks(self, face_landmarks, eye_indices: list, h: int, w: int) -> np.ndarray:
        """Extract eye landmark coordinates from MediaPipe"""
        eye_points = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * w
            y = landmark.y * h
            eye_points.append([x, y])
        
        return np.array(eye_points)
    
    def detect_blinks(self, num_blinks_required: int = 2, 
                     timeout: float = 5.0) -> Tuple[bool, str]:
        """
        Detect blinks from webcam feed
        
        Args:
            num_blinks_required: Number of blinks to detect
            timeout: Maximum time to wait for blinks
            
        Returns:
            (success, message) tuple
        """
        if self.use_mediapipe:
            return self._detect_blinks_mediapipe(num_blinks_required, timeout)
        elif self.use_dlib:
            return self._detect_blinks_dlib(num_blinks_required, timeout)
        elif self.use_opencv:
            return self._detect_blinks_opencv(num_blinks_required, timeout)
        else:
            return False, "No blink detection method available"
    
    def _detect_blinks_mediapipe(self, num_blinks_required: int, timeout: float) -> Tuple[bool, str]:
        """MediaPipe-based blink detection using Face Mesh landmarks"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        # MediaPipe eye landmark indices
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        MOUTH = [13, 14, 78, 308] # Top, Bottom, Left, Right
        
        blink_count = 0
        mouth_open_count = 0
        closed_frames = 0
        mouth_open_frames = 0
        was_open = True
        mouth_was_closed = True
        
        # Randomly choose a challenge (or do both)
        challenge_type = np.random.choice(['blink', 'mouth'])
        target_count = num_blinks_required if challenge_type == 'blink' else 1
        
        start_time = time.time()
        last_print = 0
        
        instruction = f"Please blink {num_blinks_required} times" if challenge_type == 'blink' else "Please open your mouth"
        logger.info(f"Challenge started: {instruction}")
        
        while (blink_count < num_blinks_required if challenge_type == 'blink' else mouth_open_count < 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # Detect face landmarks
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # DRAW UI
                cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, instruction, (w // 2 - 150, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if challenge_type == 'blink':
                    # Get eye landmarks
                    left_eye = self.get_eye_landmarks(face_landmarks, LEFT_EYE, h, w)
                    right_eye = self.get_eye_landmarks(face_landmarks, RIGHT_EYE, h, w)
                    
                    # Calculate EAR for both eyes
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Check if eyes are closed
                    if avg_ear < self.ear_threshold:
                        closed_frames += 1
                    else:
                        # Eyes opened again after being closed
                        if closed_frames >= self.consec_frames and not was_open:
                            blink_count += 1
                            logger.info(f"Blink {blink_count} detected!")
                            was_open = True
                        closed_frames = 0
                        if closed_frames == 0:
                            was_open = True
                    
                    if closed_frames >= self.consec_frames:
                        was_open = False
                        
                    cv2.putText(frame, f"Blinks: {blink_count}/{num_blinks_required}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif challenge_type == 'mouth':
                    # Get mouth landmarks
                    mouth_pts = self.get_eye_landmarks(face_landmarks, MOUTH, h, w)
                    mar = self.calculate_mar(mouth_pts)
                    
                    if mar > self.mar_threshold:
                        mouth_open_frames += 1
                    else:
                        if mouth_open_frames >= self.consec_frames and not mouth_was_closed:
                            mouth_open_count += 1
                            logger.info("Mouth open detected!")
                            mouth_was_closed = True
                        mouth_open_frames = 0
                        if mouth_open_frames == 0:
                            mouth_was_closed = True
                    
                    if mouth_open_frames >= self.consec_frames:
                        mouth_was_closed = False
                    
                    cv2.putText(frame, "Mouth detection active", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Active Liveness Challenge', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.imshow('Active Liveness Challenge', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        success = (blink_count >= num_blinks_required if challenge_type == 'blink' else mouth_open_count >= 1)
        return success, "Challenge successful" if success else "Challenge failed"
    
    def _detect_blinks_dlib(self, num_blinks_required: int, timeout: float) -> Tuple[bool, str]:
        """dlib-based blink detection using 68 facial landmarks"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        # dlib eye landmark indices (68-point model)
        LEFT_EYE_POINTS = list(range(36, 42))  # Points 36-41
        RIGHT_EYE_POINTS = list(range(42, 48))  # Points 42-47
        
        blink_count = 0
        closed_frames = 0
        was_open = True
        
        start_time = time.time()
        
        print(f"\n👁️  Please blink {num_blinks_required} times")
        print("Look at the camera naturally and blink...")
        
        while blink_count < num_blinks_required:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray)
            
            if len(faces) > 0:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                
                # Extract eye coordinates
                left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                                    for i in LEFT_EYE_POINTS])
                right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] 
                                     for i in RIGHT_EYE_POINTS])
                
                # Calculate EAR
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Blink detection logic
                if avg_ear < self.ear_threshold:
                    closed_frames += 1
                else:
                    if closed_frames >= self.consec_frames and not was_open:
                        blink_count += 1
                        logger.info(f"Blink {blink_count} detected!")
                        was_open = True
                    closed_frames = 0
                    if closed_frames == 0:
                        was_open = True
                
                if closed_frames >= self.consec_frames:
                    was_open = False
                
                # Draw feedback
                cv2.putText(frame, f"Blinks: {blink_count}/{num_blinks_required}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {avg_ear:.3f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No face detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            time_left = timeout - (time.time() - start_time)
            cv2.putText(frame, f"Time: {time_left:.1f}s", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Active Liveness - Blink Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if blink_count >= num_blinks_required:
            return True, f"Success: {blink_count} blinks detected"
        else:
            return False, f"Failed: Only {blink_count}/{num_blinks_required} blinks"
    
    def _detect_blinks_opencv(self, num_blinks_required: int, timeout: float) -> Tuple[bool, str]:
        """Simple OpenCV-based blink detection using eye cascade"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False, "Cannot open camera"
        
        blink_count = 0
        prev_eyes = 2
        closed_frames = 0
        
        start_time = time.time()
        last_print = 0
        
        logger.info(f"Challenge started: Please blink {num_blinks_required} times")
        
        while blink_count < num_blinks_required:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect face first
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            num_eyes = 0
            if len(faces) > 0:
                # Take largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Search for eyes only in face region
                face_roi = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(20, 20))
                num_eyes = len(eyes)
            
            # Blink logic: eyes present -> eyes closed -> eyes present
            if num_eyes < 2 and prev_eyes >= 2:
                closed_frames += 1
            else:
                if closed_frames >= self.consec_frames:
                    blink_count += 1
                    logger.info(f"Blink {blink_count} detected!")
                closed_frames = 0
            
            prev_eyes = num_eyes
            
            # Print progress every 1 second
            elapsed = time.time() - start_time
            time_left = timeout - elapsed
            logger.debug(f"Blinks: {blink_count}/{num_blinks_required} | Eyes: {num_eyes} | Time left: {time_left:.1f}s")
            last_print = elapsed
        
        cap.release()
        
        
        if blink_count >= num_blinks_required:
            return True, f"Success: {blink_count} blinks detected"
        else:
            return False, f"Failed: Only {blink_count}/{num_blinks_required} blinks"
    
    def __del__(self):
        """Cleanup"""
        if self.use_mediapipe and hasattr(self, 'face_mesh'):
            self.face_mesh.close()