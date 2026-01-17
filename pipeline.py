"""
GUARANTEED Real-Time UI Liveness Detection
This version WILL show UI windows or tell you why not
"""

import cv2
import numpy as np
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LivenessPipeline")

# Test if UI is available
def test_ui_support():
    """Test if OpenCV UI is available"""
    logger.info("="*70)
    logger.info("TESTING UI SUPPORT")
    logger.info("="*70)
    
    try:
        # Create a small test window
        test_img = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(test_img, "Testing UI...", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('UI Test', test_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        logger.info("✅ UI Support: AVAILABLE")
        logger.info("   Windows will be displayed during detection")
        return True
        
    except cv2.error as e:
        logger.error("❌ UI Support: NOT AVAILABLE")
        logger.error(f"   Error: {e}")
        logger.info("\n💡 Solutions:")
        logger.info("   1. Install: pip uninstall opencv-python")
        logger.info("              pip install opencv-contrib-python")
        logger.info("   2. Or use headless mode (no UI)")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


# Run the test before starting
if __name__ == "__main__":
    has_ui = test_ui_support()
    
    if not has_ui:
        logger.warning("="*70)
        logger.warning("⚠️  WARNING: UI not available")
        logger.warning("="*70)
        response = input("\nContinue without UI? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            sys.exit(0)
    
    logger.info("="*70)
    logger.info("STARTING LIVENESS DETECTION WITH UI")
    logger.info("="*70)


from config import Config
from face_detector import FaceDetector
from frame_processor import FrameProcessor
from passive_model import PassiveLivenessModel
from active_detector import ActiveLivenessDetector


class LivenessDetectionWithUI:
    """Liveness detection with GUARANTEED UI display"""
    
    def __init__(self, config=None):
        self.config = config or Config()
        
        # Initialize components
        self.face_detector = FaceDetector(
            detector_type=self.config.FACE_DETECTOR_TYPE,
            confidence=self.config.FACE_DETECTION_CONFIDENCE
        )
        
        self.frame_processor = FrameProcessor(
            target_size=self.config.FRAME_SIZE,
            num_frames=self.config.NUM_FRAMES,
            mean=self.config.MEAN,
            std=self.config.STD
        )
        
        self.passive_model = PassiveLivenessModel(
            model_path=self.config.MODEL_PATH,
            model_format=self.config.MODEL_FORMAT
        )
        
        self.active_detector = ActiveLivenessDetector(
            ear_threshold=self.config.BLINK_EAR_THRESHOLD,
            mar_threshold=self.config.MOUTH_MAR_THRESHOLD,
            consec_frames=self.config.EAR_CONSEC_FRAMES
        )
    
    def draw_ui_overlay(self, frame, info):
        """Draw complete UI overlay"""
        h, w = frame.shape[:2]
        display = frame.copy()
        
        # Top panel
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Title
        cv2.putText(display, "LIVE LIVENESS DETECTION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Face status
        if info.get('face_detected'):
            cv2.putText(display, "Face: DETECTED ✓", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Face: NOT FOUND", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Score
        score = info.get('score', 0.0)
        threshold = self.config.PASSIVE_THRESHOLD
        
        cv2.putText(display, f"Score: {score:.3f} / Threshold: {threshold:.2f}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Score bar
        bar_x, bar_y = 20, 140
        bar_w, bar_h = w - 40, 30
        
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        fill_w = int(bar_w * min(score, 1.0))
        color = (0, 255, 0) if score >= threshold else (0, 0, 255)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        
        # Threshold line
        thresh_x = bar_x + int(bar_w * threshold)
        cv2.line(display, (thresh_x, bar_y), (thresh_x, bar_y + bar_h), (255, 255, 255), 2)
        
        # Guide box
        box_w, box_h = int(w * 0.4), int(h * 0.5)
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2 + 50
        
        cv2.rectangle(display, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                     (100, 200, 255), 2)
        cv2.putText(display, "Place face here", 
                   (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Result indicator
        if score > 0:
            result_text = "REAL" if score >= threshold else "FAKE"
            result_color = (0, 255, 0) if score >= threshold else (0, 0, 255)
            
            result_x, result_y = w - 250, h - 120
            result_w, result_h = 230, 100
            
            cv2.rectangle(display, (result_x, result_y), 
                         (result_x + result_w, result_y + result_h), result_color, 3)
            cv2.putText(display, result_text, 
                       (result_x + 50, result_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 3)
        
        # Frame count
        cv2.putText(display, f"Frames: {info.get('frame_count', 0)}", 
                   (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instruction text (new)
        instruction = info.get('instruction', 'Please wait...')
        if instruction:
            # Draw a banner for instruction
            cv2.rectangle(display, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(display, instruction, 
                       (20, h - 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Progress bar if applicable
            if 'progress' in info:
                prog = info['progress']
                cv2.rectangle(display, (0, h - 10), (int(w * prog), h), (0, 255, 255), -1)
        
        return display
    
    def run_with_ui(self):
        """Run detection with multi-phase UI pipeline"""
        logger.info("📹 Opening camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        
        logger.info("✅ Camera ready")
        
        frames = []
        face_crops = []
        current_score = 0.0
        
        # UI State Control
        phase = "ALIGNMENT"
        phase_start = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                
                # Face Detection (Common for all phases)
                faces = self.face_detector.detect(frame)
                face_detected = len(faces) > 0
                box = None
                
                if face_detected:
                    largest = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                    x, y, fw, fh = largest['box']
                    box = (x, y, fw, fh)
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                    
                    # Crop face for analysis
                    pad = 20
                    crop = frame[max(0, y-pad):min(h, y+fh+pad), max(0, x-pad):min(w, x+fw+pad)]
                    if crop.size > 0 and phase == "SCANNING":
                        face_crops.append(crop)
                
                # --- PHASE LOGIC ---
                elapsed = time.time() - phase_start
                ui_info = {'face_detected': face_detected, 'score': current_score, 'frame_count': len(frames)}
                
                if phase == "ALIGNMENT":
                    ui_info['instruction'] = "ALIGNMENT: Place your face in the center box"
                    ui_info['progress'] = min(1.0, elapsed / self.config.WARMUP_DURATION)
                    if face_detected and elapsed >= 1.0: # Wait at least 1s for stability
                        phase = "SCANNING"
                        phase_start = time.time()
                        logger.info("→ Starting passive scan...")
                
                elif phase == "SCANNING":
                    ui_info['instruction'] = "SCANNING: Hold still... Analyzing liveness"
                    ui_info['progress'] = elapsed / self.config.ANALYSIS_DURATION
                    if elapsed >= self.config.ANALYSIS_DURATION:
                        phase = "ANALYZING"
                        phase_start = time.time()
                    
                    # Periodic mid-scan inference
                    if len(face_crops) >= self.config.NUM_FRAMES and len(frames) % 15 == 0:
                        try:
                            stacked = self.frame_processor.process_sequence(face_crops[-self.config.NUM_FRAMES:])
                            current_score = self.passive_model.predict(stacked)
                        except: pass
                
                elif phase == "ANALYZING":
                    ui_info['instruction'] = "ANALYZING: Processing results..."
                    if elapsed >= 1.0: break # Exit loop to run final inference
                
                # Draw and Show
                display = self.draw_ui_overlay(frame, ui_info)
                cv2.imshow('Liveness Detection', display)
                frames.append(frame.copy())
                
                if cv2.waitKey(1) & 0xFF == ord('q'): return None

            # --- FINAL PASSIVE ANALYSIS ---
            if not face_crops:
                return {'success': False, 'decision': 'ERROR', 'passive_score': 0.0}
                
            logger.info("🤖 Running final passive inference...")
            stacked = self.frame_processor.process_sequence(face_crops)
            final_score = self.passive_model.predict(stacked)
            logger.info(f"✅ Final score: {final_score:.4f}")
            
            # --- DECISION & TRANSITION ---
            result = {'passive_score': final_score, 'active_triggered': False, 'success': False, 'decision': None}
            
            if final_score >= self.config.HIGH_CONFIDENCE_THRESHOLD:
                result.update({'success': True, 'decision': 'LIVE'})
                logger.info("✅ RESULT: LIVE (High Confidence)")
            else:
                # Need Active Liveness or Passive check passed but low confidence
                if final_score < self.config.PASSIVE_THRESHOLD:
                    logger.info("⚠️  Passive check failed. Delaying for active challenge...")
                else:
                    logger.info("📝 Passive check borderline. Verifying with active challenge...")
                
                # PREPARATION PHASE (GET READY)
                prep_start = time.time()
                while time.time() - prep_start < self.config.PREPARATION_DELAY:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.flip(frame, 1)
                    info = {'instruction': "GET READY: Challenge starting in {}s...".format(int(self.config.PREPARATION_DELAY - (time.time() - prep_start)) + 1), 'face_detected': True, 'score': final_score}
                    cv2.imshow('Liveness Detection', self.draw_ui_overlay(frame, info))
                    if cv2.waitKey(1) & 0xFF == ord('q'): return None

                # ACTIVE CHALLENGE
                result['active_triggered'] = True
                cv2.destroyAllWindows() # Reset windows for challenge
                
                blink_success, blink_msg = self.active_detector.detect_blinks(
                    num_blinks_required=self.config.BLINKS_REQUIRED,
                    timeout=self.config.ACTIVE_TIMEOUT
                )
                
                result['active_result'] = blink_msg
                if blink_success:
                    result.update({'success': True, 'decision': 'LIVE'})
                else:
                    result.update({'success': False, 'decision': 'SPOOF'})
            
            return result
            
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    config = Config()
    pipeline = LivenessDetectionWithUI(config)
    
    result = pipeline.run_with_ui()
    
    if result:
        logger.info("="*70)
        logger.info("FINAL REPORT")
        logger.info("="*70)
        logger.info(f"Decision: {result['decision']}")
        logger.info(f"Score: {result['passive_score']:.4f}")
        logger.info(f"Status: {'✅ APPROVED' if result['success'] else '❌ REJECTED'}")
        logger.info("="*70)


if __name__ == "__main__":
    main()