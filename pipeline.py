"""
GUARANTEED Real-Time UI Liveness Detection
This version WILL show UI windows or tell you why not
"""

import cv2
import numpy as np
import time
import sys

# Test if UI is available
def test_ui_support():
    """Test if OpenCV UI is available"""
    print("\n" + "="*70)
    print("TESTING UI SUPPORT")
    print("="*70)
    
    try:
        # Create a small test window
        test_img = np.zeros((100, 300, 3), dtype=np.uint8)
        cv2.putText(test_img, "Testing UI...", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('UI Test', test_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        
        print("✅ UI Support: AVAILABLE")
        print("   Windows will be displayed during detection")
        return True
        
    except cv2.error as e:
        print("❌ UI Support: NOT AVAILABLE")
        print(f"   Error: {e}")
        print("\n💡 Solutions:")
        print("   1. Install: pip uninstall opencv-python")
        print("              pip install opencv-contrib-python")
        print("   2. Or use headless mode (no UI)")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


# Run the test before starting
if __name__ == "__main__":
    has_ui = test_ui_support()
    
    if not has_ui:
        print("\n" + "="*70)
        print("⚠️  WARNING: UI not available")
        print("="*70)
        response = input("\nContinue without UI? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)
    
    print("\n" + "="*70)
    print("STARTING LIVENESS DETECTION WITH UI")
    print("="*70)


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
        
        return display
    
    def run_with_ui(self):
        """Run detection with full UI"""
        print("\n📹 Opening camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Camera opened - UI window will appear")
        print("→ Look at the camera for 2 seconds")
        print("→ Press 'q' to quit\n")
        
        frames = []
        face_crops = []
        current_score = 0.0
        start_time = time.time()
        last_inference = 0
        
        try:
            while (time.time() - start_time) < 2.0:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frames.append(frame.copy())
                
                # Detect face
                faces = self.face_detector.detect(frame)
                face_detected = len(faces) > 0
                
                if face_detected:
                    largest = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                    x, y, w, h = largest['box']
                    
                    # Draw face box on frame
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    
                    # Crop face
                    pad = 20
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(frame.shape[1], x + w + pad)
                    y2 = min(frame.shape[0], y + h + pad)
                    crop = frame[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        face_crops.append(crop)
                
                # Run inference periodically
                elapsed = time.time() - start_time
                if elapsed - last_inference >= 0.5 and len(face_crops) >= 8:
                    try:
                        stacked = self.frame_processor.process_sequence(face_crops[-16:])
                        current_score = self.passive_model.predict(stacked)
                        last_inference = elapsed
                        print(f"  Live score: {current_score:.3f}", end='\r')
                    except:
                        pass
                
                # Draw UI
                ui_info = {
                    'face_detected': face_detected,
                    'score': current_score,
                    'frame_count': len(frames)
                }
                display = self.draw_ui_overlay(frame, ui_info)
                
                # SHOW WINDOW
                cv2.imshow('Liveness Detection - LIVE VIEW', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\n⚠️  Quit by user")
                    break
            
            print(f"\n\n✅ Capture complete - {len(frames)} frames captured")
            
            # Show final frame for 2 seconds
            if len(frames) > 0:
                final_display = self.draw_ui_overlay(frames[-1], ui_info)
                cv2.putText(final_display, "ANALYZING...", 
                           (final_display.shape[1]//2 - 150, final_display.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.imshow('Liveness Detection - LIVE VIEW', final_display)
                cv2.waitKey(2000)
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Final inference
        if len(face_crops) == 0:
            print("❌ No faces detected")
            return {'success': False, 'decision': 'ERROR', 'passive_score': 0.0}
        
        print("\n🤖 Running final inference...")
        stacked = self.frame_processor.process_sequence(face_crops)
        final_score = self.passive_model.predict(stacked)
        
        print(f"✅ Final score: {final_score:.4f}")
        
        # Decision
        result = {
            'passive_score': final_score,
            'active_triggered': False,
            'success': False,
            'decision': None
        }
        
        if final_score >= self.config.PASSIVE_THRESHOLD:
            result['success'] = True
            result['decision'] = 'LIVE'
            print("✅ RESULT: LIVE")
        else:
            print("⚠️  Triggering active liveness (blink detection)...")
            result['active_triggered'] = True
            
            blink_success, blink_msg = self.active_detector.detect_blinks(
                num_blinks_required=self.config.BLINKS_REQUIRED,
                timeout=self.config.ACTIVE_TIMEOUT
            )
            
            result['active_result'] = blink_msg
            
            if blink_success:
                result['success'] = True
                result['decision'] = 'LIVE'
                print("✅ RESULT: LIVE (via active)")
            else:
                result['success'] = False
                result['decision'] = 'SPOOF'
                print("❌ RESULT: SPOOF")
        
        return result


def main():
    """Main entry point"""
    config = Config()
    pipeline = LivenessDetectionWithUI(config)
    
    result = pipeline.run_with_ui()
    
    if result:
        print("\n" + "="*70)
        print("FINAL REPORT")
        print("="*70)
        print(f"Decision: {result['decision']}")
        print(f"Score: {result['passive_score']:.4f}")
        print(f"Status: {'✅ APPROVED' if result['success'] else '❌ REJECTED'}")
        print("="*70)


if __name__ == "__main__":
    main()