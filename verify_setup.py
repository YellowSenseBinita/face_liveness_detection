
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Verification")

def verify_imports():
    logger.info("Verifying imports...")
    try:
        import torch
        logger.info("✅ torch available")
    except ImportError as e:
        logger.error(f"❌ torch missing: {e}")
        return False

    try:
        import cv2
        logger.info("✅ cv2 available")
    except ImportError as e:
        logger.error(f"❌ cv2 missing: {e}")
        return False
        
    try:
        import mediapipe
        logger.info("✅ mediapipe available")
    except ImportError as e:
        logger.error(f"❌ mediapipe missing: {e}")
        return False
        
    return True

def verify_modules():
    logger.info("Verifying module initialization...")
    try:
        from passive_model import PassiveLivenessModel
        # We don't want to actually load the heavy model if we can avoid it, 
        # but the init will try to load it. 
        # Let's just check the class exists and imports work.
        logger.info("✅ PassiveLivenessModel imported")
    except Exception as e:
        logger.error(f"❌ PassiveLivenessModel check failed: {e}")
        return False
        
    try:
        from active_detector import ActiveLivenessDetector
        # Try initializing with defaults (should use mediapipe if available)
        detector = ActiveLivenessDetector()
        logger.info("✅ ActiveLivenessDetector initialized")
    except Exception as e:
        logger.error(f"❌ ActiveLivenessDetector check failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    logger.info("Starting setup verification...")
    if verify_imports() and verify_modules():
        logger.info("="*30)
        logger.info("SETUP VERIFICATION SUCCESSFUL")
        logger.info("="*30)
        sys.exit(0)
    else:
        logger.error("="*30)
        logger.error("SETUP VERIFICATION FAILED")
        logger.error("="*30)
        sys.exit(1)
