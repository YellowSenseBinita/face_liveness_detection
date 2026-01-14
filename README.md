# Face Liveness Detection Pipeline

**Complete end-to-end system for Aadhaar-like face authentication**

Uses pretrained MobileNet+LSTM model for passive liveness detection with active blink detection fallback.

---

## 📁 Project Structure

```
face-liveness-detection/
│
├── pipeline.py              # Main orchestrator - RUN THIS
├── config.py                # Configuration (all hyperparameters)
├── face_detector.py         # Face detection (MediaPipe/MTCNN/OpenCV)
├── frame_processor.py       # Preprocessing and stacking
├── passive_model.py         # MobileNet+LSTM model loader
├── active_detector.py       # Blink detection (EAR-based)
├── requirements.txt         # Dependencies
├── README.md               # This file
│
└── models/                 # Place your pretrained models here
    └── liveness_mobilenet_lstm.h5  # (or .onnx, .pth)
```

---

## 🎯 Pipeline Flow

```
Webcam → Capture 2s video (~ 60 frames)
       ↓
Face Detection & Crop (MediaPipe)
       ↓
Preprocess & Stack → (16, 112, 112, 3)
       ↓
MobileNet+LSTM Inference → score ∈ [0, 1]
       ↓
Decision:
  - score ≥ 0.55 → LIVE ✅
  - score < 0.55 → Active Liveness
       ↓
Active: Blink Detection (EAR-based)
  - 2 blinks detected → LIVE ✅
  - timeout/fail → SPOOF ❌
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Choose model backend (based on your model format):**

```bash
# For Keras models (.h5)
pip install tensorflow-cpu==2.15.0

# OR for ONNX models (.onnx) - RECOMMENDED
pip install onnxruntime==1.16.3

# OR for PyTorch models (.pth)
pip install torch torchvision
```

### 2. (Optional) Place Your Pretrained Model

If you have a trained model:

```bash
mkdir models
cp /path/to/liveness_mobilenet_lstm.h5 models/
```

Edit `config.py`:
```python
MODEL_PATH = './models/liveness_mobilenet_lstm.h5'
MODEL_FORMAT = 'keras'  # or 'onnx', 'pytorch'
```

**Note:** Pipeline works WITHOUT a model file (uses dummy for testing)

### 3. Run Pipeline

```bash
python pipeline.py
```

That's it! 🎉

---

## 📊 Model Specifications

### Input Format (Expected)
- **Shape:** `(1, 16, 112, 112, 3)` - batch=1, frames=16, size=112x112, RGB
- **Type:** `float32`
- **Normalization:** ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Output Format (Expected)
- **Shape:** `(1, 1)` - single score
- **Range:** `[0, 1]` where 1=LIVE, 0=SPOOF

### Architecture (Typical)
```
Input (1, 16, 112, 112, 3)
  ↓
MobileNetV2 (per-frame features)
  ↓
LSTM (temporal modeling)
  ↓
Dense + Sigmoid → score
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Video capture
CAPTURE_DURATION = 2.0  # seconds

# Face detection
FACE_DETECTOR_TYPE = 'mediapipe'  # 'mediapipe', 'mtcnn', 'opencv'

# Preprocessing
NUM_FRAMES = 16  # Temporal dimension
FRAME_SIZE = (112, 112)  # Spatial size

# Model
MODEL_PATH = './models/liveness_mobilenet_lstm.h5'
MODEL_FORMAT = 'keras'  # 'keras', 'onnx', 'pytorch'

# Thresholds
PASSIVE_THRESHOLD = 0.55  # Decision threshold
HIGH_CONFIDENCE_THRESHOLD = 0.75

# Active liveness
BLINKS_REQUIRED = 2
ACTIVE_TIMEOUT = 5.0
```

---

## 🔧 Model Format Support

### Keras (.h5)
```python
MODEL_FORMAT = 'keras'
```
```bash
pip install tensorflow-cpu==2.15.0
```

### ONNX (.onnx) - **RECOMMENDED**
```python
MODEL_FORMAT = 'onnx'
```
```bash
pip install onnxruntime==1.16.3
```

**Convert Keras→ONNX:**
```python
import tf2onnx
model = tf.keras.models.load_model('model.h5')
spec = (tf.TensorSpec((None, 16, 112, 112, 3), tf.float32),)
tf2onnx.convert.from_keras(model, input_signature=spec, output_path="model.onnx")
```

### PyTorch (.pth)
```python
MODEL_FORMAT = 'pytorch'
```
```bash
pip install torch
```

---

## 🎬 Demo Mode (No Model Required)

Pipeline includes a **dummy model** for immediate testing:
- Uses heuristic-based detection (texture + motion analysis)
- Returns scores ∈ [0, 1]
- Perfect for testing the pipeline before training

```bash
# Just run - no model file needed!
python pipeline.py
```

---

## 📝 Example Output

```
======================================================================
FACE LIVENESS DETECTION SYSTEM
MobileNet+LSTM Pipeline for Aadhaar Authentication
======================================================================

✓ Face detector initialized: mediapipe
✓ Frame processor: 16 frames @ (112, 112)
✓ Passive model initialized: keras
✓ Active liveness initialized: Blink detection (EAR)

📹 Capturing video for 2.0s...
✓ Captured 59 frames

🔍 Detecting faces...
✓ Detected faces in 58/59 frames (98.3%)

🔧 Preprocessing frames...
✓ Stacked shape: (16, 112, 112, 3)

🤖 Running MobileNet+LSTM inference...
✓ Passive liveness score: 0.8234

✅ RESULT: LIVE (Passive detection passed)

======================================================================
LIVENESS DETECTION REPORT
======================================================================

📋 Final Decision: LIVE
Status: ✅ APPROVED

📊 Scores:
  Passive Score: 0.8234
  Threshold: 0.55

🔄 Active Liveness: Not triggered

⏱️  Total Time: 3.45s
======================================================================
```

---

## 🔄 Plug-and-Play Model Replacement

The pipeline is designed for **easy model swapping**:

1. Train your MobileNet+LSTM on Kaggle
2. Export to .h5, .onnx, or .pth
3. Place in `models/` folder
4. Update `config.py`
5. Run - **no code changes needed!**

---

## 🛠️ Troubleshooting

### "Model not found"
```
⚠️  Model not found: ./models/liveness_mobilenet_lstm.h5
→ Using dummy model for POC demonstration
```
**Solution:** This is normal! Dummy model works for testing. Add real model when ready.

### "Cannot open webcam"
```bash
# Check cameras
ls /dev/video*

# Try different index
# In pipeline.py, line 56:
cap = cv2.VideoCapture(1)  # Try 0, 1, 2
```

### Low detection rate
- Improve lighting
- Remove glasses
- Center face in frame

### Import errors
```bash
pip install mediapipe opencv-python numpy
```

---

## 🎯 Performance Tips

### For Edge Devices (Speed)
```python
NUM_FRAMES = 8  # Reduce from 16
FRAME_SIZE = (96, 96)  # Reduce from 112
MODEL_FORMAT = 'onnx'  # Fastest
```

### For Accuracy
```python
NUM_FRAMES = 32  # Increase
PASSIVE_THRESHOLD = 0.50  # Lower
FACE_DETECTOR_TYPE = 'mtcnn'  # More accurate
```

---

## 📈 Development Workflow

1. ✅ **Test with dummy model**
   ```bash
   python pipeline.py
   ```

2. ✅ **Train MobileNet+LSTM** (separate script/Kaggle)

3. ✅ **Export model**
   ```python
   model.save('models/liveness_mobilenet_lstm.h5')
   ```

4. ✅ **Update config.py**
   ```python
   MODEL_PATH = './models/liveness_mobilenet_lstm.h5'
   MODEL_FORMAT = 'keras'
   ```

5. ✅ **Run with trained model**
   ```bash
   python pipeline.py
   ```

6. ✅ **Fine-tune thresholds** based on validation data

---

## 🚀 Ready to Run!

```bash
# Minimal setup
pip install opencv-python numpy mediapipe

# Run immediately
python pipeline.py
```

**Works out of the box with dummy model for testing!**

---

## 📄 License
MIT

## 🤝 Support
- Ensure camera is accessible
- Check dependencies installed
- Test with dummy model first
- Verify model path/format if using pretrained model

---

**That's it! Simple, modular, production-ready.** 🎉