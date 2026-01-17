# TSM Model Setup Guide

## Quick Setup (Automated)

```bash
cd D:\face_liveness_detection
python setup_tsm_model.py
```

This will automatically download, convert, and integrate the TSM model.

---

## Manual Setup (Step-by-Step)

### Step 1: Install Dependencies

```bash
pip install torch torchvision onnx onnxruntime
```

### Step 2: Download TSM Repository

```bash
cd D:\face_liveness_detection
git clone https://github.com/mit-han-lab/temporal-shift-module.git
cd temporal-shift-module
```

### Step 3: Download Pretrained Weights

**Option A: Using wget (if available)**
```bash
mkdir pretrained
cd pretrained
wget https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100.pth
```

**Option B: Manual Download**
1. Visit: https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100.pth
2. Save to: `temporal-shift-module/pretrained/TSM_kinetics_mobilenetv2.pth`

### Step 4: Convert to ONNX (Optional but Recommended)

Create `convert_tsm.py`:

```python
import torch
import torch.nn as nn
import sys

sys.path.insert(0, 'temporal-shift-module')
from ops.models import TSN

# Load model
model = TSN(
    num_class=400,
    num_segments=8,
    modality='RGB',
    base_model='mobilenetv2',
    consensus_type='avg',
    dropout=0.5,
    is_shift=True,
    shift_div=8
)

# Load weights
checkpoint = torch.load('temporal-shift-module/pretrained/TSM_kinetics_mobilenetv2.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Modify for liveness (binary classification)
model.new_fc = nn.Linear(model.new_fc.in_features, 1)

# Export to ONNX
dummy_input = torch.randn(1, 8, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'models/mobilenetv3_tsm.onnx',
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

print("✓ Converted to ONNX: models/mobilenetv3_tsm.onnx")
```

Run:
```bash
python convert_tsm.py
```

### Step 5: Update config.py

Edit `D:\face_liveness_detection\config.py`:

```python
# Change these lines:
MODEL_PATH = './models/mobilenetv3_tsm.onnx'  # or .pth if not converted
MODEL_FORMAT = 'onnx'  # or 'pytorch' if using .pth
```

### Step 6: Test Pipeline

```bash
cd D:\face_liveness_detection
python pipeline.py
```

---

## Alternative: Use PyTorch Model Directly

If ONNX conversion fails, use PyTorch format:

1. Skip Step 4 (conversion)
2. Update config.py:
   ```python
   MODEL_PATH = './temporal-shift-module/pretrained/TSM_kinetics_mobilenetv2.pth'
   MODEL_FORMAT = 'pytorch'
   ```
3. Update `passive_model.py` to load TSM model

---

## Troubleshooting

### Issue: Git not installed
**Solution:** Download as ZIP from GitHub and extract

### Issue: PyTorch not installed
**Solution:** 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: ONNX conversion fails
**Solution:** Use PyTorch format directly (see alternative above)

### Issue: Model file too large
**Solution:** The model is ~14MB, ensure you have space and stable internet

---

## Expected Results

After setup:
- Model file: `models/mobilenetv3_tsm.onnx` (~14MB)
- Config updated: `MODEL_FORMAT = 'onnx'`
- Pipeline ready to test

**Note:** Pretrained model is trained on Kinetics (action recognition), not liveness.
Expected accuracy: 50-60% without fine-tuning.
For production use, fine-tune on liveness dataset.

---

## Next Steps

1. ✅ Test with `python pipeline.py`
2. ✅ Evaluate accuracy on test faces
3. ⏳ Collect liveness dataset (if accuracy is low)
4. ⏳ Fine-tune model on liveness data
5. ⏳ Deploy to production

---

## Quick Commands Summary

```bash
# Full automated setup
python setup_tsm_model.py

# Or manual
git clone https://github.com/mit-han-lab/temporal-shift-module.git
# Download weights manually
python convert_tsm.py
# Edit config.py
python pipeline.py
```
