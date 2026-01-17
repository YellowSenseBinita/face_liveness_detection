import torch
import sys
import os
import torch.nn as nn

# Add TSM module to path
tsm_path = os.path.join(os.getcwd(), 'temporal-shift-module')
if tsm_path not in sys.path:
    sys.path.insert(0, tsm_path)

from ops.models import TSN

model_path = "models/tsm_mobilenetv3_kinetics.pth"

try:
    print(f"Testing load from {model_path}...")
    
    # Initialize architecture
    model = TSN(
        num_class=400,
        num_segments=8,
        modality='RGB',
        base_model='mobilenetv2',
        consensus_type='avg',
        dropout=0.5,
        is_shift=True,
        shift_div=8,
        shift_place='blockres'
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    # Modify for binary classification
    model.new_fc = nn.Linear(model.new_fc.in_features, 1)
    model.sigmoid = nn.Sigmoid()
    
    model.eval()
    print("✓ Model loaded successfully!")
    
    # Test dummy inference
    input_tensor = torch.randn(1, 8, 3, 224, 224)
    with torch.no_grad():
        output = model(input_tensor)
    print(f"✓ Inference test successful! Output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Load failed: {e}")
    import traceback
    traceback.print_exc()
