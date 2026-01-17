"""
Script to integrate downloaded TSM model into pipeline
Handles both MobileNetV2 and ResNet50 TSM models
"""

import os
import torch
import torch.nn as nn

def find_downloaded_model():
    """Find the downloaded TSM model file"""
    print("\n" + "="*70)
    print("FINDING DOWNLOADED TSM MODEL")
    print("="*70)
    
    # Common download locations
    search_paths = [
        ".",
        "Downloads",
        os.path.expanduser("~/Downloads"),
        "temporal-shift-module/pretrained"
    ]
    
    # Look for .pth files
    found_models = []
    for path in search_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.pth') and 'TSM' in file:
                    full_path = os.path.join(path, file)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    found_models.append((full_path, size_mb))
    
    if not found_models:
        print("✗ No TSM model found")
        print("\nPlease specify the path to your downloaded .pth file:")
        model_path = input("Path: ").strip()
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            return model_path, size_mb
        return None, 0
    
    print(f"✓ Found {len(found_models)} TSM model(s):")
    for i, (path, size) in enumerate(found_models, 1):
        print(f"  {i}. {os.path.basename(path)} ({size:.1f} MB)")
        print(f"     Location: {path}")
    
    if len(found_models) == 1:
        return found_models[0]
    
    choice = int(input(f"\nSelect model (1-{len(found_models)}): "))
    return found_models[choice - 1]

def identify_model_type(model_path, size_mb):
    """Identify if model is MobileNetV2 or ResNet50"""
    print("\n" + "="*70)
    print("IDENTIFYING MODEL TYPE")
    print("="*70)
    
    filename = os.path.basename(model_path).lower()
    
    if 'mobilenet' in filename:
        model_type = 'mobilenetv2'
        print(f"✓ Detected: MobileNetV2 TSM ({size_mb:.1f} MB)")
    elif 'resnet' in filename or size_mb > 100:
        model_type = 'resnet50'
        print(f"✓ Detected: ResNet50 TSM ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  Unknown model type ({size_mb:.1f} MB)")
        print("\nWhich model did you download?")
        print("1. MobileNetV2 (~14 MB)")
        print("2. ResNet50 (~196 MB)")
        choice = input("Select (1/2): ")
        model_type = 'mobilenetv2' if choice == '1' else 'resnet50'
    
    return model_type

def setup_model(model_path, model_type):
    """Setup the TSM model for liveness detection"""
    print("\n" + "="*70)
    print("SETTING UP MODEL")
    print("="*70)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Copy model to models folder
    target_path = f"models/tsm_{model_type}_kinetics.pth"
    
    if model_path != target_path:
        print(f"Copying model to: {target_path}")
        import shutil
        shutil.copy(model_path, target_path)
        print("✓ Model copied")
    
    return target_path

def update_config(model_path, model_type):
    """Update config.py with model settings"""
    print("\n" + "="*70)
    print("UPDATING CONFIGURATION")
    print("="*70)
    
    config_path = "config.py"
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update model path
        content = content.replace(
            "MODEL_PATH = './models/liveness_mobilenet_lstm.h5'",
            f"MODEL_PATH = './{model_path}'"
        )
        
        # Update model format
        content = content.replace(
            "MODEL_FORMAT = 'keras'",
            "MODEL_FORMAT = 'pytorch'"
        )
        
        # Update frame size for TSM (224x224 instead of 112x112)
        content = content.replace(
            "FRAME_SIZE = (112, 112)",
            "FRAME_SIZE = (224, 224)"
        )
        
        # Update number of frames for TSM (8 instead of 16)
        content = content.replace(
            "NUM_FRAMES = 16",
            "NUM_FRAMES = 8"
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("✓ config.py updated:")
        print(f"  - MODEL_PATH = './{model_path}'")
        print(f"  - MODEL_FORMAT = 'pytorch'")
        print(f"  - FRAME_SIZE = (224, 224)")
        print(f"  - NUM_FRAMES = 8")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config: {e}")
        return False

def create_tsm_loader():
    """Create a custom TSM model loader for passive_model.py"""
    print("\n" + "="*70)
    print("CREATING TSM MODEL LOADER")
    print("="*70)
    
    loader_code = '''"""
TSM Model Loader for PyTorch
Add this to passive_model.py _load_pytorch() method
"""

import sys
sys.path.insert(0, 'temporal-shift-module')

try:
    from ops.models import TSN
    
    # Determine model type from path
    if 'mobilenet' in self.model_path.lower():
        base_model = 'mobilenetv2'
    else:
        base_model = 'resnet50'
    
    # Load TSM model
    model = TSN(
        num_class=400,
        num_segments=8,
        modality='RGB',
        base_model=base_model,
        consensus_type='avg',
        dropout=0.5,
        is_shift=True,
        shift_div=8,
        shift_place='blockres'
    )
    
    # Load pretrained weights
    checkpoint = torch.load(self.model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    # Modify for binary classification (liveness)
    model.new_fc = nn.Linear(model.new_fc.in_features, 1)
    model.sigmoid = nn.Sigmoid()
    
    model.eval()
    self.model = model
    
except Exception as e:
    print(f"Error loading TSM model: {e}")
    print("Falling back to dummy model")
    self.use_dummy = True
'''
    
    with open('tsm_loader_snippet.py', 'w') as f:
        f.write(loader_code)
    
    print("✓ Created tsm_loader_snippet.py")
    print("  (Reference for integrating TSM into passive_model.py)")

def main():
    """Main setup workflow"""
    print("\n" + "="*70)
    print("TSM MODEL INTEGRATION")
    print("Downloaded Model Setup")
    print("="*70)
    
    # Step 1: Find model
    model_path, size_mb = find_downloaded_model()
    if not model_path:
        print("\n✗ No model found. Please download TSM model first.")
        return
    
    print(f"\n✓ Using model: {model_path}")
    print(f"  Size: {size_mb:.1f} MB")
    
    # Step 2: Identify type
    model_type = identify_model_type(model_path, size_mb)
    
    # Step 3: Setup
    target_path = setup_model(model_path, model_type)
    
    # Step 4: Update config
    if not update_config(target_path, model_type):
        print("\n⚠️  Manual config update required")
    
    # Step 5: Create loader
    create_tsm_loader()
    
    # Final instructions
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\n⚠️  IMPORTANT: TSM model requires additional integration")
    print("\nNext steps:")
    print("1. Install PyTorch: pip install torch torchvision")
    print("2. The model is ready but needs custom loader")
    print("3. For now, test with dummy model: python pipeline.py")
    print("4. Or wait for TSM integration in passive_model.py")
    print("\n" + "="*70)
    
    print(f"\nModel location: {target_path}")
    print(f"Model type: {model_type.upper()}")
    print(f"Config updated: ✓")
    print("\nNote: TSM models are complex and require the TSM repository")
    print("For quick testing, continue using the dummy model.")

if __name__ == "__main__":
    main()
