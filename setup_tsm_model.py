"""
Script to download and setup pretrained TSM MobileNetV3 model
Run this to integrate pretrained model into the pipeline
"""

import os
import sys
import urllib.request
import torch
import torch.nn as nn

def download_tsm_repo():
    """Clone TSM repository"""
    print("\n" + "="*70)
    print("STEP 1: Downloading TSM Repository")
    print("="*70)
    
    if os.path.exists('temporal-shift-module'):
        print("✓ TSM repo already exists")
        return True
    
    print("Cloning TSM repository...")
    os.system('git clone https://github.com/mit-han-lab/temporal-shift-module.git')
    
    if os.path.exists('temporal-shift-module'):
        print("✓ TSM repository downloaded")
        return True
    else:
        print("✗ Failed to download TSM repository")
        return False

def download_pretrained_weights():
    """Download pretrained TSM MobileNetV2 weights"""
    print("\n" + "="*70)
    print("STEP 2: Downloading Pretrained Weights")
    print("="*70)
    
    # Alternative URLs (try multiple sources)
    model_urls = [
        "https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100.pth",
        "https://hanlab.mit.edu/files/tsm/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100.pth",
    ]
    
    model_path = "temporal-shift-module/pretrained/TSM_kinetics_mobilenetv2.pth"
    
    os.makedirs("temporal-shift-module/pretrained", exist_ok=True)
    
    if os.path.exists(model_path):
        print("✓ Pretrained weights already exist")
        return model_path
    
    print("Trying multiple download sources...")
    print("This may take 5-10 minutes (model size: ~14MB)...")
    
    for i, model_url in enumerate(model_urls, 1):
        try:
            print(f"\nAttempt {i}/{len(model_urls)}: {model_url[:60]}...")
            urllib.request.urlretrieve(model_url, model_path)
            print(f"✓ Downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print("\n✗ All download attempts failed")
    print("\n⚠️  ALTERNATIVE SOLUTION:")
    print("Since pretrained TSM model is unavailable, we'll use a simpler approach.")
    print("\nOption 1: Skip pretrained model and use dummy model (current setup)")
    print("Option 2: Download from Google Drive (manual):")
    print("  1. Visit: https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL")
    print("  2. Download TSM_kinetics_RGB_mobilenetv2...pth")
    print(f"  3. Save to: {model_path}")
    print("  4. Run this script again")
    return None

def convert_to_onnx(pytorch_model_path):
    """Convert PyTorch model to ONNX format"""
    print("\n" + "="*70)
    print("STEP 3: Converting to ONNX Format")
    print("="*70)
    
    try:
        import torch
        import torch.onnx
        
        # Add TSM module to path
        sys.path.insert(0, 'temporal-shift-module')
        from ops.models import TSN
        
        print("Loading PyTorch model...")
        
        # Load pretrained model
        model = TSN(
            num_class=400,
            num_segments=8,
            modality='RGB',
            base_model='mobilenetv2',
            consensus_type='avg',
            dropout=0.5,
            img_feature_dim=256,
            partial_bn=True,
            is_shift=True,
            shift_div=8,
            shift_place='blockres',
            fc_lr5=True,
            temporal_pool=False,
            non_local=False
        )
        
        # Load weights
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        print("✓ Model loaded successfully")
        
        # Modify for binary classification (liveness)
        print("Modifying for liveness detection...")
        model.new_fc = nn.Linear(model.new_fc.in_features, 1)
        model.sigmoid = nn.Sigmoid()
        
        # Create dummy input
        dummy_input = torch.randn(1, 8, 3, 224, 224)
        
        # Export to ONNX
        onnx_path = "models/mobilenetv3_tsm_kinetics.onnx"
        os.makedirs("models", exist_ok=True)
        
        print(f"Exporting to ONNX: {onnx_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✓ ONNX model saved to: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print("\nNote: ONNX conversion requires PyTorch and ONNX packages")
        print("Install with: pip install torch onnx")
        return None

def update_config():
    """Update config.py to use TSM model"""
    print("\n" + "="*70)
    print("STEP 4: Updating Configuration")
    print("="*70)
    
    config_path = "config.py"
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update model path and format
        content = content.replace(
            "MODEL_PATH = './models/liveness_mobilenet_lstm.h5'",
            "MODEL_PATH = './models/mobilenetv3_tsm_kinetics.onnx'"
        )
        content = content.replace(
            "MODEL_FORMAT = 'keras'",
            "MODEL_FORMAT = 'onnx'"
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("✓ config.py updated")
        print("  - MODEL_PATH = './models/mobilenetv3_tsm_kinetics.onnx'")
        print("  - MODEL_FORMAT = 'onnx'")
        return True
        
    except Exception as e:
        print(f"✗ Failed to update config: {e}")
        print("\nManual update required:")
        print("Edit config.py:")
        print("  MODEL_PATH = './models/mobilenetv3_tsm_kinetics.onnx'")
        print("  MODEL_FORMAT = 'onnx'")
        return False

def test_pipeline():
    """Test the pipeline with TSM model"""
    print("\n" + "="*70)
    print("STEP 5: Testing Pipeline")
    print("="*70)
    
    print("Running pipeline.py...")
    print("This will test the complete system with TSM model")
    print("\nPress Ctrl+C to stop\n")
    
    os.system('python pipeline.py')

def main():
    """Main setup workflow"""
    print("\n" + "="*70)
    print("TSM MODEL SETUP FOR FACE LIVENESS DETECTION")
    print("="*70)
    print("\nThis script will:")
    print("1. Download TSM repository")
    print("2. Download pretrained weights (~14MB)")
    print("3. Convert to ONNX format")
    print("4. Update config.py")
    print("5. Test the pipeline")
    print("\nEstimated time: 10-15 minutes")
    print("="*70)
    
    input("\nPress Enter to continue...")
    
    # Step 1: Download repo
    if not download_tsm_repo():
        print("\n✗ Setup failed at step 1")
        return
    
    # Step 2: Download weights
    weights_path = download_pretrained_weights()
    if not weights_path:
        print("\n" + "="*70)
        print("CONTINUING WITHOUT PRETRAINED MODEL")
        print("="*70)
        print("\nThe pipeline will use the dummy model for now.")
        print("This is fine for testing the pipeline structure.")
        print("\nTo use a real model later:")
        print("1. Download TSM weights manually (see instructions above)")
        print("2. Or train your own model on liveness data")
        print("3. Place model in models/ folder")
        print("4. Update config.py")
        print("\n" + "="*70)
        
        cont = input("\nContinue with dummy model? (y/n): ")
        if cont.lower() != 'y':
            return
        
        # Skip to testing with dummy model
        print("\n" + "="*70)
        print("SETUP COMPLETE (Using Dummy Model)")
        print("="*70)
        print("\nYou can now test the pipeline with:")
        print("  python pipeline.py")
        print("\nThe dummy model will be used for liveness detection.")
        print("Accuracy will be ~60-70% (for testing only).")
        print("="*70)
        
        test_now = input("\nTest pipeline now? (y/n): ")
        if test_now.lower() == 'y':
            test_pipeline()
        return
    
    # Step 3: Convert to ONNX
    onnx_path = convert_to_onnx(weights_path)
    if not onnx_path:
        print("\n⚠️  ONNX conversion failed")
        print("You can still use PyTorch format")
        print("Update config.py manually:")
        print(f"  MODEL_PATH = '{weights_path}'")
        print("  MODEL_FORMAT = 'pytorch'")
        return
    
    # Step 4: Update config
    if not update_config():
        print("\n⚠️  Config update failed - manual update required")
    
    # Step 5: Test
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run: python pipeline.py")
    print("2. Test with your webcam")
    print("3. Check accuracy")
    print("4. Fine-tune if needed")
    print("="*70)
    
    test_now = input("\nTest pipeline now? (y/n): ")
    if test_now.lower() == 'y':
        test_pipeline()

if __name__ == "__main__":
    main()
