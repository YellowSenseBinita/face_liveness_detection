import os

model_path = "d:/face_liveness_detection/models/liveness_mobilenet_lstm.hm"

print(f"Checking {model_path}...")

# Try Keras
try:
    from tensorflow import keras
    model = keras.models.load_model(model_path, compile=False)
    print("✓ Success: This is a Keras model.")
except Exception as e:
    print(f"✗ Not a standard Keras model: {e}")

# Try PyTorch
try:
    import torch
    model = torch.load(model_path, map_location='cpu')
    print("✓ Success: This is a PyTorch model.")
except Exception as e:
    print(f"✗ Not a standard PyTorch model: {e}")
