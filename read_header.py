model_path = "d:/face_liveness_detection/models/liveness_mobilenet_lstm.hm"
try:
    with open(model_path, "rb") as f:
        header = f.read(16)
        print(f"Header: {header}")
except Exception as e:
    print(f"Error: {e}")
