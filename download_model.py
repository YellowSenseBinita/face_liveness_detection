import urllib.request
import os

url = "https://hanlab.mit.edu/files/tsm/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100.pth"
dest = "d:/face_liveness_detection/models/tsm_mobilenetv3_kinetics.pth"

os.makedirs(os.path.dirname(dest), exist_ok=True)

print(f"Downloading from {url}...")
try:
    urllib.request.urlretrieve(url, dest)
    print(f"Successfully downloaded to {dest}")
except Exception as e:
    print(f"Error downloading: {e}")
