import sys
import torch
from ultralytics import YOLO

model = YOLO(model="models/yolo/yolo11m.pt")

device = torch.device("mps" if torch.mps.is_available() else "cpu")

if len(sys.argv) < 2:
    raise ValueError("Usage: python image_tracking.py image_path")

image_path = sys.argv[1]

results = model.track(source=image_path, conf=0.7, device=device)

result = results[0]

result.show()
