import sys
import torch
from ultralytics import YOLO

model = YOLO(model="models/yolo_pose/yolo11x-pose.pt")

device = torch.device("mps" if torch.mps.is_available() else "cpu")

if len(sys.argv) < 2:
    raise ValueError("Usage: python yolo_video_pose.py video_path")

video_path = sys.argv[1]

results = model.track(source=video_path, device=device, conf=0.5, show=True, save=True)
