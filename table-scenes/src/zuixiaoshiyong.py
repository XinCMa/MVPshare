import os
import sys
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 添加CLIP-main目录到Python路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)
from ultralytics import YOLOE

model = YOLOE("yoloe-11l-seg.onnx")
results = model.predict(source=1, conf=0.3, show=True)
results[0].show()