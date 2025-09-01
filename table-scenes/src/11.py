import os
import sys
import numpy as np
import time
import cv2
from ultralytics import YOLOE

# 添加CLIP-main目录到Python路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "bus"]

model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict(source=1, conf=0.3, show=True)  # Use 0.3 confidence threshold

# Show results
results[0].show()