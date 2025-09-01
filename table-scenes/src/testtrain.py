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

# 加载训练好的模型
model_path = r"D:\portfolio\table-scenes\runs\detect\train3\weights\best.pt"
print(f"加载训练好的模型: {model_path}")

model = YOLOE(model_path)

# 输出模型的类别信息
print(f"模型类别: {model.names}")
print(f"类别数量: {len(model.names) if hasattr(model, 'names') else '未知'}")

# Run detection on the given image
results = model.predict(source=1, conf=0.3, show=True)  # Use 0.3 confidence threshold

# Show results
results[0].show()