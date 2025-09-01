import os
import sys
import numpy as np
import time
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.models.yolo.yoloe import YOLOEPETrainer
import cv2

# 添加CLIP库路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)


# 初始化检测模型
model = YOLOE("yoloe-11l.yaml")

# 加载分割预训练权重
model.load("yoloe-11l-seg.pt")

# 检测数据集上微调 - 使用更适合保留知识的参数
results = model.train(
    data="D:\\portfolio\\table-scenes\\data\\yolo\\data.yaml",  # 修改后的data.yaml包含所有COCO类别
    epochs=30,                # 减少训练轮数避免过拟合
    patience=10,              # 早停耐心值
    lr0=0.0005,               # 更小的学习率保留原始知识
    lrf=0.01,                 # 学习率最小值因子
    batch=8,                  # 小批量尺寸
    freeze=10,                # 冻结前10层保留特征提取能力
    imgsz=640,                # 输入图像尺寸
    optimizer="AdamW",        # 使用AdamW优化器
    trainer=YOLOEPETrainer,   # 检测任务用detection trainer
    name="train_preserve"     # 自定义实验名
)