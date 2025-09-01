"""
测试导出的模型
这个脚本将加载已导出的带有扑克牌记忆的YOLOE模型并进行测试
"""

import os
import sys
import cv2
import numpy as np
import time
from ultralytics import YOLOE
import torch

# 添加CLIP库路径
clip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CLIP-main")
if clip_path not in sys.path:
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")

# 设置导出模型的路径
MODEL_PATH = "D:\\portfolio\\table-scenes\\yoloe-11l-poker-memory_full.pt"
if not os.path.exists(MODEL_PATH):
    print(f"错误: 导出的模型文件不存在: {MODEL_PATH}")
    # 尝试其他可能的文件名
    alt_paths = [
        "D:\\portfolio\\table-scenes\\yoloe-11l-poker-memory_exported.pt",
        "D:\\portfolio\\table-scenes\\yoloe-11l-poker-memory.pt",
        "D:\\portfolio\\table-scenes\\yoloe-11l-seg_exported.pt"
    ]
    for path in alt_paths:
        if os.path.exists(path):
            MODEL_PATH = path
            print(f"找到替代模型: {MODEL_PATH}")
            break
    else:
        print("错误: 找不到可用的模型文件，请先运行导出脚本")
        exit(1)

def main():
    # 加载已导出的模型
    print(f"加载导出的模型: {MODEL_PATH}")
    try:
        if MODEL_PATH.endswith("_full.pt"):
            # 加载完整模型（包含记忆状态）
            model_state = torch.load(MODEL_PATH)
            model = YOLOE("D:\\portfolio\\table-scenes\\yoloe-11l-seg.pt")
            model.model = model_state
            print("已加载完整模型（包含记忆状态）")
        else:
            # 加载常规导出模型
            model = YOLOE(MODEL_PATH)
            print("已加载导出模型")
            
        # 设置类别名称（确保类别与导出前相同）
        class_names = [
            "poker cards", "playing cards", "deck of cards", "card deck",
            "chopsticks", "wooden chopsticks", "bamboo chopsticks", 
            "cup", "wine glass", "book"
        ]
        model.set_classes(class_names)
        
        # 打开摄像头
        cap = cv2.VideoCapture(1)  # 外接摄像头，使用0表示内置摄像头
        if not cap.isOpened():
            print("无法打开摄像头，尝试使用内置摄像头...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("无法打开任何摄像头!")
                exit(1)

        print("摄像头已打开，按 'q' 退出")
        print("使用导出的模型，无需再次提供视觉提示")

        # 用于FPS计算
        prev_time = 0
        fps = 0

        # 循环获取视频帧并进行预测
        while True:
            # 计算FPS
            current_time = time.time()
            if prev_time > 0:
                fps = 1/(current_time - prev_time)
            prev_time = current_time
            
            ret, frame = cap.read()
            if not ret:
                print("无法获取视频帧")
                break
            
            # 增强图像预处理
            enhanced_frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=15)
            
            # 执行预测 - 不需要提供视觉提示，因为模型已包含记忆
            result = model.predict(
                enhanced_frame,
                conf=0.15,  # 置信度阈值
                verbose=True  # 开启详细输出
            )
            
            # 显示结果
            annotated_frame = result[0].plot()
            
            # 在画面上显示统计信息
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"导出的模型: {os.path.basename(MODEL_PATH)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示窗口
            cv2.imshow("YOLOE 导出模型测试", annotated_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()
