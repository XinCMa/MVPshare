import sys
import os
import cv2
import time
import numpy as np
from ultralytics import YOLOE

# 添加CLIP-main目录到Python路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)

# 初始化YOLOE模型
model_path = "D:\\portfolio\\table-scenes\\yoloe-11l-seg.pt"
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    exit(1)

print(f"加载模型: {model_path}")
model = YOLOE(model_path)

# 设置要检测的对象
names = [
    "person", "keyboard", "mouse", "laptop", "cup",
    "wine glass", "book", "cell phone", 
    "chopsticks", "wooden chopsticks", "bamboo chopsticks", "asian eating utensils",
    "playing cards", "poker cards", "card deck", "deck of cards",
    "bottle", "fork", "spoon", "knife"
]
print(f"设置检测类别: {', '.join(names)}")

# 设置类别
try:
    model.set_classes(names, model.get_text_pe(names))
    print("成功设置检测类别")
except Exception as e:
    print(f"设置类别时发生错误: {e}")
    # 如果失败，使用默认的COCO类别
    print("将使用默认COCO类别")

# 打开摄像头
cap = cv2.VideoCapture(1)  # 外接摄像头，使用0表示内置摄像头
if not cap.isOpened():
    print("无法打开摄像头，尝试使用内置摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开任何摄像头!")
        exit(1)

print("摄像头已打开，按 'q' 退出")

# 用于FPS计算
prev_time = 0
fps = 0

# 循环获取视频帧并进行预测
try:
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
            
        # 执行预测
        results = model.predict(
            frame, 
            conf=0.30,  # 置信度阈值
            verbose=True  # 开启详细输出
        )
        
        # 显示结果
        annotated_frame = results[0].plot()
        
        # 创建检测统计信息
        detection_stats = {}
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls)
            name = results[0].names[cls_id]
            conf = float(box.conf)
            
            # 统计每种物体的数量
            if name in detection_stats:
                detection_stats[name] += 1
            else:
                detection_stats[name] = 1
                
            # 特别关注的物体
            special_objects = ["chopsticks", "playing cards", "cup", "wine glass", "book"]
            if name in special_objects:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 为不同物体使用不同颜色
                if "chopsticks" in name or "chop" in name:
                    color = (255, 0, 0)  # 蓝色用于筷子
                elif "playing cards" in name or "poker" in name or "card" in name:
                    color = (255, 0, 255)  # 紫色用于扑克牌
                else:
                    color = (0, 0, 255)  # 红色用于其他特定物体
                
                # 绘制更醒目的边框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                # 添加标签
                label = f"{name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # 在控制台输出信息
                print(f"检测到特定物体: {name} (置信度: {conf:.2f})")
        
        # 在画面上显示统计信息
        y_pos = 30
        
        # 显示FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        
        # 显示物体总数
        total_objects = sum(detection_stats.values())
        cv2.putText(annotated_frame, f"总物体数: {total_objects}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        
        # 显示各种物体的数量
        for obj_name, count in detection_stats.items():
            # 为特殊物体使用不同颜色
            if obj_name in special_objects:
                color = (0, 255, 255)  # 黄色
            else:
                color = (0, 255, 0)  # 绿色
                
            text = f"{obj_name}: {count}"
            cv2.putText(annotated_frame, text, (20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
        
        # 显示窗口
        cv2.imshow("YOLOE-11L-SEG 开放词汇检测", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")