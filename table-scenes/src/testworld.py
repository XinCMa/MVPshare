from ultralytics import YOLO
import os
import cv2
import numpy as np
import time

# 使用YOLOE-11L-SEG模型，但作为普通的YOLO模型使用
model_path = "D:\\portfolio\\table-scenes\\yoloe-11l-seg.pt"

if not os.path.exists(model_path):
    print(f"错误: 模型文件 {model_path} 不存在!")
    print(f"请确保下载模型文件: {model_path}")
    exit(1)
else:
    print(f"加载模型: {model_path}")
    # 初始化YOLO模型
    model = YOLO(model_path)  # 使用 YOLOE-11L-SEG 模型

# YOLOE 是一个开放词汇表模型
# 为筷子创建更丰富的描述集合，提高识别率
chopsticks_descriptions = [
    "chopsticks", 
    "wooden chopsticks", 
    "bamboo chopsticks", 
    "wooden eating utensils",
    "long thin wooden sticks for eating", 
    "asian eating utensils",
    "chinese chopsticks",
    "japanese chopsticks", 
    "korean chopsticks",
    "black chopsticks",
    "pair of thin wooden sticks",
    "traditional asian eating tools",
    "wooden rods used for eating",
    "slender wooden eating implements",
    "straight thin wooden sticks used in asian cuisine"
]

# 为扑克牌创建更丰富的描述集合，提高识别率
poker_descriptions = [
    "poker", 
    "playing cards",
    "deck of cards",
    "poker cards", 
    "playing card deck",
    "card game",
    "card deck",
    "cards for gambling",
    "casino cards",
    "rectangular paper cards with numbers and suits",
    "hearts spades clubs diamonds cards",
    "face cards",
    "poker game cards",
    "bridge cards",
    "standard 52-card deck",
    "playing card set",
    "gaming cards"
]

# 标准物体类别
standard_objects = [
    "person", "keyboard", "mouse", "laptop", "book",
    "cup", "wine glass", "fork", "spoon", "knife", "bowl", 
    "dining table", "cell phone", "remote", "scissors",
    "chair", "bottle", "chess board", "board game pieces"
]

# 使用YOLO预训练模型的标准类别
print("使用预训练模型的标准类别...")
model_classes = model.names
print(f"模型支持 {len(model_classes)} 个类别")
# 显示一些类别示例
sample_classes = list(model_classes.values())[:10]
print(f"类别示例: {', '.join(sample_classes)}")

# 打开摄像头
cap = cv2.VideoCapture(1)  # 外接摄像头，使用0表示内置摄像头
if not cap.isOpened():
    print("无法打开摄像头，尝试使用内置摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开任何摄像头!")
        exit(1)

print("摄像头已打开，按 'q' 退出")

# 用于FPS计算的变量
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
            
        # 执行预测 - 使用YOLOE模型
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
                
            # 特别检测和处理特定的物体
            # 检测感兴趣的物体，包括筷子和扑克牌
            is_chopsticks = any(desc.lower() in name.lower() for desc in chopsticks_descriptions)
            is_poker = any(desc.lower() in name.lower() for desc in poker_descriptions)
            
            if name == "book" or name == "vase" or name == "cup" or is_chopsticks or is_poker:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 根据物体类型选择不同颜色
                if is_chopsticks:
                    color = (255, 0, 0)  # 蓝色用于筷子
                elif is_poker:
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
        
        # 在画面左上角显示统计信息
        # 显示FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(annotated_frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示检测统计
        stats_text = "检测统计: "
        y_pos = 60
        cv2.putText(annotated_frame, stats_text, (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        
        # 显示总检测数量
        total_objects = len(detection_stats)
        if total_objects > 0:
            total_text = f"共检测到 {total_objects} 种物体"
            cv2.putText(annotated_frame, total_text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 30
            
            # 显示各物体数量
            for obj_name, count in detection_stats.items():
                text = f"{obj_name}: {count}"
                cv2.putText(annotated_frame, text, (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25
        
        cv2.imshow("YOLOE-11L-SEG 物体检测", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")