"""
YOLOE开放词汇表模型 - 增强型文本提示实时检测脚本

利用丰富的文本描述集合来提高特定物体的识别率，特别是扑克牌和筷子等物品。
YOLOE是一个开放词汇表模型，能够通过文本提示理解新的概念。
"""

import os
import sys
import numpy as np
import time
import cv2
from ultralytics import YOLOE

# 添加CLIP库路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)

# ====================== 丰富的文本描述集合 ======================

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

# ====================== 纯文本描述模式 ======================

# 本脚本移除了视觉提示功能，只使用丰富的文本描述集合来提高检测效果
# 参考图像部分已被移除，该功能在 yoloe_visual_prompt_realtime.py 中提供

# 合并所有类别描述，构建完整的类别列表
def build_complete_class_list():
    # 所有标准对象保留为单独的类别
    all_classes = standard_objects.copy()
    
    # 添加所有扑克牌描述
    all_classes.extend(poker_descriptions)
    
    # 添加所有筷子描述
    all_classes.extend(chopsticks_descriptions)
    
    print(f"构建了 {len(all_classes)} 个类别描述")
    return all_classes

# ====================== 模型初始化与推理 ======================

# 初始化YOLOE模型并仅使用丰富文本描述（不使用视觉提示）
def initialize_model_with_enhanced_descriptions():
    # 加载模型
    model_path = "D:\\portfolio\\table-scenes\\yoloe-11l-seg.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        exit(1)

    print(f"加载模型: {model_path}")
    model = YOLOE(model_path)
    
    # 构建丰富的类别描述列表
    all_class_descriptions = build_complete_class_list()
    
    # 设置类别名称和文本嵌入
    print(f"设置增强的类别描述 (共 {len(all_class_descriptions)} 个描述)...")
    model.set_classes(all_class_descriptions)
    print("类别描述设置完成")
    
    print("\n模型准备就绪 - 仅使用文本提示增强，没有视觉提示")
    return model

# 检测并显示FPS
def detect_and_display(model):
    # 直接使用索引1打开摄像头
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开外置摄像头(索引1)，尝试使用内置摄像头...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开任何摄像头!")
            return
    
    print("摄像头已打开，按 'q' 退出，'s' 保存当前帧")
    
    # 用于FPS计算
    prev_time = 0
    fps = 0
    
    # 主循环
    while True:
        # 读取视频帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
        
        # 计算FPS
        current_time = time.time()
        if prev_time > 0:
            fps = 1/(current_time - prev_time)
        prev_time = current_time
        
        # 图像预处理
        # 1. 调整亮度对比度
        alpha = 1.2  # 对比度增强因子
        beta = 10    # 亮度增强因子
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        try:
            # 执行预测
            results = model.predict(
                enhanced_frame,
                conf=0.25,       # 置信度阈值
                iou=0.45,        # NMS IOU阈值
                verbose=False,   # 不显示预测日志
                show_labels=True,# 显示标签名称而不是"object"
                show_conf=True   # 显示置信度
            )
            
            # 获取预测结果
            result = results[0]
            
            # 绘制结果
            annotated_frame = result.plot()
            
            # 在画面上显示FPS
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # 显示检测结果
            cv2.imshow("YOLOE 增强型文本提示检测", annotated_frame)
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 显示原始帧
            cv2.putText(
                frame, 
                "预测失败", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            cv2.imshow("YOLOE 增强型文本提示检测", frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 按 'q' 退出
            break
        elif key == ord('s'):
            # 按 's' 保存当前帧
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"D:\\portfolio\\table-scenes\\captures\\detection_{timestamp}.jpg"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, annotated_frame)
            print(f"已保存当前帧到: {save_path}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# ====================== 主程序 ======================

def main():
    print("=== YOLOE 开放词汇表模型纯文本提示检测 ===")
    print("初始化模型并加载丰富文本描述...")
    
    # 初始化模型并加载增强描述
    model = initialize_model_with_enhanced_descriptions()
    if model is None:
        return
    
    # 启动实时检测
    print("\n启动实时检测...")
    detect_and_display(model)
    
    print("程序已退出")

if __name__ == "__main__":
    main()
