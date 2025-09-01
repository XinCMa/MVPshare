"""
YOLOE模型视觉提示增强脚本
- 加载预训练YOLOE模型
- 使用12个扑克牌参考图像更新模型记忆
- 同时保留原始模型对其他物体的检测能力
- 实时使用摄像头进行检测
"""

import os
import sys
import numpy as np
import time
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 添加CLIP库路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)

# 定义参考图像文件列表
reference_images = [
    "微信图片_2025-08-27_111006_038.jpg",
    "微信图片_2025-08-27_111016_778.jpg",
    "微信图片_2025-08-27_110947_058.jpg",
    "微信图片_2025-08-27_110951_321.jpg",
    "微信图片_2025-08-27_110954_228.jpg",
    "微信图片_2025-08-27_110956_979.jpg",
    "微信图片_2025-08-27_110959_908.jpg",
    "微信图片_2025-08-27_111011_372.jpg",
    "微信图片_2025-08-27_111123_002.jpg",
    "微信图片_20250827133628_138_101.jpg",
    "微信图片_20250827133630_139_101.jpg",
    "微信图片_20250827133632_140_101.jpg"
]

# 解析参考图像的边界框
def parse_reference_image(image_filename):
    refer_image_path = f"D:\\portfolio\\table-scenes\\data\\voc\\images\\train\\{image_filename}"
    annotation_file = f"D:\\portfolio\\table-scenes\\data\\voc\\annotations\\train\\{image_filename.replace('.jpg', '.xml')}"
    
    image_bboxes = []
    image_classes = []
    
    if not os.path.exists(refer_image_path):
        print(f"警告: 参考图像不存在: {refer_image_path}")
        return None, None, None
    
    if not os.path.exists(annotation_file):
        print(f"警告: 注释文件不存在: {annotation_file}")
        return None, None, None
    
    print(f"解析注释文件: {annotation_file}")
    
    try:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        
        for obj in root.findall('./object'):
            name = obj.find('name').text
            
            # 只选择扑克牌相关的对象
            if "poker" in name.lower() or "card" in name.lower() or "playing" in name.lower():
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                image_bboxes.append([xmin, ymin, xmax, ymax])
                image_classes.append(0)  # 类ID为0（扑克牌）
                
                print(f"找到目标: {name}, 边界框: [{xmin}, {ymin}, {xmax}, {ymax}]")
        
        if not image_bboxes:
            print(f"注释文件 {annotation_file} 中未找到扑克牌相关的对象，跳过此参考图像")
            return None, None, None
        
        return refer_image_path, np.array(image_bboxes), np.array(image_classes)
            
    except Exception as e:
        print(f"解析注释文件时出错: {e}")
        return None, None, None

# 加载所有有效的参考图像
def load_reference_images():
    all_reference_data = []
    for image_file in reference_images:
        image_path, image_bboxes, image_classes = parse_reference_image(image_file)
        if image_path is not None and len(image_bboxes) > 0:
            all_reference_data.append({
                'path': image_path,
                'bboxes': image_bboxes,
                'classes': image_classes
            })
    
    print(f"找到 {len(all_reference_data)} 个有效的参考图像")
    return all_reference_data

# 初始化YOLOE模型并使用参考图像更新模型记忆
def initialize_model_with_memory():
    # 加载模型
    model_path = "D:\\portfolio\\table-scenes\\yoloe-11l-seg.pt"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        exit(1)

    print(f"加载模型: {model_path}")
    model = YOLOE(model_path)
    
    # 设置完整的类别名称 - 包含COCO类别和自定义类别
    # 使用扩展后的类别列表，确保包含原始模型的所有类别和自定义类别
    names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush", "poker cards", "chopsticks", "wooden chopsticks", "bamboo chopsticks"
    ]
    print(f"设置检测类别: {len(names)} 类")
    
    # 重要：使用类别名称列表和YOLOE的set_classes方法来设置类别名称
    model.set_classes(names)
    print("已设置类别名称映射")
    
    # 加载所有参考图像
    all_reference_data = load_reference_images()
    if not all_reference_data:
        print("错误: 没有找到有效的参考图像和边界框")
        return None
    
    # 创建一个空白图像用于初始化视觉提示
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 依次使用每个参考图像更新模型记忆
    print("\n开始依次使用所有参考图像更新模型记忆...")
    for i, reference in enumerate(all_reference_data):
        refer_image_path = reference['path']
        bboxes = reference['bboxes']
        classes = reference['classes']
        
        print(f"[{i+1}/{len(all_reference_data)}] 使用参考图像: {os.path.basename(refer_image_path)}")
        print(f"  使用的边界框数量: {len(bboxes)}")
        
        # 创建视觉提示字典
        visual_prompts = {
            "bboxes": bboxes,
            "cls": classes
        }
        
        # 使用当前参考图像更新模型记忆
        result = model.predict(
            dummy_image,
            refer_image=refer_image_path,
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor
        )
        print(f"  已更新模型记忆")
        time.sleep(0.5)  # 短暂暂停，确保模型更新完成
    
    print("\n所有参考图像已被用于更新模型记忆，模型准备就绪")
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
            cv2.imshow("YOLOE 实时检测 (带扑克牌记忆)", annotated_frame)
            
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
            cv2.imshow("YOLOE 实时检测 (带扑克牌记忆)", frame)
        
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

# 主函数
def main():
    print("=== YOLOE模型视觉提示增强脚本 ===")
    print("初始化模型并加载12个扑克牌参考图像记忆...")
    
    # 初始化模型并加载记忆
    model = initialize_model_with_memory()
    if model is None:
        return
    
    # 启动实时检测
    print("\n启动实时检测...")
    detect_and_display(model)
    
    print("程序已退出")

if __name__ == "__main__":
    main()
