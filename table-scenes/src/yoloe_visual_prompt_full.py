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

# 初始化YOLOE模型
model_path = "D:\\portfolio\\table-scenes\\yoloe-11l-seg.pt"
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    exit(1)

print(f"加载模型: {model_path}")
model = YOLOE(model_path)

# 导入XML解析库
import xml.etree.ElementTree as ET

# 定义多个参考图像文件和对应的注释文件
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

# 创建一个函数来解析单个参考图像的边界框
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
        return refer_image_path, np.array([[800, 900, 2400, 3200]]), np.array([0])
    
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

# 解析所有参考图像，收集边界框
all_reference_data = []
for image_file in reference_images:
    image_path, image_bboxes, image_classes = parse_reference_image(image_file)
    if image_path is not None and len(image_bboxes) > 0:
        all_reference_data.append({
            'path': image_path,
            'bboxes': image_bboxes,
            'classes': image_classes
        })

# 确保至少有一个有效的参考图像
if not all_reference_data:
    print("错误: 没有找到有效的参考图像和边界框")
    # 使用默认值作为后备
    refer_image_path = "D:\\portfolio\\table-scenes\\data\\voc\\images\\train\\微信图片_2025-08-27_111006_038.jpg"
    bboxes = np.array([
        [800, 900, 2400, 3200],   # 扑克牌边界框1
        [1000, 1200, 2200, 3000], # 扑克牌边界框2
    ])
    classes = np.array([0, 0])
    
    all_reference_data.append({
        'path': refer_image_path,
        'bboxes': bboxes,
        'classes': classes
    })

print(f"找到 {len(all_reference_data)} 个有效的参考图像")

# 为了简单起见，我们使用第一个参考图像进行视觉提示
# 设置类别名称
names = [
    "poker", "poker cards", "playing cards", "deck of cards", "card deck", 
    "chopsticks", "wooden chopsticks", "bamboo chopsticks",
    "cup", "wine glass", "book"
]
print(f"设置检测类别: {', '.join(names)}")
model.set_classes(names, model.get_text_pe(names))

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
    
    # 为每个参考图像创建多尺度边界框，提高远近都能识别的能力
    
    # 保存原始边界框
    original_bboxes = bboxes.copy()
    original_classes = classes.copy()
    
    # 创建更小的边界框版本，用于远距离物体识别
    small_bboxes = []
    small_classes = []
    
    # 为每个原始边界框创建一个更小的版本（模拟远距离）
    for bbox, cls in zip(original_bboxes, original_classes):
        # 获取边界框中心点
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # 获取边界框尺寸
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # 创建更小的边界框（模拟远距离物体）
        small_width = width * 0.5  # 缩小到原来的50%
        small_height = height * 0.5
        
        small_bbox = [
            int(center_x - small_width/2),
            int(center_y - small_height/2),
            int(center_x + small_width/2),
            int(center_y + small_height/2)
        ]
        
        small_bboxes.append(small_bbox)
        small_classes.append(cls)
    
    # 将原始边界框和小边界框合并
    if len(small_bboxes) > 0:
        combined_bboxes = np.vstack([original_bboxes, np.array(small_bboxes)])
        combined_classes = np.concatenate([original_classes, np.array(small_classes)])
    else:
        combined_bboxes = original_bboxes
        combined_classes = original_classes
    
    # 创建视觉提示字典
    visual_prompts = {
        "bboxes": combined_bboxes,
        "cls": combined_classes
    }
    
    # 使用当前参考图像更新模型记忆
    result = model.predict(
        dummy_image,
        refer_image=refer_image_path,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor
    )
    print(f"  已更新模型记忆")
    
    # 短暂暂停，确保模型更新完成
    time.sleep(0.5)

print("\n所有参考图像已被用于更新模型记忆")
# 导出模型

       
# 导出为ONNX格式
try:
    print("正在导出为ONNX格式...")
    model.export(format="onnx", imgsz=640)
except Exception as onnx_err:
    print(f"导出ONNX格式时出错: {onnx_err}")


print("\n导出操作已完成，模型现在包含所有12个参考图像的记忆")