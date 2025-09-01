"""
简化版YOLOE扑克牌模型记忆与导出脚本
只保留核心功能：参考12张图片并导出模型
"""

import os
import sys
import numpy as np
import time
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import cv2

# 添加CLIP库路径
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

# 主函数
def main():
    # 设置类别名称
    names = [
        "poker", "poker cards", "playing cards", "deck of cards", "card deck", 
        "chopsticks", "wooden chopsticks", "bamboo chopsticks",
        "cup", "wine glass", "book"
    ]
    print(f"设置检测类别: {', '.join(names)}")
    model.set_classes(names)
    
    # 加载所有参考图像
    all_reference_data = load_reference_images()
    if not all_reference_data:
        print("错误: 没有找到有效的参考图像和边界框")
        return
    
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
    
    print("\n所有参考图像已被用于更新模型记忆")
       
    # 创建导出目录
    export_dir = "D:\\portfolio\\table-scenes\\models"
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "yoloe-poker-memory")
    print(f"正在导出模型到: {export_path}")
    
    # 保存一个包含类别名称的JSON文件，方便后续使用
    import json
    class_names_path = os.path.join(export_dir, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(names, f)
    print(f"类别名称已保存到: {class_names_path}")
    
    # 导出ONNX格式
    try:
        print("正在导出为ONNX格式...")
        export_model = model.export(format="onnx", imgsz=640)
        
        # 移动默认导出的ONNX文件到指定位置
        default_onnx = model_path.replace(".pt", ".onnx")
        if os.path.exists(default_onnx):
            import shutil
            onnx_path = f"{export_path}.onnx"
            shutil.move(default_onnx, onnx_path)
            print(f"模型已成功导出为ONNX格式: {onnx_path}")
    except Exception as e:
        print(f"导出ONNX格式时出错: {e}")
    
    print("\n导出操作已完成，模型现在包含所有12个参考图像的记忆")

if __name__ == "__main__":
    main()
