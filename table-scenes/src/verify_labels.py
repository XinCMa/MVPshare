"""
验证图像和标注文件的匹配
"""

import os
import glob

# 图像和标注文件目录
train_images_dir = r"D:\portfolio\table-scenes\data\yolo\images\train"
train_labels_dir = r"D:\portfolio\table-scenes\data\yolo\labels\train"
val_images_dir = r"D:\portfolio\table-scenes\data\yolo\images\val"
val_labels_dir = r"D:\portfolio\table-scenes\data\yolo\labels\val"

def check_matching_files(images_dir, labels_dir, set_name):
    if not os.path.exists(images_dir):
        print(f"{set_name}图像目录不存在: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"{set_name}标注目录不存在: {labels_dir}")
        return
    
    # 获取所有图像文件
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    
    # 获取所有标注文件
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    # 检查每个图像是否有对应的标注
    images_without_labels = []
    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(labels_dir, base_name + ".txt")
        if not os.path.exists(label_file):
            images_without_labels.append(os.path.basename(image_file))
    
    # 检查每个标注是否有对应的图像
    labels_without_images = []
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_found = False
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_file = os.path.join(images_dir, base_name + ext)
            if os.path.exists(image_file):
                image_found = True
                break
        if not image_found:
            labels_without_images.append(os.path.basename(label_file))
    
    # 输出结果
    print(f"=== {set_name}集验证结果 ===")
    print(f"图像文件总数: {len(image_files)}")
    print(f"标注文件总数: {len(label_files)}")
    
    if images_without_labels:
        print(f"警告: 发现 {len(images_without_labels)} 个图像没有对应的标注文件:")
        for img in images_without_labels[:5]:  # 只显示前5个
            print(f"  - {img}")
        if len(images_without_labels) > 5:
            print(f"  ... 以及其他 {len(images_without_labels) - 5} 个文件")
    else:
        print("所有图像都有对应的标注文件")
    
    if labels_without_images:
        print(f"警告: 发现 {len(labels_without_images)} 个标注文件没有对应的图像:")
        for lbl in labels_without_images[:5]:
            print(f"  - {lbl}")
        if len(labels_without_images) > 5:
            print(f"  ... 以及其他 {len(labels_without_images) - 5} 个文件")
    else:
        print("所有标注文件都有对应的图像")

# 检查训练集和验证集
check_matching_files(train_images_dir, train_labels_dir, "训练")
check_matching_files(val_images_dir, val_labels_dir, "验证")
