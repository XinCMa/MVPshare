"""
更新YOLO标注文件中的类别ID
将所有poker类别从ID 0更改为ID 80
"""

import os
import glob
import shutil

# 标注文件目录
train_labels_dir = r"D:\portfolio\table-scenes\data\yolo\labels\train"
val_labels_dir = r"D:\portfolio\table-scenes\data\yolo\labels\val"

# 创建备份目录
train_backup_dir = os.path.join(os.path.dirname(train_labels_dir), "train_labels_backup")
val_backup_dir = os.path.join(os.path.dirname(val_labels_dir), "val_labels_backup")

os.makedirs(train_backup_dir, exist_ok=True)
os.makedirs(val_backup_dir, exist_ok=True)

# 处理训练集标注
def update_labels_in_dir(labels_dir, backup_dir):
    # 找到所有txt文件
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    print(f"找到 {len(label_files)} 个标注文件在 {labels_dir}")
    
    updated_count = 0
    
    for label_file in label_files:
        # 备份原始文件
        backup_file = os.path.join(backup_dir, os.path.basename(label_file))
        shutil.copy2(label_file, backup_file)
        
        # 读取文件内容
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # 更新类别ID
        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == '0':  # 如果类别ID是0
                parts[0] = '80'  # 更改为80
                updated_lines.append(' '.join(parts) + '\n')
            else:
                updated_lines.append(line)
        
        # 写回文件
        with open(label_file, 'w') as f:
            f.writelines(updated_lines)
        
        updated_count += 1
        
    print(f"已更新 {updated_count} 个标注文件，类别ID从0更改为80")
    print(f"原始标注文件已备份到 {backup_dir}")

# 更新训练集和验证集
if os.path.exists(train_labels_dir):
    update_labels_in_dir(train_labels_dir, train_backup_dir)
else:
    print(f"训练集标注目录不存在: {train_labels_dir}")

if os.path.exists(val_labels_dir):
    update_labels_in_dir(val_labels_dir, val_backup_dir)
else:
    print(f"验证集标注目录不存在: {val_labels_dir}")

print("标注文件更新完成!")
