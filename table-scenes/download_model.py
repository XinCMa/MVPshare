# 下载 YOLOE 模型脚本
from ultralytics import YOLOE

# 这将下载模型（如果尚未下载）
model = YOLOE('yoloe-11l-seg.pt')
print(f"模型已下载: {model.model.yaml_file}")
print(f"模型路径: {model.ckpt_path}")
