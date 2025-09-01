# Ultralytics YOLOE 检测器封装
from ultralytics import YOLOE
import os
import cv2
import sys

# 添加CLIP库路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)


class YoloDetector:
	def __init__(self, model_cfg, classes):
		print("model_cfg:", model_cfg)  # 调试用
		
		# 检查模型文件是否存在
		model_path = model_cfg["weights"]
		if not os.path.exists(model_path):
			print(f"错误: 模型文件 {model_path} 不存在!")
			raise FileNotFoundError(f"找不到 YOLOE 模型文件: {model_path}")
		
		self.model = YOLOE(model_path)
		self.imgsz = model_cfg.get("imgsz", 640)
		self.conf = model_cfg.get("conf", 0.4)  # 使用 YOLOE 默认置信度阈值
		self.iou = model_cfg.get("iou", 0.5)
		
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
		
		# 构建丰富的类别描述列表
		all_classes = standard_objects.copy()
		all_classes.extend(poker_descriptions)
		all_classes.extend(chopsticks_descriptions)
		print(f"构建了 {len(all_classes)} 个类别描述")
		
		# 按照官方用法设置类别和文本嵌入
		print("设置增强的类别描述...")
		self.model.set_classes(all_classes)
		print("类别描述设置完成")
		
		# 为了兼容性，仍然处理传入的类别
		if isinstance(classes, dict):
			class_names = list(classes.values())
		else:
			class_names = classes
		print(f"可识别的类别: {class_names[:10]}...")
		self.classes = set(class_names)

	def infer(self, frame):
		# 图像预处理
		# 1. 调整亮度对比度
		alpha = 1.2  # 对比度增强因子
		beta = 10    # 亮度增强因子
		enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
		
		# 使用 YOLOE 模型进行预测
		results = self.model.predict(enhanced_frame, conf=self.conf, iou=self.iou, verbose=False)
		res = results[0]
		out = []
		
		# 调试：查看所有检测结果
		all_detections = []
		for i, b in enumerate(res.boxes):
			cls_id = int(b.cls)
			name = res.names[cls_id]
			confidence = float(b.conf)
			all_detections.append(f"{name} ({confidence:.2f})")
		
		if all_detections:
			print(f"检测到: {', '.join(all_detections)}")
		else:
			print("未检测到任何物体")
		
		# 处理检测结果，转换为标准格式
		for i, b in enumerate(res.boxes):
			name = res.names[int(b.cls)]
			
			# 添加所有检测到的物体（它们已经是我们请求检测的类别）
			out.append({
				"name": name, 
				"xyxy": b.xyxy[0].tolist(), 
				"conf": float(b.conf)
			})
		
		return out