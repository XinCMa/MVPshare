# 可视化工具
import cv2
import numpy as np


def overlay_scene(frame, scene: str):
	cv2.putText(frame, f"SCENE: {scene}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


def overlay_counts(frame, counts, max_items=8):
	y = 70
	for k, v in counts.most_common(max_items):
		cv2.putText(frame, f"{k}:{v}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		y += 24


def draw_detections(frame, detections):
	"""在画面中框出检测到的物体
	Args:
		frame: 视频帧
		detections: 检测结果列表，每个元素包含 'name', 'xyxy', 'conf' 等键
	"""
	colors = {
		'person': (0, 255, 0),    # 绿色
		'laptop': (255, 0, 0),    # 蓝色
		'keyboard': (255, 165, 0), # 橙色
		'mouse': (0, 255, 255),   # 黄色
		'cell phone': (255, 0, 255), # 紫色
		'book': (128, 0, 128),    # 深紫色
		'bottle': (0, 128, 255),  # 橙色
		'cup': (255, 128, 0),     # 橙红色
		'tv': (0, 0, 255),        # 红色
		'tie': (255, 255, 0),     # 青色
		'wine glass': (128, 0, 0), # 深蓝色
		'fork': (0, 128, 128),    # 棕色
		'knife': (128, 128, 0)    # 深绿色
	}
	
	for det in detections:
		# 获取物体类别和坐标
		name = det['name']
		box = det['xyxy']
		conf = det['conf']
		
		# 转换为整数坐标
		x1, y1, x2, y2 = map(int, box)
		
		# 确定颜色 (默认白色)
		color = colors.get(name, (255, 255, 255))
		
		# 绘制矩形框
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
		
		# 添加类别标签和置信度
		label = f"{name} {conf:.2f}"
		label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 5
		
		# 绘制文本背景
		cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
					  (x1 + label_size[0], label_y + 5), color, -1)
		
		# 绘制文本
		cv2.putText(frame, label, (x1, label_y), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)