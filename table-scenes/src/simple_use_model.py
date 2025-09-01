import os
import sys
import numpy as np
import time
import cv2
from ultralytics import YOLOE

# 添加CLIP-main目录到Python路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)

# 定义类别名称映射 - 这必须与导出模型时使用的类别名称顺序一致
names = [
    "poker", "poker cards", "playing cards", "deck of cards", "card deck", 
    "chopsticks", "wooden chopsticks", "bamboo chopsticks",
    "cup", "wine glass", "book"
]
print(f"准备类别名称映射: {', '.join(names)}")

# 加载ONNX模型
model_path = "D:\\portfolio\\table-scenes\\models\\yoloe-poker-memory.onnx"
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    exit(1)
    
print(f"加载ONNX模型: {model_path}")
model = YOLOE(model_path)

# 手动设置类别名称映射
model.names = {i: name for i, name in enumerate(names)}

def main():
    # 打开摄像头
    print("尝试打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开默认摄像头，尝试外接摄像头...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("无法打开任何摄像头，退出程序")
            exit(1)
    
    print("摄像头已打开，按'q'键退出")
    
    # FPS计算变量
    prev_time = 0
    fps = 0
    
    try:
        while True:
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break
                
            # 计算FPS
            current_time = time.time()
            if prev_time > 0:
                fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # 使用模型预测
            try:
                results = model.predict(frame, conf=0.25, verbose=False)
                
                # 绘制结果
                annotated_frame = results[0].plot()
                
                # 显示FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # 显示结果
                cv2.imshow("YOLOE 扑克牌检测", annotated_frame)
            except Exception as e:
                print(f"预测或显示过程出错: {e}")
                # 显示原始帧
                cv2.putText(
                    frame,
                    f"预测错误: {str(e)[:50]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("YOLOE 扑克牌检测", frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()