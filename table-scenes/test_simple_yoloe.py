# YOLOE 简单测试
from ultralytics import YOLOE
import cv2

# 加载模型
model = YOLOE('yoloe-11l-seg.pt')

# 打开摄像头
cap = cv2.VideoCapture(1)  # 外接摄像头，使用0表示内置摄像头
if not cap.isOpened():
    print("无法打开摄像头，尝试使用内置摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开任何摄像头!")
        exit(1)

print("摄像头已打开，按 'q' 退出")

# 循环获取视频帧并进行预测
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取视频帧")
            break
            
        # 简单使用默认参数进行预测
        results = model.predict(frame, conf=0.3, verbose=True)
        
        # 显示结果
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOE 检测", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")
