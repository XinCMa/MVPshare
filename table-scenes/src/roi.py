# ROI 区域相关
import cv2, numpy as np


def load_roi(cfg_roi):
    if not cfg_roi.get('enabled', False):
        return None
    poly = np.array(cfg_roi['polygon'], dtype=np.int32)
    def inside(xyxy):
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cv2.pointPolygonTest(poly, (cx, cy), False) >= 0
    return inside


def draw_roi(frame, cfg_roi):
    if not cfg_roi.get('enabled', False):
        return
    poly = np.array(cfg_roi['polygon'], dtype=np.int32)
    cv2.polylines(frame, [poly], isClosed=True, color=(0,255,255), thickness=2)