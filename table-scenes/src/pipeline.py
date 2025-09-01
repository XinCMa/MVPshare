# 串联 detector 和 scene_rules
import cv2, time, yaml
from aggregator import SlidingCounter
from detector import YoloDetector
from scene_rules import decide_scene
from roi import load_roi, draw_roi
from viz import overlay_scene, overlay_counts, draw_detections


def run_pipeline(cfg_path="config/config.yaml", classes_path="config/classes_coco.yaml", scene_change_callback=None):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    classes = yaml.safe_load(open(classes_path, encoding="utf-8"))["names"]

    cap = cv2.VideoCapture(cfg["camera"]["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    cap.set(cv2.CAP_PROP_FPS, cfg["camera"]["fps"])

    det = YoloDetector(cfg["model"], classes)
    roi_fn = load_roi(cfg["roi"])
    win = SlidingCounter(seconds=cfg["window"]["seconds"], fps=cfg["camera"]["fps"])

    last_scene, stable_since = None, 0
    hysteresis_s = cfg["window"]["switch_hysteresis_s"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = det.infer(frame)  # [{'name','xyxy','conf'}...]
        if roi_fn:
            dets = [d for d in dets if roi_fn(d["xyxy"])]
        counts = win.update_and_sum([d["name"] for d in dets])
        
        # 打印每一帧中滑动窗口内的检测计数
        if any(counts.values()):
            print(f"当前检测计数: {dict(counts)}")
            
        scene = decide_scene(counts)
        print(f"当前场景判断: {scene}")
        
        # 从counts中提取关键物品信息
        has_poker = any(counts.get(term, 0) > 0 for term in [
            "poker", "playing cards", "deck of cards", "poker cards", 
            "card game", "card deck", "gaming cards"
        ])
        has_chopsticks = any(counts.get(term, 0) > 0 for term in [
            "chopsticks", "wooden chopsticks", "bamboo chopsticks"
        ])
        has_work_items = any(counts.get(term, 0) > 0 for term in [
            "laptop", "book", "mouse", "keyboard"
        ])
        has_dining_items = any(counts.get(term, 0) > 0 for term in [
            "bowl", "cup", "wine glass", "spoon", "fork", "knife", "dining table"
        ])
        
        # 检测当前画面中是否有任何物品
        current_has_items = sum(counts.values()) > 0

        # 检查场景变化
        if scene != last_scene:
            # 场景切换逻辑
            # 1. 如果当前没有任何物品，可能是因为物品被移出画面，我们保持最后的场景
            if not current_has_items and last_scene != 'nothing':
                print(f"当前画面无物品，保持上一场景: {last_scene}")
                scene = last_scene  # 保持上一个场景
            else:
                # 2. 对于有明确特征物品的场景变化，可以更快地切换
                force_change = False
                
                # 特征物品优先级：扑克牌 > 工作物品 > 餐具+筷子
                # 增加场景切换的敏感度，几乎所有场景变化都可以立即切换
                if has_poker:
                    force_change = True
                    reason = "扑克牌"
                elif has_work_items and scene == 'work':
                    force_change = True
                    reason = "工作物品"
                elif has_dining_items and scene == 'dining':
                    force_change = True
                    reason = "餐饮物品"
                # 对其他任何场景的变化也快速响应
                elif scene != 'nothing' and last_scene != scene:
                    force_change = True
                    reason = "场景改变"
                    reason = "餐具和筷子"
                
                # 处理场景切换
                if force_change:
                    print(f"强特征物品({reason})出现，立即切换场景: {last_scene} -> {scene}")
                    win.reset()  # 重置滑动窗口，立即开始新场景的检测
                    
                    # 调用场景变化回调函数
                    if scene_change_callback:
                        old_scene = last_scene if last_scene is not None else ""
                        scene_change_callback(old_scene, scene)
                        
                    last_scene = scene
                    stable_since = 0
                elif stable_since == 0:
                    # 开始计时，记录新场景开始稳定的时间点
                    stable_since = time.time()
                    print(f"场景可能变化: {last_scene} -> {scene}，开始确认...")
                elif time.time() - stable_since >= hysteresis_s:
                    # 新场景已经稳定足够长的时间，确认切换
                    print(f"场景稳定切换: {last_scene} -> {scene} | 物品: {dict(counts)}")
                    # 检查是否需要重置滑动窗口
                    buffer_fullness = win.get_buffer_fullness()
                    if buffer_fullness > 0.7:  # 如果缓冲区已填充超过70%
                        win.reset()  # 重置滑动窗口，避免历史检测结果影响新场景
                        print("重置检测历史，开始新场景统计")
                    
                    # 调用场景变化回调函数
                    if scene_change_callback:
                        old_scene = last_scene if last_scene is not None else ""
                        scene_change_callback(old_scene, scene)
                        
                    last_scene = scene
                    stable_since = 0
        else:
            # 当前判断与上次场景相同，但需要处理特殊情况
            
            # 特殊情况：当前场景物品消失但有其他物品出现
            if (last_scene == 'work' and not has_work_items and current_has_items) or \
               (last_scene == 'dining' and not has_dining_items and current_has_items) or \
               (last_scene == 'entertaining' and not has_poker and current_has_items):
                # 几乎立即响应场景变化
                if stable_since == 0:
                    stable_since = time.time()
                    print(f"特征物品已消失，准备切换场景...")
                elif time.time() - stable_since >= hysteresis_s * 0.5:  # 非常快的切换
                    print(f"工作物品持续消失，重置检测窗口")
                    win.reset()  # 重置检测窗口，使系统能更快响应新场景
                    stable_since = 0
            else:
                # 当前判断与上次场景相同，重置稳定时间
                stable_since = 0

        # 调试可视化
        if cfg.get("viz", {}).get("enabled", True):
            # 在画面中框出检测到的物体
            draw_detections(frame, dets)
            overlay_scene(frame, scene)
            overlay_counts(frame, counts, max_items=cfg.get("viz", {}).get("max_items", 8))
            draw_roi(frame, cfg["roi"])
            cv2.imshow("table-scenes-s1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("用户按下了q或ESC，程序退出")
                break

    cap.release()
    cv2.destroyAllWindows()