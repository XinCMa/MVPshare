#!/usr/bin/env python3
# 主程序入口 - 增强型文本提示检测版本
"""
Table Scenes - 桌面场景识别系统
使用增强型文本提示的YOLOE模型进行桌面物体检测和场景识别
"""

import os
import sys
import time
import pyttsx3  # 添加语音合成库
import threading  # 添加线程支持
import queue  # 添加队列支持，用于语音播报

# 添加CLIP库路径
clip_path = "D:\\portfolio\\table-scenes\\CLIP-main"
if os.path.exists(clip_path):
    sys.path.append(clip_path)
    print(f"已添加CLIP库路径: {clip_path}")
else:
    print(f"错误: CLIP库路径不存在: {clip_path}")
    exit(1)

from pipeline import run_pipeline

# 全局语音队列和线程控制变量
voice_queue = queue.Queue()
voice_thread_running = False
voice_thread = None
has_voice = False

# 语音播报线程函数
def voice_worker():
    try:
        # 在线程中创建一个全局的语音引擎实例
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        
        # 尝试设置中文声音（如果有的话）
        voices = engine.getProperty('voices')
        for voice in voices:
            if "chinese" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
                
        print("语音线程已启动，等待播报...")
        
        # 持续监听语音队列
        global voice_thread_running
        while voice_thread_running:
            try:
                # 非阻塞方式获取队列消息，超时1秒
                message = voice_queue.get(block=True, timeout=1)
                print(f"语音线程接收到消息: {message}")
                
                # 播放语音
                engine.say(message)
                engine.runAndWait()
                print(f"语音播报完成: {message}")
                
                # 标记任务完成
                voice_queue.task_done()
            except queue.Empty:
                # 队列超时，继续循环
                pass
            except Exception as e:
                print(f"语音播报出错: {e}")
                
        print("语音线程已退出")
    except Exception as e:
        print(f"语音线程初始化失败: {e}")

# 初始化语音线程
try:
    # 启动语音线程
    voice_thread_running = True
    voice_thread = threading.Thread(target=voice_worker, daemon=True)
    voice_thread.start()
    has_voice = True
    print("语音提示功能已初始化")
except Exception as e:
    print(f"初始化语音线程失败: {e}")
    print("将在无语音模式下运行")
    has_voice = False

# 场景变化回调函数
def on_scene_change(old_scene, new_scene):
    scene_names = {
        "work": "工作场景",
        "dining": "就餐场景",
        "entertaining": "娱乐场景",
        "relax": "休闲场景",
        "nothing": "未识别场景"
    }
    
    # 获取中文场景名称
    new_scene_cn = scene_names.get(new_scene, "未知场景")
    
    # 打印场景变化信息
    print(f"\n场景变化: {old_scene} -> {new_scene} ({new_scene_cn})")
    
    # 如果语音引擎可用，发出语音提示
    if has_voice:
        # 构建语音提示内容
        if old_scene == "":  # 初始场景
            message = f"已检测到{new_scene_cn}"
        else:
            message = f"场景已变化为{new_scene_cn}"
            
        # 将语音消息加入队列，由专用线程处理
        print(f"添加语音消息到队列: {message}")
        try:
            voice_queue.put(message)
            print("语音消息已加入队列")
        except Exception as e:
            print(f"添加语音消息到队列失败: {e}")

def main():
    print("\n=== Table Scenes MVP - 增强型文本提示检测版本 ===")
    
    # 检查必要的目录结构
    if not os.path.exists("config"):
        print("错误: 找不到配置目录")
        return
        
    # 检查配置文件
    config_path = "config/config.yaml"
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        return
    
    classes_path = "config/classes_coco.yaml"
    if not os.path.exists(classes_path):
        print(f"错误: 找不到类别文件 {classes_path}")
        return
        
    # 检查模型文件存在
    if not os.path.exists("yoloe-11l-seg.pt"):
        print("错误: 找不到YOLOE模型文件 yoloe-11l-seg.pt")
        return
        
    print("\n场景识别说明:")
    print("1. 系统使用滑动窗口统计一段时间内的检测结果")
    print("2. 当场景变化时，需要新场景稳定一段时间才会切换")
    print("3. 如果物品被遮挡或移出视野，系统会在滑动窗口结束后重新判断场景")
    print("4. 检测到扑克牌相关物品时会立即判定为娱乐场景")
    if has_voice:
        print("5. 场景变化时会有语音提示\n")
    else:
        print("\n")
    
    # 测试语音功能
    if has_voice:
        print("测试语音功能...")
        voice_queue.put("语音系统已就绪，开始检测")
        # 等待语音测试消息处理完
        time.sleep(1)
        
    # 运行检测和场景识别管道
    try:
        run_pipeline(config_path, classes_path, scene_change_callback=on_scene_change)
    except Exception as e:
        print(f"运行过程中出错: {e}")
    finally:
        # 确保退出时停止语音线程
        global voice_thread_running
        if has_voice and voice_thread_running:
            voice_thread_running = False
            if voice_thread is not None:
                print("等待语音线程结束...")
                voice_thread.join(timeout=2)  # 等待最多2秒
                print("语音线程已结束")
        
    print("程序已退出")

if __name__ == "__main__":
    main()
