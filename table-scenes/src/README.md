# YOLOE 开放词汇表模型检测脚本

本项目包含一系列使用YOLOE开放词汇表模型的实时检测脚本，专注于提高扑克牌和筷子等特定物体的检测能力。

## 主要脚本

### 1. 纯文本提示检测 (`yoloe_text_only_detection.py`)

该脚本**仅使用丰富的文本描述集合**来增强模型对特定物体（如扑克牌、筷子）的检测能力，不使用任何视觉提示。

特点：
- 使用多种描述变体提高识别率
- 保留模型对标准物体的检测能力
- 实时摄像头检测，显示FPS
- 支持保存检测结果图像

使用方法：
```
python src/yoloe_text_only_detection.py
```

### 2. 增强型文本提示检测 (`yoloe_enhanced_text_prompts.py`)

这个脚本是文本提示检测的改进版本，移除了视觉提示功能，更加精简和专注。

### 3. 视觉提示检测 (`yoloe_visual_prompt_realtime.py`)

这个脚本使用参考图像作为视觉提示来增强检测能力，结合了文本和视觉信息。

## 关键功能

### 丰富的文本描述

每个脚本都包含针对扑克牌和筷子的多种文本描述，以提高模型识别特定物体的能力：

```python
# 为筷子创建更丰富的描述集合
chopsticks_descriptions = [
    "chopsticks", 
    "wooden chopsticks", 
    "bamboo chopsticks",
    # ...更多描述
]

# 为扑克牌创建更丰富的描述集合
poker_descriptions = [
    "poker", 
    "playing cards",
    "deck of cards",
    # ...更多描述
]
```

### 使用技巧

1. 按 `q` 键退出检测
2. 按 `s` 键保存当前检测帧
3. 脚本会自动尝试使用外置摄像头，如果失败则使用内置摄像头

## 依赖

- Ultralytics YOLOE
- CLIP (需要将CLIP-main库添加到路径中)
- OpenCV
- NumPy

## 模型文件

脚本需要以下模型文件：
- `yoloe-11l-seg.pt` - 基础分割模型
