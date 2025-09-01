# Table-Scenes

桌面场景识别系统。使用 Ultralytics YOLOE 为检测器，能够识别四类场景：`work / dining / entertaining / relax`，并通过语音提示场景变化。

## 系统特点

- 基于物体识别的场景推断
- 滑动窗口聚合检测结果
- 场景变化语音提示
- 实时可视化

## 快速开始

### 环境配置

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 下载必要模型文件

由于模型文件较大，未包含在代码仓库中，请从以下地址下载：

1. 主要检测模型：[yoloe-11l-seg.pt](https://链接到您存放模型的地址)
   - 将下载的文件放在项目根目录下

2. 扑克牌记忆增强模型（可选）：[yoloe-11l-poker-memory.pt](https://链接到您存放该模型的地址)
   - 用于增强对扑克牌场景的识别
   - 将下载的文件放在项目根目录下

### 运行程序

```bash
# 增强版本（带语音提示）
python src/main_enhanced.py

# 基础版本
python src/main.py
```
