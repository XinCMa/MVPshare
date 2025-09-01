# 结果聚合
from collections import Counter, deque


class SlidingCounter:
    def __init__(self, seconds=10, fps=20):
        self.maxlen = max(1, seconds*fps)
        self.buf = deque(maxlen=self.maxlen)
        
        # 定义文本变体映射，将多个描述映射到主要类别
        self.text_variant_mapping = {
            # 扑克牌相关的所有描述映射到"poker cards"
            "poker": "poker cards",
            "playing cards": "poker cards",
            "deck of cards": "poker cards", 
            "playing card deck": "poker cards",
            "card game": "poker cards",
            "card deck": "poker cards",
            "cards for gambling": "poker cards",
            "casino cards": "poker cards",
            "rectangular paper cards with numbers and suits": "poker cards",
            "hearts spades clubs diamonds cards": "poker cards",
            "face cards": "poker cards",
            "poker game cards": "poker cards",
            "bridge cards": "poker cards",
            "standard 52-card deck": "poker cards",
            "playing card set": "poker cards",
            "gaming cards": "poker cards",
            
            # 筷子相关的所有描述映射到"chopsticks"
            "wooden chopsticks": "chopsticks",
            "bamboo chopsticks": "chopsticks",
            "wooden eating utensils": "chopsticks",
            "long thin wooden sticks for eating": "chopsticks",
            "asian eating utensils": "chopsticks",
            "chinese chopsticks": "chopsticks",
            "japanese chopsticks": "chopsticks",
            "korean chopsticks": "chopsticks",
            "black chopsticks": "chopsticks",
            "pair of thin wooden sticks": "chopsticks",
            "traditional asian eating tools": "chopsticks",
            "wooden rods used for eating": "chopsticks",
            "slender wooden eating implements": "chopsticks",
            "straight thin wooden sticks used in asian cuisine": "chopsticks"
        }
    
    def normalize_detection_name(self, name):
        """将检测到的物体名称标准化为主要类别名称"""
        return self.text_variant_mapping.get(name, name)
    
    def reset(self):
        """重置滑动窗口，清空所有历史检测"""
        self.buf.clear()
        
    def get_buffer_fullness(self):
        """获取缓冲区填充程度 (0.0-1.0)"""
        return len(self.buf) / self.maxlen if self.maxlen > 0 else 0
    
    def update_and_sum(self, names):
        # 对所有名称进行标准化处理
        normalized_names = [self.normalize_detection_name(name) for name in names]
        
        c = Counter(normalized_names)
        self.buf.append(c)
        total = Counter()
        for x in self.buf:
            total.update(x)
        return total