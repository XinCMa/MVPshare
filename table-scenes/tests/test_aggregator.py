# 测试聚合器
from src.aggregator import SlidingCounter


def test_sliding_counter_basic():
	win = SlidingCounter(seconds=2, fps=5)
	for _ in range(5):
		win.update_and_sum(['cup'])
	counts = win.update_and_sum(['cup'])
	assert counts['cup'] >= 6