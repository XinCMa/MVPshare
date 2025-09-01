# 测试场景规则
from collections import Counter
from src.scene_rules import decide_scene


def test_work():
	assert decide_scene(Counter({'person':1,'laptop':1})) == 'work'


def test_dining():
	assert decide_scene(Counter({'person':2,'bowl':2,'cup':1})) == 'dining'


def test_entertaining():
	assert decide_scene(Counter({'person':3,'cup':2})) == 'entertaining'


def test_relax():
	assert decide_scene(Counter({'person':1,'cup':1})) == 'relax'