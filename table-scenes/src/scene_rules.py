# 场景判定规则 - 基于当前帧的检测结果计算场景分数
def decide_scene(c):
	# 扑克牌相关的所有可能文本描述
	poker_terms = [
		"poker", "playing cards", "deck of cards", "poker cards", 
		"playing card deck", "card game", "card deck", "cards for gambling",
		"casino cards", "rectangular paper cards with numbers and suits",
		"hearts spades clubs diamonds cards", "face cards",
		"poker game cards", "bridge cards", "standard 52-card deck",
		"playing card set", "gaming cards"
	]
	
	# 筷子相关的所有可能文本描述
	chopsticks_terms = [
		"chopsticks", "wooden chopsticks", "bamboo chopsticks", 
		"wooden eating utensils", "long thin wooden sticks for eating", 
		"asian eating utensils", "chinese chopsticks",
		"japanese chopsticks", "korean chopsticks",
		"black chopsticks", "pair of thin wooden sticks",
		"traditional asian eating tools", "wooden rods used for eating",
		"slender wooden eating implements", 
		"straight thin wooden sticks used in asian cuisine"
	]
	
	# 辅助函数，计算多个关键词的检测总数
	g = lambda *ks: sum(c.get(k, 0) for k in ks)
	
	# 辅助函数，检查是否存在至少一个与给定术语列表匹配的检测
	def check_terms(terms_list):
		return sum(c.get(term, 0) for term in terms_list)
	
	# 计算每个场景的分数
	scores = {}
	
	# WORK - 工作场景
	work_items = ['laptop', 'book', 'mouse', 'keyboard']
	scores['work'] = g(*work_items) * 10  # 给予工作场景项目较高权重
	
	# DINING - 用餐场景
	dining_items = ['bowl', 'cup', 'wine glass', 'spoon', 'fork', 'knife', 'dining table']
	scores['dining'] = g(*dining_items) * 5  # 基础餐具分数
	
	# 为筷子增加额外分数
	chopsticks_score = check_terms(chopsticks_terms) * 10  # 筷子是强烈的餐饮场景指示
	scores['dining'] += chopsticks_score
	
	# ENTERTAINING - 娱乐场景
	entertainment_items = ['remote', 'cell phone', 'chess board', 'board game pieces']
	scores['entertaining'] = g(*entertainment_items) * 5  # 基础娱乐项目分数
	
	# 为扑克牌增加额外分数
	poker_score = check_terms(poker_terms) * 15  # 扑克牌是强烈的娱乐场景指示
	scores['entertaining'] += poker_score
	
	# RELAX - 休闲场景
	scores['relax'] = c.get('person', 0) * 3  # 人的存在是休闲场景的基础
	
	# NOTHING - 兜底场景
	scores['nothing'] = 1 if sum(scores.values()) == 0 else 0
	
	# 检查是否有任何显著特征明确指向特定场景
	if poker_score > 0 and poker_score >= max(scores.values()) * 0.8:
		return 'entertaining'  # 扑克牌是娱乐场景的明确标志
	
	if chopsticks_score > 0 and scores['dining'] >= max(scores.values()) * 0.8:
		return 'dining'  # 筷子配合其他餐具是用餐场景的明确标志
	
	if g(*work_items) >= 1:  # 多个工作物品同时出现
		return 'work'
	
	# 在没有明确特征的情况下，选择得分最高的场景
	max_score = max(scores.values())
	if max_score > 0:
		for scene, score in scores.items():
			if score == max_score:
				return scene
	
	# 默认情况
	return 'nothing'