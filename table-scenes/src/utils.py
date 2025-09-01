"""
场景优先级: WORK > DINING > ENTERTAINING > RELAX > NOLIGHT
判定规则：
1. WORK: laptop, mouse, keyboard, book, newspaper 中至少1个
2. DINING: (不在WORK) chopsticks, bowl, plate, spoon, fork 中至少2个
3. ENTERTAINING: (不在前两种) painter, paint, chess 中至少1个
4. RELAX: (不在前三种) cup, person 中至少1个
5. NOLIGHT: (画面静止且无人，且不属于前4种)
"""

WORK_ITEMS = {"laptop", "mouse", "keyboard", "book", "newspaper"}
DINING_ITEMS = {"chopsticks", "bowl", "plate", "spoon", "fork"}
ENTERTAIN_ITEMS = {"painter", "paint", "chess"}
RELAX_ITEMS = {"cup", "person"}

def decide_scene(counts, is_static=False):
    # 1. WORK
    if any(counts.get(item, 0) > 0 for item in WORK_ITEMS):
        return "WORK"
    # 2. DINING
    if sum(counts.get(item, 0) > 0 for item in DINING_ITEMS) >= 2:
        return "DINING"
    # 3. ENTERTAINING
    if any(counts.get(item, 0) > 0 for item in ENTERTAIN_ITEMS):
        return "ENTERTAINING"
    # 4. RELAX
    if any(counts.get(item, 0) > 0 for item in RELAX_ITEMS):
        return "RELAX"
    # 5. NOLIGHT
    if is_static and counts.get("person", 0) == 0:
        return "NOLIGHT"
    return "NOLIGHT"
