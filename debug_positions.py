"""Диагностика распределения позиций по полкам."""
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from app.item_validation import normalize_name
from collections import defaultdict

# Загружаем reference result
ref_dir = Path('data/sku_results/reference/20260423_181949_image-08-04-26-11-48')
result = json.loads((ref_dir / 'result.json').read_text(encoding='utf-8'))
positions = result['reference_positions']['positions']

# Planogram
plano_path = Path('data/planogram_editor/3eec33ef832542d38e00cba99aecd5d7.json')
plano = json.loads(plano_path.read_text(encoding='utf-8'))
slots = plano.get('slots', [])

# Observed: shelf → sorted list of position_in_shelf
obs_by_shelf = defaultdict(list)
for p in positions:
    obs_by_shelf[int(p.get('shelf_id', 0))].append(int(p.get('position_in_shelf', 0)))
for s in obs_by_shelf:
    obs_by_shelf[s].sort()

# Expected: shelf → sorted list of slot_index
exp_by_shelf = defaultdict(list)
for s in slots:
    exp_by_shelf[int(s.get('shelf_id', 0))].append(int(s.get('slot_index', 0)))
for s in exp_by_shelf:
    exp_by_shelf[s].sort()

print("=== Position ranges per shelf ===")
print(f"{'Shelf':>6} | {'Observed (count, min-max)':>30} | {'Expected (count, min-max)':>30}")
print("-" * 75)
all_shelves = sorted(set(list(obs_by_shelf.keys()) + list(exp_by_shelf.keys())))
for sh in all_shelves:
    obs = obs_by_shelf.get(sh, [])
    exp = exp_by_shelf.get(sh, [])
    obs_str = f"{len(obs)} [{min(obs)}-{max(obs)}]" if obs else "none"
    exp_str = f"{len(exp)} [{min(exp)}-{max(exp)}]" if exp else "none"
    print(f"  {sh:>4} | {obs_str:>30} | {exp_str:>30}")
    if obs and exp:
        # Покажем реальное перекрытие
        obs_set = set(obs)
        exp_set = set(exp)
        intersect = obs_set & exp_set
        needed_shift = [o - e for e in exp for o in obs_set if abs(o - e) == 1][:3]
        print(f"       | Direct overlap (shift=0): {len(intersect)} positions")

print()
print("=== Detailed observed positions by shelf ===")
for sh in sorted(obs_by_shelf.keys()):
    print(f"  Shelf {sh}: {obs_by_shelf[sh]}")

print()
print("=== Detailed expected slots by shelf ===")
for sh in sorted(exp_by_shelf.keys()):
    print(f"  Shelf {sh}: {exp_by_shelf[sh]}")

# Оценим наилучший сдвиг per shelf (max overlap)
print()
print("=== Best shift analysis per shelf ===")
for sh in all_shelves:
    obs_list = obs_by_shelf.get(sh, [])
    exp_list = exp_by_shelf.get(sh, [])
    if not obs_list or not exp_list:
        continue
    obs_set = set(obs_list)
    min_obs, max_obs = min(obs_list), max(obs_list)
    min_exp, max_exp = min(exp_list), max(exp_list)
    
    best_shift = 0
    best_overlap = 0
    shift_range = range(min_obs - max_exp, max_obs - min_exp + 1)
    shift_results = []
    for shift in shift_range:
        overlap = sum(1 for e in exp_list if (e + shift) in obs_set)
        shift_results.append((shift, overlap))
        if overlap > best_overlap:
            best_overlap = overlap
            best_shift = shift
    
    top3 = sorted(shift_results, key=lambda x: -x[1])[:3]
    print(f"  Shelf {sh}: best_shift={best_shift} overlap={best_overlap}/{len(exp_list)}")
    print(f"    Top shifts: {top3}")
