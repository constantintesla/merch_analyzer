"""Debug: какие позиции помечены overlay'ем и какого цвета."""
import json, urllib.request
from pathlib import Path

plano_path = Path('data/planogram_editor/3eec33ef832542d38e00cba99aecd5d7.json')
plano = json.loads(plano_path.read_text(encoding='utf-8'))
slots = plano.get('slots', [])

bySku = {}
for s in slots:
    sku = s.get('sku_id', '')
    name = s.get('item_name', sku)
    img = s.get('reference_image_path', '')
    if sku not in bySku:
        bySku[sku] = {'sku_id': sku, 'canonical_name': name, 'brand': '', 'aliases': [], 'reference_images': []}
    if img and img not in bySku[sku]['reference_images']:
        bySku[sku]['reference_images'].append(img)

sku_catalog = {'items': list(bySku.values())}
ref_plano = {'slots': [
    {'shelf_id': s['shelf_id'], 'slot_index': s['slot_index'],
     'expected_sku_id': s['sku_id'], 'expected_facings': s.get('expected_facings', 1)}
    for s in slots
]}

import os
reference_result_dir = os.getenv('REF_DIR', 'data/sku_results/reference/20260423_181949_image-08-04-26-11-48')

payload = {
    'reference_result_dir': reference_result_dir,
    'reference_planogram': ref_plano,
    'sku_catalog': sku_catalog,
    'options': {
        'matching_level': 'brand_level',
        'foreign_sku_policy': 'hard_fail',
        'confidence_threshold': 0.62,
        'llm_name_weight': 0.12,
        'presence_weight': 0.4,
        'position_weight': 0.35,
        'facings_weight': 0.25,
    }
}

data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request('http://localhost:8000/compliance/check', data=data, method='POST')
req.add_header('Content-Type', 'application/json')
with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())

positions = result.get('observed_positions', [])
deviations = result.get('deviations', [])

# Индекс отклонений по (shelf_id, aligned_observed_slot_index)
dev_by_aligned = {}
for d in deviations:
    key = (int(d.get('shelf_id', 0)), int(d.get('aligned_observed_slot_index', d.get('slot_index', 0))))
    dev_by_aligned[key] = d

# Построим full view положения 3-й полки
shelf3 = [p for p in positions if int(p.get('shelf_id', 0)) == 3]
shelf3.sort(key=lambda p: int(p.get('position_in_shelf', 0)))

print(f"\n=== REF: {reference_result_dir} ===")
print(f"=== Shelf 3 ({len(shelf3)} positions) ===")
print(f"{'idx':>4} {'pos':>4} {'predicted_name':<28} {'lm_hint':<28} {'status':<10}")
for p in shelf3:
    idx = int(p.get('index', 0))
    pos = int(p.get('position_in_shelf', 0))
    pred = str(p.get('predicted_name', ''))[:26]
    lm = str(p.get('lm_hint', ''))[:26]
    key = (3, pos)
    if key in dev_by_aligned:
        d = dev_by_aligned[key]
        status = d['type'][:10]
    else:
        status = "OK"
    print(f"{idx:>4} {pos:>4} {pred:<28} {lm:<28} {status:<10}")

print("\n=== Target indices ===")
for target_idx in (48, 43, 47):
    p = next((x for x in positions if int(x.get("index", 0)) == target_idx), None)
    if not p:
        print(f"idx={target_idx}: not found")
        continue
    shelf = int(p.get("shelf_id", 0))
    pos = int(p.get("position_in_shelf", 0))
    pred_sku = str(p.get("predicted_sku_id", ""))
    pred_name = str(p.get("predicted_name", ""))
    d = dev_by_aligned.get((shelf, pos))
    d_type = d.get("type") if isinstance(d, dict) else "OK"
    print(f"idx={target_idx} shelf={shelf} pos={pos} sku={pred_sku} name={pred_name} deviation={d_type}")

# Покажем все expected для полки 3
print(f"\n=== Expected на полке 3 ===")
sh3_expected = sorted([s for s in slots if s['shelf_id'] == 3], key=lambda s: s['slot_index'])
shift = result.get('metrics', {}).get('alignment_debug_by_shelf', {}).get('3', {}).get('selected_shift', 0)
print(f"Shift: {shift:+d}")
for s in sh3_expected:
    print(f"  slot {s['slot_index']:>2} +{shift}={s['slot_index']+shift:>2}: {s.get('item_name', '')[:40]}")
