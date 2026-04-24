"""Диагностика выравнивания планограммы."""
import json, sys
from pathlib import Path
from PIL import Image, ImageOps
from collections import defaultdict, Counter
sys.path.insert(0, '.')

from app.step3_compliance import (
    parse_sku_catalog, build_reference_embeddings, match_sku_for_crop,
    infer_similar_sku_groups, parse_reference_planogram
)
from app.item_validation import normalize_name

# Каталог
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
catalog = parse_sku_catalog(sku_catalog)
BASE_DIR = Path('.')
ref_embeddings = build_reference_embeddings(catalog, base_dir=BASE_DIR)
catalog_for_matching = [item for item in catalog if item.sku_id in ref_embeddings]
similar_groups = infer_similar_sku_groups(catalog)

sku_to_name = {item.sku_id: normalize_name(item.canonical_name) for item in catalog}

# ─── Что планограмма ожидает per shelf ───────────────────────────
print("=== Planogram expected per shelf ===")
exp_by_shelf = defaultdict(list)
for s in slots:
    exp_by_shelf[int(s['shelf_id'])].append(s)
for sh_id in sorted(exp_by_shelf.keys()):
    sh_slots = sorted(exp_by_shelf[sh_id], key=lambda x: int(x['slot_index']))
    name_counter = Counter(s.get('item_name', '') for s in sh_slots)
    print(f"\nShelf {sh_id}: {len(sh_slots)} slots")
    for name, cnt in name_counter.most_common():
        print(f"  {name}: {cnt}")

# ─── Матчинг ─────────────────────────────────────────────────────
ref_dir = Path('data/sku_results/reference/20260423_181949_image-08-04-26-11-48')
result = json.loads((ref_dir / 'result.json').read_text(encoding='utf-8'))
positions = result['reference_positions']['positions']

with Image.open(ref_dir / 'input.jpg') as src:
    full_img = ImageOps.exif_transpose(src).convert('RGB')

# Матчим позиции
obs_positions = []
for pos in positions:
    bbox = pos.get('bbox', {})
    x1, y1, x2, y2 = int(bbox.get('x1',0)), int(bbox.get('y1',0)), int(bbox.get('x2',0)), int(bbox.get('y2',0))
    crop = full_img.crop((x1, y1, x2, y2))
    shelf_id = int(pos.get('shelf_id', 0) or 0)
    pos_in_shelf = int(pos.get('position_in_shelf', 0) or 0)

    m = match_sku_for_crop(
        crop, catalog_for_matching, ref_embeddings,
        confidence_threshold=0.62, similar_groups=similar_groups,
        llm_name_hint='', llm_name_weight=0.12
    )
    obs_positions.append({
        "shelf_id": shelf_id,
        "position_in_shelf": pos_in_shelf,
        "predicted_sku_id": m["predicted_sku_id"],
        "predicted_name": m["predicted_name"],
        "confidence": m["confidence"],
        "visual_score": m["visual_score"],
    })

# ─── Детальный анализ выравнивания per shelf ──────────────────────
print("\n\n=== Detailed alignment analysis per shelf ===")

for sh_id in sorted(exp_by_shelf.keys()):
    exp_slots = sorted(exp_by_shelf[sh_id], key=lambda x: int(x['slot_index']))
    obs_on_shelf = [p for p in obs_positions if p['shelf_id'] == sh_id]
    obs_on_shelf.sort(key=lambda p: p['position_in_shelf'])
    
    obs_index = {p['position_in_shelf']: p for p in obs_on_shelf}
    
    print(f"\n--- Shelf {sh_id} ---")
    print(f"Expected slots: {[s['slot_index'] for s in exp_slots]}")
    print(f"Observed positions: {[p['position_in_shelf'] for p in obs_on_shelf]}")
    
    # Simulate alignment for different shifts
    best_shift = 0
    best_key = None
    print("Shift → (matches, mismatches, aligned):")
    
    min_obs = min(obs_index.keys()) if obs_index else 1
    max_obs = max(obs_index.keys()) if obs_index else 1
    min_exp = min(int(s['slot_index']) for s in exp_slots)
    max_exp = max(int(s['slot_index']) for s in exp_slots)
    
    shift_results = []
    for shift in range(min_obs - max_exp, max_obs - min_exp + 1):
        match_count = 0
        mismatch_count = 0
        aligned_count = 0
        for s in exp_slots:
            exp_idx = int(s['slot_index'])
            obs = obs_index.get(exp_idx + shift)
            if not obs:
                continue
            aligned_count += 1
            pred = obs.get('predicted_sku_id', 'unknown')
            if pred == 'unknown':
                continue
            exp_name = normalize_name(s.get('item_name', ''))
            pred_name = normalize_name(obs.get('predicted_name', ''))
            if exp_name and pred_name and exp_name == pred_name:
                match_count += 1
            else:
                mismatch_count += 1
        shift_results.append((shift, match_count, mismatch_count, aligned_count))
    
    # Sort by old key (match first) and new key (coverage first)
    old_key_best = max(shift_results, key=lambda x: (x[1], -x[2], x[3], -abs(x[0])))
    new_key_best = max(shift_results, key=lambda x: (x[3], x[1], -x[2], -abs(x[0])))
    
    print(f"  Old key best (match first): shift={old_key_best[0]} m={old_key_best[1]} mm={old_key_best[2]} al={old_key_best[3]}")
    print(f"  New key best (coverage first): shift={new_key_best[0]} m={new_key_best[1]} mm={new_key_best[2]} al={new_key_best[3]}")
    
    # Show shift=0 details
    s0 = next((x for x in shift_results if x[0] == 0), None)
    if s0:
        print(f"  At shift=0: m={s0[1]} mm={s0[2]} al={s0[3]}")
    
    # Show expected vs observed at shift=0
    print(f"\n  Expected vs Observed at shift=0:")
    for s in exp_slots[:8]:  # First 8 to keep brief
        exp_idx = int(s['slot_index'])
        obs = obs_index.get(exp_idx)
        exp_name = s.get('item_name', '?')
        if obs:
            pred_name = obs.get('predicted_name', '?')
            match = '✓' if normalize_name(exp_name) == normalize_name(pred_name) else '✗'
            print(f"    slot {exp_idx}: exp={exp_name[:20]} obs={pred_name[:20]} {match}")
        else:
            print(f"    slot {exp_idx}: exp={exp_name[:20]} obs=MISSING")
