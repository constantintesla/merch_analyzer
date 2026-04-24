"""Симуляция compliance check с LM-именами."""
import json, sys
from pathlib import Path
from PIL import Image, ImageOps
from collections import Counter, defaultdict
sys.path.insert(0, '.')

from app.step3_compliance import (
    parse_sku_catalog, build_reference_embeddings, match_sku_for_crop,
    infer_similar_sku_groups, parse_reference_planogram, build_observed_planogram,
    compare_planograms_step3
)
from app.item_validation import normalize_name

# ─── Каталог ─────────────────────────────────────────────────────
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

ref_plano_payload = {"slots": [
    {"shelf_id": s["shelf_id"], "slot_index": s["slot_index"],
     "expected_sku_id": s["sku_id"], "expected_facings": s.get("expected_facings", 1)}
    for s in slots
]}
reference_slots = parse_reference_planogram(ref_plano_payload)
BASE_DIR = Path('.')
ref_embeddings = build_reference_embeddings(catalog, base_dir=BASE_DIR)
catalog_for_matching = [item for item in catalog if item.sku_id in ref_embeddings]
similar_groups = infer_similar_sku_groups(catalog)

# ─── LM-имена из шага 2 ──────────────────────────────────────────
target_ref = "20260423_181949_image-08-04-26-11-48"
lm_hint_by_index = {}
lm_root = Path('data/sku_results/lm_recognition')
for lm_run in sorted(lm_root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
    rf = lm_run / 'result.json'
    if not rf.is_file():
        continue
    lm_data = json.loads(rf.read_text(encoding='utf-8'))
    lm_ref = str(lm_data.get('reference_result_dir', '')).replace('\\', '/')
    if target_ref not in lm_ref:
        continue
    for pp in lm_data.get('per_position', []):
        idx = int(pp.get('index', 0) or 0)
        if idx <= 0:
            continue
        lm_field = pp.get('lm', {})
        if not isinstance(lm_field, dict):
            continue
        name = str(lm_field.get('item_name', '')).strip()
        status = str(lm_field.get('status', '')).lower()
        if name and name.lower() not in ('unknown', 'неизвестный бренд', '') and status == 'ok':
            lm_hint_by_index[idx] = name
    print(f"LM names loaded from: {lm_run.name}")
    print(f"LM names count: {len(lm_hint_by_index)}")
    break

# ─── Reference positions ─────────────────────────────────────────
ref_dir = Path('data/sku_results/reference/20260423_181949_image-08-04-26-11-48')
result = json.loads((ref_dir / 'result.json').read_text(encoding='utf-8'))
positions = result['reference_positions']['positions']

with Image.open(ref_dir / 'input.jpg') as src:
    full_img = ImageOps.exif_transpose(src).convert('RGB')

# ─── Матчинг с LM-именами ────────────────────────────────────────
confidence_threshold = 0.62
lm_name_weight = 0.75  # высокий вес для LM-имён

observed_positions = []
lm_used_count = 0

for pos in positions:
    bbox = pos.get('bbox', {})
    x1, y1, x2, y2 = int(bbox.get('x1',0)), int(bbox.get('y1',0)), int(bbox.get('x2',0)), int(bbox.get('y2',0))
    crop = full_img.crop((x1, y1, x2, y2))
    shelf_id = int(pos.get('shelf_id', 0) or 0)
    pos_in_shelf = int(pos.get('position_in_shelf', 0) or 0)
    idx = int(pos.get('index', 0) or 0)

    llm_hint = lm_hint_by_index.get(idx, '')
    if llm_hint:
        lm_used_count += 1

    # Если нет LM-хинта, используем базовый вес (иначе визуал упадёт ниже порога)
    pos_lm_weight = lm_name_weight if llm_hint else 0.12
    # Используем полный каталог (включая SKU без ref-изображений: Буратино, Вкус Детства, ...)
    m = match_sku_for_crop(
        crop, catalog, ref_embeddings,
        confidence_threshold=confidence_threshold, similar_groups=similar_groups,
        llm_name_hint=llm_hint, llm_name_weight=pos_lm_weight
    )
    observed_positions.append({
        "index": idx,
        "shelf_id": shelf_id,
        "position_in_shelf": pos_in_shelf,
        "bbox": bbox,
        "predicted_sku_id": m["predicted_sku_id"],
        "predicted_name": m["predicted_name"],
        "confidence": m["confidence"],
        "visual_score": m["visual_score"],
        "observed_facings": 1,
        "lm_hint_used": llm_hint,
    })

unknown_count = sum(1 for p in observed_positions if p["predicted_sku_id"] == "unknown")
print(f"\nAccepted: {len(observed_positions)-unknown_count}/{len(observed_positions)}")
print(f"Unknown: {unknown_count}")
print(f"LM hints used: {lm_used_count}/{len(observed_positions)}")

# Показываем предсказания per shelf
print("\n=== Predicted names per shelf ===")
by_shelf = defaultdict(list)
for p in observed_positions:
    by_shelf[p['shelf_id']].append(p)
for sh in sorted(by_shelf.keys()):
    names = Counter(p['predicted_name'][:25] for p in by_shelf[sh])
    print(f"  Shelf {sh}: {len(by_shelf[sh])} positions")
    for n, c in names.most_common(5):
        print(f"    {n}: {c}")

# ─── Compliance check ─────────────────────────────────────────────
observed_runs = build_observed_planogram(observed_positions)
by_slot_run = {(int(r["shelf_id"]), int(r["slot_index"])): r for r in observed_runs}
for p in observed_positions:
    k = (p.get("shelf_id", 0), p.get("position_in_shelf", 0))
    p["observed_facings"] = int(by_slot_run.get(k, {}).get("observed_facings", 1))

metrics = compare_planograms_step3(
    reference_slots=reference_slots,
    observed_positions=observed_positions,
    presence_weight=0.4,
    position_weight=0.35,
    facings_weight=0.25,
    catalog=catalog,
    matching_level="brand_level",
    foreign_sku_policy="hard_fail",
)

score = metrics["compliance_score"]
print(f"\n=== COMPLIANCE SCORE: {score:.2f}/100 ===")
print(f"  Presence: {metrics['presence_ratio']:.3f}")
print(f"  Position: {metrics['position_ratio']:.3f}")
print(f"  Facings: {metrics['facings_ratio']:.3f}")

dev_types = Counter(d["type"] for d in metrics["deviations"])
print(f"  Deviations: {dict(dev_types)}")

print("\n=== Alignment debug by shelf ===")
for sh, info in sorted(metrics.get("alignment_debug_by_shelf", {}).items()):
    shift = info.get('selected_shift', 0)
    mc = info.get('match_count', 0)
    mm = info.get('mismatch_count', 0)
    al = info.get('aligned_count', 0)
    print(f"  Shelf {sh}: shift={shift:+d} matches={mc} mismatches={mm} aligned={al}")

print("\n=== Expected vs Observed (per shelf, first 8 slots, shift applied) ===")
exp_by_shelf = defaultdict(list)
for s in slots:
    exp_by_shelf[int(s['shelf_id'])].append(s)

sku_to_name = {item.sku_id: item.canonical_name for item in catalog}
shifts = metrics.get("slot_alignment_shift_by_shelf", {})
for sh in sorted(exp_by_shelf.keys()):
    shift = int(shifts.get(str(sh), 0))
    exp_slots = sorted(exp_by_shelf[sh], key=lambda x: int(x['slot_index']))
    obs_on_shelf = sorted([p for p in observed_positions if p['shelf_id'] == sh], key=lambda p: p['position_in_shelf'])
    obs_index = {p['position_in_shelf']: p for p in obs_on_shelf}
    
    print(f"\n  Shelf {sh} (shift={shift:+d}):")
    for s in exp_slots[:8]:
        exp_idx = int(s['slot_index'])
        obs_pos = exp_idx + shift
        obs = obs_index.get(obs_pos)
        exp_name = s.get('item_name', '?')[:22]
        if obs:
            pred_name = obs.get('predicted_name', '?')[:22]
            lm_h = obs.get('lm_hint_used', '')[:20]
            match = 'OK' if normalize_name(exp_name) == normalize_name(pred_name) else '--'
            print(f"    slot{exp_idx:2d}->obs{obs_pos:2d}: exp={exp_name:<22} obs={pred_name:<22} lm={lm_h:<20} {match}")
        else:
            print(f"    slot{exp_idx:2d}->obs{obs_pos:2d}: exp={exp_name:<22} MISSING")
