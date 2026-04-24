"""Симуляция compliance check для диагностики итогового score."""
import json, sys
from pathlib import Path
from PIL import Image, ImageOps
sys.path.insert(0, '.')

from app.step3_compliance import (
    parse_sku_catalog, build_reference_embeddings, match_sku_for_crop,
    infer_similar_sku_groups, parse_reference_planogram, build_observed_planogram,
    compare_planograms_step3
)
from app.item_validation import normalize_name

# ─── Строим каталог ────────────────────────────────────────────────
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

# reference_planogram
ref_plano_payload = {
    "slots": [
        {"shelf_id": s["shelf_id"], "slot_index": s["slot_index"],
         "expected_sku_id": s["sku_id"], "expected_facings": s.get("expected_facings", 1)}
        for s in slots
    ]
}
reference_slots = parse_reference_planogram(ref_plano_payload)

BASE_DIR = Path('.')
ref_embeddings = build_reference_embeddings(catalog, base_dir=BASE_DIR)
catalog_for_matching = [item for item in catalog if item.sku_id in ref_embeddings]
similar_groups = infer_similar_sku_groups(catalog)

# ─── Загружаем reference result ──────────────────────────────────
ref_dir = Path('data/sku_results/reference/20260423_181949_image-08-04-26-11-48')
result = json.loads((ref_dir / 'result.json').read_text(encoding='utf-8'))
positions = result['reference_positions']['positions']

with Image.open(ref_dir / 'input.jpg') as src:
    full_img = ImageOps.exif_transpose(src).convert('RGB')

print(f"Reference positions: {len(positions)}, Reference slots: {len(reference_slots)}")

# ─── Матчинг кропов ──────────────────────────────────────────────
observed_positions = []
confidence_threshold = 0.62

for pos in positions:
    bbox = pos.get('bbox', {})
    x1, y1, x2, y2 = int(bbox.get('x1',0)), int(bbox.get('y1',0)), int(bbox.get('x2',0)), int(bbox.get('y2',0))
    crop = full_img.crop((x1, y1, x2, y2))
    shelf_id = int(pos.get('shelf_id', 0) or 0)
    pos_in_shelf = int(pos.get('position_in_shelf', 0) or 0)

    m = match_sku_for_crop(
        crop, catalog_for_matching, ref_embeddings,
        confidence_threshold=confidence_threshold, similar_groups=similar_groups,
        llm_name_hint='', llm_name_weight=0.12
    )
    observed_positions.append({
        "index": int(pos.get('index', 0) or 0),
        "shelf_id": shelf_id,
        "position_in_shelf": pos_in_shelf,
        "bbox": bbox,
        "predicted_sku_id": m["predicted_sku_id"],
        "predicted_name": m["predicted_name"],
        "confidence": m["confidence"],
        "visual_score": m["visual_score"],
        "observed_facings": 1,
    })

unknown_count = sum(1 for p in observed_positions if p["predicted_sku_id"] == "unknown")
print(f"Accepted: {len(observed_positions)-unknown_count}/{len(observed_positions)}, Unknown: {unknown_count}")

# ─── Распределение по полкам ─────────────────────────────────────
from collections import Counter, defaultdict
by_shelf = defaultdict(list)
for p in observed_positions:
    by_shelf[p['shelf_id']].append(p)

print("\n=== Observed by shelf ===")
for shelf_id in sorted(by_shelf.keys()):
    items = by_shelf[shelf_id]
    name_count = Counter(p['predicted_name'][:20] for p in items)
    print(f"  Shelf {shelf_id}: {len(items)} positions")
    for name, cnt in name_count.most_common(5):
        print(f"    {name}: {cnt}")

print("\n=== Expected by shelf (planogram) ===")
exp_by_shelf = defaultdict(list)
for s in reference_slots:
    exp_by_shelf[s.shelf_id].append(s)
for shelf_id in sorted(exp_by_shelf.keys()):
    items = exp_by_shelf[shelf_id]
    sku_count = Counter(s.sku_id for s in items)
    print(f"  Shelf {shelf_id}: {len(items)} expected slots, slot_index range [{min(s.slot_index for s in items)}, {max(s.slot_index for s in items)}]")

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
print(f"  Total deviations: {len(metrics['deviations'])}")

print("\n=== Alignment debug by shelf ===")
print(json.dumps(metrics.get("alignment_debug_by_shelf", {}), indent=2, ensure_ascii=False))

print("\n=== Shift by shelf ===")
print(json.dumps(metrics.get("slot_alignment_shift_by_shelf", {}), indent=2))
