"""Диагностический скрипт для анализа матчинга шага 3."""
import json, sys
from pathlib import Path
from PIL import Image, ImageOps
sys.path.insert(0, '.')

from app.step3_compliance import (
    parse_sku_catalog, build_reference_embeddings, match_sku_for_crop,
    infer_similar_sku_groups, image_embedding, _cosine, compare_planograms_step3,
    parse_reference_planogram
)
from app.item_validation import normalize_name
from collections import defaultdict

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
BASE_DIR = Path('.')
ref_embeddings = build_reference_embeddings(catalog, base_dir=BASE_DIR)
catalog_for_matching = [item for item in catalog if item.sku_id in ref_embeddings]
similar_groups = infer_similar_sku_groups(catalog)

print(f"Catalog: {len(catalog)} items, {len(ref_embeddings)} with embeddings")

# ─── Уникальные эмбеддинги ────────────────────────────────────────
import numpy as np
emb_by_name = defaultdict(list)
for item in catalog_for_matching:
    embs = ref_embeddings.get(item.sku_id, [])
    name = normalize_name(item.canonical_name)
    emb_by_name[name].extend(embs)

print(f"Unique canonical names with embeddings: {len(emb_by_name)}")
print()

# ─── Попарные косинусные сходства между группами ─────────────────
name_list = list(emb_by_name.keys())
print("=== Cosine similarities between unique product groups ===")
for i, n1 in enumerate(name_list):
    e1 = emb_by_name[n1][0]
    for n2 in name_list[i+1:]:
        e2 = emb_by_name[n2][0]
        sim = _cosine(e1, e2)
        if sim > 0.85:
            print(f"  HIGH: {n1} vs {n2}: {sim:.3f}")
print()

# ─── Загружаем reference result ──────────────────────────────────
ref_dir = Path('data/sku_results/reference/20260423_181949_image-08-04-26-11-48')
result = json.loads((ref_dir / 'result.json').read_text(encoding='utf-8'))
positions = result['reference_positions']['positions']

with Image.open(ref_dir / 'input.jpg') as src:
    full_img = ImageOps.exif_transpose(src).convert('RGB')

# ─── Тест матчинга первых 15 кропов ──────────────────────────────
print("=== Matching test (first 15 positions) ===")
unknown_count = 0
accepted_count = 0
for pos in positions[:15]:
    bbox = pos.get('bbox', {})
    x1, y1, x2, y2 = int(bbox.get('x1',0)), int(bbox.get('y1',0)), int(bbox.get('x2',0)), int(bbox.get('y2',0))
    crop = full_img.crop((x1, y1, x2, y2))
    shelf_id = pos.get('shelf_id', '?')
    pos_in_shelf = pos.get('position_in_shelf', '?')

    m = match_sku_for_crop(
        crop, catalog_for_matching, ref_embeddings,
        confidence_threshold=0.62, similar_groups=similar_groups,
        llm_name_hint='', llm_name_weight=0.12
    )
    top3 = m.get('top_k', [])[:3]
    status_mark = "OK" if m['predicted_sku_id'] != 'unknown' else "UNKNOWN"
    if m['predicted_sku_id'] == 'unknown':
        unknown_count += 1
    else:
        accepted_count += 1

    print(f"  s{shelf_id}p{pos_in_shelf} [{status_mark}] => {m['predicted_sku_id']} ({m['predicted_name'][:20]})")
    print(f"    conf={m['confidence']:.3f} vis={m['visual_score']:.3f} gap={m['score_gap']:.3f} thr={m['applied_threshold']:.3f}")
    for t in top3:
        print(f"    top: {t['sku_id']} {t['canonical_name'][:20]} vis={t['visual_score']:.3f} conf={t['confidence']:.3f}")

print()
print(f"Accepted: {accepted_count}, Unknown: {unknown_count}")

# ─── Полный прогон всех позиций ──────────────────────────────────
print()
print("=== Full match run (all positions) ===")
all_unknown = 0
all_accepted = 0
vis_scores = []
gaps = []
for pos in positions:
    bbox = pos.get('bbox', {})
    x1, y1, x2, y2 = int(bbox.get('x1',0)), int(bbox.get('y1',0)), int(bbox.get('x2',0)), int(bbox.get('y2',0))
    crop = full_img.crop((x1, y1, x2, y2))
    m = match_sku_for_crop(
        crop, catalog_for_matching, ref_embeddings,
        confidence_threshold=0.62, similar_groups=similar_groups,
        llm_name_hint='', llm_name_weight=0.12
    )
    vis_scores.append(m['visual_score'])
    gaps.append(m['score_gap'])
    if m['predicted_sku_id'] == 'unknown':
        all_unknown += 1
    else:
        all_accepted += 1

print(f"Accepted: {all_accepted}/{len(positions)}, Unknown: {all_unknown}/{len(positions)}")
print(f"Visual score: min={min(vis_scores):.3f} max={max(vis_scores):.3f} mean={sum(vis_scores)/len(vis_scores):.3f}")
print(f"Score gap: min={min(gaps):.3f} max={max(gaps):.3f} mean={sum(gaps)/len(gaps):.3f}")
gaps_below_threshold = sum(1 for g in gaps if g < 0.07)
print(f"Gaps below 0.07 (rejected by gap): {gaps_below_threshold}/{len(gaps)}")
vis_below_threshold = sum(1 for v in vis_scores if v < 0.50)
print(f"Visual below 0.50 (rejected by vis): {vis_below_threshold}/{len(vis_scores)}")
