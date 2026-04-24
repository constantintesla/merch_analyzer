"""API test for compliance check."""
import json, sys, urllib.request
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

payload = {
    'reference_result_dir': 'data/sku_results/reference/20260423_181949_image-08-04-26-11-48',
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

print("Score:", result["compliance_score"])
print("Status:", result["status"])
diag = result.get('diagnostics', {})
print("LM names loaded:", diag.get("lm_names_loaded", 0))
print("LM run:", diag.get("lm_run_dir", "N/A"))
print("LM weight used:", diag.get("lm_name_weight_used", 0))
dt = {}
for d in result.get('deviations', []):
    t = d['type']
    dt[t] = dt.get(t, 0) + 1
print("Deviations:", dt)
align = result.get('metrics', {}).get('alignment_debug_by_shelf', {})
print("Alignment:")
for sh, info in sorted(align.items()):
    print("  Shelf", sh, "shift=", info.get("selected_shift", 0), "aligned=", info.get("aligned_count", 0))
