from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.planogram_editor import build_reference_planogram_json, editor_slots_to_csv, normalize_editor_slots


def test_normalize_editor_slots_ok() -> None:
    slots = normalize_editor_slots(
        [
            {
                "shelf_id": 1,
                "slot_index": 1,
                "item_name": "Cola 0.5",
                "sku_id": "sku_1",
                "expected_facings": 2,
                "bbox_norm": {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.5},
            }
        ]
    )
    assert len(slots) == 1
    assert slots[0]["sku_id"] == "sku_1"


def test_editor_slots_to_csv() -> None:
    csv_text = editor_slots_to_csv(
        [
            {"shelf_id": 1, "slot_index": 2, "item_name": "B"},
            {"shelf_id": 1, "slot_index": 1, "item_name": "A"},
        ]
    )
    assert "shelf_id,position_in_shelf,item_name" in csv_text
    assert "1,1,A" in csv_text
    assert "1,2,B" in csv_text


def test_build_reference_planogram_json() -> None:
    out = build_reference_planogram_json(
        [
            {"shelf_id": 2, "slot_index": 1, "sku_id": "sku_x", "expected_facings": 1},
            {"shelf_id": 1, "slot_index": 2, "sku_id": "sku_y", "expected_facings": 3},
        ]
    )
    assert "slots" in out
    assert len(out["slots"]) == 2
    assert out["slots"][0]["shelf_id"] == 1
