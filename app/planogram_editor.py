from __future__ import annotations

from typing import Any


def normalize_editor_slots(raw_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw_slots, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"slot #{i} must be object")
        shelf_id = int(row.get("shelf_id", 0) or 0)
        slot_index = int(row.get("slot_index", 0) or 0)
        item_name = str(row.get("item_name", "")).strip()
        sku_id = str(row.get("sku_id", "")).strip()
        expected_facings = int(row.get("expected_facings", 1) or 1)
        bbox_norm = row.get("bbox_norm")
        if shelf_id <= 0 or slot_index <= 0:
            raise ValueError(f"slot #{i}: shelf_id and slot_index must be positive")
        if not item_name:
            raise ValueError(f"slot #{i}: item_name is required")
        if not sku_id:
            raise ValueError(f"slot #{i}: sku_id is required")
        if expected_facings <= 0:
            raise ValueError(f"slot #{i}: expected_facings must be positive")
        if not isinstance(bbox_norm, dict):
            raise ValueError(f"slot #{i}: bbox_norm is required")
        x1 = float(bbox_norm.get("x1", 0.0))
        y1 = float(bbox_norm.get("y1", 0.0))
        x2 = float(bbox_norm.get("x2", 0.0))
        y2 = float(bbox_norm.get("y2", 0.0))
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"slot #{i}: bbox_norm values must be within [0,1] and x1<x2,y1<y2")
        out.append(
            {
                "shelf_id": shelf_id,
                "slot_index": slot_index,
                "item_name": item_name,
                "sku_id": sku_id,
                "expected_facings": expected_facings,
                "bbox_norm": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "reference_image_path": str(row.get("reference_image_path", "")).strip(),
                "reference_image_url": str(row.get("reference_image_url", "")).strip(),
            }
        )
    if not out:
        raise ValueError("at least one slot is required")
    keys = {(int(s["shelf_id"]), int(s["slot_index"])) for s in out}
    if len(keys) != len(out):
        raise ValueError("duplicate shelf_id+slot_index pairs")
    return out


def editor_slots_to_csv(slots: list[dict[str, Any]]) -> str:
    lines = ["shelf_id,position_in_shelf,item_name"]
    for s in sorted(slots, key=lambda x: (int(x["shelf_id"]), int(x["slot_index"]))):
        lines.append(f'{int(s["shelf_id"])},{int(s["slot_index"])},{str(s["item_name"]).replace(",", " ")}')
    return "\n".join(lines) + "\n"


def build_reference_planogram_json(slots: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "slots": [
            {
                "shelf_id": int(s["shelf_id"]),
                "slot_index": int(s["slot_index"]),
                "expected_sku_id": str(s["sku_id"]),
                "expected_facings": int(s["expected_facings"]),
            }
            for s in sorted(slots, key=lambda x: (int(x["shelf_id"]), int(x["slot_index"])))
        ]
    }
