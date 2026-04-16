from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.planogram import normalize_shelf_positions_from_geometry


def test_normalize_positions_left_to_right_on_shelf() -> None:
    items = [
        {
            "shelf_id": 1,
            "position_in_shelf": 3,
            "row": 1,
            "bbox": {"x1": 200.0, "y1": 10.0, "x2": 280.0, "y2": 90.0},
        },
        {
            "shelf_id": 1,
            "position_in_shelf": 1,
            "row": 1,
            "bbox": {"x1": 10.0, "y1": 10.0, "x2": 90.0, "y2": 90.0},
        },
        {
            "shelf_id": 1,
            "position_in_shelf": 2,
            "row": 1,
            "bbox": {"x1": 100.0, "y1": 10.0, "x2": 180.0, "y2": 90.0},
        },
    ]
    normalize_shelf_positions_from_geometry(items)
    assert items[0]["position_in_shelf"] == 3
    assert items[1]["position_in_shelf"] == 1
    assert items[2]["position_in_shelf"] == 2
    by_x = sorted(items, key=lambda it: it["bbox"]["x1"])
    assert [it["position_in_shelf"] for it in by_x] == [1, 2, 3]


def test_normalize_shelves_top_to_bottom() -> None:
    """Нижняя полка с большим Y получает больший shelf_id после нормализации."""
    items = [
        {
            "shelf_id": 2,
            "position_in_shelf": 1,
            "row": 2,
            "bbox": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 40.0},
        },
        {
            "shelf_id": 1,
            "position_in_shelf": 1,
            "row": 1,
            "bbox": {"x1": 10.0, "y1": 200.0, "x2": 50.0, "y2": 240.0},
        },
    ]
    normalize_shelf_positions_from_geometry(items)
    top = next(it for it in items if it["bbox"]["y1"] < 100)
    bottom = next(it for it in items if it["bbox"]["y1"] > 100)
    assert top["shelf_id"] == 1
    assert bottom["shelf_id"] == 2
