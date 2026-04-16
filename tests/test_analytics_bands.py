from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.analytics import (
    Detection,
    assign_shelves_from_grid_bands,
    assign_shelves_from_horizontal_bands,
)


def test_assign_shelves_from_horizontal_bands_two_shelves() -> None:
    dets = [
        Detection(10, 10, 30, 40, 0.9),
        Detection(50, 10, 70, 40, 0.9),
        Detection(10, 110, 30, 140, 0.9),
        Detection(60, 110, 80, 140, 0.9),
    ]
    shelf_ids = [1, 2]
    y_splits = [0.0, 0.5, 1.0]
    assignments = assign_shelves_from_horizontal_bands(
        dets, image_width=100, image_height=200, shelf_ids_ordered=shelf_ids, y_splits_norm=y_splits
    )
    by_idx = {a["detection_index"]: a for a in assignments}
    assert by_idx[0]["shelf_id"] == 1
    assert by_idx[1]["shelf_id"] == 1
    assert by_idx[0]["position_in_shelf"] == 1
    assert by_idx[1]["position_in_shelf"] == 2
    assert by_idx[2]["shelf_id"] == 2
    assert by_idx[3]["shelf_id"] == 2


def test_assign_shelves_from_grid_bands_columns() -> None:
    """Две полки по Y, две колонки по X — позиция 1..2 по центру bbox."""
    dets = [
        Detection(10, 10, 40, 40, 0.9),
        Detection(60, 10, 90, 40, 0.9),
        Detection(10, 60, 40, 90, 0.9),
    ]
    shelf_ids = [1, 2]
    y_splits = [0.0, 0.5, 1.0]
    x_splits = [0.0, 0.5, 1.0]
    assignments = assign_shelves_from_grid_bands(
        dets,
        image_width=100,
        image_height=100,
        shelf_ids_ordered=shelf_ids,
        y_splits_norm=y_splits,
        x_splits_norm=x_splits,
    )
    by_idx = {a["detection_index"]: a for a in assignments}
    assert by_idx[0]["shelf_id"] == 1
    assert by_idx[0]["position_in_shelf"] == 1
    assert by_idx[1]["shelf_id"] == 1
    assert by_idx[1]["position_in_shelf"] == 2
    assert by_idx[2]["shelf_id"] == 2
    assert by_idx[2]["position_in_shelf"] == 1


def test_assign_shelves_outside_band_gets_zero() -> None:
    dets = [Detection(10, 5, 20, 15, 0.9)]
    assignments = assign_shelves_from_horizontal_bands(
        dets,
        image_width=100,
        image_height=100,
        shelf_ids_ordered=[1],
        y_splits_norm=[0.2, 0.8],
    )
    assert assignments[0]["shelf_id"] == 0
