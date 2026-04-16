from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import planogram_store


def test_planogram_store_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    images = tmp_path / "img"
    csv = "shelf_id,position_in_shelf,item_name\n1,1,Test\n"
    row = planogram_store.create_planogram(
        db,
        name="t",
        csv_text=csv,
        image_bytes=None,
        images_dir=images,
    )
    assert row.id
    loaded = planogram_store.get_planogram(db, row.id)
    assert loaded is not None
    assert loaded.csv_text == csv
    assert planogram_store.delete_planogram(db, row.id) is True
    assert planogram_store.get_planogram(db, row.id) is None


def test_planogram_store_update_csv(tmp_path: Path) -> None:
    db = tmp_path / "p2.db"
    images = tmp_path / "img2"
    csv1 = "shelf_id,position_in_shelf,item_name\n1,1,A\n"
    row = planogram_store.create_planogram(
        db,
        name="orig",
        csv_text=csv1,
        image_bytes=None,
        images_dir=images,
    )
    csv2 = "shelf_id,position_in_shelf,item_name\n1,1,B\n2,1,C\n"
    updated = planogram_store.update_planogram(
        db,
        row.id,
        csv_text=csv2,
        name=None,
        image_bytes=None,
        images_dir=images,
    )
    assert updated is not None
    assert updated.csv_text == csv2
    assert updated.name == "orig"
    again = planogram_store.update_planogram(
        db,
        row.id,
        csv_text=csv2,
        name="renamed",
        image_bytes=None,
        images_dir=images,
    )
    assert again is not None
    assert again.name == "renamed"
