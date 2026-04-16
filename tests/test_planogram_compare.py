from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.planogram import (
    build_planogram_template_from_items,
    planogram_template_to_csv_text,
    parse_manual_planogram_text,
    parse_planogram_csv,
    parse_planogram_json,
    resolve_planogram_source,
    slot_overlay_entries_from_items,
)
from app.planogram_compare import compare_planogram


def test_parse_manual_planogram_text() -> None:
    template = parse_manual_planogram_text(
        "1,1,Coca Cola 0.5\n"
        "1,2,Coca Cola Zero 0.5\n"
    )
    assert template.name == "manual_planogram"
    assert len(template.slots) == 2
    assert template.slots[0].shelf_id == 1
    assert template.slots[0].position_in_shelf == 1


def test_parse_planogram_csv() -> None:
    csv_data = (
        "shelf_id,position_in_shelf,item_name\n"
        "1,1,Coca Cola 0.5\n"
        "1,2,Coca Cola Zero 0.5\n"
    ).encode("utf-8")
    template = parse_planogram_csv(csv_data, name="imported")
    assert template.name == "imported"
    assert len(template.slots) == 2


def test_parse_planogram_json() -> None:
    json_data = (
        '{"name":"demo","version":"2","slots":['
        '{"shelf_id":1,"position_in_shelf":1,"item_name":"A"},'
        '{"shelf_id":1,"position_in_shelf":2,"item_name":"B"}]}'
    ).encode("utf-8")
    template = parse_planogram_json(json_data)
    assert template.name == "demo"
    assert template.version == "2"
    assert len(template.slots) == 2


def test_compare_planogram_detects_main_issue_types() -> None:
    template = parse_manual_planogram_text(
        "1,1,Sprite 0.5\n"
        "1,2,Coca Cola Zero 0.5\n"
        "2,1,Fanta 0.5\n"
    )
    actual_items = [
        {"item_id": 1, "shelf_id": 1, "position_in_shelf": 1, "lm_item_name": "Coca Cola Zero 0.5"},
        {"item_id": 2, "shelf_id": 1, "position_in_shelf": 2, "lm_item_name": "Sprite 0.5"},
        {"item_id": 3, "shelf_id": 3, "position_in_shelf": 1, "lm_item_name": "Pepsi 0.5"},
    ]
    report = compare_planogram(template, actual_items)
    issue_types = {issue["issue_type"] for issue in report["issues"]}

    assert report["enabled"] is True
    assert "wrong_position" in issue_types
    assert "missing" in issue_types
    assert "extra" in issue_types
    assert 0.0 <= report["compliance_ratio"] <= 1.0


def test_compare_planogram_name_mismatch() -> None:
    template = parse_manual_planogram_text("1,1,Coca Cola 0.5\n")
    actual_items = [
        {"item_id": 1, "shelf_id": 1, "position_in_shelf": 1, "lm_item_name": "Pepsi 0.5"},
    ]
    report = compare_planogram(template, actual_items)
    assert len(report["issues"]) == 1
    assert report["issues"][0]["issue_type"] == "name_mismatch"
    assert report["compliance_ratio"] == 0.0


def test_build_planogram_template_from_items_uses_best_confidence_per_slot() -> None:
    template = build_planogram_template_from_items(
        [
            {
                "shelf_id": 1,
                "position_in_shelf": 1,
                "lm_item_name": "Water A",
                "lm_confidence_final": 0.42,
            },
            {
                "shelf_id": 1,
                "position_in_shelf": 1,
                "lm_item_name": "Water B",
                "lm_confidence_final": 0.91,
            },
            {
                "shelf_id": 1,
                "position_in_shelf": 2,
                "lm_item_name": "unknown",
                "lm_confidence_final": 0.99,
                "item_id": 7,
            },
        ],
        name="ref",
    )
    assert template.name == "ref"
    assert len(template.slots) == 2
    by_pos = {s.position_in_shelf: s for s in template.slots}
    assert by_pos[1].item_name == "Water B"
    assert by_pos[2].item_name == "не распознано #7"


def test_resolve_planogram_source_priority() -> None:
    assert (
        resolve_planogram_source(
            prepared_planogram_csv="shelf_id,position_in_shelf,item_name\n1,1,A\n",
            reference_planogram_image_content=b"img",
            reference_planogram_image_filename="ref.jpg",
            planogram_file_content=b"csv",
            planogram_filename="plan.csv",
            planogram_text="1,1,A",
        )
        == "prepared_csv"
    )
    assert (
        resolve_planogram_source(
            prepared_planogram_csv="",
            reference_planogram_image_content=b"img",
            reference_planogram_image_filename="ref.jpg",
            planogram_file_content=b"csv",
            planogram_filename="plan.csv",
            planogram_text="1,1,A",
        )
        == "reference_image"
    )
    assert (
        resolve_planogram_source(
            prepared_planogram_csv="",
            reference_planogram_image_content=None,
            reference_planogram_image_filename=None,
            planogram_file_content=b"csv",
            planogram_filename="plan.csv",
            planogram_text="1,1,A",
        )
        == "file"
    )
    assert (
        resolve_planogram_source(
            prepared_planogram_csv="",
            reference_planogram_image_content=None,
            reference_planogram_image_filename=None,
            planogram_file_content=None,
            planogram_filename=None,
            planogram_text="1,1,A",
        )
        == "text"
    )


def test_planogram_template_to_csv_text() -> None:
    template = parse_manual_planogram_text("1,2,B\n1,1,A\n")
    csv_text = planogram_template_to_csv_text(template)
    assert "shelf_id,position_in_shelf,item_name" in csv_text
    assert "1,1,A" in csv_text
    assert "1,2,B" in csv_text


def test_slot_overlay_entries_bbox_norm() -> None:
    items = [
        {
            "shelf_id": 1,
            "position_in_shelf": 1,
            "lm_item_name": "A",
            "lm_confidence": 0.9,
            "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 50},
            "item_id": 1,
        }
    ]
    entries = slot_overlay_entries_from_items(items, 200, 100)
    assert len(entries) == 1
    assert entries[0]["bbox_norm"]["x2"] == 0.5
    assert entries[0]["bbox_norm"]["y2"] == 0.5


def test_slot_overlay_entries_includes_unknown_with_bbox() -> None:
    items = [
        {
            "shelf_id": 1,
            "position_in_shelf": 1,
            "lm_item_name": "unknown",
            "lm_confidence": 0.5,
            "bbox": {"x1": 10, "y1": 10, "x2": 90, "y2": 90},
            "item_id": 3,
        }
    ]
    entries = slot_overlay_entries_from_items(items, 100, 100)
    assert len(entries) == 1
    assert entries[0]["item_name"] == "не распознано #3"
    assert entries[0]["bbox_norm"]["x1"] == 0.1
