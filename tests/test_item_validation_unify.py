from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.item_validation import ValidationConfig, apply_item_validation, unify_cluster_item_display_names


def _base_item(
    group_id: int,
    lm_item_name: str,
    *,
    lm_confidence: float = 0.9,
) -> dict:
    return {
        "group_id": group_id,
        "lm_item_name": lm_item_name,
        "lm_raw_name": lm_item_name,
        "lm_status": "ok",
        "lm_confidence": lm_confidence,
    }


def test_apply_unifies_two_cluster_members_by_confidence_tiebreak() -> None:
    items = [
        _base_item(1, "Brand A variant", lm_confidence=0.7),
        _base_item(1, "Brand B variant", lm_confidence=0.95),
    ]
    out = apply_item_validation(items, ValidationConfig(catalog_path=""))
    assert out[0]["lm_item_name"] == out[1]["lm_item_name"] == "Brand B variant"
    assert "group_unified" in out[0]["lm_warning_reason"] or "group_unified" in out[1]["lm_warning_reason"]


def test_apply_unify_keeps_cyrillic_canonical_under_tie() -> None:
    cyr = "\u043a\u043e\u043a\u0430 \u043a\u043e\u043b\u0430"
    items = [
        _base_item(1, "Coca Cola 0.5", lm_confidence=0.7),
        _base_item(1, cyr, lm_confidence=0.95),
    ]
    out = apply_item_validation(items, ValidationConfig(catalog_path=""))
    assert out[0]["lm_item_name"] == out[1]["lm_item_name"] == cyr


def test_single_member_group_unchanged() -> None:
    items = [_base_item(1, "Only one", lm_confidence=0.9)]
    out = apply_item_validation(items, ValidationConfig(catalog_path=""))
    assert out[0]["lm_item_name"] == "Only one"
    assert "group_unified" not in out[0].get("lm_warning_reason", "")


def test_same_names_no_group_unified_flag() -> None:
    items = [
        _base_item(1, "Pepsi 1л", lm_confidence=0.8),
        _base_item(1, "Pepsi 1л", lm_confidence=0.85),
    ]
    out = apply_item_validation(items, ValidationConfig(catalog_path=""))
    assert out[0]["lm_item_name"] == "Pepsi 1л"
    assert out[1]["lm_item_name"] == "Pepsi 1л"
    assert "group_unified" not in out[0].get("lm_warning_reason", "")
    assert "group_unified" not in out[1].get("lm_warning_reason", "")


def test_catalog_backed_member_wins_canonical() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        delete=False,
        encoding="utf-8",
        newline="",
    ) as tmp:
        w = csv.DictWriter(tmp, fieldnames=["name"])
        w.writeheader()
        w.writerow({"name": "Sprite 0.5"})
        path = tmp.name
    try:
        items = [
            _base_item(2, "Sprite 0.5", lm_confidence=0.9),
            _base_item(2, "random guess", lm_confidence=0.5),
        ]
        out = apply_item_validation(
            items,
            ValidationConfig(catalog_path=path, catalog_match_threshold=0.75),
        )
        assert out[0]["lm_item_name"] == "Sprite 0.5"
        assert out[1]["lm_item_name"] == "Sprite 0.5"
    finally:
        Path(path).unlink(missing_ok=True)


def test_unify_cluster_standalone_after_manual_fields() -> None:
    """После ручной подстановки полей как после первого прохода валидации."""
    items = [
        {
            "group_id": 3,
            "lm_item_name": "A",
            "lm_catalog_match_score": 0.0,
            "lm_warning_reason": "not_in_catalog",
            "lm_confidence": 0.5,
            "lm_confidence_final": 0.5,
        },
        {
            "group_id": 3,
            "lm_item_name": "B",
            "lm_catalog_match_score": 0.0,
            "lm_warning_reason": "not_in_catalog",
            "lm_confidence": 0.9,
            "lm_confidence_final": 0.9,
        },
    ]
    unify_cluster_item_display_names(items, ValidationConfig(catalog_match_threshold=0.82))
    assert items[0]["lm_item_name"] == items[1]["lm_item_name"] == "B"
