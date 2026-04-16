from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.lmstudio_client import LMStudioClient


def test_parse_beverage_skips_english_meta_then_reads_napitok() -> None:
    raw = (
        'The user wants me to identify the beverage in the image based on specific rules.\n'
        "Напиток: Пьютти газированная"
    )
    r = LMStudioClient._parse_beverage_line(raw)
    assert r is not None
    assert r.item_name == "Пьютти газированная"
    assert r.status == "ok"


def test_parse_beverage_meta_only_returns_none() -> None:
    raw = "The user wants me to identify the beverage in the image based on specific rules."
    assert LMStudioClient._parse_beverage_line(raw) is None


def test_coerce_result_keeps_cyrillic_when_normalized_is_latin() -> None:
    r = LMStudioClient._coerce_result(
        {
            "item_name": "Молоко 3.2%",
            "normalized_name": "Milk 3.2%",
            "confidence_raw": 0.9,
            "status": "ok",
        }
    )
    assert r.normalized_name == "Молоко 3.2%"
    assert r.item_name == "Молоко 3.2%"


def test_batch_classify_from_model_text_json_results() -> None:
    raw = (
        '{"results":[{"slot":0,"line":"Напиток: Coca-Cola"},'
        '{"slot":1,"line":"Ничего не обнаружено"}]}'
    )
    out = LMStudioClient._batch_classify_from_model_text(raw, 2)
    assert len(out) == 2
    assert out[0].item_name == "Coca-Cola"
    assert out[0].status == "ok"
    assert "ничего" in out[1].item_name.lower()


def test_batch_classify_from_model_text_top_level_array() -> None:
    raw = '[{"slot":0,"line":"Напиток: Пьютти газированная"}]'
    out = LMStudioClient._batch_classify_from_model_text(raw, 1)
    assert len(out) == 1
    assert out[0].item_name == "Пьютти газированная"


def test_planogram_items_from_payload_shelves() -> None:
    payload = {
        "shelves": [
            {
                "shelf_id": 1,
                "items": [
                    {
                        "position_in_shelf": 1,
                        "item_name": "Тест А",
                        "bbox_norm": {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
                        "confidence": 0.9,
                    }
                ],
            }
        ]
    }
    items = LMStudioClient._planogram_items_from_payload(payload, 1000, 800)
    assert len(items) == 1
    assert items[0]["shelf_id"] == 1
    assert items[0]["position_in_shelf"] == 1
    assert items[0]["lm_item_name"] == "Тест А"
    assert items[0]["bbox"]["x2"] == 200.0
