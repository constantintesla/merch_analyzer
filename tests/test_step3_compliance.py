from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.step3_compliance import (  # noqa: E402
    build_observed_planogram,
    build_reference_embeddings,
    calibrate_thresholds_and_weights,
    collect_uncertainty_flags,
    compare_planograms_step3,
    match_sku_for_crop,
    parse_reference_planogram,
    parse_sku_catalog,
)


def _solid_image(path: Path, rgb: tuple[int, int, int]) -> None:
    Image.new("RGB", (80, 160), color=rgb).save(path, format="JPEG", quality=90)


def test_parse_contracts() -> None:
    plan = parse_reference_planogram(
        {
            "slots": [
                {"shelf_id": 1, "slot_index": 1, "expected_sku_id": "sku_1", "expected_facings": 2},
                {"shelf_id": 1, "slot_index": 2, "expected_sku_id": "sku_2", "expected_facings": 1},
            ]
        }
    )
    cat = parse_sku_catalog(
        {
            "items": [
                {
                    "sku_id": "sku_1",
                    "canonical_name": "Cola Classic",
                    "aliases": ["Classic Cola"],
                    "reference_images": ["x.jpg"],
                }
            ]
        }
    )
    assert len(plan) == 2
    assert len(cat) == 1
    assert cat[0].sku_id == "sku_1"


def test_visual_matching_prefers_closest_reference(tmp_path: Path) -> None:
    ref_red = tmp_path / "red.jpg"
    ref_green = tmp_path / "green.jpg"
    _solid_image(ref_red, (220, 20, 20))
    _solid_image(ref_green, (20, 180, 20))

    catalog = parse_sku_catalog(
        {
            "items": [
                {
                    "sku_id": "red_sku",
                    "canonical_name": "Red Bottle",
                    "aliases": [],
                    "reference_images": [str(ref_red)],
                },
                {
                    "sku_id": "green_sku",
                    "canonical_name": "Green Bottle",
                    "aliases": [],
                    "reference_images": [str(ref_green)],
                },
            ]
        }
    )
    emb = build_reference_embeddings(catalog, base_dir=ROOT_DIR)
    crop = Image.new("RGB", (90, 180), color=(210, 30, 30))
    out = match_sku_for_crop(crop, catalog, emb, confidence_threshold=0.0)
    assert out["top_k"][0]["sku_id"] == "red_sku"
    assert out["predicted_sku_id"] in {"red_sku", "unknown"}
    assert len(out["top_k"]) == 2


def test_planogram_compare_score_and_deviations() -> None:
    reference = parse_reference_planogram(
        {
            "slots": [
                {"shelf_id": 1, "slot_index": 1, "expected_sku_id": "sku_a", "expected_facings": 1},
                {"shelf_id": 1, "slot_index": 2, "expected_sku_id": "sku_b", "expected_facings": 1},
            ]
        }
    )
    observed = [
        {
            "index": 1,
            "shelf_id": 1,
            "position_in_shelf": 1,
            "predicted_sku_id": "sku_a",
            "observed_facings": 1,
        },
        {
            "index": 2,
            "shelf_id": 1,
            "position_in_shelf": 2,
            "predicted_sku_id": "sku_x",
            "observed_facings": 1,
        },
    ]
    report = compare_planograms_step3(
        reference_slots=reference,
        observed_positions=observed,
        presence_weight=0.4,
        position_weight=0.35,
        facings_weight=0.25,
    )
    assert 0.0 <= report["compliance_score"] <= 100.0
    types = {d["type"] for d in report["deviations"]}
    assert "wrong_sku" in types


def test_observed_runs_and_uncertainty() -> None:
    observed = [
        {"index": 1, "shelf_id": 1, "position_in_shelf": 1, "predicted_sku_id": "sku_a", "confidence": 0.7},
        {"index": 2, "shelf_id": 1, "position_in_shelf": 2, "predicted_sku_id": "sku_a", "confidence": 0.45},
        {"index": 3, "shelf_id": 1, "position_in_shelf": 3, "predicted_sku_id": "unknown", "confidence": 0.2},
    ]
    runs = build_observed_planogram(observed)
    assert runs[0]["observed_facings"] == 2
    flags = collect_uncertainty_flags(observed, confidence_threshold=0.6)
    assert len(flags) >= 2


def test_calibration_returns_baseline_fields() -> None:
    baseline = calibrate_thresholds_and_weights(
        [
            {
                "expected_ok": True,
                "target_score": 92.0,
                "metrics": {"presence_ratio": 1.0, "position_ratio": 0.9, "facings_ratio": 0.95},
            },
            {
                "expected_ok": False,
                "target_score": 38.0,
                "metrics": {"presence_ratio": 0.4, "position_ratio": 0.35, "facings_ratio": 0.5},
            },
        ]
    )
    assert "best_confidence_threshold" in baseline
    assert "baseline_accuracy" in baseline
