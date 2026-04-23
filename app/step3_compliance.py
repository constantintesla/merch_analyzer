from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.item_validation import normalize_name


@dataclass(frozen=True)
class SKUCatalogItem:
    sku_id: str
    canonical_name: str
    aliases: list[str]
    reference_images: list[str]


@dataclass(frozen=True)
class PlanogramExpectedSlot:
    shelf_id: int
    slot_index: int
    sku_id: str
    expected_facings: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.shelf_id, self.slot_index)


def parse_sku_catalog(payload: dict[str, Any]) -> list[SKUCatalogItem]:
    raw = payload.get("items", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("sku_catalog.items must be a non-empty list")
    out: list[SKUCatalogItem] = []
    seen: set[str] = set()
    for idx, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"sku_catalog.items[{idx}] must be object")
        sku_id = str(row.get("sku_id", "")).strip()
        name = str(row.get("canonical_name", "")).strip()
        aliases_raw = row.get("aliases", [])
        refs_raw = row.get("reference_images", [])
        if not sku_id or not name:
            raise ValueError(f"sku_catalog.items[{idx}] requires sku_id and canonical_name")
        if sku_id in seen:
            raise ValueError(f"duplicate sku_id in catalog: {sku_id}")
        seen.add(sku_id)
        aliases = [str(x).strip() for x in aliases_raw if str(x).strip()] if isinstance(aliases_raw, list) else []
        refs = [str(x).strip() for x in refs_raw if str(x).strip()] if isinstance(refs_raw, list) else []
        if not refs:
            raise ValueError(f"sku_catalog.items[{idx}] requires at least one reference_images path")
        out.append(SKUCatalogItem(sku_id=sku_id, canonical_name=name, aliases=aliases, reference_images=refs))
    return out


def parse_reference_planogram(payload: dict[str, Any]) -> list[PlanogramExpectedSlot]:
    raw = payload.get("slots", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("reference_planogram.slots must be a non-empty list")
    out: list[PlanogramExpectedSlot] = []
    seen: set[tuple[int, int]] = set()
    for idx, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"reference_planogram.slots[{idx}] must be object")
        shelf_id = int(row.get("shelf_id", 0) or 0)
        slot_index = int(row.get("slot_index", 0) or 0)
        sku_id = str(row.get("expected_sku_id", "")).strip()
        expected_facings = int(row.get("expected_facings", 1) or 1)
        if shelf_id <= 0 or slot_index <= 0 or not sku_id:
            raise ValueError(
                f"reference_planogram.slots[{idx}] requires positive shelf_id, slot_index and expected_sku_id"
            )
        if expected_facings <= 0:
            raise ValueError(f"reference_planogram.slots[{idx}] expected_facings must be positive")
        key = (shelf_id, slot_index)
        if key in seen:
            raise ValueError(f"duplicate planogram slot: shelf={shelf_id}, slot={slot_index}")
        seen.add(key)
        out.append(
            PlanogramExpectedSlot(
                shelf_id=shelf_id,
                slot_index=slot_index,
                sku_id=sku_id,
                expected_facings=expected_facings,
            )
        )
    return out


def load_contracts_from_json(
    *,
    reference_planogram_path: str,
    sku_catalog_path: str,
) -> tuple[list[PlanogramExpectedSlot], list[SKUCatalogItem]]:
    plan_payload = json.loads(Path(reference_planogram_path).read_text(encoding="utf-8"))
    cat_payload = json.loads(Path(sku_catalog_path).read_text(encoding="utf-8"))
    if not isinstance(plan_payload, dict) or not isinstance(cat_payload, dict):
        raise ValueError("reference_planogram and sku_catalog files must be JSON objects")
    return parse_reference_planogram(plan_payload), parse_sku_catalog(cat_payload)


def image_embedding(img: Image.Image) -> np.ndarray:
    rgb = ImageOps.exif_transpose(img).convert("RGB").resize((40, 40))
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    hist_parts: list[np.ndarray] = []
    for i in range(3):
        h, _ = np.histogram(arr[:, :, i], bins=12, range=(0.0, 1.0))
        hist_parts.append(h.astype(np.float32))
    gx = np.diff(arr.mean(axis=2), axis=1)
    gy = np.diff(arr.mean(axis=2), axis=0)
    edge_energy = np.array([float(np.mean(np.abs(gx))), float(np.mean(np.abs(gy)))], dtype=np.float32)
    vec = np.concatenate([*hist_parts, edge_energy], axis=0)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / den)


def _name_similarity(hint: str, item: SKUCatalogItem) -> float:
    n_hint = normalize_name(hint)
    if n_hint == "unknown":
        return 0.0
    candidates = [item.canonical_name, *item.aliases]
    best = 0.0
    hint_tokens = set(n_hint.split())
    for cand in candidates:
        n_c = normalize_name(cand)
        c_tokens = set(n_c.split())
        if not c_tokens:
            continue
        inter = len(hint_tokens & c_tokens)
        union = len(hint_tokens | c_tokens)
        score = inter / union if union else 0.0
        if score > best:
            best = score
    return float(best)


def build_reference_embeddings(
    catalog: list[SKUCatalogItem],
    *,
    base_dir: Path,
) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for item in catalog:
        vectors: list[np.ndarray] = []
        for ref in item.reference_images:
            p = (base_dir / ref).resolve() if not Path(ref).is_absolute() else Path(ref)
            if not p.is_file():
                continue
            with Image.open(p) as src:
                vectors.append(image_embedding(src))
        if not vectors:
            raise ValueError(f"catalog sku_id={item.sku_id} has no readable reference images")
        out[item.sku_id] = vectors
    return out


def match_sku_for_crop(
    crop: Image.Image,
    catalog: list[SKUCatalogItem],
    embeddings_by_sku: dict[str, list[np.ndarray]],
    *,
    confidence_threshold: float,
    similar_groups: list[set[str]] | None = None,
    llm_name_hint: str = "",
    llm_name_weight: float = 0.12,
) -> dict[str, Any]:
    emb = image_embedding(crop)
    candidates: list[tuple[SKUCatalogItem, float, float, float]] = []
    for item in catalog:
        refs = embeddings_by_sku.get(item.sku_id, [])
        visual = max((_cosine(emb, v) for v in refs), default=0.0)
        visual_conf = math.sqrt(max(0.0, min(1.0, visual)))
        text = _name_similarity(llm_name_hint, item) if llm_name_hint else 0.0
        score = max(0.0, min(1.0, visual_conf * (1.0 - llm_name_weight) + text * llm_name_weight))
        candidates.append((item, visual, visual_conf, score))
    candidates.sort(key=lambda t: t[3], reverse=True)
    best_item, best_visual, best_visual_conf, best_score = candidates[0]
    second_score = candidates[1][3] if len(candidates) > 1 else 0.0
    best_score = max(0.0, min(1.0, best_score + (best_score - second_score) * 0.2))
    is_hard_pair = False
    if similar_groups:
        for group in similar_groups:
            if best_item.sku_id in group:
                is_hard_pair = True
                break
    threshold = min(0.99, confidence_threshold + (0.08 if is_hard_pair else 0.0))
    accepted = best_score >= threshold
    return {
        "predicted_sku_id": best_item.sku_id if accepted else "unknown",
        "predicted_name": best_item.canonical_name if accepted else "unknown",
        "confidence": float(best_score),
        "visual_score": float(best_visual),
        "status": "ok" if accepted else "unknown",
        "top_k": [
            {
                "sku_id": c[0].sku_id,
                "canonical_name": c[0].canonical_name,
                "confidence": float(c[3]),
                "visual_score": float(c[1]),
                "visual_confidence": float(c[2]),
            }
            for c in candidates[:3]
        ],
        "applied_threshold": float(threshold),
        "visual_confidence": float(best_visual_conf),
    }


def build_observed_planogram(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_shelf: dict[int, list[dict[str, Any]]] = {}
    for pos in positions:
        shelf = int(pos.get("shelf_id", 0) or 0)
        if shelf <= 0:
            continue
        by_shelf.setdefault(shelf, []).append(pos)
    out: list[dict[str, Any]] = []
    for shelf_id in sorted(by_shelf):
        shelf_positions = sorted(by_shelf[shelf_id], key=lambda x: int(x.get("position_in_shelf", 0) or 0))
        run_sku = ""
        run_start = 0
        run_len = 0
        for p in shelf_positions:
            slot = int(p.get("position_in_shelf", 0) or 0)
            sku_id = str(p.get("predicted_sku_id", "unknown"))
            if sku_id != run_sku:
                if run_len > 0:
                    out.append(
                        {
                            "shelf_id": shelf_id,
                            "slot_index": run_start,
                            "sku_id": run_sku,
                            "observed_facings": run_len,
                        }
                    )
                run_sku = sku_id
                run_start = slot
                run_len = 1
            else:
                run_len += 1
        if run_len > 0:
            out.append(
                {
                    "shelf_id": shelf_id,
                    "slot_index": run_start,
                    "sku_id": run_sku,
                    "observed_facings": run_len,
                }
            )
    return out


def compare_planograms_step3(
    *,
    reference_slots: list[PlanogramExpectedSlot],
    observed_positions: list[dict[str, Any]],
    presence_weight: float,
    position_weight: float,
    facings_weight: float,
) -> dict[str, Any]:
    expected_by_slot = {(s.shelf_id, s.slot_index): s for s in reference_slots}
    observed_by_slot: dict[tuple[int, int], dict[str, Any]] = {}
    for pos in observed_positions:
        key = (int(pos.get("shelf_id", 0) or 0), int(pos.get("position_in_shelf", 0) or 0))
        if key[0] > 0 and key[1] > 0:
            observed_by_slot[key] = pos

    matched_presence = 0
    matched_position = 0
    facings_score_sum = 0.0
    deviations: list[dict[str, Any]] = []

    observed_slots_by_sku: dict[str, list[tuple[int, int]]] = {}
    for slot, item in observed_by_slot.items():
        sku_id = str(item.get("predicted_sku_id", "unknown"))
        observed_slots_by_sku.setdefault(sku_id, []).append(slot)

    for slot_key, expected in expected_by_slot.items():
        observed = observed_by_slot.get(slot_key)
        if observed is None:
            deviations.append(
                {
                    "type": "missing_sku",
                    "sku_id": expected.sku_id,
                    "shelf_id": expected.shelf_id,
                    "slot_index": expected.slot_index,
                    "expected": {"sku_id": expected.sku_id, "facings": expected.expected_facings},
                    "actual": {"sku_id": "", "facings": 0},
                    "reason": "SKU expected by planogram is absent at slot",
                }
            )
            continue

        actual_sku = str(observed.get("predicted_sku_id", "unknown"))
        observed_facings = int(observed.get("observed_facings", 1) or 1)

        if actual_sku == expected.sku_id:
            matched_presence += 1
            matched_position += 1
            facings_ratio = min(observed_facings, expected.expected_facings) / max(
                observed_facings, expected.expected_facings
            )
            facings_score_sum += facings_ratio
        else:
            found_elsewhere = slot_key not in observed_slots_by_sku.get(expected.sku_id, []) and bool(
                observed_slots_by_sku.get(expected.sku_id)
            )
            deviations.append(
                {
                    "type": "wrong_position" if found_elsewhere else "wrong_sku",
                    "sku_id": expected.sku_id,
                    "shelf_id": expected.shelf_id,
                    "slot_index": expected.slot_index,
                    "expected": {"sku_id": expected.sku_id, "facings": expected.expected_facings},
                    "actual": {"sku_id": actual_sku, "facings": observed_facings},
                    "reason": "Expected SKU found on another slot" if found_elsewhere else "Different SKU in slot",
                }
            )

    for slot_key, observed in observed_by_slot.items():
        if slot_key in expected_by_slot:
            continue
        deviations.append(
            {
                "type": "extra_sku",
                "sku_id": str(observed.get("predicted_sku_id", "unknown")),
                "shelf_id": slot_key[0],
                "slot_index": slot_key[1],
                "expected": {"sku_id": "", "facings": 0},
                "actual": {
                    "sku_id": str(observed.get("predicted_sku_id", "unknown")),
                    "facings": int(observed.get("observed_facings", 1) or 1),
                },
                "reason": "Observed SKU does not exist in reference planogram slot set",
            }
        )

    total = max(1, len(expected_by_slot))
    presence_ratio = matched_presence / total
    position_ratio = matched_position / total
    facings_ratio = facings_score_sum / total
    w_sum = max(1e-9, presence_weight + position_weight + facings_weight)
    score = (
        (presence_ratio * presence_weight + position_ratio * position_weight + facings_ratio * facings_weight)
        / w_sum
        * 100.0
    )
    return {
        "presence_ratio": presence_ratio,
        "position_ratio": position_ratio,
        "facings_ratio": facings_ratio,
        "compliance_score": float(max(0.0, min(100.0, score))),
        "deviations": deviations,
    }


def collect_uncertainty_flags(observed_positions: list[dict[str, Any]], *, confidence_threshold: float) -> list[dict]:
    out: list[dict] = []
    for pos in observed_positions:
        conf = float(pos.get("confidence", 0.0))
        sku_id = str(pos.get("predicted_sku_id", "unknown"))
        reasons: list[str] = []
        if sku_id == "unknown":
            reasons.append("unknown_sku")
        if conf < confidence_threshold:
            reasons.append("low_confidence")
        bbox = pos.get("bbox", {})
        if isinstance(bbox, dict):
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
            if (x2 - x1) * (y2 - y1) < 48.0 * 48.0:
                reasons.append("small_crop")
        if reasons:
            out.append(
                {
                    "index": int(pos.get("index", 0) or 0),
                    "shelf_id": int(pos.get("shelf_id", 0) or 0),
                    "position_in_shelf": int(pos.get("position_in_shelf", 0) or 0),
                    "reasons": reasons,
                }
            )
    return out


def calibrate_thresholds_and_weights(labeled_samples: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Grid search baseline для confidence threshold и весов score.
    labeled_samples: [{"expected_ok": bool, "metrics":{"presence_ratio":..,"position_ratio":..,"facings_ratio":..}, "target_score":float?}]
    """
    if not labeled_samples:
        return {
            "best_confidence_threshold": 0.6,
            "presence_weight": 0.4,
            "position_weight": 0.35,
            "facings_weight": 0.25,
            "baseline_mae": 0.0,
            "baseline_accuracy": 0.0,
        }

    best = None
    for c_thr in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        for pw, ow, fw in (
            (0.4, 0.35, 0.25),
            (0.45, 0.35, 0.2),
            (0.35, 0.4, 0.25),
            (0.4, 0.4, 0.2),
        ):
            mae = 0.0
            ok_hits = 0
            total = 0
            for sample in labeled_samples:
                m = sample.get("metrics", {})
                p = float(m.get("presence_ratio", 0.0))
                o = float(m.get("position_ratio", 0.0))
                f = float(m.get("facings_ratio", 0.0))
                pred_score = (p * pw + o * ow + f * fw) / (pw + ow + fw) * 100.0
                target_score = float(sample.get("target_score", 100.0 if sample.get("expected_ok") else 0.0))
                mae += abs(pred_score - target_score)
                expected_ok = bool(sample.get("expected_ok", target_score >= 80.0))
                pred_ok = pred_score >= 80.0
                if pred_ok == expected_ok:
                    ok_hits += 1
                total += 1
            mae = mae / max(1, total)
            acc = ok_hits / max(1, total)
            key = (acc, -mae)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "best_confidence_threshold": c_thr,
                    "presence_weight": pw,
                    "position_weight": ow,
                    "facings_weight": fw,
                    "baseline_mae": round(mae, 4),
                    "baseline_accuracy": round(acc, 4),
                    "samples_count": total,
                }
    assert best is not None
    best.pop("key", None)
    return best


def pass_fail_from_score(score: float, threshold: float) -> str:
    return "pass" if score >= threshold else "fail"


def infer_similar_sku_groups(catalog: list[SKUCatalogItem]) -> list[set[str]]:
    out: list[set[str]] = []
    by_head: dict[str, set[str]] = {}
    for item in catalog:
        n = normalize_name(item.canonical_name)
        head = n.split(" ")[0] if n and n != "unknown" else ""
        if not head:
            continue
        by_head.setdefault(head, set()).add(item.sku_id)
    for g in by_head.values():
        if len(g) >= 2:
            out.append(g)
    return out
