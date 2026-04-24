from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageOps

from app.analytics import Detection, assign_shelves_and_positions
from app.lmstudio_client import ItemClassification, LMStudioClient
from app.merch_logging import configure_merch_logging
from app.planogram_editor import (
    build_reference_planogram_json,
    editor_slots_to_csv,
    normalize_editor_slots,
    renumber_slots_within_shelves,
)
from app.planogram_store import (
    create_planogram,
    delete_planogram,
    get_planogram,
    list_planograms,
    update_planogram,
)
from app.similarity import cluster_crop_indices_by_similarity
from app.sku110k_adapter import SKU110KDetector
from app.step3_compliance import (
    build_observed_planogram,
    build_reference_embeddings,
    calibrate_thresholds_and_weights,
    collect_uncertainty_flags,
    compare_planograms_step3,
    infer_similar_sku_groups,
    match_by_lm_name,
    match_sku_for_crop,
    parse_reference_planogram,
    parse_sku_catalog,
    pass_fail_from_score,
    score_crop_against_sku,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_DB_PATH = DATA_DIR / "reference_by_sku.json"
SKU_RESULTS_DIR = DATA_DIR / "sku_results"
PLANOGRAM_DB_PATH = DATA_DIR / "planograms.db"
PLANOGRAM_IMAGES_DIR = DATA_DIR / "planogram_images"
PLANOGRAM_EDITOR_META_DIR = DATA_DIR / "planogram_editor"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

DEFAULT_LMSTUDIO_URL = "http://desktop-oh7jn1i:1234"
DEFAULT_LMSTUDIO_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_REPO_PATH = str(BASE_DIR / "third_party" / "SKU110K_CVPR19")
DEFAULT_WEIGHTS_PATH = str(BASE_DIR / "models" / "sku110k_pretrained.h5")
DEFAULT_PYTHON_BIN = str(BASE_DIR / ".venv_sku" / "Scripts" / "python.exe")
DEFAULT_RUN_MODE = "docker"
DEFAULT_WSL_PYTHON_BIN = "python3"
DEFAULT_DOCKER_IMAGE = "merch-analyzer-sku110k:tf1.15"
DEFAULT_DOCKER_MOUNT_HOST = str(BASE_DIR)
DEFAULT_DOCKER_MOUNT_TARGET = "/workspace"
DEFAULT_DOCKER_USE_GPU = True
DEFAULT_SCORE_THRESHOLD = 0.7

app = FastAPI(title="Merch Analyzer (simplified)")
logger = logging.getLogger("merch_analyzer")


@app.on_event("startup")
async def _merch_configure_logging() -> None:
    configure_merch_logging()


def _sanitize_filename_stem(original_name: str, *, max_len: int = 80) -> str:
    base = Path(original_name).name
    stem = Path(base).stem or "file"
    stem = re.sub(r"[^\w\u0400-\u04FF\-]+", "_", stem, flags=re.UNICODE)
    stem = stem.strip("_") or "file"
    return stem[:max_len]


def _make_unique_run_dir(category: str, original_filename: str) -> Path:
    """Папка вида sku_results/{category}/YYYYMMDD_HHMMSS_имя_файла/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = _sanitize_filename_stem(original_filename)
    base_name = f"{ts}_{safe}"
    root = SKU_RESULTS_DIR / category
    root.mkdir(parents=True, exist_ok=True)
    path = root / base_name
    if not path.exists():
        path.mkdir(parents=False, exist_ok=False)
        return path
    n = 1
    while True:
        candidate = root / f"{base_name}_{n}"
        if not candidate.exists():
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        n += 1


def _load_normalized_rgb_image(uploaded_bytes: bytes) -> Image.Image:
    from io import BytesIO

    with Image.open(BytesIO(uploaded_bytes)) as src:
        img = ImageOps.exif_transpose(src)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


def _parse_zone_bbox_norm(raw: str) -> dict[str, float] | None:
    txt = (raw or "").strip()
    if not txt:
        return None
    try:
        if txt.startswith("{"):
            data = json.loads(txt)
            x1 = float(data.get("x1"))
            y1 = float(data.get("y1"))
            x2 = float(data.get("x2"))
            y2 = float(data.get("y2"))
        else:
            parts = [float(p.strip()) for p in txt.split(",")]
            if len(parts) != 4:
                raise ValueError
            x1, y1, x2, y2 = parts
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Некорректный формат зоны: ожидается JSON {x1,y1,x2,y2} в долях [0..1]") from exc

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Некорректная зона: x2>x1 и y2>y1 обязательны")
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _point_in_bbox_norm(x: float, y: float, box: dict[str, float]) -> bool:
    return bool(box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"])


def _save_reference_positions_sku(
    *,
    img: Image.Image,
    image_path: Path,
    detector: SKU110KDetector,
    run_dir: Path,
    own_zone_bbox_norm: dict[str, float] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    logger.info(
        "reference: вызов SKU110K detect_image path=%s run_dir=%s image_size=%sx%s",
        image_path,
        run_dir,
        img.width,
        img.height,
    )
    detections = detector.detect_image(str(image_path))
    logger.info(
        "reference: SKU110K вернул %d сырых боксов за %.1f с",
        len(detections),
        time.perf_counter() - t0,
    )
    det_objects: list[Detection] = [
        Detection(x1=float(d.x1), y1=float(d.y1), x2=float(d.x2), y2=float(d.y2), score=float(d.score), label=str(d.label))
        for d in detections
    ]
    assignments = assign_shelves_and_positions(det_objects, img.width, img.height)
    assignment_by_idx = {int(a["detection_index"]): a for a in assignments}
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    saved_positions: list[dict[str, Any]] = []
    for idx, det in enumerate(detections, start=1):
        x1 = int(max(0, min(det.x1, img.width - 1)))
        y1 = int(max(0, min(det.y1, img.height - 1)))
        x2 = int(max(0, min(det.x2, img.width)))
        y2 = int(max(0, min(det.y2, img.height)))
        if x2 <= x1 or y2 <= y1:
            continue
        cx_norm = ((x1 + x2) / 2.0) / max(1.0, float(img.width))
        cy_norm = ((y1 + y2) / 2.0) / max(1.0, float(img.height))
        is_own_zone = True
        if own_zone_bbox_norm is not None:
            is_own_zone = _point_in_bbox_norm(cx_norm, cy_norm, own_zone_bbox_norm)
        zone_tag = "own" if is_own_zone else "foreign"
        crop_name = f"crop_{idx:03d}.jpg"
        crop_rel = Path("crops") / crop_name
        img.crop((x1, y1, x2, y2)).save(str(run_dir / crop_rel), format="JPEG", quality=92)
        label = f"{det.label}:{det.score:.2f} [{zone_tag}]"
        box_color = "red" if is_own_zone else "orange"
        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{idx}. {label}", fill=box_color)
        saved_positions.append(
            {
                "index": idx,
                "label": str(det.label),
                "score": float(det.score),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_path": str(crop_rel).replace("\\", "/"),
                "shelf_id": int(assignment_by_idx.get(idx - 1, {}).get("shelf_id", 0) or 0),
                "position_in_shelf": int(
                    assignment_by_idx.get(idx - 1, {}).get("position_in_shelf", 0) or 0
                ),
                "is_own_zone": bool(is_own_zone),
                "zone_tag": zone_tag,
            }
        )
    logger.info("reference: после фильтра bbox сохранено позиций=%d", len(saved_positions))
    shelf_layout = _estimate_shelf_layout(saved_positions)
    if shelf_layout:
        # Полупрозрачные зоны + жирные контуры, чтобы полки были заметны на любом фоне.
        overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay, "RGBA")
        for i, sh in enumerate(shelf_layout, start=1):
            y_top = int(max(0, min(sh["y_top"], img.height - 1)))
            y_bottom = int(max(y_top + 1, min(sh["y_bottom"], img.height - 1)))
            fill_rgba = (47, 107, 255, 56) if i % 2 == 1 else (0, 181, 173, 56)
            odraw.rectangle([(0, y_top), (img.width - 1, y_bottom)], fill=fill_rgba)
        draw_img = Image.alpha_composite(draw_img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(draw_img)

        for i, sh in enumerate(shelf_layout, start=1):
            y_top = int(max(0, min(sh["y_top"], img.height - 1)))
            y_bottom = int(max(y_top + 1, min(sh["y_bottom"], img.height - 1)))
            line_color = "#0f4de2" if i % 2 == 1 else "#0a8d87"
            draw.line([(0, y_top), (img.width - 1, y_top)], fill=line_color, width=5)
            draw.line([(0, y_bottom), (img.width - 1, y_bottom)], fill=line_color, width=5)
            label = f"ПОЛКА {i} · {int(sh['count'])} шт."
            tx = 10
            ty = max(0, y_top + 6)
            draw.rectangle([(tx - 6, ty - 4), (tx + 230, ty + 24)], fill=(255, 255, 255))
            draw.text((tx, ty), label, fill=line_color)

    shelf_rows = [int(s["count"]) for s in shelf_layout]
    marked_name = "annotated.jpg"
    draw_img.save(str(run_dir / marked_name), format="JPEG", quality=92)
    logger.info(
        "reference: шаг эталона завершён за %.1f с (детекция+кропы+разметка): полок=%d annotated=%s",
        time.perf_counter() - t0,
        len(shelf_layout),
        marked_name,
    )
    own_positions_count = sum(1 for p in saved_positions if bool(p.get("is_own_zone", True)))
    foreign_positions_count = max(0, len(saved_positions) - own_positions_count)
    return {
        "positions_count": len(saved_positions),
        "own_positions_count": own_positions_count,
        "foreign_positions_count": foreign_positions_count,
        "own_zone_bbox_norm": own_zone_bbox_norm or {},
        "shelf_count": len(shelf_rows),
        "objects_per_shelf": [{"shelf": i + 1, "count": c} for i, c in enumerate(shelf_rows)],
        "positions": saved_positions,
        "annotated_image_path": marked_name,
    }


def _estimate_shelf_layout(positions: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    Грубая оценка полок по вертикальным центрам bbox:
    объединяем объекты в ряды по близости y-центров.
    """
    centers: list[tuple[float, float, float, float]] = []
    heights: list[float] = []
    for p in positions:
        bbox = p.get("bbox")
        if not isinstance(bbox, dict):
            continue
        try:
            y1 = float(bbox.get("y1", 0.0))
            y2 = float(bbox.get("y2", 0.0))
        except (TypeError, ValueError):
            continue
        h = max(1.0, y2 - y1)
        cy = (y1 + y2) / 2.0
        centers.append((cy, h, y1, y2))
        heights.append(h)

    if not centers:
        return []

    heights.sort()
    median_h = heights[len(heights) // 2]
    # Порог объединения в одну полку: доля типовой высоты объекта.
    row_merge_threshold = max(18.0, median_h * 0.60)

    centers.sort(key=lambda x: x[0])
    # sum_cy, count, y_min, y_max
    rows: list[list[float]] = []
    for cy, _h, y1, y2 in centers:
        if not rows:
            rows.append([cy, 1.0, y1, y2])
            continue
        avg_cy = rows[-1][0] / rows[-1][1]
        if abs(cy - avg_cy) <= row_merge_threshold:
            rows[-1][0] += cy
            rows[-1][1] += 1.0
            rows[-1][2] = min(rows[-1][2], y1)
            rows[-1][3] = max(rows[-1][3], y2)
        else:
            rows.append([cy, 1.0, y1, y2])

    out: list[dict[str, float]] = []
    for sum_cy, count, y_min, y_max in rows:
        out.append(
            {
                "count": float(count),
                "y_center": sum_cy / count,
                "y_top": y_min,
                "y_bottom": y_max,
            }
        )
    return out


def _db_load() -> dict[str, Any]:
    if not REFERENCE_DB_PATH.exists():
        return {}
    raw = REFERENCE_DB_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _db_save(payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _db_get_sku_runs(db: dict[str, Any], sku: str) -> list[dict[str, Any]]:
    raw = db.get(sku, {})
    if isinstance(raw, dict) and isinstance(raw.get("history"), list):
        return [r for r in raw["history"] if isinstance(r, dict)]
    if isinstance(raw, dict) and raw.get("result_dir"):
        return [raw]
    return []


def _db_set_sku_runs(db: dict[str, Any], sku: str, runs: list[dict[str, Any]]) -> None:
    hist = runs[-50:]
    latest = hist[-1] if hist else None
    db[sku] = {"latest": latest, "history": hist}


def _load_runs_from_disk(sku: str | None = None) -> list[dict[str, Any]]:
    ref_root = SKU_RESULTS_DIR / "reference"
    if not ref_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for run_dir in ref_root.iterdir():
        if not run_dir.is_dir():
            continue
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        sku_in_file = str(payload.get("sku", "")).strip()
        if sku is not None and sku_in_file != sku:
            continue
        rec = dict(payload)
        rec.setdefault("result_dir", str(Path("data/sku_results/reference") / run_dir.name))
        rec.setdefault("sku", sku_in_file)
        out.append(rec)
    out.sort(key=lambda x: str(x.get("result_dir", "")))
    return out


def _load_lm_recognition_runs_from_disk(sku: str | None = None) -> list[dict[str, Any]]:
    """Прогоны /recognize с диска: data/sku_results/lm_recognition/<run>/result.json."""
    lm_root = SKU_RESULTS_DIR / "lm_recognition"
    if not lm_root.exists():
        return []
    run_dirs = [p for p in lm_root.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name, reverse=True)
    out: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("kind") != "lm_recognition":
            continue
        sku_in_file = str(payload.get("sku", "")).strip()
        if sku is not None and sku_in_file != sku:
            continue
        rel_dir = str(payload.get("result_dir", "")).strip().replace("\\", "/")
        if not rel_dir:
            rel_dir = (Path("data/sku_results/lm_recognition") / run_dir.name).as_posix()
        visual = payload.get("visual")
        if not isinstance(visual, dict):
            visual = {}
        positions_count = int(payload.get("positions_lm_count", 0) or 0)
        per_position = payload.get("per_position")
        if not isinstance(per_position, list):
            per_position = []
        out.append(
            {
                "sku": sku_in_file,
                "result_dir": rel_dir,
                "reference_result_dir": str(payload.get("reference_result_dir", "")).replace("\\", "/"),
                "positions_count": positions_count,
                "visual": visual,
                "per_position": per_position,
            }
        )
    return out


def _get_all_runs_for_sku(db: dict[str, Any], sku: str) -> list[dict[str, Any]]:
    by_dir: dict[str, dict[str, Any]] = {}
    for r in _db_get_sku_runs(db, sku):
        rd = str(r.get("result_dir", "")).strip()
        if rd:
            by_dir[rd] = r
    for r in _load_runs_from_disk(sku):
        rd = str(r.get("result_dir", "")).strip()
        if rd:
            by_dir[rd] = r
    runs = list(by_dir.values())
    runs.sort(key=lambda x: str(x.get("result_dir", "")))
    return runs


def _result_dir_to_project_path(result_dir: str) -> Path:
    raw = (result_dir or "").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()


def _load_reference_record_from_disk(result_dir: str) -> dict[str, Any] | None:
    base = _result_dir_to_project_path(result_dir)
    f = base / "result.json"
    if not f.is_file():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _reference_run_id_from_result_dir(result_dir: str) -> tuple[str, str] | None:
    """Возвращает (category, run_folder) для /result-file/..."""
    p = Path(str(result_dir).replace("\\", "/"))
    parts = [x for x in p.parts if x]
    try:
        i = parts.index("sku_results")
    except ValueError:
        return None
    if i + 2 >= len(parts):
        return None
    return parts[i + 1], parts[i + 2]


def _file_url_under_sku_results(category: str, run_id: str, *subpath: str) -> str:
    tail = "/".join(subpath)
    return f"/result-file/{category}/{run_id}/{tail}"


def _draw_compliance_overlay(
    *,
    image: Image.Image,
    observed_positions: list[dict[str, Any]],
    deviations: list[dict[str, Any]],
    ideal_similarity_flags: list[dict[str, Any]],
    uncertainty_flags: list[dict[str, Any]],
    matching_level: str = "sku_only",
) -> Image.Image:
    base_img = image.convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Проверка бренда актуальна только когда в каталоге реально есть заполненный brand
    # хотя бы у нескольких позиций. Иначе ВСЕ удачные матчи будут жёлтыми, т.к.
    # каталог из planogram_editor не содержит brand-поля.
    any_brand_in_positions = any(
        str(p.get("predicted_brand", "") or "").strip()
        for p in observed_positions
        if isinstance(p, dict)
    )
    slot_severity: dict[tuple[int, int], str] = {}
    severity_rank = {"info": 1, "warn": 2, "fail": 3}
    for d in deviations:
        if not isinstance(d, dict):
            continue
        shelf_id = int(d.get("shelf_id", 0) or 0)
        slot_index = int(d.get("aligned_observed_slot_index", d.get("slot_index", 0)) or 0)
        if shelf_id <= 0 or slot_index <= 0:
            continue
        sev = str(d.get("visual_severity", "fail") or "fail").strip().lower()
        if sev not in severity_rank:
            sev = "fail"
        key = (shelf_id, slot_index)
        prev = slot_severity.get(key, "")
        if not prev or severity_rank[sev] > severity_rank.get(prev, 0):
            slot_severity[key] = sev
    # Визуальная "идеальная похожесть" считается до выравнивания и может давать
    # ложный red для фактически корректных позиций (особенно при match по имени).
    # Поэтому цвет рамки строим только по итоговым deviations.

    uncertainty_idx = {
        int(f.get("index", 0) or 0)
        for f in uncertainty_flags
        if isinstance(f, dict)
    }

    for p in observed_positions:
        if not isinstance(p, dict):
            continue
        bbox = p.get("bbox")
        if not isinstance(bbox, dict):
            continue
        box = _bbox_to_int_crop(bbox, base_img.width, base_img.height)
        if not box:
            continue
        x1, y1, x2, y2 = box
        shelf_id = int(p.get("shelf_id", 0) or 0)
        slot_index = int(p.get("position_in_shelf", 0) or 0)
        idx = int(p.get("index", 0) or 0)
        sev = slot_severity.get((shelf_id, slot_index), "")
        is_info = sev == "info"
        is_warn = sev == "warn"
        bad = sev == "fail"
        uncertain = idx in uncertainty_idx
        # Если сравнение на уровне бренда и бренд не определен — визуально помечаем как uncertain.
        # Только когда brand-данные реально присутствуют хотя бы у части позиций
        # (иначе для каталога из planogram_editor все OK-совпадения получат жёлтый).
        if matching_level == "brand_level" and any_brand_in_positions:
            pred_brand = str(p.get("predicted_brand", "") or "").strip()
            pred_sku = str(p.get("predicted_sku_id", "") or "").strip().lower()
            if not pred_brand and pred_sku not in {"", "unknown"} and not bad and not is_info and not is_warn:
                uncertain = True
        if bad:
            stroke_color = (217, 4, 41, 255)
            fill_color = (217, 4, 41, 52)
        elif is_info:
            stroke_color = (37, 99, 235, 255)
            fill_color = (37, 99, 235, 44)
        elif is_warn:
            stroke_color = (139, 92, 246, 255)
            fill_color = (139, 92, 246, 44)
        elif uncertain:
            stroke_color = (245, 158, 11, 255)
            fill_color = (245, 158, 11, 44)
        else:
            stroke_color = (22, 163, 74, 255)
            fill_color = (22, 163, 74, 44)
        mark = "FAIL" if bad else ("INFO" if is_info else ("WARN" if is_warn else ("UNCERTAIN" if uncertain else "OK")))
        pred = str(p.get("predicted_sku_id", "")).strip() or "unknown"
        draw.rectangle([(x1, y1), (x2, y2)], fill=fill_color, outline=stroke_color, width=4)

        label_text = f"{idx}. {mark} {pred}"
        text_bbox = draw.textbbox((0, 0), label_text)
        text_w = max(1, int(text_bbox[2] - text_bbox[0]))
        text_h = max(1, int(text_bbox[3] - text_bbox[1]))
        label_x1 = x1 + 2
        label_y1 = max(0, y1 - text_h - 8)
        label_x2 = min(base_img.width - 1, label_x1 + text_w + 8)
        label_y2 = min(base_img.height - 1, label_y1 + text_h + 4)
        # Полупрозрачная плашка, чтобы подпись читалась на любой этикетке.
        draw.rectangle([(label_x1, label_y1), (label_x2, label_y2)], fill=(15, 23, 42, 170))
        draw.text((label_x1 + 4, label_y1 + 2), label_text, fill=stroke_color)

    composed = Image.alpha_composite(base_img, overlay)
    return composed.convert("RGB")


def _record_visual(record: dict[str, Any]) -> dict[str, Any]:
    rel_dir = Path(str(record.get("result_dir", ""))).as_posix()
    ref_pos = record.get("reference_positions", {})
    pos_meta = ref_pos.get("positions", [])
    return {
        "kind": "reference",
        "input_url": f"/{rel_dir}/input.jpg".replace("/data/sku_results", "/result-file"),
        "annotated_url": f"/{rel_dir}/annotated.jpg".replace("/data/sku_results", "/result-file"),
        "analysis": {
            "shelf_count": int(ref_pos.get("shelf_count", 0) or 0),
            "positions_count": int(ref_pos.get("positions_count", 0) or 0),
            "own_positions_count": int(ref_pos.get("own_positions_count", 0) or 0),
            "foreign_positions_count": int(ref_pos.get("foreign_positions_count", 0) or 0),
            "own_zone_bbox_norm": ref_pos.get("own_zone_bbox_norm", {}),
            "objects_per_shelf": ref_pos.get("objects_per_shelf", []),
        },
        "crops": [
            {
                **p,
                "crop_url": f"/{rel_dir}/{str(p.get('crop_path', '')).lstrip('/')}".replace(
                    "/data/sku_results",
                    "/result-file",
                ),
            }
            for p in pos_meta
        ],
    }


def _classify_crops_parallel(
    lm: LMStudioClient,
    crops: list[Image.Image],
    concurrent: int,
) -> list[ItemClassification]:
    if not crops:
        return []
    if concurrent <= 1:
        return [lm.classify_crop_with_recheck(c) for c in crops]
    with ThreadPoolExecutor(max_workers=concurrent) as pool:
        return list(pool.map(lm.classify_crop_with_recheck, crops))


def _truthy_env(name: str, default: str = "0") -> bool:
    return (os.getenv(name) or default).strip().lower() in {"1", "true", "yes", "on"}


def _classify_crops_shared_groups(
    lm: LMStudioClient,
    crops: list[Image.Image],
    concurrent: int,
    similarity_threshold: float,
) -> list[ItemClassification]:
    """Один classify_crop_with_recheck на представителя кластера похожих кропов."""
    groups = cluster_crop_indices_by_similarity(crops, similarity_threshold)
    n = len(crops)
    if n == 0:
        return []

    def _one(grp: list[int]) -> tuple[list[int], ItemClassification]:
        return grp, lm.classify_crop_with_recheck(crops[grp[0]])

    if concurrent <= 1:
        pairs = [_one(g) for g in groups]
    else:
        with ThreadPoolExecutor(max_workers=concurrent) as pool:
            pairs = list(pool.map(_one, groups))

    by_idx: dict[int, ItemClassification] = {}
    for grp, ic in pairs:
        for i in grp:
            by_idx[i] = ic
    return [by_idx[i] for i in range(n)]


def _bbox_to_int_crop(bbox: dict[str, Any], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    """Пиксельные координаты кропа на изображении разбора (как в SKU110K)."""
    try:
        x1 = float(bbox.get("x1", 0))
        y1 = float(bbox.get("y1", 0))
        x2 = float(bbox.get("x2", 0))
        y2 = float(bbox.get("y2", 0))
    except (TypeError, ValueError):
        return None
    iw = max(int(img_w), 1)
    ih = max(int(img_h), 1)
    ix1 = max(0, min(int(round(x1)), iw - 1))
    iy1 = max(0, min(int(round(y1)), ih - 1))
    ix2 = max(ix1 + 1, min(int(round(x2)), iw))
    iy2 = max(iy1 + 1, min(int(round(y2)), ih))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return None
    return ix1, iy1, ix2, iy2


def _fit_image_to_cell(src: Image.Image, *, width: int, height: int) -> Image.Image:
    """Масштабирует изображение в рамку ячейки с центрированием."""
    out = Image.new("RGB", (max(1, width), max(1, height)), (245, 247, 250))
    if src.width <= 0 or src.height <= 0:
        return out
    scale = min(width / src.width, height / src.height)
    new_w = max(1, int(round(src.width * scale)))
    new_h = max(1, int(round(src.height * scale)))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    px = (width - new_w) // 2
    py = (height - new_h) // 2
    out.paste(resized, (px, py))
    return out


def _build_shelf_comparison_visuals(
    *,
    full_img: Image.Image,
    aligned_slot_rows: list[dict[str, Any]],
    observed_positions: list[dict[str, Any]],
    catalog: list[Any],
    compliance_run_dir: Path,
) -> list[dict[str, Any]]:
    """Собирает картинки по полкам: верхняя строка — эталон, нижняя — факт."""
    if not aligned_slot_rows:
        return []

    observed_by_index: dict[int, dict[str, Any]] = {}
    for pos in observed_positions:
        if not isinstance(pos, dict):
            continue
        idx = int(pos.get("index", 0) or 0)
        if idx > 0:
            observed_by_index[idx] = pos

    sku_ref_path: dict[str, Path] = {}
    for item in catalog:
        sku_id = str(getattr(item, "sku_id", "") or "")
        if not sku_id:
            continue
        ref_list = list(getattr(item, "reference_images", []) or [])
        chosen: Path | None = None
        for rel in ref_list:
            p = Path(str(rel))
            if not p.is_absolute():
                p = (BASE_DIR / p).resolve()
            if p.is_file():
                chosen = p
                break
        if chosen is not None:
            sku_ref_path[sku_id] = chosen

    ref_cache: dict[Path, Image.Image] = {}

    def _expected_thumb(expected_sku_id: str, *, cell_w: int, img_h: int) -> Image.Image:
        p = sku_ref_path.get(expected_sku_id)
        if p is None:
            ph = Image.new("RGB", (cell_w, img_h), (240, 240, 240))
            d = ImageDraw.Draw(ph)
            d.text((6, 6), expected_sku_id[:18] or "N/A", fill=(80, 80, 80))
            return ph
        src = ref_cache.get(p)
        if src is None:
            with Image.open(p) as im:
                src = im.convert("RGB").copy()
            ref_cache[p] = src
        return _fit_image_to_cell(src, width=cell_w, height=img_h)

    def _actual_thumb(observed_index: int, *, cell_w: int, img_h: int) -> Image.Image:
        pos = observed_by_index.get(int(observed_index))
        if not pos:
            ph = Image.new("RGB", (cell_w, img_h), (236, 239, 244))
            d = ImageDraw.Draw(ph)
            d.text((6, 6), "empty", fill=(91, 102, 117))
            return ph
        bbox = pos.get("bbox")
        if not isinstance(bbox, dict):
            return Image.new("RGB", (cell_w, img_h), (236, 239, 244))
        box = _bbox_to_int_crop(bbox, full_img.width, full_img.height)
        if not box:
            return Image.new("RGB", (cell_w, img_h), (236, 239, 244))
        crop = full_img.crop(box).convert("RGB")
        return _fit_image_to_cell(crop, width=cell_w, height=img_h)

    by_shelf: dict[int, list[dict[str, Any]]] = {}
    for row in aligned_slot_rows:
        if not isinstance(row, dict):
            continue
        shelf_id = int(row.get("shelf_id", 0) or 0)
        if shelf_id <= 0:
            continue
        by_shelf.setdefault(shelf_id, []).append(row)

    shelf_images: list[dict[str, Any]] = []
    cell_w = 140
    img_h = 108
    header_h = 30
    row_title_h = 20
    label_h = 18
    col_gap = 6
    side_pad = 10

    for shelf_id in sorted(by_shelf.keys()):
        rows = sorted(by_shelf[shelf_id], key=lambda r: int(r.get("slot_index", 0) or 0))
        if not rows:
            continue
        shift = int(rows[0].get("alignment_shift", 0) or 0)
        n = len(rows)
        width = side_pad * 2 + n * cell_w + max(0, n - 1) * col_gap
        height = header_h + row_title_h + img_h + label_h + row_title_h + img_h + label_h + 12
        canvas = Image.new("RGB", (width, height), (252, 253, 255))
        draw = ImageDraw.Draw(canvas)

        draw.rectangle([(0, 0), (width - 1, header_h)], fill=(18, 26, 41))
        # Используем ASCII-метки: встроенный PIL-шрифт не всегда умеет кириллицу.
        draw.text((10, 8), f"Shelf {shelf_id} | shift {shift:+d}", fill=(236, 240, 245))

        exp_y = header_h
        act_y = header_h + row_title_h + img_h + label_h
        draw.text((10, exp_y + 2), "REF", fill=(70, 88, 112))
        draw.text((10, act_y + 2), "ACT", fill=(70, 88, 112))

        for i, row in enumerate(rows):
            x = side_pad + i * (cell_w + col_gap)
            expected_sku = str(row.get("expected_sku_id", "") or "")
            observed_index = int(row.get("observed_index", 0) or 0)
            actual_sku = str(row.get("actual_sku_id", "") or "")
            status = str(row.get("status", "") or "")
            match_ok = bool(row.get("match_ok", False))
            if status == "ok":
                border = (22, 163, 74)
            elif status == "empty":
                border = (107, 114, 128)
            else:
                border = (217, 4, 41)
            exp_thumb = _expected_thumb(expected_sku, cell_w=cell_w, img_h=img_h)
            act_thumb = _actual_thumb(observed_index, cell_w=cell_w, img_h=img_h)
            canvas.paste(exp_thumb, (x, exp_y + row_title_h))
            canvas.paste(act_thumb, (x, act_y + row_title_h))
            draw.rectangle(
                [(x, act_y + row_title_h), (x + cell_w - 1, act_y + row_title_h + img_h - 1)],
                outline=border,
                width=3,
            )

            slot_idx = int(row.get("slot_index", 0) or 0)
            draw.text((x + 2, exp_y + row_title_h + img_h + 1), f"S{slot_idx}", fill=(74, 85, 104))
            actual_lbl = actual_sku if actual_sku else "empty"
            if len(actual_lbl) > 14:
                actual_lbl = actual_lbl[:14] + "…"
            draw.text((x + 2, act_y + row_title_h + img_h + 1), actual_lbl, fill=border)

            # Небольшая отметка совпадения для быстрого сканирования ряда
            if match_ok:
                draw.text((x + cell_w - 16, act_y + 2), "OK", fill=(22, 163, 74))
            elif status == "empty":
                draw.text((x + cell_w - 30, act_y + 2), "EMPTY", fill=(107, 114, 128))
            else:
                draw.text((x + cell_w - 28, act_y + 2), "MISS", fill=(217, 4, 41))

        out_name = f"shelf_compare_{shelf_id}.jpg"
        out_path = compliance_run_dir / out_name
        canvas.save(str(out_path), format="JPEG", quality=90)
        shelf_images.append(
            {
                "shelf_id": int(shelf_id),
                "alignment_shift": int(shift),
                "url": _file_url_under_sku_results("compliance", compliance_run_dir.name, out_name),
            }
        )

    return shelf_images


def _sku_detector() -> SKU110KDetector:
    return SKU110KDetector(
        repo_path=os.getenv("SKU110K_REPO_PATH", DEFAULT_REPO_PATH),
        weights_path=os.getenv("SKU110K_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH),
        python_bin=os.getenv("SKU110K_PYTHON_BIN", DEFAULT_PYTHON_BIN),
        score_threshold=float(os.getenv("SKU110K_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD))),
        run_mode=os.getenv("SKU110K_RUN_MODE", DEFAULT_RUN_MODE),
        wsl_python_bin=os.getenv("SKU110K_WSL_PYTHON_BIN", DEFAULT_WSL_PYTHON_BIN),
        docker_image=os.getenv("SKU110K_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE),
        docker_use_gpu=os.getenv("SKU110K_DOCKER_USE_GPU", str(DEFAULT_DOCKER_USE_GPU)).lower()
        in {"1", "true", "yes", "on"},
        docker_mount_host=os.getenv("SKU110K_DOCKER_MOUNT_HOST", DEFAULT_DOCKER_MOUNT_HOST),
        docker_mount_target=os.getenv("SKU110K_DOCKER_MOUNT_TARGET", DEFAULT_DOCKER_MOUNT_TARGET),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "reference_db_path": str(REFERENCE_DB_PATH),
            "sku_results_dir": str(SKU_RESULTS_DIR),
        },
    )


@app.get("/planogram/editor", response_class=HTMLResponse)
async def planogram_editor_screen(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "planogram_editor.html", {"request": request})


@app.get("/result-file/{category}/{run_id}/{filename:path}")
async def result_file(category: str, run_id: str, filename: str) -> FileResponse:
    safe_base = SKU_RESULTS_DIR.resolve()
    target = (safe_base / category / run_id / filename).resolve()
    try:
        target.relative_to(safe_base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Некорректный путь к файлу результата") from exc
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Файл результата не найден")
    return FileResponse(str(target))


@app.get("/planogram/editor/image/{planogram_id}")
async def planogram_editor_image(planogram_id: str) -> FileResponse:
    row = get_planogram(PLANOGRAM_DB_PATH, planogram_id.strip())
    if row is None or not row.image_path:
        raise HTTPException(status_code=404, detail="Image not found")
    p = Path(row.image_path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(p))


@app.get("/planogram/editor/list")
async def planogram_editor_list() -> JSONResponse:
    items = list_planograms(PLANOGRAM_DB_PATH)
    out: list[dict[str, Any]] = []
    for it in items:
        pid = str(it.get("id", ""))
        meta_exists = (PLANOGRAM_EDITOR_META_DIR / f"{pid}.json").is_file()
        out.append({**it, "has_editor_metadata": meta_exists})
    return JSONResponse({"ok": True, "items": out})


@app.get("/planogram/editor/{planogram_id}")
async def planogram_editor_get(planogram_id: str) -> JSONResponse:
    pid = planogram_id.strip()
    row = get_planogram(PLANOGRAM_DB_PATH, pid)
    if row is None:
        return JSONResponse({"ok": False, "error": "planogram not found"}, status_code=404)
    meta_path = PLANOGRAM_EDITOR_META_DIR / f"{pid}.json"
    metadata: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                metadata = raw
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return JSONResponse(
        {
            "ok": True,
            "planogram": {
                "id": row.id,
                "name": row.name,
                "csv_text": row.csv_text,
                "created_at": row.created_at,
                "image_url": f"/planogram/editor/image/{row.id}" if row.image_path else "",
            },
            "metadata": metadata,
        }
    )


@app.delete("/planogram/editor/{planogram_id}")
async def planogram_editor_delete(planogram_id: str) -> JSONResponse:
    pid = planogram_id.strip()
    if not pid:
        return JSONResponse({"ok": False, "error": "planogram_id is required"}, status_code=400)
    deleted = delete_planogram(PLANOGRAM_DB_PATH, pid)
    if not deleted:
        return JSONResponse({"ok": False, "error": "planogram not found"}, status_code=404)

    meta_path = PLANOGRAM_EDITOR_META_DIR / f"{pid}.json"
    try:
        meta_path.unlink(missing_ok=True)
    except OSError:
        pass
    ideal_dir = PLANOGRAM_EDITOR_META_DIR / "ideal_images" / pid
    if ideal_dir.is_dir():
        for p in ideal_dir.rglob("*"):
            if p.is_file():
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        for p in sorted(ideal_dir.rglob("*"), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
        try:
            ideal_dir.rmdir()
        except OSError:
            pass

    return JSONResponse({"ok": True, "deleted_planogram_id": pid})


@app.post("/planogram/editor/save")
async def planogram_editor_save(
    planogram_id: str = Form(default=""),
    name: str = Form(...),
    slots_json: str = Form(...),
    ideal_images_map: str = Form(default="[]"),
    ideal_images: list[UploadFile] = File(default=[]),
    source_image: UploadFile | None = File(default=None),
) -> JSONResponse:
    title = name.strip() or "planogram"
    try:
        raw_slots = json.loads(slots_json)
        if not isinstance(raw_slots, list):
            raise ValueError("slots_json must be a JSON array")
        slots = normalize_editor_slots(raw_slots)
        slots = renumber_slots_within_shelves(slots)
    except (ValueError, json.JSONDecodeError) as exc:
        return JSONResponse({"ok": False, "error": f"invalid slots_json: {exc}"}, status_code=400)

    image_bytes: bytes | None = None
    if source_image is not None:
        image_bytes = await source_image.read()
        if image_bytes:
            try:
                _ = _load_normalized_rgb_image(image_bytes)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({"ok": False, "error": f"invalid image: {exc}"}, status_code=400)

    csv_text = editor_slots_to_csv(slots)
    pid = planogram_id.strip()
    if pid:
        updated = update_planogram(
            PLANOGRAM_DB_PATH,
            pid,
            csv_text=csv_text,
            name=title,
            image_bytes=image_bytes,
            images_dir=PLANOGRAM_IMAGES_DIR,
        )
        if updated is None:
            return JSONResponse({"ok": False, "error": "planogram not found for update"}, status_code=404)
        stored = updated
    else:
        stored = create_planogram(
            PLANOGRAM_DB_PATH,
            name=title,
            csv_text=csv_text,
            image_bytes=image_bytes,
            images_dir=PLANOGRAM_IMAGES_DIR,
        )
    PLANOGRAM_EDITOR_META_DIR.mkdir(parents=True, exist_ok=True)
    def _canon_name(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())
    upload_map_raw = []
    try:
        parsed_map = json.loads(ideal_images_map or "[]")
        if isinstance(parsed_map, list):
            upload_map_raw = [x for x in parsed_map if isinstance(x, dict)]
    except json.JSONDecodeError:
        upload_map_raw = []
    ideal_dir = PLANOGRAM_EDITOR_META_DIR / "ideal_images" / stored.id
    ideal_dir.mkdir(parents=True, exist_ok=True)
    for m in upload_map_raw:
        try:
            slot_i = int(m.get("slot_array_index", -1))
            upload_i = int(m.get("upload_index", -1))
        except (TypeError, ValueError):
            continue
        if slot_i < 0 or slot_i >= len(slots):
            continue
        if upload_i < 0 or upload_i >= len(ideal_images):
            continue
        uf = ideal_images[upload_i]
        if uf.content_type and not str(uf.content_type).lower().startswith("image/"):
            continue
        data = await uf.read()
        if not data:
            continue
        safe = _sanitize_filename_stem(uf.filename or f"slot_{slot_i+1}")
        ext = Path(uf.filename or "").suffix.lower().strip()
        if not ext or len(ext) > 12 or not re.fullmatch(r"\.[a-z0-9]+", ext):
            ext = ".jpg"
        dest = ideal_dir / f"slot_{slot_i+1:03d}_{safe}{ext}"
        Path(dest).write_bytes(data)
        rel = dest.relative_to(BASE_DIR).as_posix()
        slots[slot_i]["reference_image_path"] = rel
        slots[slot_i]["reference_image_url"] = f"/planogram/editor/asset/{rel}"

    # Если у одинаковых item_name только часть позиций получила идеальное фото,
    # разносить путь/URL на все позиции с тем же названием.
    grouped_best: dict[str, tuple[str, str]] = {}
    for s in slots:
        key = _canon_name(s.get("item_name", ""))
        if not key:
            continue
        rp = str(s.get("reference_image_path", "")).strip()
        ru = str(s.get("reference_image_url", "")).strip()
        if rp:
            grouped_best[key] = (rp, ru)
    for s in slots:
        key = _canon_name(s.get("item_name", ""))
        if not key or key not in grouped_best:
            continue
        rp, ru = grouped_best[key]
        s["reference_image_path"] = rp
        s["reference_image_url"] = ru

    meta = {
        "kind": "planogram_editor",
        "planogram_id": stored.id,
        "name": title,
        "slots": slots,
        "reference_planogram": build_reference_planogram_json(slots),
    }
    (PLANOGRAM_EDITOR_META_DIR / f"{stored.id}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return JSONResponse(
        {
            "ok": True,
            "planogram_id": stored.id,
            "name": stored.name,
            "csv_text": stored.csv_text,
            "image_url": f"/planogram/editor/image/{stored.id}" if stored.image_path else "",
            "metadata": meta,
        }
    )


@app.get("/planogram/editor/asset/{asset_path:path}")
async def planogram_editor_asset(asset_path: str) -> FileResponse:
    p = (BASE_DIR / asset_path).resolve()
    try:
        p.relative_to(BASE_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc
    if not p.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(str(p))


@app.post("/planogram/editor/detect-sku110k")
async def planogram_editor_detect_sku110k(
    source_image: UploadFile = File(...),
) -> JSONResponse:
    img_bytes = await source_image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty image file"}, status_code=400)
    try:
        img = _load_normalized_rgb_image(img_bytes)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"invalid image: {exc}"}, status_code=400)

    detector = _sku_detector()
    run_dir = _make_unique_run_dir("planogram_editor_detect", source_image.filename or "editor")
    input_path = run_dir / "input.jpg"
    img.save(str(input_path), format="JPEG", quality=92)
    try:
        detections = detector.detect_image(str(input_path))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"SKU110K error: {exc}"}, status_code=400)

    det_objects: list[Detection] = [
        Detection(
            x1=float(d.x1),
            y1=float(d.y1),
            x2=float(d.x2),
            y2=float(d.y2),
            score=float(d.score),
            label=str(d.label),
        )
        for d in detections
    ]
    assignments = assign_shelves_and_positions(det_objects, img.width, img.height)

    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    slots: list[dict[str, Any]] = []
    for i, a in enumerate(assignments, start=1):
        d = a["detection"]
        x1 = int(max(0, min(d.x1, img.width - 1)))
        y1 = int(max(0, min(d.y1, img.height - 1)))
        x2 = int(max(0, min(d.x2, img.width)))
        y2 = int(max(0, min(d.y2, img.height)))
        if x2 <= x1 or y2 <= y1:
            continue
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text(
            (x1 + 4, max(0, y1 - 14)),
            f"#{i} P{int(a['shelf_id'])}:{int(a['position_in_shelf'])}",
            fill="red",
        )
        slots.append(
            {
                "index": i,
                "shelf_id": int(a["shelf_id"]),
                "slot_index": int(a["position_in_shelf"]),
                "item_name": f"Позиция {i}",
                "sku_id": f"sku_{i}",
                "expected_facings": 1,
                "score": float(d.score),
                "bbox_norm": {
                    "x1": x1 / max(1, img.width),
                    "y1": y1 / max(1, img.height),
                    "x2": x2 / max(1, img.width),
                    "y2": y2 / max(1, img.height),
                },
            }
        )

    annotated_path = run_dir / "annotated.jpg"
    draw_img.save(str(annotated_path), format="JPEG", quality=92)
    return JSONResponse(
        {
            "ok": True,
            "image_width": int(img.width),
            "image_height": int(img.height),
            "slots": slots,
            "visual": {
                "kind": "planogram_editor_detect",
                "annotated_url": _file_url_under_sku_results("planogram_editor_detect", run_dir.name, "annotated.jpg"),
            },
        }
    )


@app.post("/planogram/editor/enrich-lm")
async def planogram_editor_enrich_lm(
    slots_json: str = Form(...),
    source_image: UploadFile = File(...),
) -> JSONResponse:
    try:
        raw_slots = json.loads(slots_json)
        if not isinstance(raw_slots, list):
            raise ValueError("slots_json must be list")
        slots = normalize_editor_slots(raw_slots)
        slots = renumber_slots_within_shelves(slots)
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": f"invalid slots_json: {exc}"}, status_code=400)

    img_bytes = await source_image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "empty image file"}, status_code=400)
    try:
        img = _load_normalized_rgb_image(img_bytes)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"invalid image: {exc}"}, status_code=400)

    lm = LMStudioClient(
        base_url=os.getenv("LMSTUDIO_URL", DEFAULT_LMSTUDIO_URL),
        model=os.getenv("LMSTUDIO_MODEL", DEFAULT_LMSTUDIO_MODEL),
        timeout_sec=float(os.getenv("LMSTUDIO_TIMEOUT_SEC", "25")),
    )
    def _score_ic(ic: ItemClassification) -> tuple[int, float]:
        status_rank = {
            "ok": 3,
            "uncertain": 2,
            "unknown": 1,
            "lmstudio_error": 0,
        }.get(str(ic.status or "").strip().lower(), 1)
        return status_rank, float(ic.confidence or 0.0)

    def _expand_box(box: tuple[int, int, int, int], w: int, h: int, px: int, py: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        ex1 = max(0, x1 - px)
        ey1 = max(0, y1 - py)
        ex2 = min(w, x2 + px)
        ey2 = min(h, y2 + py)
        if ex2 - ex1 < 4 or ey2 - ey1 < 4:
            return box
        return ex1, ey1, ex2, ey2

    enriched: list[dict[str, Any]] = []
    for s in slots:
        b = s["bbox_norm"]
        box = _bbox_to_int_crop(
            {
                "x1": float(b["x1"]) * img.width,
                "y1": float(b["y1"]) * img.height,
                "x2": float(b["x2"]) * img.width,
                "y2": float(b["y2"]) * img.height,
            },
            img.width,
            img.height,
        )
        if not box:
            enriched.append(s)
            continue
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(3, int(round(bw * 0.10)))
        pad_y = max(3, int(round(bh * 0.12)))
        candidate_boxes = [
            _expand_box(box, img.width, img.height, pad_x, pad_y),
            box,
            _expand_box(box, img.width, img.height, pad_x * 2, pad_y * 2),
        ]
        unique_boxes: list[tuple[int, int, int, int]] = []
        seen_boxes: set[tuple[int, int, int, int]] = set()
        for cb in candidate_boxes:
            if cb not in seen_boxes:
                seen_boxes.add(cb)
                unique_boxes.append(cb)

        best_ic: ItemClassification | None = None
        for cb in unique_boxes:
            crop = img.crop(cb)
            ic = lm.classify_crop_with_recheck(crop)
            if best_ic is None or _score_ic(ic) > _score_ic(best_ic):
                best_ic = ic
            if best_ic.status == "ok" and best_ic.confidence >= 0.85:
                break
        if best_ic is None:
            enriched.append(s)
            continue

        x = dict(s)
        if best_ic.item_name and best_ic.item_name != "unknown":
            x["item_name"] = best_ic.item_name
        x["lm_confidence"] = float(best_ic.confidence)
        x["lm_status"] = best_ic.status
        enriched.append(x)
    return JSONResponse({"ok": True, "slots": enriched})


@app.post("/reference/save")
async def save_reference(
    sku: str = Form(...),
    reference_image: UploadFile = File(...),
    our_zone_norm: str = Form(""),
) -> JSONResponse:
    req_t0 = time.perf_counter()
    sku_key = sku.strip()
    if not sku_key:
        return JSONResponse({"ok": False, "error": "Название разметки не задано"}, status_code=400)

    img_bytes = await reference_image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "Пустой файл эталона"}, status_code=400)

    orig_name = reference_image.filename or "reference.jpg"
    logger.info(
        "POST /reference/save sku=%r file=%r bytes=%d",
        sku_key,
        orig_name,
        len(img_bytes),
    )

    img = _load_normalized_rgb_image(img_bytes)
    logger.info(
        "reference: изображение нормализовано %sx%s",
        img.width,
        img.height,
    )
    detector = _sku_detector()
    try:
        own_zone_bbox_norm = _parse_zone_bbox_norm(our_zone_norm)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    run_dir = _make_unique_run_dir("reference", orig_name)
    input_path = run_dir / "input.jpg"
    img.save(str(input_path), format="JPEG", quality=92)
    logger.info("reference: сохранён input.jpg run_dir=%s", run_dir)
    try:
        ref_positions_meta = _save_reference_positions_sku(
            img=img,
            image_path=input_path,
            detector=detector,
            run_dir=run_dir,
            own_zone_bbox_norm=own_zone_bbox_norm,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("POST /reference/save: ошибка SKU110K после %.1f с", time.perf_counter() - req_t0)
        return JSONResponse({"ok": False, "error": f"SKU110K ошибка: {exc}"}, status_code=400)

    record = {
        "sku": sku_key,
        "reference_filename": orig_name,
        "reference_width": img.width,
        "reference_height": img.height,
        "reference_detection": {
            "status": "ok",
            "positions_count": ref_positions_meta.get("positions_count", 0),
            "own_positions_count": ref_positions_meta.get("own_positions_count", 0),
            "foreign_positions_count": ref_positions_meta.get("foreign_positions_count", 0),
        },
        "reference_classification": {
            "item_name": "by_sku_detection",
            "normalized_name": "by_sku_detection",
            "raw_name": "by_sku_detection",
            "confidence": 1.0,
            "status": "by_sku_detection",
            "positions_count": ref_positions_meta.get("positions_count", 0),
        },
        "reference_positions": ref_positions_meta,
        "result_dir": str(run_dir.relative_to(BASE_DIR)),
    }
    (run_dir / "result.json").write_text(
        json.dumps({"kind": "reference", **record}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    db = _db_load()
    runs = _db_get_sku_runs(db, sku_key)
    runs.append(record)
    _db_set_sku_runs(db, sku_key, runs)
    _db_save(db)
    visual = _record_visual(record)
    logger.info(
        "POST /reference/save ok sku=%r positions=%d за %.1f с result_dir=%s",
        sku_key,
        int(ref_positions_meta.get("positions_count", 0)),
        time.perf_counter() - req_t0,
        record.get("result_dir", ""),
    )
    return JSONResponse(
        {
            "ok": True,
            "sku": sku_key,
            "saved": record,
            "result_dir": record["result_dir"],
            "visual": visual,
        }
    )


@app.get("/reference/{sku}")
async def get_reference(sku: str) -> JSONResponse:
    db = _db_load()
    runs = _get_all_runs_for_sku(db, sku.strip())
    row = runs[-1] if runs else None
    if row is None:
        return JSONResponse({"ok": False, "error": "Разбор фото не найден"}, status_code=404)
    return JSONResponse({"ok": True, "reference": row})


@app.get("/reference/history/{sku}")
async def get_reference_history(sku: str) -> JSONResponse:
    db = _db_load()
    runs = _get_all_runs_for_sku(db, sku.strip())
    rows = [
        {
            "result_dir": r.get("result_dir", ""),
            "reference_filename": r.get("reference_filename", ""),
            "positions_count": int(r.get("reference_detection", {}).get("positions_count", 0)),
            "visual": _record_visual(r),
        }
        for r in reversed(runs)
    ]
    return JSONResponse({"ok": True, "sku": sku.strip(), "items": rows})


@app.get("/reference-folder/history")
async def get_reference_history_folder(sku: str = "") -> JSONResponse:
    sku_key = sku.strip()
    runs = _load_runs_from_disk(sku_key if sku_key else None)
    rows = [
        {
            "sku": r.get("sku", ""),
            "result_dir": r.get("result_dir", ""),
            "reference_filename": r.get("reference_filename", ""),
            "positions_count": int(r.get("reference_detection", {}).get("positions_count", 0)),
            "visual": _record_visual(r),
        }
        for r in reversed(runs)
    ]
    return JSONResponse({"ok": True, "sku": sku_key, "items": rows})


@app.get("/lm-recognition/history")
async def get_lm_recognition_history(sku: str = "") -> JSONResponse:
    """Список сохранённых прогонов распознавания LM (шаг 2)."""
    sku_key = sku.strip()
    runs = _load_lm_recognition_runs_from_disk(sku_key if sku_key else None)
    rows = [
        {
            "sku": r.get("sku", ""),
            "result_dir": r.get("result_dir", ""),
            "reference_result_dir": r.get("reference_result_dir", ""),
            "positions_count": int(r.get("positions_count", 0)),
            "visual": r.get("visual") or {},
            "per_position": r.get("per_position") or [],
        }
        for r in runs
    ]
    return JSONResponse({"ok": True, "sku": sku_key, "items": rows})


def _resolve_reference_record(reference_result_dir: str) -> dict[str, Any] | None:
    """Найти разбор шага 1 только по пути result_dir; название распознавания (sku) не проверяется."""
    selected = reference_result_dir.strip()
    if not selected:
        return None
    disk = _load_reference_record_from_disk(selected)
    if disk is None:
        return None
    if str(disk.get("kind", "")).strip() != "reference":
        return None
    ref_pos = disk.get("reference_positions")
    if not isinstance(ref_pos, dict):
        return None
    positions = ref_pos.get("positions")
    if not isinstance(positions, list) or not positions:
        return None
    return disk


@app.post("/recognize")
async def recognize_reference_crops(
    sku: str = Form(...),
    reference_result_dir: str = Form(...),
) -> JSONResponse:
    """Шаг 2: кропы из выбранного разбора (шаг 1) → LM Studio, без отдельного фото стеллажа."""
    req_t0 = time.perf_counter()
    sku_key = sku.strip()
    if not sku_key:
        return JSONResponse({"ok": False, "error": "Название распознавания не задано"}, status_code=400)

    selected_dir = reference_result_dir.strip()
    if not selected_dir:
        return JSONResponse({"ok": False, "error": "Выберите сохранённый разбор (reference_result_dir)"}, status_code=400)

    logger.info(
        "POST /recognize sku=%r reference_result_dir=%r",
        sku_key,
        selected_dir,
    )

    reference = _resolve_reference_record(selected_dir)
    if reference is None:
        return JSONResponse(
            {
                "ok": False,
                "error": "Разбор не найден: проверьте путь к папке шага 1 и наличие result.json с позициями",
            },
            status_code=404,
        )

    ref_base = _result_dir_to_project_path(str(reference.get("result_dir", "")))
    ref_run_id = ref_base.name
    input_path = ref_base / "input.jpg"
    if not input_path.is_file():
        return JSONResponse({"ok": False, "error": "В папке разбора нет input.jpg"}, status_code=400)

    with Image.open(input_path) as src:
        full_img = ImageOps.exif_transpose(src)
        if full_img.mode != "RGB":
            full_img = full_img.convert("RGB")
        full_img = full_img.copy()

    positions_src = reference.get("reference_positions", {}).get("positions", [])
    if not isinstance(positions_src, list) or not positions_src:
        return JSONResponse({"ok": False, "error": "В выбранном разборе нет позиций"}, status_code=400)

    max_raw = (os.getenv("LM_RECOGNIZE_MAX_POSITIONS") or os.getenv("LM_ASSESS_MAX_POSITIONS") or "0").strip()
    max_positions = int(max_raw) if max_raw.isdigit() else 0
    if max_positions > 0:
        positions_src = positions_src[:max_positions]

    lm_timeout = float(os.getenv("LMSTUDIO_TIMEOUT_SEC", "25"))
    lm = LMStudioClient(
        base_url=os.getenv("LMSTUDIO_URL", DEFAULT_LMSTUDIO_URL),
        model=os.getenv("LMSTUDIO_MODEL", DEFAULT_LMSTUDIO_MODEL),
        timeout_sec=lm_timeout,
    )
    logger.info(
        "recognize: LM base_url=%s model=%s timeout_sec=%s",
        os.getenv("LMSTUDIO_URL", DEFAULT_LMSTUDIO_URL),
        os.getenv("LMSTUDIO_MODEL", DEFAULT_LMSTUDIO_MODEL),
        lm_timeout,
    )

    stem = f"{sku_key}_{ref_run_id}"
    run_dir = _make_unique_run_dir("lm_recognition", stem)
    lm_run_id = run_dir.name
    (run_dir / "crops").mkdir(parents=True, exist_ok=True)

    draw_img = full_img.copy()
    draw = ImageDraw.Draw(draw_img)

    tasks: list[dict[str, Any]] = []
    for pos in positions_src:
        if not isinstance(pos, dict):
            continue
        idx = int(pos.get("index", 0) or 0)
        crop_rel_raw = pos.get("crop_path")
        crop_rel_str = str(crop_rel_raw).strip().replace("\\", "/") if crop_rel_raw else ""

        crop_pil: Image.Image | None = None
        if crop_rel_str:
            p_crop = ref_base / Path(crop_rel_str)
            if p_crop.is_file():
                with Image.open(p_crop) as csrc:
                    crop_pil = csrc.convert("RGB").copy()

        bbox_ref = pos.get("bbox")
        if crop_pil is None and isinstance(bbox_ref, dict):
            box = _bbox_to_int_crop(bbox_ref, full_img.width, full_img.height)
            if box:
                crop_pil = full_img.crop(box)

        if crop_pil is None:
            continue

        tasks.append(
            {
                "pos": pos,
                "index": idx,
                "crop_pil": crop_pil,
                "crop_rel_str": crop_rel_str,
                "bbox_ref": bbox_ref,
            }
        )

    concurrent_raw = (os.getenv("LM_CONCURRENT") or "1").strip()
    try:
        concurrent = int(concurrent_raw)
    except ValueError:
        concurrent = 1
    concurrent = max(1, min(32, concurrent))

    crops_for_lm = [t["crop_pil"] for t in tasks]
    try:
        sim_th = float((os.getenv("SIMILARITY_THRESHOLD") or "0.88").strip())
    except ValueError:
        sim_th = 0.88

    batch_single = _truthy_env("LM_BATCH_CLASSIFY_SINGLE_REQUEST", default="1")
    shared_group = _truthy_env("LM_SHARED_CLASSIFY_PER_SIMILARITY_GROUP")
    logger.info(
        "recognize: кропов для LM=%d concurrent=%d batch_single=%s shared_group=%s sim_th=%s ref_run=%s",
        len(crops_for_lm),
        concurrent,
        batch_single,
        shared_group,
        sim_th,
        ref_run_id,
    )
    t_lm = time.perf_counter()
    if batch_single and crops_for_lm:
        lm_results = lm.classify_crops_batch_chunked(crops_for_lm)
    elif shared_group and crops_for_lm:
        lm_results = _classify_crops_shared_groups(lm, crops_for_lm, concurrent, sim_th)
    else:
        lm_results = _classify_crops_parallel(lm, crops_for_lm, concurrent)
    logger.info(
        "recognize: вызовы LM завершены за %.1f с (ответов=%d)",
        time.perf_counter() - t_lm,
        len(lm_results),
    )

    per_position: list[dict[str, Any]] = []
    for task, lm_res in zip(tasks, lm_results, strict=True):
        pos = task["pos"]
        idx = task["index"]
        crop_rel_str = task["crop_rel_str"]
        bbox_ref = task["bbox_ref"]
        crop_pil = task["crop_pil"]

        out_crop_rel = Path("crops") / f"crop_{idx:03d}.jpg"
        crop_pil.save(str(run_dir / out_crop_rel), format="JPEG", quality=92)
        crop_url = _file_url_under_sku_results(
            "lm_recognition", lm_run_id, str(out_crop_rel).replace("\\", "/")
        )
        crop_saved_rel = str(out_crop_rel).replace("\\", "/")

        if isinstance(bbox_ref, dict):
            box_d = _bbox_to_int_crop(bbox_ref, full_img.width, full_img.height)
            if box_d:
                sx1, sy1, sx2, sy2 = box_d
                short_name = (lm_res.item_name or "?")[:28]
                is_own_zone = bool(pos.get("is_own_zone", True))
                zone_tag = "own" if is_own_zone else "foreign"
                label_txt = f"{idx}. {short_name} ({lm_res.confidence:.2f}) [{zone_tag}]"
                box_color = "lime" if is_own_zone else "orange"
                draw.rectangle([(sx1, sy1), (sx2, sy2)], outline=box_color, width=3)
                draw.text((sx1 + 4, max(0, sy1 - 14)), label_txt, fill=box_color)

        per_position.append(
            {
                "index": idx,
                "reference_bbox": bbox_ref if isinstance(bbox_ref, dict) else {},
                "sku110k_label": str(pos.get("label", "")),
                "sku110k_score": float(pos.get("score", 0.0) or 0.0),
                "is_own_zone": bool(pos.get("is_own_zone", True)),
                "zone_tag": str(pos.get("zone_tag", "own") or "own"),
                "crop_path": crop_saved_rel,
                "reference_crop_path": crop_rel_str,
                "crop_url": crop_url,
                "lm": {
                    "item_name": lm_res.item_name,
                    "normalized_name": lm_res.normalized_name,
                    "raw_name": lm_res.raw_name,
                    "confidence": lm_res.confidence,
                    "status": lm_res.status,
                },
            }
        )

    if not per_position:
        return JSONResponse(
            {"ok": False, "error": "Не удалось получить ни одного кропа для распознавания"},
            status_code=400,
        )

    draw_img.save(str(run_dir / "annotated_lm.jpg"), format="JPEG", quality=92)

    rel_to_base = run_dir.relative_to(BASE_DIR).as_posix()
    visual: dict[str, Any] = {
        "kind": "lm_recognition",
        "input_url": _file_url_under_sku_results("reference", ref_run_id, "input.jpg"),
        "annotated_url": _file_url_under_sku_results("lm_recognition", lm_run_id, "annotated_lm.jpg"),
        "positions": per_position,
    }

    payload: dict[str, Any] = {
        "ok": True,
        "sku": sku_key,
        "reference_markup_sku": str(reference.get("sku", "")).strip(),
        "reference_result_dir": str(reference.get("result_dir", "")),
        "positions_lm_count": len(per_position),
        "per_position": per_position,
        "visual": visual,
        "result_dir": rel_to_base,
    }
    (run_dir / "result.json").write_text(
        json.dumps({"kind": "lm_recognition", **payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "POST /recognize ok positions_lm=%d за %.1f с result_dir=%s",
        len(per_position),
        time.perf_counter() - req_t0,
        rel_to_base,
    )
    return JSONResponse(payload)


@app.post("/compliance/check")
async def check_planogram_compliance(request: Request) -> JSONResponse:
    """
    Шаг 3: сравнение выкладки с эталоном.
    Ожидает JSON:
    {
      "reference_result_dir": "data/sku_results/reference/<run>",
      "reference_planogram": {"slots":[...]},
      "sku_catalog": {"items":[...]},
      "options": {
        "confidence_threshold": 0.62,
        "llm_name_weight": 0.12,
        "presence_weight": 0.4,
        "position_weight": 0.35,
        "facings_weight": 0.25,
        "pass_threshold": 80,
        "matching_level": "brand_level",
        "foreign_sku_policy": "info_only",
        "ideal_similarity_threshold": 0.55
      }
    }
    """
    req_t0 = time.perf_counter()
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"ok": False, "error": "JSON payload expected"}, status_code=400)

    ref_dir = str(payload.get("reference_result_dir", "")).strip()
    if not ref_dir:
        return JSONResponse({"ok": False, "error": "reference_result_dir is required"}, status_code=400)

    reference = _resolve_reference_record(ref_dir)
    if reference is None:
        return JSONResponse(
            {"ok": False, "error": "reference_result_dir not found or has no positions"},
            status_code=404,
        )

    logger.info("POST /compliance/check reference_result_dir=%r", ref_dir)

    try:
        reference_planogram_payload = payload.get("reference_planogram", {})
        sku_catalog_payload = payload.get("sku_catalog", {})
        if not isinstance(reference_planogram_payload, dict) or not isinstance(sku_catalog_payload, dict):
            raise ValueError("reference_planogram and sku_catalog must be objects")
        reference_slots = parse_reference_planogram(reference_planogram_payload)
        catalog = parse_sku_catalog(sku_catalog_payload)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    logger.info(
        "compliance: слотов в эталоне=%d позиций SKU в каталоге=%d",
        len(reference_slots),
        len(catalog),
    )

    opts = payload.get("options", {})
    if not isinstance(opts, dict):
        opts = {}
    confidence_threshold = float(opts.get("confidence_threshold", 0.62) or 0.62)
    llm_name_weight = float(opts.get("llm_name_weight", 0.12) or 0.12)
    presence_weight = float(opts.get("presence_weight", 0.4) or 0.4)
    position_weight = float(opts.get("position_weight", 0.35) or 0.35)
    facings_weight = float(opts.get("facings_weight", 0.25) or 0.25)
    pass_threshold = float(opts.get("pass_threshold", 80.0) or 80.0)
    matching_level = str(opts.get("matching_level", "sku_only") or "sku_only").strip().lower()
    foreign_sku_policy = str(opts.get("foreign_sku_policy", "hard_fail") or "hard_fail").strip().lower()
    if matching_level not in {"sku_only", "brand_level"}:
        matching_level = "sku_only"
    if foreign_sku_policy not in {"hard_fail", "soft_substitute", "info_only"}:
        foreign_sku_policy = "hard_fail"
    ideal_similarity_threshold = float(opts.get("ideal_similarity_threshold", 0.55) or 0.55)
    ideal_similarity_threshold = max(0.0, min(1.0, ideal_similarity_threshold))

    ref_base = _result_dir_to_project_path(str(reference.get("result_dir", "")))
    positions_src = reference.get("reference_positions", {}).get("positions", [])
    if not isinstance(positions_src, list) or not positions_src:
        return JSONResponse({"ok": False, "error": "No positions in reference_result_dir"}, status_code=400)

    # Автоматически подгружаем LM-имена из шага 2 (если такой прогон существует).
    # Ищем последний lm_recognition-прогон, ссылающийся на тот же reference_result_dir.
    lm_hint_by_index: dict[int, str] = {}
    _lm_run_dir_found: str = ""
    try:
        lm_recognition_root = SKU_RESULTS_DIR / "lm_recognition"
        if lm_recognition_root.is_dir():
            target_ref_dir = str(reference.get("result_dir", "")).replace("\\", "/")
            lm_runs = sorted(lm_recognition_root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
            for lm_run in lm_runs[:20]:  # проверяем последние 20 прогонов
                lm_result_path = lm_run / "result.json"
                if not lm_result_path.is_file():
                    continue
                lm_data = json.loads(lm_result_path.read_text(encoding="utf-8"))
                lm_ref = str(lm_data.get("reference_result_dir", "")).replace("\\", "/")
                if lm_ref and (lm_ref == target_ref_dir or lm_ref.endswith(target_ref_dir.split("/")[-1])):
                    for pp in lm_data.get("per_position", []):
                        if not isinstance(pp, dict):
                            continue
                        idx = int(pp.get("index", 0) or 0)
                        if idx <= 0:
                            continue
                        lm_field = pp.get("lm", {})
                        if not isinstance(lm_field, dict):
                            continue
                        name = str(lm_field.get("item_name", "")).strip()
                        status = str(lm_field.get("status", "")).strip().lower()
                        # Включаем ВСЕ имена (в т.ч. «Неизвестный бренд»): match_by_lm_name
                        # корректно вернёт «foreign» для чужих брендов без дополнительной логики.
                        if name and status == "ok":
                            lm_hint_by_index[idx] = name
                    _lm_run_dir_found = lm_run.name
                    break
    except Exception as _lm_exc:
        logger.warning("compliance: не удалось загрузить LM-имена: %s", _lm_exc)

    lm_names_found = len(lm_hint_by_index)
    lm_name_weight_effective = llm_name_weight
    if lm_names_found > 0:
        # При наличии LM-имён повышаем их вес: они гораздо точнее визуального матчера
        lm_name_weight_effective = max(llm_name_weight, 0.75)
        logger.info(
            "compliance: загружены LM-имена из %s (%d позиций), weight=%.2f",
            _lm_run_dir_found, lm_names_found, lm_name_weight_effective,
        )

    try:
        t_emb = time.perf_counter()
        ref_embeddings = build_reference_embeddings(catalog, base_dir=BASE_DIR)
        logger.info("compliance: эмбеддинги референсов за %.1f с", time.perf_counter() - t_emb)
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    matchable_sku_ids = set(ref_embeddings.keys())
    catalog_for_matching = [item for item in catalog if item.sku_id in matchable_sku_ids]
    if not catalog_for_matching:
        return JSONResponse(
            {
                "ok": False,
                "error": "В sku_catalog нет ни одного SKU с доступными reference_images",
            },
            status_code=400,
        )
    catalog_without_refs = [item.sku_id for item in catalog if item.sku_id not in matchable_sku_ids]
    similar_groups = infer_similar_sku_groups(catalog)

    with Image.open(ref_base / "input.jpg") as src:
        full_img = ImageOps.exif_transpose(src)
        if full_img.mode != "RGB":
            full_img = full_img.convert("RGB")
        full_img = full_img.copy()

    # Совместимость со старыми/частично заполненными result.json шага 1:
    # если часть shelf_id/position_in_shelf отсутствует, восстанавливаем по bbox.
    slot_assignment_source = "original"
    recovered_slots_count = 0
    total_slots_with_bbox = 0
    missing_slot_indices: set[int] = set()
    det_objects: list[Detection] = []
    det_indices: list[int] = []
    for i, pos in enumerate(positions_src):
        if not isinstance(pos, dict):
            continue
        bbox = pos.get("bbox")
        if not isinstance(bbox, dict):
            continue
        total_slots_with_bbox += 1
        sid = int(pos.get("shelf_id", 0) or 0)
        pis = int(pos.get("position_in_shelf", 0) or 0)
        if sid <= 0 or pis <= 0:
            missing_slot_indices.add(i)
        try:
            det_objects.append(
                Detection(
                    x1=float(bbox.get("x1", 0)),
                    y1=float(bbox.get("y1", 0)),
                    x2=float(bbox.get("x2", 0)),
                    y2=float(bbox.get("y2", 0)),
                    score=float(pos.get("score", 1.0) or 1.0),
                    label=str(pos.get("label", "item") or "item"),
                )
            )
            det_indices.append(i)
        except (TypeError, ValueError):
            continue

    if missing_slot_indices and det_objects:
        recovered = assign_shelves_and_positions(det_objects, full_img.width, full_img.height)
        for a in recovered:
            di = int(a.get("detection_index", -1) or -1)
            if di < 0 or di >= len(det_indices):
                continue
            src_idx = det_indices[di]
            if src_idx not in missing_slot_indices:
                continue
            p = positions_src[src_idx]
            if not isinstance(p, dict):
                continue
            p["shelf_id"] = int(a.get("shelf_id", 0) or 0)
            p["position_in_shelf"] = int(a.get("position_in_shelf", 0) or 0)
            recovered_slots_count += 1
        if recovered_slots_count > 0:
            slot_assignment_source = "recovered"

    observed_positions: list[dict[str, Any]] = []
    expected_by_slot = {(s.shelf_id, s.slot_index): s.sku_id for s in reference_slots}
    ideal_similarity_flags: list[dict[str, Any]] = []
    t_match = time.perf_counter()
    for pos in positions_src:
        if not isinstance(pos, dict):
            continue
        idx = int(pos.get("index", 0) or 0)
        shelf_id = int(pos.get("shelf_id", 0) or 0)
        position_in_shelf = int(pos.get("position_in_shelf", 0) or 0)
        bbox_ref = pos.get("bbox")
        if not isinstance(bbox_ref, dict):
            continue
        box = _bbox_to_int_crop(bbox_ref, full_img.width, full_img.height)
        if not box:
            continue
        crop = full_img.crop(box)
        # LM-имя: сначала из объединённого прогона шага 2, затем из inline поля позиции
        llm_hint = lm_hint_by_index.get(idx, "") or str(pos.get("lm_item_name", "")).strip()
        expected_sku = expected_by_slot.get((shelf_id, position_in_shelf), "")
        ideal_similarity = score_crop_against_sku(crop, expected_sku, ref_embeddings) if expected_sku else 0.0
        is_similar_to_ideal = bool(expected_sku and ideal_similarity >= ideal_similarity_threshold)
        if expected_sku and not is_similar_to_ideal:
            ideal_similarity_flags.append(
                {
                    "index": idx,
                    "shelf_id": shelf_id,
                    "position_in_shelf": position_in_shelf,
                    "expected_sku_id": expected_sku,
                    "ideal_similarity": float(ideal_similarity),
                    "ideal_similarity_threshold": float(ideal_similarity_threshold),
                    "reason": "low_ideal_similarity",
                }
            )
        # Основной путь: если LLM-имя есть — прямой name-маппинг (точнее гистограмм).
        # Fallback: визуальный матчер для позиций без LLM-имени.
        if llm_hint:
            m = match_by_lm_name(llm_hint, catalog)
        else:
            m = match_sku_for_crop(
                crop,
                catalog_for_matching,
                ref_embeddings,
                confidence_threshold=confidence_threshold,
                similar_groups=similar_groups,
                llm_name_hint="",
                llm_name_weight=llm_name_weight,
            )
        observed_positions.append(
            {
                "index": idx,
                "shelf_id": shelf_id,
                "position_in_shelf": position_in_shelf,
                "bbox": bbox_ref,
                "predicted_sku_id": m["predicted_sku_id"],
                "predicted_name": m["predicted_name"],
                "predicted_brand": m["predicted_brand"],
                "confidence": m["confidence"],
                "top_k": m["top_k"],
                "status": m["status"],
                "observed_facings": 1,
                "expected_sku_id": expected_sku,
                "ideal_similarity": float(ideal_similarity),
                "is_similar_to_ideal_position": is_similar_to_ideal,
            }
        )

    logger.info(
        "compliance: сопоставление кропов завершено за %.1f с позиций=%d",
        time.perf_counter() - t_match,
        len(observed_positions),
    )

    observed_runs = build_observed_planogram(observed_positions)
    by_slot_run = {(int(r["shelf_id"]), int(r["slot_index"])): r for r in observed_runs}
    for p in observed_positions:
        k = (int(p.get("shelf_id", 0)), int(p.get("position_in_shelf", 0)))
        p["observed_facings"] = int(by_slot_run.get(k, {}).get("observed_facings", 1))

    metrics = compare_planograms_step3(
        reference_slots=reference_slots,
        observed_positions=observed_positions,
        presence_weight=presence_weight,
        position_weight=position_weight,
        facings_weight=facings_weight,
        catalog=catalog,
        matching_level=matching_level,
        foreign_sku_policy=foreign_sku_policy,
    )
    uncertainty_flags = collect_uncertainty_flags(
        observed_positions,
        confidence_threshold=confidence_threshold,
    )
    compliance_run_dir = _make_unique_run_dir("compliance", ref_base.name)
    compliance_input = compliance_run_dir / "input.jpg"
    full_img.save(str(compliance_input), format="JPEG", quality=92)
    compliance_overlay = _draw_compliance_overlay(
        image=full_img,
        observed_positions=observed_positions,
        deviations=metrics["deviations"],
        ideal_similarity_flags=ideal_similarity_flags,
        uncertainty_flags=uncertainty_flags,
        matching_level=matching_level,
    )
    compliance_overlay.save(str(compliance_run_dir / "annotated_compliance.jpg"), format="JPEG", quality=92)
    aligned_slot_rows = metrics.get("aligned_slot_rows", [])
    shelf_compare_images = _build_shelf_comparison_visuals(
        full_img=full_img,
        aligned_slot_rows=aligned_slot_rows if isinstance(aligned_slot_rows, list) else [],
        observed_positions=observed_positions,
        catalog=catalog,
        compliance_run_dir=compliance_run_dir,
    )

    score = float(metrics["compliance_score"])
    result = {
        "ok": True,
        "reference_result_dir": str(reference.get("result_dir", "")),
        "compliance_score": score,
        "status": pass_fail_from_score(score, pass_threshold),
        "metrics": {
            "presence_ratio": metrics["presence_ratio"],
            "position_ratio": metrics["position_ratio"],
            "facings_ratio": metrics["facings_ratio"],
            "slot_alignment_shift_by_shelf": metrics.get("slot_alignment_shift_by_shelf", {}),
            "alignment_debug_by_shelf": metrics.get("alignment_debug_by_shelf", {}),
        },
        "deviations": metrics["deviations"],
        "foreign_brand_count": metrics.get("foreign_brand_count", 0),
        "brand_substitute_count": metrics.get("brand_substitute_count", 0),
        "info_only_deviation_count": metrics.get("info_only_deviation_count", 0),
        "catalog_items_without_reference_images": catalog_without_refs,
        "ideal_similarity_flags": ideal_similarity_flags,
        "uncertainty_flags": uncertainty_flags,
        "observed_positions": observed_positions,
        "aligned_slot_rows": aligned_slot_rows,
        "diagnostics": {
            "slot_assignment_source": slot_assignment_source,
            "recovered_slots_count": int(recovered_slots_count),
            "total_slots_with_bbox": int(total_slots_with_bbox),
            "missing_slots_before_recovery": int(len(missing_slot_indices)),
            "observed_positions_with_valid_slot": int(
                sum(
                    1
                    for p in observed_positions
                    if int(p.get("shelf_id", 0) or 0) > 0 and int(p.get("position_in_shelf", 0) or 0) > 0
                )
            ),
            "lm_names_loaded": int(lm_names_found),
            "lm_run_dir": _lm_run_dir_found,
            "lm_name_weight_used": float(lm_name_weight_effective),
        },
        "visual": {
            "kind": "compliance",
            "input_url": _file_url_under_sku_results("compliance", compliance_run_dir.name, "input.jpg"),
            "annotated_url": _file_url_under_sku_results(
                "compliance", compliance_run_dir.name, "annotated_compliance.jpg"
            ),
            "shelf_compare_images": shelf_compare_images,
        },
    }
    logger.info(
        "POST /compliance/check ok score=%.2f status=%s за %.1f с deviations=%d",
        score,
        result["status"],
        time.perf_counter() - req_t0,
        len(metrics["deviations"]),
    )
    return JSONResponse(result)


@app.post("/compliance/calibrate")
async def calibrate_compliance(request: Request) -> JSONResponse:
    payload = await request.json()
    if not isinstance(payload, dict):
        return JSONResponse({"ok": False, "error": "JSON payload expected"}, status_code=400)
    labeled_samples = payload.get("labeled_samples", [])
    if not isinstance(labeled_samples, list):
        return JSONResponse({"ok": False, "error": "labeled_samples must be list"}, status_code=400)
    baseline = calibrate_thresholds_and_weights(labeled_samples)
    return JSONResponse({"ok": True, "baseline": baseline})
