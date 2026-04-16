from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw, ImageOps

from app.lmstudio_client import ItemClassification, LMStudioClient
from app.similarity import cluster_crop_indices_by_similarity
from app.sku110k_adapter import SKU110KDetector

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REFERENCE_DB_PATH = DATA_DIR / "reference_by_sku.json"
SKU_RESULTS_DIR = DATA_DIR / "sku_results"
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


def _save_reference_positions_sku(
    *,
    img: Image.Image,
    image_path: Path,
    detector: SKU110KDetector,
    run_dir: Path,
) -> dict[str, Any]:
    detections = detector.detect_image(str(image_path))
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
        crop_name = f"crop_{idx:03d}.jpg"
        crop_rel = Path("crops") / crop_name
        img.crop((x1, y1, x2, y2)).save(str(run_dir / crop_rel), format="JPEG", quality=92)
        label = f"{det.label}:{det.score:.2f}"
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1 + 4, max(0, y1 - 14)), f"{idx}. {label}", fill="red")
        saved_positions.append(
            {
                "index": idx,
                "label": str(det.label),
                "score": float(det.score),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "crop_path": str(crop_rel).replace("\\", "/"),
            }
        )
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
    return {
        "positions_count": len(saved_positions),
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


@app.post("/reference/save")
async def save_reference(
    sku: str = Form(...),
    reference_image: UploadFile = File(...),
) -> JSONResponse:
    sku_key = sku.strip()
    if not sku_key:
        return JSONResponse({"ok": False, "error": "Название разметки не задано"}, status_code=400)

    img_bytes = await reference_image.read()
    if not img_bytes:
        return JSONResponse({"ok": False, "error": "Пустой файл эталона"}, status_code=400)

    img = _load_normalized_rgb_image(img_bytes)
    detector = _sku_detector()

    orig_name = reference_image.filename or "reference.jpg"
    run_dir = _make_unique_run_dir("reference", orig_name)
    input_path = run_dir / "input.jpg"
    img.save(str(input_path), format="JPEG", quality=92)
    try:
        ref_positions_meta = _save_reference_positions_sku(
            img=img,
            image_path=input_path,
            detector=detector,
            run_dir=run_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"SKU110K ошибка: {exc}"}, status_code=400)

    record = {
        "sku": sku_key,
        "reference_filename": orig_name,
        "reference_width": img.width,
        "reference_height": img.height,
        "reference_detection": {
            "status": "ok",
            "positions_count": ref_positions_meta.get("positions_count", 0),
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


def _resolve_reference_record(sku_key: str, reference_result_dir: str, db: dict[str, Any]) -> dict[str, Any] | None:
    selected = reference_result_dir.strip()
    if not selected:
        return None
    want = _result_dir_to_project_path(selected)
    runs = _get_all_runs_for_sku(db, sku_key)
    for r in runs:
        try:
            if _result_dir_to_project_path(str(r.get("result_dir", ""))) == want:
                return r
        except OSError:
            continue
    disk = _load_reference_record_from_disk(selected)
    if disk is None:
        return None
    if str(disk.get("sku", "")).strip() != sku_key:
        return None
    return disk


@app.post("/recognize")
async def recognize_reference_crops(
    sku: str = Form(...),
    reference_result_dir: str = Form(...),
) -> JSONResponse:
    """Шаг 2: кропы из выбранного разбора (шаг 1) → LM Studio, без отдельного фото стеллажа."""
    sku_key = sku.strip()
    if not sku_key:
        return JSONResponse({"ok": False, "error": "Название распознавания не задано"}, status_code=400)

    selected_dir = reference_result_dir.strip()
    if not selected_dir:
        return JSONResponse({"ok": False, "error": "Выберите сохранённый разбор (reference_result_dir)"}, status_code=400)

    db = _db_load()
    reference = _resolve_reference_record(sku_key, selected_dir, db)
    if reference is None:
        return JSONResponse(
            {
                "ok": False,
                "error": "Разбор не найден или название распознавания не совпадает с сохранённой разметкой",
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

    batch_single = _truthy_env("LM_BATCH_CLASSIFY_SINGLE_REQUEST")
    shared_group = _truthy_env("LM_SHARED_CLASSIFY_PER_SIMILARITY_GROUP")
    if batch_single and crops_for_lm:
        lm_results = lm.classify_crops_batch_chunked(crops_for_lm)
    elif shared_group and crops_for_lm:
        lm_results = _classify_crops_shared_groups(lm, crops_for_lm, concurrent, sim_th)
    else:
        lm_results = _classify_crops_parallel(lm, crops_for_lm, concurrent)

    per_position: list[dict[str, Any]] = []
    for task, lm_res in zip(tasks, lm_results, strict=True):
        pos = task["pos"]
        idx = task["index"]
        crop_rel_str = task["crop_rel_str"]
        bbox_ref = task["bbox_ref"]
        crop_pil = task["crop_pil"]

        crop_url: str
        if crop_rel_str and (ref_base / Path(crop_rel_str)).is_file():
            crop_url = _file_url_under_sku_results("reference", ref_run_id, crop_rel_str)
        else:
            recrop_rel = Path("crops") / f"recrop_{idx:03d}.jpg"
            crop_pil.save(str(run_dir / recrop_rel), format="JPEG", quality=92)
            crop_url = _file_url_under_sku_results("lm_recognition", lm_run_id, str(recrop_rel).replace("\\", "/"))

        if isinstance(bbox_ref, dict):
            box_d = _bbox_to_int_crop(bbox_ref, full_img.width, full_img.height)
            if box_d:
                sx1, sy1, sx2, sy2 = box_d
                short_name = (lm_res.item_name or "?")[:28]
                label_txt = f"{idx}. {short_name} ({lm_res.confidence:.2f})"
                draw.rectangle([(sx1, sy1), (sx2, sy2)], outline="lime", width=3)
                draw.text((sx1 + 4, max(0, sy1 - 14)), label_txt, fill="lime")

        per_position.append(
            {
                "index": idx,
                "reference_bbox": bbox_ref if isinstance(bbox_ref, dict) else {},
                "sku110k_label": str(pos.get("label", "")),
                "sku110k_score": float(pos.get("score", 0.0) or 0.0),
                "crop_path": crop_rel_str,
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

    return JSONResponse(payload)
