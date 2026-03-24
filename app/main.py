from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw

from app.analytics import shelf_metrics
from app.similarity import analyze_similar_positions
from app.sku110k_adapter import SKU110KDetector

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Merch Analyzer (SKU110K MVP)")

# Дефолтные пути для локальной установки в этом проекте.
DEFAULT_REPO_PATH = str(BASE_DIR / "third_party" / "SKU110K_CVPR19")
DEFAULT_WEIGHTS_PATH = str(BASE_DIR / "models" / "sku110k_pretrained.h5")
DEFAULT_PYTHON_BIN = str(BASE_DIR / ".venv_sku" / "Scripts" / "python.exe")
DEFAULT_RUN_MODE = "docker"
DEFAULT_WSL_PYTHON_BIN = "python3"
DEFAULT_DOCKER_IMAGE = "merch-analyzer-sku110k:tf1.15"
DEFAULT_DOCKER_MOUNT_HOST = str(BASE_DIR)
DEFAULT_DOCKER_MOUNT_TARGET = "/workspace"
DEFAULT_DOCKER_USE_GPU = True
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_SHELF_ROWS = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.88


def _image_to_base64(pil_image: Image.Image) -> str:
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
) -> HTMLResponse:
    upload_tmp_dir = BASE_DIR / "tmp" / "uploads"
    upload_tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(upload_tmp_dir)) as tmp:
        tmp.write(await image.read())
        temp_img_path = tmp.name

    try:
        repo_path = os.getenv("SKU110K_REPO_PATH", DEFAULT_REPO_PATH)
        weights_path = os.getenv("SKU110K_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)
        python_bin = os.getenv("SKU110K_PYTHON_BIN", DEFAULT_PYTHON_BIN)
        run_mode = os.getenv("SKU110K_RUN_MODE", DEFAULT_RUN_MODE)
        wsl_python_bin = os.getenv("SKU110K_WSL_PYTHON_BIN", DEFAULT_WSL_PYTHON_BIN)
        docker_image = os.getenv("SKU110K_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
        docker_mount_host = os.getenv("SKU110K_DOCKER_MOUNT_HOST", DEFAULT_DOCKER_MOUNT_HOST)
        docker_mount_target = os.getenv("SKU110K_DOCKER_MOUNT_TARGET", DEFAULT_DOCKER_MOUNT_TARGET)
        docker_use_gpu = os.getenv("SKU110K_DOCKER_USE_GPU", str(DEFAULT_DOCKER_USE_GPU)).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        score_threshold = float(os.getenv("SKU110K_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD)))
        shelf_rows = int(os.getenv("SHELF_ROWS", str(DEFAULT_SHELF_ROWS)))
        similarity_threshold = float(
            os.getenv("SIMILARITY_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
        )

        detector = SKU110KDetector(
            repo_path=repo_path,
            weights_path=weights_path,
            python_bin=python_bin,
            score_threshold=score_threshold,
            run_mode=run_mode,
            wsl_python_bin=wsl_python_bin,
            docker_image=docker_image,
            docker_use_gpu=docker_use_gpu,
            docker_mount_host=docker_mount_host,
            docker_mount_target=docker_mount_target,
        )
        detections = detector.detect_image(temp_img_path)

        with Image.open(temp_img_path) as src:
            img = src.convert("RGB")
        draw = ImageDraw.Draw(img)
        for det in detections:
            draw.rectangle([(det.x1, det.y1), (det.x2, det.y2)], outline="red", width=3)
            draw.text((det.x1 + 3, max(0, det.y1 - 14)), f"{det.label}:{det.score:.2f}", fill="red")

        metrics = shelf_metrics(
            detections=detections,
            image_width=img.width,
            image_height=img.height,
            shelf_rows=shelf_rows,
        )
        similar_groups = analyze_similar_positions(
            image=img,
            detections=detections,
            similarity_threshold=similarity_threshold,
            shelf_rows=shelf_rows,
        )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "metrics": metrics,
                "similar_groups": similar_groups,
                "result_image_b64": _image_to_base64(img),
            },
        )
    except Exception as exc:  # noqa: BLE001 - для MVP отдаём текст ошибки в UI
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(exc),
            },
            status_code=400,
        )
    finally:
        try:
            os.unlink(temp_img_path)
        except OSError:
            pass
