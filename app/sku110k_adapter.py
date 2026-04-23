from __future__ import annotations

import csv
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image

from app.analytics import Detection

logger = logging.getLogger("merch_analyzer.sku110k")


def _tail_text(text: str | None, max_chars: int = 3500) -> str:
    if not text:
        return ""
    s = text.strip()
    if len(s) <= max_chars:
        return s
    return "…" + s[-max_chars:]


class SKU110KDetector:
    """
    Адаптер для запуска предикта через репозиторий SKU110K_CVPR19.

    Ожидается, что у пользователя:
    - есть локальная копия SKU110K_CVPR19;
    - есть .h5 веса;
    - доступна команда python для окружения с зависимостями repo.
    """

    def __init__(
        self,
        repo_path: str,
        weights_path: str,
        python_bin: str = "python",
        score_threshold: float = 0.25,
        run_mode: str = "auto",
        wsl_python_bin: str = "python3",
        docker_image: str = "merch-analyzer-sku110k:tf1.15",
        docker_use_gpu: bool = True,
        docker_mount_host: str | None = None,
        docker_mount_target: str = "/workspace",
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.weights_path = Path(weights_path).resolve()
        self.python_bin = python_bin
        self.score_threshold = score_threshold
        self.run_mode = run_mode.lower()
        self.wsl_python_bin = wsl_python_bin
        self.docker_image = docker_image
        self.docker_use_gpu = docker_use_gpu
        self.docker_mount_host = (
            Path(docker_mount_host).resolve() if docker_mount_host else self.repo_path.parent.parent
        )
        self.docker_mount_target = docker_mount_target

    @staticmethod
    def _to_wsl_path(path: Path | str) -> str:
        p = str(path).replace("\\", "/")
        match = re.match(r"^([A-Za-z]):/(.*)$", p)
        if not match:
            return p
        drive = match.group(1).lower()
        rest = match.group(2)
        return f"/mnt/{drive}/{rest}"

    def _resolve_run_mode(self) -> str:
        if self.run_mode in {"native", "wsl", "docker"}:
            return self.run_mode
        if shutil.which("docker"):
            return "docker"
        if os.name == "nt" and shutil.which("wsl"):
            return "wsl"
        return "native"

    def _to_docker_mounted_path(self, path: Path | str) -> str:
        p = Path(path).resolve()
        try:
            rel = p.relative_to(self.docker_mount_host)
        except ValueError as exc:
            raise ValueError(
                f"Path {p} is outside docker mount root {self.docker_mount_host}. "
                "Set SKU110K_DOCKER_MOUNT_HOST to include repo/weights/tmp paths."
            ) from exc
        rel_posix = str(rel).replace("\\", "/")
        return f"{self.docker_mount_target.rstrip('/')}/{rel_posix}"

    def detect_image(self, image_path: str) -> List[Detection]:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"SKU110K repo not found: {self.repo_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"weights file not found: {self.weights_path}")

        image_path = str(Path(image_path).resolve())
        run_mode = self._resolve_run_mode()
        timeout_sec_raw = (os.getenv("SKU110K_PREDICT_TIMEOUT_SEC") or "1800").strip()
        timeout_sec = int(timeout_sec_raw) if timeout_sec_raw.isdigit() else 1800
        t_pipeline = time.perf_counter()
        logger.info(
            "SKU110K: старт detect_image path=%s run_mode=%s (configured=%s) score_threshold=%s timeout_sec=%s",
            image_path,
            run_mode,
            self.run_mode,
            self.score_threshold,
            timeout_sec,
        )

        tmp_root = self.repo_path.parent / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sku110k_web_", dir=str(tmp_root)) as tmpdir:
            tmp = Path(tmpdir)
            dataset_base = tmp / "dataset"
            image_rel_dir = "upload"
            image_rel_name = "input" + Path(image_path).suffix.lower()
            image_rel_path = f"{image_rel_dir}/{image_rel_name}"
            image_dst_dir = dataset_base / image_rel_dir
            image_dst_dir.mkdir(parents=True, exist_ok=True)
            image_dst = image_dst_dir / image_rel_name
            shutil.copy2(image_path, image_dst)

            with Image.open(image_dst) as img:
                width, height = img.size
            logger.info(
                "SKU110K: подготовлен датасет tmpdir=%s image_rel=%s size=%sx%s",
                tmpdir,
                image_rel_path,
                width,
                height,
            )

            annotations_csv = tmp / "images.csv"
            classes_csv = tmp / "classes.csv"
            run_home = tmp / "runtime_home"
            results_dir = run_home / "Documents" / "SKU110K" / "results"
            runtime_image_dir = run_home / "Documents" / "SKU110K" / image_rel_dir
            runtime_image_dir.mkdir(parents=True, exist_ok=True)
            runtime_image_dst = runtime_image_dir / image_rel_name
            shutil.copy2(image_dst, runtime_image_dst)

            # keras-retinanet csv parser: image_path,x1,y1,x2,y2,class_name
            # Для инференса достаточно строки с путём картинки и пустыми bbox.
            with annotations_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # CSVGenerator в этом репозитории ожидает 8 полей и валидный bbox.
                writer.writerow([image_rel_path, 0, 0, 1, 1, "item", width, height])

            with classes_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["item", "0"])

            existing_results = set(results_dir.glob("detections_output_iou_*.csv")) if results_dir.exists() else set()

            if run_mode == "wsl":
                repo_wsl = self._to_wsl_path(self.repo_path)
                weights_wsl = self._to_wsl_path(self.weights_path)
                annotations_wsl = self._to_wsl_path(annotations_csv)
                classes_wsl = self._to_wsl_path(classes_csv)
                dataset_base_wsl = self._to_wsl_path(dataset_base)
                home_wsl = self._to_wsl_path(run_home)
                predict_script_wsl = f"{repo_wsl}/object_detector_retinanet/keras_retinanet/bin/predict.py"

                bash_cmd = (
                    f"cd {shlex.quote(repo_wsl)} && "
                    f"mkdir -p {shlex.quote(f'{home_wsl}/Documents/SKU110K')} && "
                    f"HOME={shlex.quote(home_wsl)} "
                    f"PYTHONPATH={shlex.quote(repo_wsl)} "
                    f"{shlex.quote(self.wsl_python_bin)} -u {shlex.quote(predict_script_wsl)} "
                    f"--base_dir {shlex.quote(dataset_base_wsl)} "
                    "csv "
                    f"--annotations {shlex.quote(annotations_wsl)} "
                    f"--classes {shlex.quote(classes_wsl)} "
                    f"{shlex.quote(weights_wsl)}"
                )
                command = ["wsl", "bash", "-lc", bash_cmd]
                logger.info(
                    "SKU110K: запуск WSL-предикта (первый прогон может занять несколько минут из-за загрузки TF/Keras)."
                )
                t0 = time.perf_counter()
                proc = subprocess.run(
                    command,
                    check=False,
                    timeout=timeout_sec,
                )
                logger.info(
                    "SKU110K: WSL завершён за %.1f с returncode=%s",
                    time.perf_counter() - t0,
                    proc.returncode,
                )
            elif run_mode == "docker":
                repo_docker = self._to_docker_mounted_path(self.repo_path)
                weights_docker = self._to_docker_mounted_path(self.weights_path)
                annotations_docker = self._to_docker_mounted_path(annotations_csv)
                classes_docker = self._to_docker_mounted_path(classes_csv)
                dataset_base_docker = self._to_docker_mounted_path(dataset_base)
                home_docker = self._to_docker_mounted_path(run_home)
                predict_script_docker = (
                    f"{repo_docker}/object_detector_retinanet/keras_retinanet/bin/predict.py"
                )

                bash_cmd = (
                    f"cd {shlex.quote(repo_docker)} && "
                    f"mkdir -p {shlex.quote(f'{home_docker}/Documents/SKU110K')} && "
                    f"HOME={shlex.quote(home_docker)} "
                    f"PYTHONPATH={shlex.quote(repo_docker)} "
                    f"python3 -u {shlex.quote(predict_script_docker)} "
                    f"--base_dir {shlex.quote(dataset_base_docker)} "
                    "csv "
                    f"--annotations {shlex.quote(annotations_docker)} "
                    f"--classes {shlex.quote(classes_docker)} "
                    f"{shlex.quote(weights_docker)}"
                )
                command = [
                    "docker",
                    "run",
                    "--rm",
                ]
                if self.docker_use_gpu:
                    command.extend(["--gpus", "all"])
                command.extend(
                    [
                        "-v",
                        f"{self.docker_mount_host}:{self.docker_mount_target}",
                        "-w",
                        repo_docker,
                        self.docker_image,
                        "bash",
                        "-lc",
                        bash_cmd,
                    ]
                )
                logger.info(
                    "SKU110K: запуск Docker image=%s gpu=%s mount=%s:%s (первый старт контейнера и загрузка весов "
                    "могут занять 1–10+ минут — это нормально).",
                    self.docker_image,
                    self.docker_use_gpu,
                    self.docker_mount_host,
                    self.docker_mount_target,
                )
                t0 = time.perf_counter()
                proc = subprocess.run(
                    command,
                    check=False,
                    timeout=timeout_sec,
                )
                logger.info(
                    "SKU110K: Docker завершён за %.1f с returncode=%s",
                    time.perf_counter() - t0,
                    proc.returncode,
                )
            else:
                command = [
                    self.python_bin,
                    "-u",
                    str(self.repo_path / "object_detector_retinanet/keras_retinanet/bin/predict.py"),
                    "--base_dir",
                    str(dataset_base),
                    "csv",
                    "--annotations",
                    str(annotations_csv),
                    "--classes",
                    str(classes_csv),
                    str(self.weights_path),
                ]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(self.repo_path)
                logger.info(
                    "SKU110K: запуск native python=%s cwd=%s (первый прогон может быть долгим).",
                    self.python_bin,
                    self.repo_path,
                )
                t0 = time.perf_counter()
                proc = subprocess.run(
                    command,
                    cwd=str(self.repo_path),
                    env=env,
                    check=False,
                    timeout=timeout_sec,
                )
                logger.info(
                    "SKU110K: native завершён за %.1f с returncode=%s",
                    time.perf_counter() - t0,
                    proc.returncode,
                )
            if proc.returncode != 0:
                logger.error("SKU110K: stdout tail:\n%s", _tail_text(proc.stdout))
                logger.error("SKU110K: stderr tail:\n%s", _tail_text(proc.stderr))
                raise RuntimeError(
                    f"SKU110K prediction failed (mode={run_mode}).\n"
                    f"Command: {' '.join(command)}\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )

            if run_mode == "native":
                user = os.getenv("USERNAME", "")
                native_results_dir = Path(f"C:/Users/{user}/Documents/SKU110K/results")
                results_csv = self._find_latest_results_csv(native_results_dir, set())
            else:
                results_csv = self._find_latest_results_csv(results_dir, existing_results)

            logger.info("SKU110K: CSV результатов %s", results_csv)
            detections = self._read_detections(results_csv)
            logger.info(
                "SKU110K: готово за %.1f с, детекций после порога: %d",
                time.perf_counter() - t_pipeline,
                len(detections),
            )
            return detections
    @staticmethod
    def _find_latest_results_csv(results_dir: Path, previous_files: set[Path]) -> Path:
        if not results_dir.exists():
            raise RuntimeError(f"Prediction finished, but results dir was not found: {results_dir}")
        current = set(results_dir.glob("detections_output_iou_*.csv"))
        new_files = [p for p in current if p not in previous_files]
        candidates = new_files if new_files else list(current)
        if not candidates:
            raise RuntimeError(f"Prediction finished, but no detections CSV found in: {results_dir}")
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _read_detections(self, csv_path: Path) -> List[Detection]:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []

        # Поддерживаем разные варианты колонок.
        candidates = {
            "x1": ["x1", "xmin"],
            "y1": ["y1", "ymin"],
            "x2": ["x2", "xmax"],
            "y2": ["y2", "ymax"],
            "score": ["score", "confidence"],
            "label": ["label", "class", "class_name"],
        }

        def pick_col(options: list[str], required: bool) -> str | None:
            for col in options:
                if col in df.columns:
                    return col
            if required:
                raise ValueError(f"Cannot find required columns in prediction CSV: {options}")
            return None

        x1_col = pick_col(candidates["x1"], required=True)
        y1_col = pick_col(candidates["y1"], required=True)
        x2_col = pick_col(candidates["x2"], required=True)
        y2_col = pick_col(candidates["y2"], required=True)
        score_col = pick_col(candidates["score"], required=False)
        label_col = pick_col(candidates["label"], required=False)

        detections: List[Detection] = []
        for _, row in df.iterrows():
            score = float(row[score_col]) if score_col else 1.0
            if score < self.score_threshold:
                continue
            detections.append(
                Detection(
                    x1=float(row[x1_col]),
                    y1=float(row[y1_col]),
                    x2=float(row[x2_col]),
                    y2=float(row[y2_col]),
                    score=score,
                    label=str(row[label_col]) if label_col else "item",
                )
            )
        return detections
