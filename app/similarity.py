from __future__ import annotations

import base64
import io
from typing import Iterable

import numpy as np
from PIL import Image

from app.analytics import (
    Detection,
    assign_shelves_and_positions,
    assign_shelves_from_grid_bands,
    assign_shelves_from_horizontal_bands,
)


def _crop_to_base64(crop: Image.Image) -> str:
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _feature_from_crop(crop: Image.Image) -> np.ndarray:
    arr = np.asarray(crop.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2).reshape(-1)
    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))
    ratio = np.array([crop.width / max(crop.height, 1), crop.height / max(crop.width, 1)], dtype=np.float32)
    feat = np.concatenate([gray, mean_rgb, std_rgb, ratio], axis=0)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def _ratio_close(v1: float, v2: float, limit: float) -> bool:
    if v1 <= 0 or v2 <= 0:
        return False
    r = max(v1, v2) / max(1e-6, min(v1, v2))
    return r <= limit


def _cluster_row_centroid(
    item_indices: list[int],
    features: np.ndarray,
    areas: np.ndarray,
    aspects: np.ndarray,
    similarity_threshold: float,
    area_ratio_limit: float,
    aspect_ratio_limit: float,
) -> list[list[int]]:
    # Без транзитивного "слипания": каждый элемент назначается в ближайший centroid.
    clusters: list[dict] = []

    for idx in item_indices:
        feat = features[idx]
        area = float(areas[idx])
        aspect = float(aspects[idx])
        best_cluster_id = -1
        best_sim = -1.0

        for cid, cluster in enumerate(clusters):
            sim = float(feat @ cluster["centroid"])
            if sim < similarity_threshold:
                continue
            if not _ratio_close(area, cluster["area_ref"], area_ratio_limit):
                continue
            if not _ratio_close(aspect, cluster["aspect_ref"], aspect_ratio_limit):
                continue
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = cid

        if best_cluster_id == -1:
            clusters.append(
                {
                    "items": [idx],
                    "centroid": feat.copy(),
                    "area_ref": area,
                    "aspect_ref": aspect,
                }
            )
            continue

        cluster = clusters[best_cluster_id]
        cluster["items"].append(idx)
        # Обновляем centroid, чтобы кластер "подстраивался", но без полного merge групп.
        k = len(cluster["items"])
        cluster["centroid"] = (cluster["centroid"] * (k - 1) + feat) / k
        norm = np.linalg.norm(cluster["centroid"])
        if norm > 0:
            cluster["centroid"] = cluster["centroid"] / norm
        cluster["area_ref"] = float((cluster["area_ref"] * (k - 1) + area) / k)
        cluster["aspect_ref"] = float((cluster["aspect_ref"] * (k - 1) + aspect) / k)

    return [c["items"] for c in clusters]


def analyze_similar_positions(
    image: Image.Image,
    detections: Iterable[Detection],
    similarity_threshold: float = 0.88,
    shelf_rows: int = 4,
    max_groups: int = 12,
    max_previews_per_group: int = 8,
    area_ratio_limit: float = 1.7,
    aspect_ratio_limit: float = 1.35,
    *,
    fact_band_shelf_ids: list[int] | None = None,
    fact_band_y_splits: list[float] | None = None,
    fact_band_x_splits: list[float] | None = None,
) -> dict:
    dets = list(detections)
    if not dets:
        return {"group_count": 0, "groups": [], "items": []}

    crops: list[Image.Image] = []
    valid_dets: list[Detection] = []
    rows: list[int] = []
    shelf_ids: list[int] = []
    positions_in_shelf: list[int] = []
    areas: list[float] = []
    aspects: list[float] = []
    if fact_band_shelf_ids is not None and fact_band_y_splits is not None:
        if fact_band_x_splits is not None:
            assignments = assign_shelves_from_grid_bands(
                dets,
                image.width,
                image.height,
                fact_band_shelf_ids,
                fact_band_y_splits,
                fact_band_x_splits,
            )
        else:
            assignments = assign_shelves_from_horizontal_bands(
                dets,
                image.width,
                image.height,
                fact_band_shelf_ids,
                fact_band_y_splits,
            )
    else:
        assignments = assign_shelves_and_positions(dets, image.width, image.height)
    placement_by_index = {
        int(a["detection_index"]): (int(a["shelf_id"]), int(a["position_in_shelf"]))
        for a in assignments
    }
    for det_index, d in enumerate(dets):
        x1 = int(max(0, min(d.x1, image.width - 1)))
        y1 = int(max(0, min(d.y1, image.height - 1)))
        x2 = int(max(0, min(d.x2, image.width)))
        y2 = int(max(0, min(d.y2, image.height)))
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        crops.append(image.crop((x1, y1, x2, y2)))
        valid_dets.append(d)
        shelf_id, position_in_shelf = placement_by_index.get(det_index, (0, 0))
        rows.append(shelf_id)
        shelf_ids.append(shelf_id)
        positions_in_shelf.append(position_in_shelf)
        w = max(1.0, d.x2 - d.x1)
        h = max(1.0, d.y2 - d.y1)
        areas.append(float(w * h))
        aspects.append(float(w / h))

    if not crops:
        return {"group_count": 0, "groups": [], "items": []}

    features = np.vstack([_feature_from_crop(c) for c in crops])
    areas_arr = np.asarray(areas, dtype=np.float32)
    aspects_arr = np.asarray(aspects, dtype=np.float32)

    # Кластеризуем по каждому ряду отдельно, чтобы не мешать разные полки.
    groups: list[list[int]] = []
    distinct_shelves = sorted(set(shelf_ids))
    for row_id in distinct_shelves:
        row_item_indices = [i for i, r in enumerate(rows) if r == row_id]
        if not row_item_indices:
            continue
        row_groups = _cluster_row_centroid(
            item_indices=row_item_indices,
            features=features,
            areas=areas_arr,
            aspects=aspects_arr,
            similarity_threshold=similarity_threshold,
            area_ratio_limit=area_ratio_limit,
            aspect_ratio_limit=aspect_ratio_limit,
        )
        groups.extend(row_groups)

    groups = sorted(groups, key=len, reverse=True)[:max_groups]

    group_id_by_item_index: dict[int, int] = {}
    for group_id, group in enumerate(groups, start=1):
        for item_idx in group:
            group_id_by_item_index[item_idx] = group_id

    result_groups = []
    for idx, group in enumerate(groups, start=1):
        previews = [_crop_to_base64(crops[i]) for i in group[:max_previews_per_group]]
        scores = [valid_dets[i].score for i in group]
        group_rows = [rows[i] for i in group]
        result_groups.append(
            {
                "id": idx,
                "count": len(group),
                "avg_score": float(np.mean(scores)) if scores else 0.0,
                "row": int(np.median(group_rows)) if group_rows else 0,
                "preview_images": previews,
            }
        )

    result_items = []
    for i, det in enumerate(valid_dets, start=1):
        item_idx = i - 1
        result_items.append(
            {
                "item_id": i,
                "group_id": group_id_by_item_index.get(item_idx, 0),
                "score": float(det.score),
                "crop_image": _crop_to_base64(crops[item_idx]),
                "bbox": {
                    "x1": float(det.x1),
                    "y1": float(det.y1),
                    "x2": float(det.x2),
                    "y2": float(det.y2),
                },
                "row": rows[item_idx],
                "shelf_id": shelf_ids[item_idx],
                "position_in_shelf": positions_in_shelf[item_idx],
            }
        )

    return {
        "group_count": len(result_groups),
        "groups": result_groups,
        "items": result_items,
    }


def cluster_crop_indices_by_similarity(
    crops: list[Image.Image],
    similarity_threshold: float,
    *,
    area_ratio_limit: float = 1.7,
    aspect_ratio_limit: float = 1.35,
) -> list[list[int]]:
    """
    Разбивает индексы кропов 0..n-1 на кластеры по визуальной похожести (как в analyze_similar_positions).
    Каждый индекс попадает ровно в один кластер.
    """
    n = len(crops)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    features = np.vstack([_feature_from_crop(c) for c in crops])
    areas = np.asarray([float(c.width * c.height) for c in crops], dtype=np.float32)
    aspects = np.asarray([float(c.width) / max(float(c.height), 1.0) for c in crops], dtype=np.float32)
    return _cluster_row_centroid(
        item_indices=list(range(n)),
        features=features,
        areas=areas,
        aspects=aspects,
        similarity_threshold=similarity_threshold,
        area_ratio_limit=area_ratio_limit,
        aspect_ratio_limit=aspect_ratio_limit,
    )
