from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = "item"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def _center_x(det: Detection) -> float:
    return (det.x1 + det.x2) / 2.0


def _center_y(det: Detection) -> float:
    return (det.y1 + det.y2) / 2.0


def _clip_box(det: Detection, width: int, height: int) -> Detection:
    return Detection(
        x1=max(0.0, min(det.x1, width - 1)),
        y1=max(0.0, min(det.y1, height - 1)),
        x2=max(0.0, min(det.x2, width)),
        y2=max(0.0, min(det.y2, height)),
        score=det.score,
        label=det.label,
    )


def assign_shelves_and_positions(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
) -> list[dict]:
    dets = [_clip_box(d, image_width, image_height) for d in detections]
    if not dets:
        return []

    median_height = float(np.median([max(1.0, d.height) for d in dets]))
    # Порог объединения по вертикали: адаптивный относительно средней высоты товара.
    y_gap_threshold = max(10.0, median_height * 0.8)
    sorted_indices = sorted(range(len(dets)), key=lambda i: _center_y(dets[i]))

    shelf_groups: list[list[int]] = []
    current_group: list[int] = []
    current_anchor_y: float | None = None
    for idx in sorted_indices:
        cy = _center_y(dets[idx])
        if current_anchor_y is None:
            current_group = [idx]
            current_anchor_y = cy
            continue
        if abs(cy - current_anchor_y) <= y_gap_threshold:
            current_group.append(idx)
            current_anchor_y = float(np.mean([_center_y(dets[i]) for i in current_group]))
        else:
            shelf_groups.append(current_group)
            current_group = [idx]
            current_anchor_y = cy
    if current_group:
        shelf_groups.append(current_group)

    assignments: list[dict] = []
    for shelf_idx, group_indices in enumerate(shelf_groups, start=1):
        left_to_right = sorted(group_indices, key=lambda i: _center_x(dets[i]))
        for position_idx, det_idx in enumerate(left_to_right, start=1):
            d = dets[det_idx]
            assignments.append(
                {
                    "detection_index": det_idx,
                    "detection": d,
                    "shelf_id": shelf_idx,
                    "position_in_shelf": position_idx,
                }
            )

    assignments.sort(key=lambda a: a["detection_index"])
    for i, item in enumerate(assignments, start=1):
        item["position_global"] = i
        item["shelf_count"] = len(shelf_groups)
    return assignments


def _band_index_norm(val: float, splits: list[float], count: int, *, vertical_end_closed: bool) -> int:
    if val < splits[0] or val > splits[-1]:
        return -1
    for i in range(count):
        lo, hi = splits[i], splits[i + 1]
        if i < count - 1:
            if lo <= val < hi:
                return i
        else:
            if vertical_end_closed:
                if lo <= val <= hi:
                    return i
            else:
                if lo <= val < hi:
                    return i
    return -1


def assign_shelves_from_horizontal_bands(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
    shelf_ids_ordered: list[int],
    y_splits_norm: list[float],
) -> list[dict]:
    """
    Назначает shelf_id по горизонтальным полосам (y_splits относительно высоты кадра, 0 — верх).
    y_splits_norm длины K+1 для K полок: границы полос [s[i], s[i+1]].
    Детекты вне полос или с центром вне [0,1] по Y получают shelf_id=0.
    """
    dets = [_clip_box(d, image_width, image_height) for d in detections]
    if not dets:
        return []

    k = len(shelf_ids_ordered)
    if k == 0:
        return []
    if len(y_splits_norm) != k + 1:
        raise ValueError(
            f"y_splits_norm must have length {k + 1} for {k} shelves, got {len(y_splits_norm)}"
        )
    splits = list(y_splits_norm)
    for i in range(len(splits) - 1):
        if splits[i] > splits[i + 1] + 1e-9:
            raise ValueError("y_splits_norm must be non-decreasing")

    shelf_by_det: list[int] = [0] * len(dets)
    for di, d in enumerate(dets):
        cy = _center_y(d)
        if image_height <= 0:
            continue
        yn = cy / float(image_height)
        bi = _band_index_norm(yn, splits, k, vertical_end_closed=True)
        if bi < 0:
            shelf_by_det[di] = 0
        else:
            shelf_by_det[di] = shelf_ids_ordered[bi]

    groups: dict[int, list[int]] = {}
    for di, sid in enumerate(shelf_by_det):
        if sid <= 0:
            continue
        groups.setdefault(sid, []).append(di)

    position_by_det = [0] * len(dets)
    for sid, indices in groups.items():
        left_to_right = sorted(indices, key=lambda i: _center_x(dets[i]))
        for pos_idx, di in enumerate(left_to_right, start=1):
            position_by_det[di] = pos_idx

    assignments: list[dict] = []
    shelf_count = len({s for s in shelf_by_det if s > 0})
    for di, d in enumerate(dets):
        sid = shelf_by_det[di]
        pos = position_by_det[di] if sid > 0 else 0
        assignments.append(
            {
                "detection_index": di,
                "detection": d,
                "shelf_id": sid,
                "position_in_shelf": pos,
            }
        )

    assignments.sort(key=lambda a: a["detection_index"])
    for i, item in enumerate(assignments, start=1):
        item["position_global"] = i
        item["shelf_count"] = shelf_count
    return assignments


def assign_shelves_from_grid_bands(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
    shelf_ids_ordered: list[int],
    y_splits_norm: list[float],
    x_splits_norm: list[float],
) -> list[dict]:
    """
    Полка по Y-полосам, позиция в ряду по X-полосам (колонки слева направо → 1..M).
    x_splits_norm и y_splits_norm — в координатах кадра [0,1].
    """
    dets = [_clip_box(d, image_width, image_height) for d in detections]
    if not dets:
        return []

    k = len(shelf_ids_ordered)
    m = len(x_splits_norm) - 1
    if k == 0 or m <= 0:
        return []
    if len(y_splits_norm) != k + 1:
        raise ValueError(
            f"y_splits_norm must have length {k + 1} for {k} shelves, got {len(y_splits_norm)}"
        )
    if len(x_splits_norm) != m + 1:
        raise ValueError(
            f"x_splits_norm must have length {m + 1} for {m} columns, got {len(x_splits_norm)}"
        )
    ys = list(y_splits_norm)
    xs = list(x_splits_norm)
    for arr in (ys, xs):
        for i in range(len(arr) - 1):
            if arr[i] > arr[i + 1] + 1e-9:
                raise ValueError("splits must be non-decreasing")

    iw = max(1, int(image_width))
    ih = max(1, int(image_height))

    assignments: list[dict] = []
    for di, d in enumerate(dets):
        cx = _center_x(d)
        cy = _center_y(d)
        xn = cx / float(iw)
        yn = cy / float(ih)
        bi_y = _band_index_norm(yn, ys, k, vertical_end_closed=True)
        shelf_id = shelf_ids_ordered[bi_y] if bi_y >= 0 else 0
        bi_x = _band_index_norm(xn, xs, m, vertical_end_closed=True)
        position = (bi_x + 1) if bi_x >= 0 and shelf_id > 0 else 0
        assignments.append(
            {
                "detection_index": di,
                "detection": d,
                "shelf_id": shelf_id,
                "position_in_shelf": position,
            }
        )

    assignments.sort(key=lambda a: a["detection_index"])
    shelf_count = len({a["shelf_id"] for a in assignments if a["shelf_id"] > 0})
    for i, item in enumerate(assignments, start=1):
        item["position_global"] = i
        item["shelf_count"] = shelf_count
    return assignments


def shelf_metrics(
    detections: Iterable[Detection],
    image_width: int,
    image_height: int,
    shelf_rows: int = 4,
) -> dict:
    dets = [_clip_box(d, image_width, image_height) for d in detections]
    item_count = len(dets)

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    for d in dets:
        x1, y1 = int(d.x1), int(d.y1)
        x2, y2 = int(d.x2), int(d.y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1

    occupancy_ratio = float(mask.mean())
    empty_ratio = 1.0 - occupancy_ratio

    assignments = assign_shelves_and_positions(dets, image_width, image_height)
    shelf_count = int(max([a["shelf_id"] for a in assignments], default=0))
    shelf_stats = []
    for shelf_idx in range(1, shelf_count + 1):
        shelf_assignments = [a for a in assignments if a["shelf_id"] == shelf_idx]
        shelf_dets = [a["detection"] for a in shelf_assignments]
        if shelf_dets:
            y_start = int(max(0.0, min(d.y1 for d in shelf_dets)))
            y_end = int(min(image_height, max(d.y2 for d in shelf_dets)))
        else:
            y_start = 0
            y_end = 0
        shelf_mask = mask[y_start:y_end, :] if y_end > y_start else np.zeros((0, image_width), dtype=np.uint8)
        avg_front_width = (
            float(np.mean([max(1.0, d.width) for d in shelf_dets])) if shelf_dets else 0.0
        )
        shelf_stats.append(
            {
                "shelf": shelf_idx,
                "items": len(shelf_dets),
                "occupancy": float(shelf_mask.mean()) if shelf_mask.size else 0.0,
                "avg_front_width": avg_front_width,
            }
        )

    # Простая оценка "плотности выкладки": чем меньше среднее расстояние между центрами, тем выше.
    centers = np.array([[(d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0] for d in dets], dtype=float)
    density_score = 0.0
    if len(centers) >= 2:
        distances = np.linalg.norm(centers[None, :, :] - centers[:, None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        mean_nearest = float(np.mean(nearest))
        norm = (image_width**2 + image_height**2) ** 0.5
        density_score = max(0.0, 1.0 - (mean_nearest / norm))

    return {
        "item_count": item_count,
        "occupancy_ratio": occupancy_ratio,
        "empty_ratio": empty_ratio,
        "density_score": density_score,
        "shelf_count": shelf_count,
        "shelves": shelf_stats,
        # Для обратной совместимости с шаблоном оставляем rows.
        "rows": [
            {"row": s["shelf"], "items": s["items"], "occupancy": s["occupancy"]}
            for s in shelf_stats
        ],
    }
