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


def _clip_box(det: Detection, width: int, height: int) -> Detection:
    return Detection(
        x1=max(0.0, min(det.x1, width - 1)),
        y1=max(0.0, min(det.y1, height - 1)),
        x2=max(0.0, min(det.x2, width)),
        y2=max(0.0, min(det.y2, height)),
        score=det.score,
        label=det.label,
    )


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

    row_height = image_height / max(1, shelf_rows)
    row_stats = []
    for row_idx in range(shelf_rows):
        y_start = int(row_idx * row_height)
        y_end = int((row_idx + 1) * row_height)
        row_dets = [
            d
            for d in dets
            if ((d.y1 + d.y2) / 2.0) >= y_start and ((d.y1 + d.y2) / 2.0) < y_end
        ]
        row_mask = mask[y_start:y_end, :]
        row_stats.append(
            {
                "row": row_idx + 1,
                "items": len(row_dets),
                "occupancy": float(row_mask.mean()) if row_mask.size else 0.0,
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
        "rows": row_stats,
    }
