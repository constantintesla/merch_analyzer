from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


def resolve_planogram_source(
    *,
    prepared_planogram_csv: str,
    reference_planogram_image_content: bytes | None,
    reference_planogram_image_filename: str | None,
    planogram_file_content: bytes | None,
    planogram_filename: str | None,
    planogram_text: str,
) -> str:
    """
    Приоритет источника планограммы: подготовленный CSV → эталонное изображение
    → загруженный файл → текстовое описание.
    """
    if (prepared_planogram_csv or "").strip():
        return "prepared_csv"
    if reference_planogram_image_content and (reference_planogram_image_filename or "").strip():
        return "reference_image"
    if planogram_file_content and (planogram_filename or "").strip():
        return "file"
    return "text"


def _normalize_item_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9а-яА-Я\s]+", " ", str(name).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass(frozen=True)
class PlanogramSlot:
    shelf_id: int
    position_in_shelf: int
    item_name: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.shelf_id, self.position_in_shelf)

    @property
    def normalized_item_name(self) -> str:
        return _normalize_item_name(self.item_name)


@dataclass(frozen=True)
class PlanogramTemplate:
    name: str
    version: str
    slots: list[PlanogramSlot]


def _validate_slots(slots: list[PlanogramSlot]) -> None:
    if not slots:
        raise ValueError("planogram must contain at least one slot")

    keys_seen: set[tuple[int, int]] = set()
    for slot in slots:
        if slot.shelf_id <= 0:
            raise ValueError("shelf_id must be positive")
        if slot.position_in_shelf <= 0:
            raise ValueError("position_in_shelf must be positive")
        if not slot.normalized_item_name:
            raise ValueError("item_name cannot be empty")
        if slot.key in keys_seen:
            raise ValueError(
                f"duplicate slot detected for shelf={slot.shelf_id}, position={slot.position_in_shelf}"
            )
        keys_seen.add(slot.key)


def build_planogram_template(
    slots: list[PlanogramSlot],
    name: str = "manual_planogram",
    version: str = "1",
) -> PlanogramTemplate:
    _validate_slots(slots)
    return PlanogramTemplate(name=name.strip() or "manual_planogram", version=version, slots=slots)


def parse_manual_planogram_text(text: str) -> PlanogramTemplate:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    slots: list[PlanogramSlot] = []
    for idx, line in enumerate(lines, start=1):
        # Формат строки: shelf_id,position_in_shelf,item_name
        parts = [p.strip() for p in line.split(",", maxsplit=2)]
        if len(parts) != 3:
            raise ValueError(
                f"invalid manual planogram line {idx}: expected 'shelf,position,item_name'"
            )
        shelf_id, position_in_shelf, item_name = parts
        try:
            shelf_num = int(shelf_id)
            pos_num = int(position_in_shelf)
        except ValueError as exc:
            raise ValueError(f"invalid numbers on line {idx}") from exc
        slots.append(
            PlanogramSlot(
                shelf_id=shelf_num,
                position_in_shelf=pos_num,
                item_name=item_name,
            )
        )
    return build_planogram_template(slots=slots, name="manual_planogram", version="1")


def _slot_from_mapping(row: dict, row_index: int) -> PlanogramSlot:
    shelf_val = str(row.get("shelf_id", row.get("shelf", ""))).strip()
    pos_val = str(row.get("position_in_shelf", row.get("position", ""))).strip()
    name_val = str(row.get("item_name", row.get("sku_name", row.get("sku", "")))).strip()
    if not shelf_val or not pos_val or not name_val:
        raise ValueError(f"missing required fields in row {row_index}")
    return PlanogramSlot(
        shelf_id=int(shelf_val),
        position_in_shelf=int(pos_val),
        item_name=name_val,
    )


def parse_planogram_csv(content: bytes, name: str = "csv_planogram") -> PlanogramTemplate:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("csv planogram has no header")
    slots = [_slot_from_mapping(row, i) for i, row in enumerate(reader, start=2)]
    return build_planogram_template(slots=slots, name=name, version="1")


def parse_planogram_json(content: bytes, name: str = "json_planogram") -> PlanogramTemplate:
    payload = json.loads(content.decode("utf-8-sig"))
    if isinstance(payload, dict):
        slots_raw = payload.get("slots", [])
        plan_name = str(payload.get("name", name))
        version = str(payload.get("version", "1"))
    elif isinstance(payload, list):
        slots_raw = payload
        plan_name = name
        version = "1"
    else:
        raise ValueError("json planogram must be object or list")

    slots: list[PlanogramSlot] = []
    for idx, row in enumerate(slots_raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"slot {idx} must be object")
        slots.append(_slot_from_mapping(row, idx))
    return build_planogram_template(slots=slots, name=plan_name, version=version)


def parse_planogram_file(filename: str, content: bytes) -> PlanogramTemplate:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return parse_planogram_csv(content, name=Path(filename).stem or "csv_planogram")
    if suffix == ".json":
        return parse_planogram_json(content, name=Path(filename).stem or "json_planogram")
    raise ValueError("unsupported planogram file format, expected .csv or .json")


def normalize_shelf_positions_from_geometry(items: list[dict]) -> None:
    """
    Перенумеровывает полки и позиции по bbox: полки сверху вниз (по среднему Y центра),
    позиции на полке слева направо (по X центра). Мутирует элементы items in-place.

    Нужна после детекции/кластеризации: порядок объектов в списке и промежуточные номера
    могут не совпадать с визуальным порядком на фото.
    """
    work: list[tuple[dict, int, float, float]] = []
    for it in items:
        sid = int(it.get("shelf_id", 0) or 0)
        pos = int(it.get("position_in_shelf", 0) or 0)
        if sid <= 0 or pos <= 0:
            continue
        bbox = it.get("bbox") or {}
        try:
            x1 = float(bbox.get("x1", 0.0))
            y1 = float(bbox.get("y1", 0.0))
            x2 = float(bbox.get("x2", 0.0))
            y2 = float(bbox.get("y2", 0.0))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        work.append((it, sid, cx, cy))

    if not work:
        return

    by_shelf: dict[int, list[tuple[dict, float, float]]] = {}
    for it, sid, cx, cy in work:
        by_shelf.setdefault(sid, []).append((it, cx, cy))

    shelf_mean_y: list[tuple[int, float]] = []
    for sid, lst in by_shelf.items():
        shelf_mean_y.append((sid, float(mean(cy for _, _, cy in lst))))
    shelf_mean_y.sort(key=lambda t: t[1])

    old_to_new_shelf = {old: idx + 1 for idx, (old, _) in enumerate(shelf_mean_y)}

    for old_sid, lst in by_shelf.items():
        new_sid = old_to_new_shelf[old_sid]
        lst.sort(key=lambda t: t[1])
        for new_pos, (it, _, _) in enumerate(lst, start=1):
            it["shelf_id"] = new_sid
            it["position_in_shelf"] = new_pos
            it["row"] = new_sid


def planogram_template_to_csv_text(template: PlanogramTemplate) -> str:
    buff = io.StringIO()
    writer = csv.writer(buff, lineterminator="\n")
    writer.writerow(["shelf_id", "position_in_shelf", "item_name"])
    for slot in sorted(template.slots, key=lambda s: (s.shelf_id, s.position_in_shelf)):
        writer.writerow([slot.shelf_id, slot.position_in_shelf, slot.item_name])
    return buff.getvalue()


def _item_display_name_for_planogram(item: dict) -> str:
    """Имя для слота в шаблоне/оверлее: LM или запасной ярлык, чтобы не терять геометрию при unknown."""
    raw = str(item.get("lm_item_name", "")).strip()
    if raw and _normalize_item_name(raw) != "unknown":
        return raw
    iid = int(item.get("item_id", 0) or 0)
    return f"не распознано #{iid}" if iid else "не распознано"


def build_planogram_template_from_items(
    items: list[dict],
    name: str = "reference_image_planogram",
    version: str = "1",
) -> PlanogramTemplate:
    best_by_slot: dict[tuple[int, int], dict] = {}
    for item in items:
        shelf_id = int(item.get("shelf_id", 0))
        position = int(item.get("position_in_shelf", 0))
        if shelf_id <= 0 or position <= 0:
            continue
        item_name = _item_display_name_for_planogram(item)
        if not _normalize_item_name(item_name):
            continue

        score = float(item.get("lm_confidence_final", item.get("lm_confidence", 0.0)) or 0.0)
        slot_key = (shelf_id, position)
        prev = best_by_slot.get(slot_key)
        if prev is None or score > prev["score"]:
            best_by_slot[slot_key] = {
                "score": score,
                "slot": PlanogramSlot(
                    shelf_id=shelf_id,
                    position_in_shelf=position,
                    item_name=item_name,
                ),
            }

    slots = [entry["slot"] for entry in best_by_slot.values()]
    return build_planogram_template(slots=slots, name=name, version=version)


def slot_overlay_entries_from_items(
    items: list[dict],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """
    Для каждого слота (shelf_id, position_in_shelf) — лучший детект с bbox,
    нормализованным к размеру кадра [0,1].
    """
    iw = max(1, int(image_width))
    ih = max(1, int(image_height))
    best_by_slot: dict[tuple[int, int], dict] = {}
    for item in items:
        shelf_id = int(item.get("shelf_id", 0))
        position = int(item.get("position_in_shelf", 0))
        if shelf_id <= 0 or position <= 0:
            continue

        score = float(item.get("lm_confidence_final", item.get("lm_confidence", 0.0)) or 0.0)
        bbox = item.get("bbox", {})
        x1 = float(bbox.get("x1", 0.0))
        y1 = float(bbox.get("y1", 0.0))
        x2 = float(bbox.get("x2", 0.0))
        y2 = float(bbox.get("y2", 0.0))
        if x2 <= x1 or y2 <= y1:
            continue

        item_name = _item_display_name_for_planogram(item)
        bbox_norm = {
            "x1": max(0.0, min(1.0, x1 / iw)),
            "y1": max(0.0, min(1.0, y1 / ih)),
            "x2": max(0.0, min(1.0, x2 / iw)),
            "y2": max(0.0, min(1.0, y2 / ih)),
        }
        slot_key = (shelf_id, position)
        prev = best_by_slot.get(slot_key)
        entry = {
            "shelf_id": shelf_id,
            "position_in_shelf": position,
            "item_name": item_name,
            "bbox_norm": bbox_norm,
            "item_id": int(item.get("item_id", 0)),
        }
        if prev is None or score > prev["score"]:
            best_by_slot[slot_key] = {"score": score, "entry": entry}

    out = [best_by_slot[k]["entry"] for k in sorted(best_by_slot.keys())]
    return out