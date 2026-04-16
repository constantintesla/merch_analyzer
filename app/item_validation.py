from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

try:
    from rapidfuzz import fuzz as _rf_fuzz
except ImportError:
    _rf_fuzz = None

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

# path resolve -> (mtime_ns, names)
_catalog_file_cache: dict[str, tuple[int, list[str]]] = {}


def _has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC_RE.search(text))


def normalize_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9а-яА-Я\s]+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "unknown"


@dataclass
class ValidationConfig:
    catalog_path: str = ""
    catalog_match_threshold: float = 0.82
    group_consistency_threshold: float = 0.45
    low_confidence_threshold: float = 0.55
    # Если задано — не читать каталог с диска (один список на запрос анализа).
    catalog_names: list[str] | None = None


def load_catalog_names(catalog_path: str) -> list[str]:
    if not catalog_path:
        return []
    path = Path(catalog_path)
    if not path.exists():
        return []

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        return []

    if path.suffix.lower() == ".csv":
        names: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return []
            for row in reader:
                if "name" in row and row["name"]:
                    names.append(row["name"].strip())
                elif "sku_name" in row and row["sku_name"]:
                    names.append(row["sku_name"].strip())
        return names

    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def load_catalog_names_cached(catalog_path: str) -> list[str]:
    """Кэш по пути и mtime файла, чтобы не перечитывать каталог при повторных вызовах валидации."""
    if not catalog_path:
        return []
    path = Path(catalog_path).resolve()
    if not path.exists():
        return []
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return load_catalog_names(catalog_path)
    key = str(path)
    hit = _catalog_file_cache.get(key)
    if hit is not None and hit[0] == mtime_ns:
        return hit[1]
    names = load_catalog_names(str(path))
    _catalog_file_cache[key] = (mtime_ns, names)
    return names


def fuzzy_best_match(name: str, catalog: list[str]) -> tuple[str, float]:
    if not catalog:
        return "", 0.0
    src = normalize_name(name)
    best_name = ""
    best_score = 0.0
    if _rf_fuzz is not None:
        for candidate in catalog:
            score = _rf_fuzz.ratio(src, normalize_name(candidate)) / 100.0
            if score > best_score:
                best_score = score
                best_name = candidate
    else:
        for candidate in catalog:
            score = SequenceMatcher(None, src, normalize_name(candidate)).ratio()
            if score > best_score:
                best_score = score
                best_name = candidate
    return best_name, best_score


def _is_unknown_display(name: str) -> bool:
    n = normalize_name(name)
    return not n or n == "unknown"


def _pick_group_canonical_name(group_items: list[dict], config: ValidationConfig) -> str:
    """
    Одно отображаемое имя для кластера похожих кропов: сначала голосование среди позиций
    с уверенным матчем к каталогу, иначе — по всем членам группы; при равенстве — выше lm_confidence_final.
    """
    thr = float(config.catalog_match_threshold)
    catalog_backed = [
        it
        for it in group_items
        if float(it.get("lm_catalog_match_score", 0.0)) >= thr and "not_in_catalog" not in (it.get("lm_warning_reason") or "")
    ]
    pool = catalog_backed if catalog_backed else list(group_items)
    counts: Counter[str] = Counter()
    for it in pool:
        raw = str(it.get("lm_item_name", "unknown")).strip() or "unknown"
        if _is_unknown_display(raw):
            continue
        counts[raw] += 1
    if not counts:
        for it in group_items:
            raw = str(it.get("lm_item_name", "unknown")).strip() or "unknown"
            if raw:
                return raw
        return "unknown"
    top_n, top_c = counts.most_common(1)[0]
    tied = [name for name, c in counts.items() if c == top_c]
    if len(tied) == 1:
        return top_n
    # При равных голосах не использовать lm_confidence_final: штрафы group_conflict
    # зависят от случайного «большинства» при счёте 1:1 в Counter.
    best_name = tied[0]
    best_conf = -1.0
    for it in pool:
        raw = str(it.get("lm_item_name", "unknown")).strip() or "unknown"
        if raw not in tied:
            continue
        cf = float(it.get("lm_confidence", 0.0))
        if cf > best_conf:
            best_conf = cf
            best_name = raw
    return best_name


def unify_cluster_item_display_names(items: list[dict], config: ValidationConfig) -> None:
    """Приводит lm_item_name к одному канону внутри каждого group_id (похожие кропы одной бутылки)."""
    by_gid: dict[int, list[dict]] = {}
    for item in items:
        gid = int(item.get("group_id", 0))
        if gid > 0:
            by_gid.setdefault(gid, []).append(item)
    for _gid, group_items in by_gid.items():
        if len(group_items) < 2:
            continue
        canonical = _pick_group_canonical_name(group_items, config)
        for item in group_items:
            old = str(item.get("lm_item_name", "unknown"))
            if normalize_name(old) != normalize_name(canonical):
                parts = {p for p in (item.get("lm_warning_reason") or "").split(",") if p}
                parts.add("group_unified")
                item["lm_warning_reason"] = ",".join(sorted(parts))
                item["lm_is_suspicious"] = True
            item["lm_item_name"] = canonical


def apply_item_validation(items: list[dict], config: ValidationConfig) -> list[dict]:
    if config.catalog_names is not None:
        catalog = config.catalog_names
    else:
        catalog = load_catalog_names_cached(config.catalog_path)

    group_majority: dict[int, str] = {}
    by_group: dict[int, list[str]] = {}
    for item in items:
        gid = int(item.get("group_id", 0))
        name = normalize_name(str(item.get("lm_item_name", "unknown")))
        by_group.setdefault(gid, []).append(name)
    for gid, names in by_group.items():
        if not names:
            continue
        group_majority[gid] = Counter(names).most_common(1)[0][0]

    for item in items:
        raw_name = str(item.get("lm_raw_name", item.get("lm_item_name", "unknown")))
        lm_name = str(item.get("lm_item_name", "unknown"))
        lm_status = str(item.get("lm_status", "unknown"))
        lm_conf = float(item.get("lm_confidence", 0.0))
        warnings: list[str] = []

        matched_name, match_score = fuzzy_best_match(lm_name, catalog)
        catalog_name = lm_name
        if catalog:
            if match_score >= config.catalog_match_threshold and matched_name:
                # Не подменять русское название латинским из каталога
                if _has_cyrillic(lm_name) and not _has_cyrillic(matched_name):
                    catalog_name = lm_name
                else:
                    catalog_name = matched_name
            else:
                warnings.append("not_in_catalog")

        gid = int(item.get("group_id", 0))
        majority = group_majority.get(gid, "unknown")
        majority_ratio = SequenceMatcher(
            None, normalize_name(catalog_name), normalize_name(majority)
        ).ratio()
        if gid > 0 and majority != "unknown" and majority_ratio < config.group_consistency_threshold:
            warnings.append("group_conflict")

        if lm_status in {"unknown", "uncertain", "lmstudio_error"}:
            warnings.append("lm_status")
        if lm_conf < config.low_confidence_threshold:
            warnings.append("low_confidence")

        confidence_final = lm_conf
        if "not_in_catalog" in warnings:
            confidence_final *= 0.7
        if "group_conflict" in warnings:
            confidence_final *= 0.7
        if "lm_status" in warnings:
            confidence_final *= 0.6
        if "low_confidence" in warnings:
            confidence_final *= 0.8
        confidence_final = max(0.0, min(1.0, confidence_final))

        item["lm_raw_name"] = raw_name
        item["lm_catalog_name"] = catalog_name
        item["lm_catalog_match_score"] = match_score
        item["lm_group_majority_name"] = majority
        item["lm_group_similarity"] = majority_ratio
        item["lm_confidence_final"] = confidence_final
        item["lm_warning_reason"] = ",".join(sorted(set(warnings))) if warnings else ""
        item["lm_is_suspicious"] = bool(warnings)
        item["lm_item_name"] = catalog_name

    unify_cluster_item_display_names(items, config)
    return items
