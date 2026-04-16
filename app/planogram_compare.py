from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.planogram import PlanogramTemplate


def _normalize_item_name(name: str) -> str:
    from app.item_validation import normalize_name

    return normalize_name(name)


@dataclass(frozen=True)
class PlanogramIssue:
    issue_type: str
    shelf_id: int
    position_in_shelf: int
    expected_item_name: str
    actual_item_name: str
    item_id: int
    details: str = ""


def compare_planogram(
    template: PlanogramTemplate,
    actual_items: list[dict],
) -> dict:
    expected_by_slot: dict[tuple[int, int], str] = {
        slot.key: slot.normalized_item_name for slot in template.slots
    }
    expected_raw_by_slot: dict[tuple[int, int], str] = {
        slot.key: slot.item_name for slot in template.slots
    }

    actual_by_slot: dict[tuple[int, int], dict] = {}
    actual_slot_by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in actual_items:
        shelf_id = int(item.get("shelf_id", 0))
        position = int(item.get("position_in_shelf", 0))
        if shelf_id <= 0 or position <= 0:
            continue
        key = (shelf_id, position)
        actual_by_slot[key] = item
        actual_name = _normalize_item_name(str(item.get("lm_item_name", "unknown")))
        if actual_name:
            actual_slot_by_name[actual_name].append(key)

    issues: list[PlanogramIssue] = []
    matched_count = 0
    seen_actual_slots: set[tuple[int, int]] = set()

    for slot_key, expected_name in expected_by_slot.items():
        shelf_id, position = slot_key
        expected_raw = expected_raw_by_slot.get(slot_key, expected_name)
        actual = actual_by_slot.get(slot_key)
        if actual is None:
            issues.append(
                PlanogramIssue(
                    issue_type="missing",
                    shelf_id=shelf_id,
                    position_in_shelf=position,
                    expected_item_name=expected_raw,
                    actual_item_name="",
                    item_id=0,
                )
            )
            continue

        seen_actual_slots.add(slot_key)
        actual_name = _normalize_item_name(str(actual.get("lm_item_name", "unknown")))
        if actual_name == expected_name:
            matched_count += 1
            continue

        wrong_positions = [k for k in actual_slot_by_name.get(expected_name, []) if k != slot_key]
        if wrong_positions:
            wrong_slot = wrong_positions[0]
            issues.append(
                PlanogramIssue(
                    issue_type="wrong_position",
                    shelf_id=shelf_id,
                    position_in_shelf=position,
                    expected_item_name=expected_raw,
                    actual_item_name=str(actual.get("lm_item_name", "")),
                    item_id=int(actual.get("item_id", 0)),
                    details=f"expected item found at shelf={wrong_slot[0]}, position={wrong_slot[1]}",
                )
            )
        else:
            issues.append(
                PlanogramIssue(
                    issue_type="name_mismatch",
                    shelf_id=shelf_id,
                    position_in_shelf=position,
                    expected_item_name=expected_raw,
                    actual_item_name=str(actual.get("lm_item_name", "")),
                    item_id=int(actual.get("item_id", 0)),
                )
            )

    for slot_key, actual in actual_by_slot.items():
        if slot_key in expected_by_slot:
            continue
        if slot_key in seen_actual_slots:
            continue
        issues.append(
            PlanogramIssue(
                issue_type="extra",
                shelf_id=slot_key[0],
                position_in_shelf=slot_key[1],
                expected_item_name="",
                actual_item_name=str(actual.get("lm_item_name", "")),
                item_id=int(actual.get("item_id", 0)),
            )
        )

    by_type: dict[str, int] = defaultdict(int)
    by_shelf: dict[int, int] = defaultdict(int)
    issue_by_item_id: dict[int, str] = {}
    for issue in issues:
        by_type[issue.issue_type] += 1
        by_shelf[issue.shelf_id] += 1
        if issue.item_id > 0:
            issue_by_item_id[issue.item_id] = issue.issue_type

    total_expected = len(expected_by_slot)
    compliance_ratio = (matched_count / total_expected) if total_expected else 0.0

    return {
        "enabled": True,
        "template_name": template.name,
        "template_version": template.version,
        "expected_slots": total_expected,
        "actual_slots": len(actual_by_slot),
        "matched_slots": matched_count,
        "compliance_ratio": compliance_ratio,
        "issues": [
            {
                "issue_type": issue.issue_type,
                "shelf_id": issue.shelf_id,
                "position_in_shelf": issue.position_in_shelf,
                "expected_item_name": issue.expected_item_name,
                "actual_item_name": issue.actual_item_name,
                "item_id": issue.item_id,
                "details": issue.details,
            }
            for issue in issues
        ],
        "issues_by_type": dict(by_type),
        "issues_by_shelf": dict(sorted(by_shelf.items())),
        "issue_by_item_id": issue_by_item_id,
    }
