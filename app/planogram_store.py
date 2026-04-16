from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredPlanogram:
    id: str
    name: str
    csv_text: str
    created_at: float
    image_path: str | None


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS planograms (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                csv_text TEXT NOT NULL,
                created_at REAL NOT NULL,
                image_path TEXT
            )
            """
        )
        conn.commit()


def create_planogram(
    db_path: Path,
    *,
    name: str,
    csv_text: str,
    image_bytes: bytes | None,
    images_dir: Path,
) -> StoredPlanogram:
    init_db(db_path)
    pid = uuid.uuid4().hex
    image_path: str | None = None
    if image_bytes:
        images_dir.mkdir(parents=True, exist_ok=True)
        dest = images_dir / f"{pid}.jpg"
        dest.write_bytes(image_bytes)
        image_path = str(dest)
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO planograms (id, name, csv_text, created_at, image_path) VALUES (?, ?, ?, ?, ?)",
            (pid, name.strip() or "planogram", csv_text, now, image_path),
        )
        conn.commit()
    return get_planogram(db_path, pid)


def update_planogram(
    db_path: Path,
    planogram_id: str,
    *,
    csv_text: str,
    name: str | None = None,
    image_bytes: bytes | None = None,
    images_dir: Path,
) -> StoredPlanogram | None:
    """Обновляет CSV; имя — только если передано непустое; превью — только если переданы байты картинки."""
    init_db(db_path)
    row = get_planogram(db_path, planogram_id)
    if row is None:
        return None
    new_name = (name.strip() if name else "") or row.name
    new_csv = csv_text
    new_image_path = row.image_path
    if image_bytes is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        if row.image_path:
            try:
                Path(row.image_path).unlink(missing_ok=True)
            except OSError:
                pass
        dest = images_dir / f"{planogram_id}.jpg"
        dest.write_bytes(image_bytes)
        new_image_path = str(dest)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE planograms SET name = ?, csv_text = ?, image_path = ? WHERE id = ?",
            (new_name, new_csv, new_image_path, planogram_id),
        )
        conn.commit()
    return get_planogram(db_path, planogram_id)


def list_planograms(db_path: Path) -> list[dict[str, object]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM planograms ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id": str(r["id"]),
            "name": str(r["name"]),
            "created_at": float(r["created_at"]),
        }
        for r in rows
    ]


def get_planogram(db_path: Path, planogram_id: str) -> StoredPlanogram | None:
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, csv_text, created_at, image_path FROM planograms WHERE id = ?",
            (planogram_id,),
        ).fetchone()
    if row is None:
        return None
    return StoredPlanogram(
        id=str(row["id"]),
        name=str(row["name"]),
        csv_text=str(row["csv_text"]),
        created_at=float(row["created_at"]),
        image_path=str(row["image_path"]) if row["image_path"] else None,
    )


def delete_planogram(db_path: Path, planogram_id: str) -> bool:
    init_db(db_path)
    row = get_planogram(db_path, planogram_id)
    if row is None:
        return False
    if row.image_path:
        try:
            Path(row.image_path).unlink(missing_ok=True)
        except OSError:
            pass
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM planograms WHERE id = ?", (planogram_id,))
        conn.commit()
    return True


def expected_slots_count(csv_text: str) -> int:
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return 0
    return len(lines) - 1
