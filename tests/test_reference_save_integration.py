"""Медленный интеграционный тест SKU110K + POST /reference/save.

Запуск вручную (нужны Docker-образ, репо SKU110K, веса и sample-фото):

  set MERCH_INTEGRATION_SKU110K=1
  pytest tests/test_reference_save_integration.py -v

Без переменной тест пропускается, чтобы не ломать CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = BASE / "img" / "image-08-04-26-11-48-2.jpg"


@pytest.mark.integration
def test_reference_save_with_sample_jpg() -> None:
    if os.getenv("MERCH_INTEGRATION_SKU110K", "").strip() != "1":
        pytest.skip("Set MERCH_INTEGRATION_SKU110K=1 to run SKU110K + /reference/save integration")

    repo = BASE / "third_party" / "SKU110K_CVPR19"
    weights = BASE / "models" / "sku110k_pretrained.h5"
    if not repo.is_dir():
        pytest.fail(f"SKU110K repo missing: {repo}")
    if not weights.is_file():
        pytest.fail(f"Weights missing: {weights}")
    if not SAMPLE_IMAGE.is_file():
        pytest.fail(f"Sample image missing: {SAMPLE_IMAGE}")

    try:
        import httpx  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("Нужен пакет httpx для TestClient: pip install httpx (указан в requirements.txt)")

    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    with SAMPLE_IMAGE.open("rb") as f:
        r = client.post(
            "/reference/save",
            data={"sku": "integration_sample"},
            files={"reference_image": (SAMPLE_IMAGE.name, f, "image/jpeg")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("ok") is True, body
    assert int(body.get("saved", {}).get("reference_detection", {}).get("positions_count", -1)) >= 0
