"""
Сравнение скорости: per-crop (как /recognize до batch) vs classify_crops_batch_chunked.
Запуск из корня проекта; переменные LMSTUDIO_* из env или окружения процесса.

Опции окружения: BENCH_CROPS_DIR, BENCH_LM_CONCURRENT (по умолчанию 4), BENCH_MAX_CROPS (0 = все).
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.lmstudio_client import LMStudioClient  # noqa: E402


def _load_dotenv_simple(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def main() -> int:
    crops_dir = Path(
        os.environ.get(
            "BENCH_CROPS_DIR",
            str(ROOT / "data/sku_results/reference/20260414_181109_input/crops"),
        )
    )
    _load_dotenv_simple(ROOT / "env")
    _load_dotenv_simple(ROOT / ".env")

    # Без повторных запросов — batch их не делает; так сравнение по «одному проходу».
    os.environ.setdefault("LM_RECHECK_UNKNOWN", "0")

    paths = sorted(crops_dir.glob("crop_*.jpg"))
    if not paths:
        print(f"Нет crop_*.jpg в {crops_dir}", file=sys.stderr)
        return 1

    max_crops_raw = (os.environ.get("BENCH_MAX_CROPS") or "0").strip()
    try:
        max_crops = int(max_crops_raw)
    except ValueError:
        max_crops = 0
    if max_crops > 0:
        paths = paths[:max_crops]

    crops: list[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            crops.append(im.convert("RGB").copy())

    n = len(crops)
    base_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234")
    model = os.environ.get("LMSTUDIO_MODEL", "local-model")
    timeout = float(os.environ.get("LMSTUDIO_TIMEOUT_SEC", "120"))
    lm = LMStudioClient(base_url=base_url, model=model, timeout_sec=timeout)

    try_concurrent = int(os.environ.get("BENCH_LM_CONCURRENT", "4"))

    print(f"Кропов: {n}, LM: {base_url}, модель: {model}")
    print(f"LM_RECHECK_UNKNOWN={os.environ.get('LM_RECHECK_UNKNOWN')}, BENCH_LM_CONCURRENT={try_concurrent}")
    print()

    # 1) Последовательно, один запрос на кроп (как LM_CONCURRENT=1)
    t0 = time.perf_counter()
    seq = [lm.classify_crop(c, retry=False) for c in crops]
    t_seq = time.perf_counter() - t0
    def _ok_cnt(xs: list) -> tuple[int, int]:
        ok = sum(1 for x in xs if x.status != "lmstudio_error")
        return ok, n - ok

    o, err = _ok_cnt(seq)
    print(
        f"Per-crop последовательно:     {t_seq:8.2f} с  ({t_seq / n:.2f} с/кроп)  "
        f"ok={o} err_lm={err}"
    )

    # 2) Параллельно (как старый режим с LM_CONCURRENT>1)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, try_concurrent)) as pool:
        par = list(pool.map(lambda c: lm.classify_crop(c, retry=False), crops))
    t_par = time.perf_counter() - t0
    o, err = _ok_cnt(par)
    print(
        f"Per-crop parallel x{try_concurrent}: {t_par:8.2f} с  ({t_par / n:.2f} с/кроп)  "
        f"ok={o} err_lm={err}"
    )

    # 3) Batch (новый режим)
    t0 = time.perf_counter()
    bat = lm.classify_crops_batch_chunked(crops)
    t_bat = time.perf_counter() - t0
    o, err = _ok_cnt(bat)
    print(
        f"Batch chunked:                {t_bat:8.2f} с  ({t_bat / n:.2f} с/кроп)  "
        f"ok={o} err_lm={err}"
    )

    if t_bat > 0:
        print()
        print(f"Ускорение vs последовательный per-crop: {t_seq / t_bat:.2f}x")
        print(f"Ускорение vs parallel x{try_concurrent}:          {t_par / t_bat:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
