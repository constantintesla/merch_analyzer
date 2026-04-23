"""Настройка логов Merch Analyzer (консоль stderr).

Уровень: переменная окружения MERCH_LOG_LEVEL (по умолчанию INFO).
Дочерние логгеры merch_analyzer.* наследуют обработчик родителя.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_merch_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    level_name = (os.getenv("MERCH_LOG_LEVEL") or "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger("merch_analyzer")
    root.setLevel(level)
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        root.addHandler(h)
    root.propagate = False
