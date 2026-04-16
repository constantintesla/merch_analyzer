"""
Точка входа для запуска вида: uvicorn src.web_app:app
(эквивалентно: uvicorn app.main:app)
"""

from app.main import app

__all__ = ["app"]
