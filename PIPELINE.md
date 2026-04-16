# Пайплайн Merch Analyzer (кратко)

Два последовательных этапа в UI/API и общий поток данных.

## Этап 1: разбор эталона (`POST /reference/save`)

1. **Вход**: JPEG/PNG полки, строковый ключ разметки (`sku`).
2. **Нормализация изображения**: Pillow — EXIF-orientation, RGB (`app/main.py`).
3. **Детекция**: `SKU110KDetector` (`app/sku110k_adapter.py`) запускает код репозитория SKU110K с весами RetinaNet; режим выполнения задаётся `SKU110K_RUN_MODE` (Docker с TF 1.15, WSL или локальный Python).
4. **Постобработка**: для каждого bbox — кроп на диск, отрисовка боксов; грубая **оценка полок** по кластеризации центров bbox по вертикали (`_estimate_shelf_layout`).
5. **Выход**: каталог прогона `data/sku_results/reference/<timestamp>_…/` — `input.jpg`, `crops/`, `annotated.jpg`, `result.json`; метаданные дублируются в `data/reference_by_sku.json`.

**Стек этапа 1**: FastAPI, Pillow, subprocess/Docker → внешний SKU110K (Keras/TF 1.x), CSV-датасет для одного кадра во временной папке репозитория.

## Этап 2: распознавание напитков (`POST /recognize`)

1. **Вход**: тот же ключ `sku` + `reference_result_dir` (папка этапа 1).
2. **Загрузка**: полное изображение и список позиций из `result.json`; кропы из файлов или повторный crop по bbox.
3. **Опционально**: ограничение числа позиций (`LM_RECOGNIZE_MAX_POSITIONS`).
4. **LM Studio**: `LMStudioClient` — HTTP `chat/completions` с `image_url` (data URL JPEG). Режимы:
   - по одному кропу с опциональным `LM_CONCURRENT` и повтором при плохом ответе (`LM_RECHECK_UNKNOWN`);
   - **batch** — `classify_crops_batch_chunked` (несколько изображений в одном сообщении, ответ JSON);
   - **shared** — кластеры похожих кропов (`app/similarity.py`, порог `SIMILARITY_THRESHOLD`), один запрос на кластер.
5. **Выход**: `data/sku_results/lm_recognition/<run>/` — `annotated_lm.jpg`, `result.json` с полями `per_position` / `visual`.

**Стек этапа 2**: urllib, Pillow (ресайз/качество JPEG из env), потоки `ThreadPoolExecutor` при параллельных запросах.

## Вспомогательная логика (вне основного двухшагового UI)

- **`app/analytics.py`**: модель `Detection`, привязка детекций к полкам и позициям в ряду (для согласованной геометрии/отчётов).
- **`app/planogram*.py`, `planogram_compare.py`**: шаблон планограммы (CSV/JSON/текст), приоритет источников `resolve_planogram_source`, сравнение с фактическими именами; SQLite в `planogram_store` — для сценариев хранения планов.
- **`app/item_validation.py`**: нормализация строк, опциональный каталог и fuzzy (RapidFuzz); переменные `ASSORTMENT_*`, `CATALOG_*`, `GROUP_*` в `env.example` ориентированы на эту подсистему.

## Зависимости «снаружи» репозитория

| Компонент | Назначение |
|-----------|------------|
| Клон `SKU110K_CVPR19` | инференс RetinaNet по их скриптам |
| Файл `.h5` весов | загрузка обученной модели |
| Docker-образ `merch-analyzer-sku110k:tf1.15` | изолированный TF 1.15 |
| LM Studio | локальный сервер с vision-моделью |

## Запуск приложения

`run.ps1` поднимает Uvicorn с `app.main:app`, подгружает переменные из файла `env` или `.env` в корне проекта.
