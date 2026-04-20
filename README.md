# Merch Analyzer

Веб-приложение на **FastAPI** для разбора фото полки с напитками: детекция позиций (**SKU110K** / RetinaNet), сохранение кропов и опциональное распознавание названий через **LM Studio** (OpenAI-совместимый vision API).

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)

Краткая схема этапов и стеков: см. [PIPELINE.md](PIPELINE.md).

## Возможности

- **Шаг 1 — разбор эталона** (`POST /reference/save`): загрузка фото → детекция боксов SKU110K → сохранение `input.jpg`, кропов, разметки с полками (оценка рядов по вертикали bbox) → `result.json` и запись в `data/reference_by_sku.json`.
- **Шаг 2 — распознавание** (`POST /recognize`): кропы из выбранного разбора → LM Studio (`/v1/chat/completions`) → подписи на изображении, `annotated_lm.jpg`, отдельный прогон в `data/sku_results/lm_recognition/`.
- **Режимы LM**: по умолчанию пакетный запрос (`LM_BATCH_CLASSIFY_SINGLE_REQUEST=true`, `classify_crops_batch_chunked`); иначе по одному кропу, или один запрос на кластер похожих кропов (`LM_SHARED_CLASSIFY_PER_SIMILARITY_GROUP` + `SIMILARITY_THRESHOLD`).
- **Вспомогательные модули** (для тестов и дальнейшей интеграции): `app/planogram.py`, `planogram_compare.py`, `planogram_store.py`, `item_validation.py` (каталог, fuzzy-сопоставление — см. код и `tests/`).

## Требования

- Python **3.10+**
- Зависимости из `requirements.txt` (в т.ч. FastAPI, Uvicorn, Pillow, NumPy, Pandas, RapidFuzz)
- Для детекции: репозиторий **[SKU110K_CVPR19](https://github.com/eg4000/SKU110K_CVPR19)** и файл весов `sku110k_pretrained.h5`
- Рекомендуется **Docker** для инференса SKU110K на **TensorFlow 1.15** (или отдельное окружение `SKU110K_PYTHON_BIN` / WSL)
- Для шага 2: запущенный **LM Studio** с vision-моделью

## Быстрый старт

### 1. Клонирование и виртуальное окружение приложения

```powershell
cd merch_analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. SKU110K и веса

```bash
git clone https://github.com/eg4000/SKU110K_CVPR19.git third_party/SKU110K_CVPR19
```

Веса (~100 MB): [Google Drive](https://drive.google.com/file/d/1f9tRzJSqjuUQzXz8WjJC0V_WD-8y_6wy/view?usp=sharing) → сохранить как `models/sku110k_pretrained.h5`.

### 3. Docker для SKU110K (по желанию)

```powershell
.\scripts\build_sku110k_docker.ps1
```

```bash
docker build -t merch-analyzer-sku110k:tf1.15 -f docker/sku110k/Dockerfile .
```

Без GPU: `SKU110K_DOCKER_USE_GPU=false`.

### 4. Конфигурация

Скопируйте `env.example` в файл `env` или `.env` в корне проекта (оба подхватываются `run.ps1`) и поправьте URL/модель LM Studio и пути SKU110K.

### 5. Запуск

```powershell
.\run.ps1
```

Порт по умолчанию **8000**; пример: `.\run.ps1 -Port 8010`.

Интерфейс: **http://127.0.0.1:8000**

Альтернативная точка входа Uvicorn (эквивалент приложения):

```bash
uvicorn src.web_app:app --reload --host 0.0.0.0 --port 8000
```

## Структура проекта

```
merch_analyzer/
├── app/
│   ├── main.py              # FastAPI: эталон, recognize, выдача файлов
│   ├── sku110k_adapter.py # вызов предикта SKU110K (docker / wsl / native)
│   ├── lmstudio_client.py # HTTP к LM Studio, batch / single crop
│   ├── analytics.py       # Detection, привязка к полкам/позициям
│   ├── similarity.py      # кластеризация похожих кропов для LM
│   ├── planogram*.py      # планограммы, SQLite-хранилище (библиотека)
│   └── item_validation.py # нормализация, каталог (библиотека)
├── templates/index.html
├── docker/sku110k/
├── scripts/                 # сборка Docker, бенчмарки LM
├── data/                    # результаты прогонов (создаётся при работе)
├── tests/
├── run.ps1
├── env.example
├── PIPELINE.md
└── requirements.txt
```

## API (кратко)

| Метод | Путь | Назначение |
|--------|------|------------|
| GET | `/` | HTML UI |
| POST | `/reference/save` | эталонное фото + имя разметки → SKU110K |
| GET | `/reference/{sku}`, `/reference/history/{sku}` | последний / история по ключу |
| GET | `/reference-folder/history` | история с диска |
| POST | `/recognize` | LM по кропам выбранного `reference_result_dir` |
| GET | `/lm-recognition/history` | сохранённые прогоны шага 2 |
| GET | `/result-file/{category}/{run_id}/{path}` | раздача файлов из `data/sku_results/` |

## Переменные окружения

Основные (полный список и комментарии — в `env.example`):

| Переменная | По умолчанию (код) | Описание |
|------------|-------------------|----------|
| `SKU110K_REPO_PATH` | `third_party/SKU110K_CVPR19` | путь к репозиторию SKU110K |
| `SKU110K_WEIGHTS_PATH` | `models/sku110k_pretrained.h5` | веса |
| `SKU110K_PYTHON_BIN` | `.venv_sku/Scripts/python.exe` | Python для native-режима (Windows) |
| `SKU110K_RUN_MODE` | `docker` | `auto` \| `docker` \| `native` \| `wsl` |
| `SKU110K_DOCKER_USE_GPU` | `true` | GPU в Docker |
| `SKU110K_SCORE_THRESHOLD` | `0.7` | порог score (переопределяется через env) |
| `LMSTUDIO_URL` | см. `app/main.py` | базовый URL LM Studio |
| `LMSTUDIO_MODEL` | см. `app/main.py` | идентификатор модели |
| `LMSTUDIO_TIMEOUT_SEC` | `25` | таймаут HTTP |
| `LM_CONCURRENT` | `1` | параллельные запросы на шаге `/recognize` |
| `SIMILARITY_THRESHOLD` | `0.88` | порог для объединения похожих кропов |
| `LM_BATCH_CLASSIFY_SINGLE_REQUEST` | `true` | один запрос на пачку кропов (JSON) |
| `LM_SHARED_CLASSIFY_PER_SIMILARITY_GROUP` | `false` | один запрос на кластер похожих кропов |
| `LM_RECOGNIZE_MAX_POSITIONS` | `0` | лимит позиций для LM (`0` = без лимита) |
| `ASSORTMENT_CATALOG_PATH` | — | в `env.example`; логика в `item_validation` (не подключена к `/recognize` в текущей версии) |

## LM Studio

Клиент обращается к `POST {LMSTUDIO_URL}/v1/chat/completions`. Формат ответа модели для одиночного кропа: одна строка вида `Напиток: …` или `Ничего не обнаружено` (см. системный промпт в `app/lmstudio_client.py`). При ошибке сети или непарсируемом ответе позиция получает `unknown` / статус ошибки, **детекция SKU110K при этом не ломается**.

## Тесты

```bash
pytest tests
```

## Лицензия

MIT
