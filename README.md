# Merch Analyzer

Веб-приложение для анализа мерчандайзинга полок: загрузка фото -> детекция товаров (SKU110K) -> метрики наполненности и похожих позиций.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)

## Возможности

- **Детекция товаров** - bounding boxes через SKU110K (RetinaNet + Soft-IoU)
- **Визуализация** - боксы поверх фото
- **Аналитика:** количество позиций, заполненность, пустота, плотность выкладки, статистика по рядам
- **Похожие позиции** - поиск визуально схожих товаров на полке

## Требования

- Python 3.10+
- Docker (рекомендуется для инференса SKU110K)
- [SKU110K_CVPR19](https://github.com/eg4000/SKU110K_CVPR19) - репозиторий и веса модели

## Быстрый старт

### 1. Клонирование и подготовка окружения

```bash
git clone https://github.com/constantintesla/merch_analyzer.git
cd merch_analyzer
```

```powershell
# Windows
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

### 2. SKU110K и веса модели

Клонируйте репозиторий SKU110K в `third_party/`:

```bash
git clone https://github.com/eg4000/SKU110K_CVPR19.git third_party/SKU110K_CVPR19
```

Скачайте веса (≈100 MB) и положите в `models/`:

- [Google Drive](https://drive.google.com/file/d/1f9tRzJSqjuUQzXz8WjJC0V_WD-8y_6wy/view?usp=sharing)
- Сохраните как `models/sku110k_pretrained.h5`

### 3. Docker-образ для инференса (рекомендуется)

SKU110K использует TensorFlow 1.15 - на современных системах проще запускать в Docker:

```powershell
# Windows
.\scripts\build_sku110k_docker.ps1
```

```bash
# Linux/macOS
docker build -t merch-analyzer-sku110k:tf1.15 -f docker/sku110k/Dockerfile .
```

Без GPU:

```powershell
$env:SKU110K_DOCKER_USE_GPU="false"
```

### 4. Запуск из корня проекта

```powershell
.\run.ps1
```

Скрипт запускает сервер из корня через локальный Python `.venv`.

Опционально можно указать порт:

```powershell
.\run.ps1 -Port 8010
```

Откройте **http://127.0.0.1:8000**

## Структура проекта

```
merch_analyzer/
├── app/
│   ├── main.py          # FastAPI сервер
│   ├── sku110k_adapter.py
│   ├── analytics.py
│   └── similarity.py
├── templates/
│   └── index.html
├── docker/sku110k/
│   └── Dockerfile
├── models/              # веса -> sku110k_pretrained.h5
├── run.ps1              # запуск из корня
├── third_party/         # SKU110K_CVPR19
└── requirements.txt
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `SKU110K_REPO_PATH` | `third_party/SKU110K_CVPR19` | Путь к SKU110K |
| `SKU110K_WEIGHTS_PATH` | `models/sku110k_pretrained.h5` | Путь к весам |
| `SKU110K_RUN_MODE` | `docker` | `auto` \| `native` \| `wsl` |
| `SKU110K_DOCKER_USE_GPU` | `true` | GPU в Docker |
| `SKU110K_SCORE_THRESHOLD` | `0.25` | Порог детекции |
| `SHELF_ROWS` | `4` | Число рядов полки |
| `SIMILARITY_THRESHOLD` | `0.88` | Порог похожести позиций |

## Лицензия

MIT
