# -*- coding: utf-8 -*-
"""Общие пути и константы проекта."""
from pathlib import Path

# корень репозитория = на уровень выше папки src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# папка с выданными данными (articles.f / calibration.f / test.f)
DATA_DIR = PROJECT_ROOT / "data" / "candidate_data"

# кеш эмбеддингов статей (dense-ретривер); безопасно удалять — пересчитается
EMB_CACHE = PROJECT_ROOT / ".emb_cache"

# горизонт метрики MAP@K и максимум статей в ответе (из условия задачи)
K = 10
