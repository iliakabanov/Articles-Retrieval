# -*- coding: utf-8 -*-
"""Базовый интерфейс ретривера.

Каждый алгоритм живёт в отдельном файле src/retrievers/<name>.py и определяет класс
`Retriever(BaseRetriever)`. Раннер получает его по имени через get_retriever().
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import K


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы top-k по убыванию score (устойчиво к k >= len)."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]        # k наибольших без полной сортировки
    return idx[np.argsort(-scores[idx])]             # упорядочиваем эти k


class BaseRetriever:
    """Контракт ретривера: fit(articles) -> self; rank(queries, k) -> предсказания."""

    name = "base"

    def fit(self, articles: pd.DataFrame) -> "BaseRetriever":
        """Построить индекс по корпусу статей. Возвращает self."""
        raise NotImplementedError

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        """{query_id: [article_id, ...]} — ранжированный топ-k для каждого запроса."""
        raise NotImplementedError
