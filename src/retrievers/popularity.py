# -*- coding: utf-8 -*-
"""Baseline «популярность» — опорная нижняя граница.

Возвращает всем запросам один и тот же список самых частых статей из разметки
calibration. Это не содержательный поиск, а reference-бейзлайн: из-за сильной
смещённости calibration он неожиданно силён (MAP@10 ~ 0.32) и служит планкой,
которую осмысленный алгоритм обязан превзойти.

Важно: использует ground_truth из calibration, поэтому годится только как ориентир,
не как решение для test.
"""
from collections import Counter
from typing import Dict, List

import pandas as pd

from ..config import K
from ..data import load_calibration
from .base import BaseRetriever


class Retriever(BaseRetriever):

    name = "popularity"

    def __init__(self, top_n: int = K):
        self.top_n = top_n
        self._ranked: List[int] = []

    def fit(self, articles: pd.DataFrame) -> "Retriever":
        # частота статей как правильных ответов в calibration
        cal = load_calibration()
        counts = Counter(i for ids in cal["gt_ids"] for i in ids)
        self._ranked = [aid for aid, _ in counts.most_common(self.top_n)]
        return self

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        top = self._ranked[:k]
        return {int(q): list(top) for q in queries["query_id"]}
