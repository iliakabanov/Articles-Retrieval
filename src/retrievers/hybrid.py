# -*- coding: utf-8 -*-
"""Гибридный ретривер: слияние BM25 (лексика) и одной-или-нескольких dense-моделей
(семантика) через RRF.

Reciprocal Rank Fusion:  score(d) = sum_r  w_r / (k_rrf + rank_r(d))

где rank_r(d) — позиция документа d в ранжировании ретривера r (1-based). RRF
работает с рангами, а не с сырыми скорами, поэтому не требует нормализации разных
по природе величин (BM25-скор vs косинус) — это его главное удобство.

По умолчанию — 3-way ансамбль: BM25 + RoSBERTa + e5-large. Разные модели ошибаются
по-разному, их согласие усиливает правильные статьи (на calibration 3-way даёт
~0.52 против ~0.50 у 2-way). Число dense-моделей задаётся списком `dense_models`.

Запускать под torch_env (нужен dense):
    "<torch_env>/python.exe" scripts/run_eval.py --algo hybrid
    ... --param dense_models=ai-forever/ru-en-RoSBERTa   # 2-way, только одна модель
"""
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from ..config import K
from .base import BaseRetriever
from .bm25 import Retriever as BM25Retriever
from .dense import Retriever as DenseRetriever

# ансамбль по умолчанию: две сильнейшие dense-модели (обе заметно сильнее BM25)
DEFAULT_DENSE = "ai-forever/ru-en-RoSBERTa,intfloat/multilingual-e5-large"


class Retriever(BaseRetriever):

    name = "hybrid"

    # дефолты из свипа: dense-источники весомее BM25 (1 : 2 : 2), k_rrf=30 —
    # «простые» round-веса, устойчивее к сдвигу, чем вылизанные под calibration.
    def __init__(self, k_rrf: int = 30, depth: int = 100,
                 w_bm25: float = 1.0, w_dense: float = 2.0,
                 dense_models: str = DEFAULT_DENSE):
        self.k_rrf = int(k_rrf)          # сглаживающая константа RRF (стандарт 60)
        self.depth = int(depth)          # сколько кандидатов брать от каждого ретривера
        self.w_bm25, self.w_dense = float(w_bm25), float(w_dense)
        self.dense_model_names = [m.strip() for m in dense_models.split(",") if m.strip()]
        self.bm25 = BM25Retriever()
        self.dense_list = [DenseRetriever(model=m) for m in self.dense_model_names]

    def fit(self, articles: pd.DataFrame) -> "Retriever":
        self.bm25.fit(articles)
        for d in self.dense_list:
            d.fit(articles)
        return self

    @staticmethod
    def _rrf_add(scores: dict, ranked: List[int], weight: float, k_rrf: int):
        """Добавляет вклад одного ранжирования в накопитель RRF-скоров."""
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] += weight / (k_rrf + rank)

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        # берём кандидатов глубже, чем нужно на выходе, чтобы слияние было осмысленным
        bm25_pred = self.bm25.rank(queries, k=self.depth)
        dense_preds = [d.rank(queries, k=self.depth) for d in self.dense_list]

        preds = {}
        for qid in queries["query_id"].astype(int):
            scores = defaultdict(float)
            self._rrf_add(scores, bm25_pred.get(qid, []), self.w_bm25, self.k_rrf)
            for dp in dense_preds:
                self._rrf_add(scores, dp.get(qid, []), self.w_dense, self.k_rrf)
            ranked = sorted(scores, key=scores.get, reverse=True)[:k]
            preds[qid] = ranked
        return preds
