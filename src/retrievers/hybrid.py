# -*- coding: utf-8 -*-
"""Гибридный ретривер: слияние BM25 (лексика) и dense (семантика) через RRF.

Reciprocal Rank Fusion:  score(d) = sum_r  w_r / (k_rrf + rank_r(d))

где rank_r(d) — позиция документа d в ранжировании ретривера r (1-based). RRF
работает с рангами, а не с сырыми скорами, поэтому не требует нормализации разных
по природе величин (BM25-скор vs косинус) — это его главное удобство.

Идея: семантика закрывает лексический разрыв (recall@10≈0.80), а BM25 добавляет
точные словесные совпадения там, где эмбеддинги путают близкие статьи.

Запускать под torch_env (нужен dense):
    "<torch_env>/python.exe" scripts/run_eval.py --algo hybrid
"""
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from ..config import K
from .base import BaseRetriever
from .bm25 import Retriever as BM25Retriever
from .dense import Retriever as DenseRetriever


class Retriever(BaseRetriever):

    name = "hybrid"

    # дефолты из свипа: усиленный вес семантики (dense сильнее BM25), k_rrf=30.
    # w_dense=2 — устойчивая точка плато (1.5–3 дают ~тот же MAP), не край грида.
    def __init__(self, k_rrf: int = 30, depth: int = 100,
                 w_bm25: float = 1.0, w_dense: float = 2.0,
                 dense_model: str = "ai-forever/ru-en-RoSBERTa"):
        self.k_rrf = int(k_rrf)          # сглаживающая константа RRF (стандарт 60)
        self.depth = int(depth)          # сколько кандидатов брать от каждого ретривера
        self.w_bm25, self.w_dense = float(w_bm25), float(w_dense)
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model=dense_model)

    def fit(self, articles: pd.DataFrame) -> "Retriever":
        self.bm25.fit(articles)
        self.dense.fit(articles)
        return self

    @staticmethod
    def _rrf_add(scores: dict, ranked: List[int], weight: float, k_rrf: int):
        """Добавляет вклад одного ранжирования в накопитель RRF-скоров."""
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] += weight / (k_rrf + rank)

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        # берём кандидатов глубже, чем нужно на выходе, чтобы слияние было осмысленным
        bm25_pred = self.bm25.rank(queries, k=self.depth)
        dense_pred = self.dense.rank(queries, k=self.depth)

        preds = {}
        for qid in queries["query_id"].astype(int):
            scores = defaultdict(float)
            self._rrf_add(scores, bm25_pred.get(qid, []), self.w_bm25, self.k_rrf)
            self._rrf_add(scores, dense_pred.get(qid, []), self.w_dense, self.k_rrf)
            ranked = sorted(scores, key=scores.get, reverse=True)[:k]
            preds[qid] = ranked
        return preds
