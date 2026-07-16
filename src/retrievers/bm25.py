# -*- coding: utf-8 -*-
"""Лексический ретривер BM25 (Okapi) — прозрачная реализация на sklearn/scipy.

score(d, Q) = sum_{t in Q} IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1 - b + b*|d|/avgdl))

Документ статьи собирается из полей title/body/anchors с полевыми весами (вес поля
реализуется повтором его токенов, что увеличивает f(t,d) для слов поля).

Дефолты выбраны по абляциям Этапа 1 (без стемминга работает лучше).
"""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ..config import K
from ..text import tokenize
from .base import BaseRetriever, top_k_indices


class _BM25Core:
    """Чистая математика BM25 поверх терм-документной матрицы счётчиков."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, tokenized_docs: List[List[str]]) -> "_BM25Core":
        # analyzer=identity: на входе уже списки токенов, не строки
        self.vec = CountVectorizer(analyzer=lambda t: t, min_df=1)
        X = self.vec.fit_transform(tokenized_docs)   # (N x V) счётчики
        self.vocab = self.vec.vocabulary_
        self.X = X.tocsc()                           # быстрый доступ по колонкам-термам

        N, V = X.shape
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        dl = np.asarray(X.sum(axis=1)).ravel()
        avgdl = dl.mean() if N else 0.0
        # знаменатель без f(t,d): B_d = k1*(1 - b + b*|d|/avgdl), считаем один раз
        self.B_d = self.k1 * (1 - self.b + self.b * dl / (avgdl or 1.0))
        self.N = N
        return self

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.N)
        for t in set(query_tokens):                  # каждый терм запроса — один раз
            j = self.vocab.get(t)
            if j is None:
                continue
            col = self.X[:, j]
            rows, f = col.indices, col.data.astype(float)
            scores[rows] += self.idf[j] * (f * (self.k1 + 1)) / (f + self.B_d[rows])
        return scores


class Retriever(BaseRetriever):

    name = "bm25"

    # дефолты — умеренный конфиг из свипа (Этап 1): прирост без экстремальных значений
    # BM25 (edge-оптимум k1=3/b=0.25 отвергнут как переобучение под calibration)
    def __init__(self, k1: float = 2.5, b: float = 0.4,
                 w_title: int = 3, w_body: int = 1, w_anchor: int = 0,
                 stem: bool = False, drop_stop: bool = True):
        self.k1, self.b = k1, b
        self.w_title, self.w_body, self.w_anchor = w_title, w_body, w_anchor
        self.tok_kw = dict(stem=stem, drop_stop=drop_stop)

    def _doc_tokens(self, row) -> List[str]:
        toks = tokenize(row["title"], **self.tok_kw) * self.w_title
        toks += tokenize(row["body_text"], **self.tok_kw) * self.w_body
        if self.w_anchor:
            toks += tokenize(row["anchors"], **self.tok_kw) * self.w_anchor
        return toks

    def fit(self, articles: pd.DataFrame) -> "Retriever":
        self.article_ids = articles["article_id"].to_numpy()
        docs = [self._doc_tokens(r) for _, r in articles.iterrows()]
        self._bm25 = _BM25Core(self.k1, self.b).fit(docs)
        return self

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        preds = {}
        for _, r in queries.iterrows():
            q = tokenize(r["query_text"], **self.tok_kw)
            scores = self._bm25.get_scores(q)
            idx = top_k_indices(scores, k)
            preds[int(r["query_id"])] = [int(self.article_ids[i]) for i in idx]
        return preds
