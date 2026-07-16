# -*- coding: utf-8 -*-
"""Переранжирование топ-кандидатов cross-encoder'ом (Этап 4).

Схема retrieve → rerank: базовый ретривер (гибрид) быстро отбирает top_n кандидатов,
затем cross-encoder оценивает каждую пару (запрос, статья) целиком и переупорядочивает
их. Cross-encoder точнее bi-encoder'а (видит запрос и документ во взаимодействии), но
дорог, поэтому применяется только к небольшому списку кандидатов.

При recall@10≈0.84 у гибрида нужные статьи уже в топе — reranker поднимает их выше.

Запускать под torch_env:
    "<torch_env>/python.exe" scripts/run_eval.py --algo rerank
"""
from collections import defaultdict
from typing import Dict, List

import pandas as pd

from ..config import K
from ..text import clean_text
from . import get_retriever
from .base import BaseRetriever


class Retriever(BaseRetriever):

    name = "rerank"

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", base: str = "hybrid",
                 top_n: int = 30, max_length: int = 512, batch_size: int = 32,
                 max_doc_chars: int = 2000, device: str = None,
                 progress: bool = False,
                 blend: bool = True, w_base: float = 1.0, w_rerank: float = 0.3,
                 k_blend: int = 10):
        self.model_name = model
        self.top_n = int(top_n)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.max_doc_chars = int(max_doc_chars)
        self.device = device
        self.progress = bool(progress)
        # blend: смешивать ранг reranker'а с рангом базового гибрида через RRF.
        # Чистый reranker на этой смещённой разметке проигрывает (демотит хаб-статьи),
        # поэтому по умолчанию смешиваем, сохраняя приор гибрида.
        self.blend = bool(blend)
        self.w_base, self.w_rerank = float(w_base), float(w_rerank)
        self.k_blend = int(k_blend)
        self.base = get_retriever(base)()      # базовый ретривер-кандидатогенератор

    def _load_model(self):
        from sentence_transformers import CrossEncoder
        import torch
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return CrossEncoder(self.model_name, device=device, max_length=self.max_length)

    def fit(self, articles: pd.DataFrame) -> "Retriever":
        self.base.fit(articles)
        # текст статьи для reranker'а: заголовок + тело (обрезаем — cross-encoder всё
        # равно усекает до max_length токенов, длинный хвост не нужен)
        self.id2text = {
            int(aid): clean_text(f"{title}. {body}")[:self.max_doc_chars]
            for aid, title, body in zip(articles["article_id"],
                                        articles["title"], articles["body_text"])
        }
        return self

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        # 1) быстрый отбор кандидатов базовым ретривером
        candidates = self.base.rank(queries, k=self.top_n)

        # 2) собираем ВСЕ пары (запрос, статья) в один батч для cross-encoder'а
        q2text = {int(q): clean_text(t)
                  for q, t in zip(queries["query_id"], queries["query_text"])}
        pairs, index = [], []
        for qid in queries["query_id"].astype(int):
            for doc in candidates.get(qid, []):
                pairs.append((q2text[qid], self.id2text.get(doc, "")))
                index.append((qid, doc))

        model = getattr(self, "_model", None) or self._load_model()
        self._model = model
        scores = model.predict(pairs, batch_size=self.batch_size,
                               show_progress_bar=self.progress)

        # 3) переупорядочиваем кандидатов каждого запроса
        by_q = defaultdict(list)
        for (qid, doc), s in zip(index, scores):
            by_q[qid].append((doc, float(s)))

        preds = {}
        for qid in queries["query_id"].astype(int):
            rer_list = [doc for doc, _ in
                        sorted(by_q[qid], key=lambda x: x[1], reverse=True)]
            if not self.blend:
                preds[qid] = rer_list[:k]
                continue
            # RRF-смешивание: ранг базового гибрида + ранг reranker'а
            base_list = candidates.get(qid, [])
            score = defaultdict(float)
            for rank, doc in enumerate(base_list, start=1):
                score[doc] += self.w_base / (self.k_blend + rank)
            for rank, doc in enumerate(rer_list, start=1):
                score[doc] += self.w_rerank / (self.k_blend + rank)
            preds[qid] = sorted(score, key=score.get, reverse=True)[:k]
        return preds
