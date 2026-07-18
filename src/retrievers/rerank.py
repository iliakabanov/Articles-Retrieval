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
                 top_n: int = 20, max_length: int = 512, batch_size: int = 32,
                 max_doc_chars: int = 2000, device: str = None,
                 progress: bool = False,
                 blend: bool = True, w_base: float = 1.0, w_rerank: float = 0.75,
                 k_blend: int = 5, best_chunk: bool = True,
                 dense_models: str = None, w_bm25: float = None,
                 w_dense: float = None, k_rrf: int = None):
        self.model_name = model
        self.top_n = int(top_n)
        # best_chunk: подавать cross-encoder'у лучший dense-чанк статьи, а не «голову»
        # (длинные статьи иначе усекаются до первых ~512 токенов, релевантный кусок теряется)
        self.best_chunk = bool(best_chunk)
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
        # опциональная конфигурация базового гибрида (FT-модели, веса RRF)
        base_kwargs = {}
        if base == "hybrid":
            if dense_models is not None: base_kwargs["dense_models"] = dense_models
            if w_bm25 is not None: base_kwargs["w_bm25"] = float(w_bm25)
            if w_dense is not None: base_kwargs["w_dense"] = float(w_dense)
            if k_rrf is not None: base_kwargs["k_rrf"] = int(k_rrf)
        self.base = get_retriever(base)(**base_kwargs)   # базовый ретривер-кандидатогенератор

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

    def _free_base_gpu(self):
        """Освобождает VRAM dense-моделей базы: после отбора кандидатов они не нужны,
        а reranker тяжёлый — иначе на 8 ГБ GPU три модели сразу дают OOM."""
        try:
            import gc
            import torch
            for d in getattr(self.base, "dense_list", []):
                d._model = None
            if getattr(self.base, "_model", None) is not None:
                self.base._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        # 1) быстрый отбор кандидатов базовым ретривером
        candidates = self.base.rank(queries, k=self.top_n)

        # (опц.) текст документа для reranker'а = лучший dense-чанк, а не «голова».
        # Считаем ДО освобождения GPU — нужна dense-модель базы (RoSBERTa) для запросов.
        pair_text = None
        if self.best_chunk and getattr(self.base, "dense_list", None):
            pair_text = self.base.dense_list[0].best_chunk_texts(queries, candidates)

        self._free_base_gpu()          # освобождаем VRAM под тяжёлый cross-encoder

        # 2) собираем ВСЕ пары (запрос, статья) в один батч для cross-encoder'а
        q2text = {int(q): clean_text(t)
                  for q, t in zip(queries["query_id"], queries["query_text"])}
        pairs, index = [], []
        for qid in queries["query_id"].astype(int):
            for doc in candidates.get(qid, []):
                doc_text = (pair_text.get((qid, doc)) if pair_text is not None
                            else None) or self.id2text.get(doc, "")
                pairs.append((q2text[qid], doc_text))
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
