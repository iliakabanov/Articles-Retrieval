# -*- coding: utf-8 -*-
"""Плотный (семантический) ретривер на sentence-transformers.

Закрывает лексический разрыв: матч по смыслу, а не по совпадению слов. Длинные
статьи режутся на пассажи (чанки) + к каждому прибавляется заголовок; скор статьи =
max косинуса по её чанкам. Эмбеддинги нормализуются, поэтому косинус = скалярное
произведение. Матрица эмбеддингов кешируется на диск.

Запускать под окружением с torch/sentence-transformers (torch_env):
    "<torch_env>/python.exe" scripts/run_eval.py --algo dense --param model=intfloat/multilingual-e5-large
"""
import hashlib
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import EMB_CACHE, K
from ..text import clean_text
from .base import BaseRetriever

# у каждой модели свои префиксы для запроса/документа (часть их протокола обучения)
MODEL_PRESETS = {
    "intfloat/multilingual-e5-large": {"query": "query: ", "passage": "passage: "},
    "intfloat/multilingual-e5-base":  {"query": "query: ", "passage": "passage: "},
    "ai-forever/ru-en-RoSBERTa": {"query": "search_query: ", "passage": "search_document: "},
    "deepvk/USER-bge-m3": {"query": "", "passage": ""},
}


class Retriever(BaseRetriever):

    name = "dense"

    # дефолт — ru-en-RoSBERTa: на calibration лучше e5-large (MAP 0.446 vs 0.422)
    def __init__(self, model: str = "ai-forever/ru-en-RoSBERTa",
                 chunk_words: int = 200, overlap: int = 40,
                 batch_size: int = 64, device: str = None,
                 progress: bool = False):
        self.model_name = model
        self.chunk_words = int(chunk_words)
        self.overlap = int(overlap)
        self.batch_size = int(batch_size)
        self.device = device
        self.progress = bool(progress)
        pre = MODEL_PRESETS.get(model, {"query": "", "passage": ""})
        self.q_prefix, self.p_prefix = pre["query"], pre["passage"]

    # ------------------------------------------------------------------ чанкинг
    def _chunks(self, title: str, body: str) -> List[str]:
        """Пассажи ~chunk_words слов с перекрытием; к каждому спереди — заголовок."""
        title = clean_text(title)
        words = clean_text(body).split()
        if len(words) <= self.chunk_words:
            segs = [" ".join(words)]
        else:
            step = max(1, self.chunk_words - self.overlap)
            segs = []
            for start in range(0, len(words), step):
                segs.append(" ".join(words[start:start + self.chunk_words]))
                if start + self.chunk_words >= len(words):
                    break
        # заголовок как контекст в каждом чанке (и единственный сигнал для пустых тел)
        return [f"{title}. {s}".strip(" .") or title for s in segs]

    # ------------------------------------------------------------------ модель
    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        import torch
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return SentenceTransformer(self.model_name, device=device)

    def _encode(self, texts: List[str]) -> np.ndarray:
        model = getattr(self, "_model", None) or self._load_model()
        self._model = model
        emb = model.encode(texts, batch_size=self.batch_size,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=self.progress)
        return emb.astype(np.float32)

    # ------------------------------------------------------------------ индекс
    def _cache_path(self, n_chunks: int):
        key = f"{self.model_name}|{self.chunk_words}|{self.overlap}|{n_chunks}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
        safe = self.model_name.replace("/", "_")
        return EMB_CACHE / f"{safe}__cw{self.chunk_words}_ov{self.overlap}__{h}.npy"

    def fit(self, articles: pd.DataFrame) -> "Retriever":
        self.article_ids = articles["article_id"].to_numpy()

        # строим чанки; seg_starts[i] — индекс первого чанка i-й статьи (для max-pool)
        chunk_texts, seg_starts = [], []
        for title, body in zip(articles["title"], articles["body_text"]):
            seg_starts.append(len(chunk_texts))
            chunk_texts.extend(self._chunks(title, body))
        self.seg_starts = np.asarray(seg_starts)
        self.chunk_texts = chunk_texts          # тексты чанков (для реранка по чанку)
        n_chunks = len(chunk_texts)

        # кеш эмбеддингов на диск (воспроизводимо и быстро при повторе)
        cache = self._cache_path(n_chunks)
        if cache.exists():
            self.chunk_emb = np.load(cache)
        else:
            passages = [self.p_prefix + t for t in chunk_texts]
            self.chunk_emb = self._encode(passages)
            EMB_CACHE.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.chunk_emb)
        return self

    def best_chunk_texts(self, queries: pd.DataFrame, candidates: Dict[int, List[int]]) -> dict:
        """{(query_id, article_id): текст лучшего по близости чанка статьи}.

        Для реранка по чанку: вместо «головы» статьи cross-encoder получает тот
        пассаж, что дал max при dense-поиске (наиболее релевантный запросу).
        """
        q_emb = self._encode([self.q_prefix + clean_text(t) for t in queries["query_text"]])
        aid2idx = {int(a): i for i, a in enumerate(self.article_ids)}
        n_chunks = len(self.chunk_texts)
        out = {}
        for row, qid in enumerate(queries["query_id"].astype(int)):
            qv = q_emb[row]
            for aid in candidates.get(qid, []):
                i = aid2idx[aid]
                s = int(self.seg_starts[i])
                e = int(self.seg_starts[i + 1]) if i + 1 < len(self.seg_starts) else n_chunks
                best = s + int(np.argmax(self.chunk_emb[s:e] @ qv))
                out[(qid, aid)] = self.chunk_texts[best]
        return out

    # ------------------------------------------------------------------ поиск
    def rank(self, queries: pd.DataFrame, k: int = K) -> Dict[int, List[int]]:
        q_texts = [self.q_prefix + clean_text(t) for t in queries["query_text"]]
        q_emb = self._encode(q_texts)                       # (Q x d), нормализованы

        sims = q_emb @ self.chunk_emb.T                     # косинус: (Q x n_chunks)
        # max-pool по чанкам каждой статьи -> (Q x n_articles)
        art_scores = np.maximum.reduceat(sims, self.seg_starts, axis=1)

        qids = queries["query_id"].to_numpy()
        preds = {}
        for row, qid in enumerate(qids):
            scores = art_scores[row]
            kk = min(k, scores.shape[0])
            top = np.argpartition(-scores, kk - 1)[:kk]
            top = top[np.argsort(-scores[top])]
            preds[int(qid)] = [int(self.article_ids[i]) for i in top]
        return preds
