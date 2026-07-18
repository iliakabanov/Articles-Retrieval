# -*- coding: utf-8 -*-
"""Генерация answer.csv на test.f ФИНАЛЬНЫМ FT-пайплайном.

Пайплайн (все гиперпараметры подобраны на dev-400, проверено на held-out test-100):
  BM25 + RoSBERTa-ft + e5-large-ft  ->  RRF (w_bm25=1.5, w_dense=2, k_rrf=10)
  ->  cross-encoder reranker (best_chunk) blend (w_rerank=0.3, k_blend=10, top_n=50)

Дообученные dense — в .emb_cache/*_ftsup (обучены супервизно на dev-400, эпоха 1).
Требует torch/sentence-transformers (torch_env). Прогон reranker на 500 запросах —
~15–20 мин GPU.

    "<torch_env>/python.exe" scripts/build_answer_ft.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMB_CACHE, K
from src.data import load_articles, load_test
from src.retrievers.rerank import Retriever
from build_answer import validate          # переиспользуем проверки формата

# дообученные dense-модели (абсолютные пути — чтобы совпал ключ кеша эмбеддингов)
ROS_FT = str(EMB_CACHE / "ai-forever_ru-en-RoSBERTa_ftsup")
E5_FT = str(EMB_CACHE / "intfloat_multilingual-e5-large_ftsup")

OUT = PROJECT_ROOT / "answer.csv"


def main():
    articles = load_articles()
    test = load_test()
    valid_ids = set(articles["article_id"])

    retriever = Retriever(
        # база: FT-гибрид с весами, подобранными на dev-400
        dense_models=f"{ROS_FT},{E5_FT}",
        w_bm25=1.5, w_dense=2, k_rrf=10,
        # reranker поверх: best_chunk + blend (параметры с dev-400)
        best_chunk=True, blend=True,
        top_n=50, w_rerank=0.3, k_blend=10,
        progress=True,
    ).fit(articles)

    preds = retriever.rank(test, k=K)

    rows = [{"query_id": int(q),
             "answer": " ".join(str(i) for i in preds.get(int(q), [])[:K])}
            for q in test["query_id"]]
    answer = pd.DataFrame(rows)

    validate(answer, test, valid_ids, K)     # все проверки формата
    answer.to_csv(OUT, index=False)
    print(f"\nOK: {OUT} ({len(answer)} строк, все проверки пройдены)")
    print(answer.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
