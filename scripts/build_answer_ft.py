# -*- coding: utf-8 -*-
"""Генерация answer.csv на test.f ФИНАЛЬНЫМ FT-пайплайном.

Пайплайн (все гиперпараметры подобраны на dev-400, проверено на held-out test-100):
  BM25 + RoSBERTa-ft + e5-large-ft  ->  RRF (w_bm25=1.5, w_dense=2, k_rrf=10)
  ->  ДООБУЧЕННЫЙ cross-encoder reranker (best_chunk) blend (w_rerank=5, k_blend=5, top_n=30)

Дообученные модели — в .emb_cache/*_ftsup и bge-reranker-v2-m3_ft (обучены супервизно
на dev-400). Инференс детерминирован (фиксируем сиды + cuDNN) → воспроизводимый
answer.csv. Требует torch/sentence-transformers (torch_env), ~15–20 мин GPU.

    "<torch_env>/python.exe" scripts/build_answer_ft.py
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMB_CACHE, K
from src.data import load_articles, load_test
from src.retrievers.rerank import Retriever
from build_answer import validate          # переиспользуем проверки формата

# Модели на HuggingFace (дообученное решение).
HF = {"ros": "iliakabanov/russian-dense-retriever",
      "e5": "iliakabanov/multillingual-dense-retriever",
      "rr": "iliakabanov/reranker-retriever"}
# Локальные (после обучения через train_dense/train_reranker).
LOCAL = {"ros": str(EMB_CACHE / "ai-forever_ru-en-RoSBERTa_ftsup"),
         "e5": str(EMB_CACHE / "intfloat_multilingual-e5-large_ftsup"),
         "rr": str(EMB_CACHE / "bge-reranker-v2-m3_ft")}

OUT = PROJECT_ROOT / "answer.csv"


def resolve_models(use_local: bool):
    """Пути моделей: --local -> из .emb_cache, иначе HF. Env ROS_FT/E5_FT/RR_FT — приоритетнее."""
    src = LOCAL if use_local else HF
    ros = os.environ.get("ROS_FT", src["ros"])
    e5 = os.environ.get("E5_FT", src["e5"])
    rr = os.environ.get("RR_FT", src["rr"])
    return ros, e5, rr


def set_determinism(seed: int = 42):
    """Фиксируем сиды и cuDNN — для воспроизводимого кодирования/инференса."""
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser(description="Генерация answer.csv финальным FT-пайплайном")
    ap.add_argument("--local", action="store_true",
                    help="использовать локальные модели из .emb_cache вместо HF")
    args = ap.parse_args()

    set_determinism(42)
    ros_ft, e5_ft, rr_ft = resolve_models(args.local)
    print(f"модели: {'локальные (.emb_cache)' if args.local else 'HuggingFace'}", flush=True)

    articles = load_articles()
    test = load_test()
    valid_ids = set(articles["article_id"])

    retriever = Retriever(
        # база: FT-гибрид с весами, подобранными на dev-400
        dense_models=f"{ros_ft},{e5_ft}",
        w_bm25=1.5, w_dense=2, k_rrf=10,
        # ДООБУЧЕННЫЙ reranker поверх: best_chunk + blend (параметры с dev-400)
        model=rr_ft, max_length=256,
        best_chunk=True, blend=True,
        top_n=30, w_rerank=5.0, k_blend=5,
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
