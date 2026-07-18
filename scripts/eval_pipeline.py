# -*- coding: utf-8 -*-
"""Финальный замер FT-пайплайна на dev-400 / test-100.

Строит FT-гибрид (дообученные RoSBERTa+e5) + дообученный reranker, честно:
веса RRF и blend подбираются на dev-400, метрика — на нетронутом test-100.
Скоры reranker'а считаются заново каждый прогон (без кеша) — чтобы после
переобучения моделей не подхватить устаревшие. Долго (~13 мин: cross-encoder по
всем кандидатам).

    "<torch_env>/python.exe" -u scripts/eval_pipeline.py
"""
import gc
import itertools
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_articles, load_calibration, build_truth, split_calibration
from src.metrics import mean_average_precision_at_k
from src.text import clean_text
from src.retrievers.hybrid import Retriever as Hybrid

# по умолчанию — дообученные модели на HuggingFace (можно переопределить env-путём)
ROS_FT = os.environ.get("ROS_FT", "iliakabanov/russian-dense-retriever")
E5_FT = os.environ.get("E5_FT", "iliakabanov/multillingual-dense-retriever")
RR_FT = os.environ.get("RR_FT", "iliakabanov/reranker-retriever")
TOP_N = 50


def main():
    import torch
    from sentence_transformers import CrossEncoder

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    arts = load_articles(); cal = load_calibration(); truth = build_truth(cal)
    dev, test = split_calibration(cal, n_test=100, seed=42)

    def map_sub(preds, df):
        ids = set(df.query_id.astype(int))
        return mean_average_precision_at_k({q: preds[q] for q in ids},
                                           {q: truth[q] for q in ids}, 10)

    # --- FT-гибрид: перетюн весов RRF на dev-400 ---
    print("строю FT-гибрид...", flush=True)
    hyb = Hybrid(dense_models=f"{ROS_FT},{E5_FT}", depth=100).fit(arts)
    # достаём ранги компонент один раз, тюним веса перебором
    bm25_r = {int(q): v for q, v in hyb.bm25.rank(cal, k=100).items()}
    dpred = [{int(q): v for q, v in d.rank(cal, k=100).items()} for d in hyb.dense_list]

    def fuse(w_bm25, w_dense, k_rrf, k=TOP_N):
        preds = {}
        for qid in cal.query_id.astype(int):
            s = defaultdict(float)
            for r, d in enumerate(bm25_r[qid], 1): s[d] += w_bm25 / (k_rrf + r)
            for dp in dpred:
                for r, d in enumerate(dp[qid], 1): s[d] += w_dense / (k_rrf + r)
            preds[qid] = sorted(s, key=s.get, reverse=True)[:k]
        return preds

    best = None
    for wb, wd, kr in itertools.product((0.5, 1.0, 1.5), (1, 2, 3, 4, 5), (10, 30, 60)):
        m = map_sub({q: v[:10] for q, v in fuse(wb, wd, kr).items()}, dev)
        if best is None or m > best[0]: best = (m, wb, wd, kr)
    _, wb, wd, kr = best
    candidates = fuse(wb, wd, kr)
    ft_hybrid = {q: candidates[q][:10] for q in candidates}
    print(f"\nFT-гибрид (веса dev: w_bm25={wb}, w_dense={wd}, k_rrf={kr}):")
    print(f"   dev {map_sub(ft_hybrid, dev):.4f} | test {map_sub(ft_hybrid, test):.4f}")

    # best_chunk по финальным кандидатам, затем освобождаем dense
    bc = hyb.dense_list[0].best_chunk_texts(cal, candidates)
    for d in hyb.dense_list: d._model = None
    gc.collect(); torch.cuda.empty_cache()

    # --- reranker: скорим заново ---
    q2text = {int(q): clean_text(t) for q, t in zip(cal.query_id, cal.query_text)}
    index = [(qid, doc) for qid in cal.query_id.astype(int) for doc in candidates[qid]]
    pairs = [(q2text[qid], bc[(qid, doc)]) for qid, doc in index]
    ce = CrossEncoder(RR_FT, max_length=256, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nскорю {len(pairs)} пар FT-reranker'ом...", flush=True)
    sc = ce.predict(pairs, batch_size=64, show_progress_bar=True)
    by_q = defaultdict(list)
    for (qid, doc), s in zip(index, sc): by_q[qid].append((doc, float(s)))

    pure = {q: [d for d, _ in sorted(by_q[q], key=lambda x: -x[1])][:10] for q in by_q}
    print(f"\nчистый FT-reranker:  dev {map_sub(pure, dev):.4f} | test {map_sub(pure, test):.4f}")

    # --- blend: перетюн на dev-400 ---
    def blend(w, kb, depth, k=10):
        preds = {}
        for qid in cal.query_id.astype(int):
            base = candidates[qid][:depth]; bs = set(base)
            rer = [d for d, _ in sorted([p for p in by_q[qid] if p[0] in bs], key=lambda x: -x[1])]
            s = defaultdict(float)
            for r, d in enumerate(base, 1): s[d] += 1.0 / (kb + r)
            for r, d in enumerate(rer, 1):  s[d] += w / (kb + r)
            preds[qid] = sorted(s, key=s.get, reverse=True)[:k]
        return preds

    best = None
    for w, kb, depth in itertools.product((0.5, 1, 2, 3, 5, 8), (5, 10, 20, 60), (20, 30, 50)):
        m = map_sub(blend(w, kb, depth), dev)
        if best is None or m > best[0]: best = (m, w, kb, depth)
    _, w, kb, depth = best
    preds = blend(w, kb, depth)
    print(f"\nFT-гибрид + FT-reranker (blend dev: w={w}, k_blend={kb}, top_n={depth}):")
    print(f"   dev(400):  {map_sub(preds, dev):.4f}")
    print(f"   test(100): {map_sub(preds, test):.4f}")


if __name__ == "__main__":
    main()
