# -*- coding: utf-8 -*-
"""Сбор датасета для доменной адаптации dense-модели (вариант B, из корпуса).

Пары (ничего из test):
  1) anchor-текст ссылки -> целевая статья  (человеческий перефраз темы)
  2) заголовок статьи    -> её тело         (self-supervised, все 793 статьи)
Плюс hard negatives через BM25, с ФИЛЬТРОМ дубликатов (в корпусе ~173 статьи с
заголовком-двойником — их нельзя брать в негативы к позитиву-дубликату).

Сохраняет .emb_cache/finetune_pairs.pkl и печатает статистику + примеры.
    python scripts/build_finetune_data.py
"""
import re
import sys
import pickle
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMB_CACHE
from src.data import load_articles
from src.retrievers.bm25 import Retriever as BM25

LINK = re.compile(r"support\.avito\.ru/articles/(\d+)")

arts = load_articles()
corpus_ids = set(arts["article_id"])
id2title = dict(zip(arts["article_id"], arts["title"]))
id2body = dict(zip(arts["article_id"], arts["body_text"]))

# ---- 1) anchor -> target -------------------------------------------------
anchor_pairs = []
for body in arts["body"].dropna():
    soup = BeautifulSoup(body, "lxml")
    for a in soup.find_all("a", href=True):
        m = LINK.search(a["href"])
        if not m:
            continue
        tid = int(m.group(1))
        if tid not in corpus_ids:
            continue
        anchor = a.get_text(" ", strip=True)
        w = anchor.split()
        if 1 <= len(w) <= 12 and any(c.isalpha() for c in anchor):
            anchor_pairs.append((anchor, tid))
anchor_pairs = list({(a.lower().strip(), t): (a, t) for a, t in anchor_pairs}.values())

# ---- 2) title -> body ----------------------------------------------------
title_pairs = [(t, aid) for aid, t in id2title.items() if id2body[aid]]

# ---- 3) hard negatives через BM25 (с фильтром дубликатов) ----------------
bm25 = BM25().fit(arts)
queries = [a for a, _ in anchor_pairs] + [t for t, _ in title_pairs]
targets = [t for _, t in anchor_pairs] + [t for _, t in title_pairs]
qdf = pd.DataFrame({"query_id": range(len(queries)), "query_text": queries})
ranked = bm25.rank(qdf, k=10)

triples, n_filtered = [], 0
for i, (q, pos) in enumerate(zip(queries, targets)):
    pos_title, pos_body = id2title[pos], id2body[pos]
    negs = []
    for d in ranked[i]:
        if d == pos:
            continue
        if id2title[d] == pos_title or id2body[d] == pos_body:   # дубликат позитива
            n_filtered += 1
            continue
        negs.append(d)
        if len(negs) >= 4:
            break
    triples.append({"query": q, "pos_id": int(pos), "neg_ids": [int(x) for x in negs]})

out = EMB_CACHE / "finetune_pairs.pkl"
EMB_CACHE.mkdir(exist_ok=True)
pickle.dump({"anchor_pairs": anchor_pairs, "title_pairs": title_pairs, "triples": triples},
            open(out, "wb"))

print(f"anchor-пар: {len(anchor_pairs)} (целевых статей {len({t for _,t in anchor_pairs})}/{len(corpus_ids)})")
print(f"title-пар:  {len(title_pairs)}")
print(f"триплетов:  {len(triples)} | отфильтровано дубль-негативов: {n_filtered}")
print(f"сохранено:  {out}")
