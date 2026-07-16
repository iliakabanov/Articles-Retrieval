# -*- coding: utf-8 -*-
"""Оценка ретривера на calibration.f по метрике MAP@10.

Примеры:
    python scripts/run_eval.py --algo bm25
    python scripts/run_eval.py --algo bm25 --param k1=2.0 --param b=0.4
    python scripts/run_eval.py --algo popularity
    python scripts/run_eval.py --list

Раннер по имени алгоритма импортирует класс Retriever из src/retrievers/<algo>.py,
строит индекс по articles.f, ранжирует запросы calibration.f и печатает MAP@10
вместе с опорными границами (популярность / оракул) и худшими запросами.
"""
import argparse
import sys
import time
from pathlib import Path

# делаем пакет src импортируемым при запуске скрипта напрямую
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import K
from src.data import build_truth, load_articles, load_calibration
from src.metrics import average_precision_at_k, mean_average_precision_at_k
from src.retrievers import available, get_retriever


def _coerce(value: str):
    """'2.0' -> 2.0, '3' -> 3, 'false' -> False, иначе строка."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_params(pairs):
    """['k1=2.0', 'stem=false'] -> {'k1': 2.0, 'stem': False}."""
    params = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--param ждёт key=value, получено: {p!r}")
        key, val = p.split("=", 1)
        params[key.strip()] = _coerce(val.strip())
    return params


def reference_bounds(calibration, truth, k):
    """Опорные MAP@10: оракул (верх) и популярность (низ) — для контекста."""
    from collections import Counter
    oracle = {qid: list(rel) for qid, rel in truth.items()}
    map_oracle = mean_average_precision_at_k(oracle, truth, k)

    counts = Counter(i for ids in calibration["gt_ids"] for i in ids)
    top = [aid for aid, _ in counts.most_common(k)]
    pop = {int(q): top for q in calibration["query_id"]}
    map_pop = mean_average_precision_at_k(pop, truth, k)
    return map_pop, map_oracle


def main():
    ap = argparse.ArgumentParser(description="MAP@10 ретривера на calibration.f")
    ap.add_argument("--algo", help="имя алгоритма (файл в src/retrievers/)")
    ap.add_argument("--k", type=int, default=K, help=f"горизонт метрики (деф. {K})")
    ap.add_argument("--param", action="append", default=[],
                    help="параметр ретривера key=value (можно повторять)")
    ap.add_argument("--worst", type=int, default=8,
                    help="сколько худших запросов показать (0 — не показывать)")
    ap.add_argument("--list", action="store_true", help="показать доступные алгоритмы")
    args = ap.parse_args()

    if args.list:
        print("доступные алгоритмы:", ", ".join(available()))
        return
    if not args.algo:
        ap.error("укажите --algo (или --list)")

    params = parse_params(args.param)

    # данные
    articles = load_articles()
    calibration = load_calibration()
    truth = build_truth(calibration)

    # ретривер по имени -> fit -> rank
    Retriever = get_retriever(args.algo)
    t0 = time.perf_counter()
    retriever = Retriever(**params).fit(articles)
    preds = retriever.rank(calibration, k=args.k)
    elapsed = time.perf_counter() - t0

    map_score = mean_average_precision_at_k(preds, truth, args.k)
    map_pop, map_oracle = reference_bounds(calibration, truth, args.k)

    print(f"\n=== {args.algo} " + (f"{params} " if params else "") + "===")
    print(f"статей: {len(articles)} | запросов: {len(calibration)} | время: {elapsed:.1f}s")
    print(f"\nMAP@{args.k} = {map_score:.4f}")
    print(f"  границы:  популярность {map_pop:.4f}  |  оракул {map_oracle:.4f}")

    # дополнительная диагностика
    per_ap = {qid: average_precision_at_k(preds.get(qid, []), truth[qid], args.k)
              for qid in truth}
    zero = sum(1 for a in per_ap.values() if a == 0.0)
    recall = sum(len(set(preds.get(q, [])) & truth[q]) / len(truth[q])
                 for q in truth) / len(truth)
    print(f"  запросов с AP=0: {zero}/{len(truth)} ({zero/len(truth)*100:.1f}%)"
          f"  |  recall@{args.k}: {recall:.3f}")

    if args.worst:
        id2title = dict(zip(articles["article_id"], articles["title"]))
        worst = sorted(per_ap.items(), key=lambda kv: kv[1])[:args.worst]
        q2text = dict(zip(calibration["query_id"], calibration["query_text"]))
        print(f"\nхудшие {args.worst} запросов:")
        for qid, ap_val in worst:
            got = preds.get(qid, [])[:3]
            print(f"  AP={ap_val:.3f} [{qid}] {q2text[qid][:70]}")
            print(f"      нужно: {[f'{g}:{id2title.get(g, chr(63))[:28]}' for g in truth[qid]]}")
            print(f"      выдал: {[f'{g}:{id2title.get(g, chr(63))[:28]}' for g in got]}")


if __name__ == "__main__":
    main()
