# -*- coding: utf-8 -*-
"""Генерация answer.csv для test.f выбранным ретривером (воспроизводимо).

Примеры:
    "<torch_env>/python.exe" scripts/build_answer.py --algo rerank --progress
    python scripts/build_answer.py --algo hybrid --out answer_hybrid.csv

Формат вывода (по условию): колонки query_id, answer, где answer — ранжированный
список article_id через пробел (до 10). Скрипт проверяет корректность: все query_id
из test присутствуют, без повторов строк, без повторов id внутри ответа, все id
существуют в articles.f, не более K на запрос.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import K
from src.data import load_articles, load_test
from src.retrievers import available, get_retriever

# переиспользуем парсер параметров раннера, чтобы --param вёл себя одинаково
from run_eval import parse_params  # noqa: E402


def validate(answer: pd.DataFrame, test: pd.DataFrame, valid_ids: set, k: int):
    """Жёсткие проверки формата ответа по условию задачи."""
    assert list(answer.columns) == ["query_id", "answer"], answer.columns.tolist()

    # ровно те же query_id, что в test: без пропусков, лишних строк и дублей
    assert not answer["query_id"].duplicated().any(), "дубли query_id в answer"
    assert set(answer["query_id"]) == set(test["query_id"]), "набор query_id != test"
    assert len(answer) == len(test), f"{len(answer)} строк вместо {len(test)}"

    for qid, ans in zip(answer["query_id"], answer["answer"]):
        ids = ans.split()
        assert ids, f"пустой ответ для query_id={qid}"
        assert len(ids) <= k, f"больше {k} статей для query_id={qid}"
        assert len(ids) == len(set(ids)), f"повтор article_id в ответе query_id={qid}"
        for i in ids:
            assert int(i) in valid_ids, f"article_id {i} нет в articles.f (query_id={qid})"


def main():
    ap = argparse.ArgumentParser(description="Сгенерировать answer.csv для test.f")
    ap.add_argument("--algo", default="rerank", help="имя алгоритма (деф. rerank)")
    ap.add_argument("--out", default="answer.csv", help="путь к выходному CSV")
    ap.add_argument("--k", type=int, default=K, help=f"сколько статей на запрос (деф. {K})")
    ap.add_argument("--param", action="append", default=[], help="параметр ретривера key=value")
    ap.add_argument("--progress", action="store_true", help="прогресс-бар (dense/rerank)")
    ap.add_argument("--list", action="store_true", help="показать доступные алгоритмы")
    args = ap.parse_args()

    if args.list:
        print("доступные алгоритмы:", ", ".join(available()))
        return

    params = parse_params(args.param)
    articles = load_articles()
    test = load_test()
    valid_ids = set(articles["article_id"])

    Retriever = get_retriever(args.algo)
    if args.progress:
        import inspect
        if "progress" in inspect.signature(Retriever.__init__).parameters:
            params.setdefault("progress", True)

    print(f"алгоритм: {args.algo} {params or ''} | запросов в test: {len(test)}")
    retriever = Retriever(**params).fit(articles)
    preds = retriever.rank(test, k=args.k)

    # собираем answer в порядке test
    rows = []
    for qid in test["query_id"]:
        ids = preds.get(int(qid), [])[:args.k]
        rows.append({"query_id": int(qid), "answer": " ".join(str(i) for i in ids)})
    answer = pd.DataFrame(rows)

    validate(answer, test, valid_ids, args.k)

    out = Path(args.out)
    answer.to_csv(out, index=False)
    print(f"OK: {out.resolve()} ({len(answer)} строк, все проверки пройдены)")
    print(answer.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
