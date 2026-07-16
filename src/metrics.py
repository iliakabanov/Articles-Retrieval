# -*- coding: utf-8 -*-
"""Метрика соревнования: MAP@K.

AP@K для одного запроса:

    AP@K = (1 / min(m, K)) * sum_{i=1..K} P(i) * rel(i)

где m — число правильных статей (ground_truth), rel(i)=1 если документ на позиции i
релевантен, P(i) — точность на отсечке i. MAP@K — среднее AP@K по всем запросам.
"""
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import K


def average_precision_at_k(predicted: Sequence[int],
                           relevant: Iterable[int],
                           k: int = K) -> float:
    """AP@k для одного запроса.

    predicted — ранжированный список article_id (от самого релевантного).
    relevant  — множество правильных article_id.
    Повторы в predicted убираются (сохраняя порядок первого вхождения): повтор не
    должен ни добивать скор дважды, ни «съедать» позицию у следующего документа.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0

    seen, deduped = set(), []
    for doc in predicted:
        if doc not in seen:
            seen.add(doc)
            deduped.append(doc)

    hits = 0
    score = 0.0
    for i, doc in enumerate(deduped[:k], start=1):
        if doc in relevant:
            hits += 1
            score += hits / i          # P(i) в момент попадания
    return score / min(len(relevant), k)


def mean_average_precision_at_k(predictions: Mapping[int, Sequence[int]],
                                truth: Mapping[int, Iterable[int]],
                                k: int = K) -> float:
    """MAP@k — среднее AP@k по всем запросам из truth."""
    aps = [average_precision_at_k(predictions.get(qid, []), rel, k)
           for qid, rel in truth.items()]
    return float(np.mean(aps)) if aps else 0.0
