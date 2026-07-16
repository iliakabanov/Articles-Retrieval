# -*- coding: utf-8 -*-
"""Загрузка и очистка данных.

Тела статей — сырой HTML со служебной разметкой (табы, спойлеры, таблицы,
factoid-блоки, картинки). Для поиска нужен чистый текст; дополнительно вытаскиваем
тексты внутренних ссылок (перефразы-синонимы) и id связанных статей (граф ссылок).
"""
import html as html_lib
import re
from typing import Dict, Set

import pandas as pd
from bs4 import BeautifulSoup

from .config import DATA_DIR

_LINK_RE = re.compile(r"support\.avito\.ru/articles/(\d+)")
_WS_RE = re.compile(r"\s+")


def html_to_text(raw: str) -> str:
    """Грубая, но устойчивая очистка HTML статьи в plain-текст.

    - выкидываем script/style;
    - <img> заменяем его alt-текстом (там бывают осмысленные подписи);
    - собираем видимый текст, разворачиваем HTML-сущности, схлопываем пробелы.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for img in soup.find_all("img"):
        img.replace_with(" " + (img.get("alt") or "") + " ")
    text = html_lib.unescape(soup.get_text(separator=" "))
    return _WS_RE.sub(" ", text).strip()


def extract_anchor_text(raw: str) -> str:
    """Тексты всех внутренних ссылок на статьи справки, склеенные в одну строку."""
    if not isinstance(raw, str) or not raw:
        return ""
    soup = BeautifulSoup(raw, "lxml")
    parts = [a.get_text(" ", strip=True)
             for a in soup.find_all("a", href=True)
             if _LINK_RE.search(a["href"])]
    return " ".join(p for p in parts if p)


def extract_linked_ids(raw: str) -> list:
    """Список article_id, на которые ссылается статья (для графа ссылок)."""
    if not isinstance(raw, str):
        return []
    return [int(x) for x in _LINK_RE.findall(raw)]


def parse_gt(s: str) -> list:
    """'1909 4396' -> [1909, 4396]."""
    if not isinstance(s, str):
        return []
    return [int(x) for x in s.split()]


def load_articles() -> pd.DataFrame:
    """Читает articles.f и добавляет очищенные текстовые поля."""
    df = pd.read_feather(DATA_DIR / "articles.f").copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["body_text"] = df["body"].map(html_to_text)
    df["anchors"] = df["body"].map(extract_anchor_text)
    df["linked_ids"] = df["body"].map(extract_linked_ids)
    df["n_words"] = df["body_text"].str.count(r"\w+")
    return df


def load_calibration() -> pd.DataFrame:
    """Читает calibration.f и парсит ground_truth в список int."""
    df = pd.read_feather(DATA_DIR / "calibration.f").copy()
    df["query_text"] = df["query_text"].fillna("").astype(str)
    df["gt_ids"] = df["ground_truth"].map(parse_gt)
    return df


def load_test() -> pd.DataFrame:
    """Читает test.f."""
    df = pd.read_feather(DATA_DIR / "test.f").copy()
    df["query_text"] = df["query_text"].fillna("").astype(str)
    return df


def build_truth(calibration: pd.DataFrame) -> Dict[int, Set[int]]:
    """{query_id: {правильные article_id}} — вход для метрики."""
    return {int(q): set(ids)
            for q, ids in zip(calibration["query_id"], calibration["gt_ids"])}
