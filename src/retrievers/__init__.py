# -*- coding: utf-8 -*-
"""Реестр ретриверов: имя алгоритма -> класс Retriever из соответствующего файла."""
import importlib
import pkgutil
from typing import List, Type

from .base import BaseRetriever


def get_retriever(name: str) -> Type[BaseRetriever]:
    """Импортирует src.retrievers.<name> и возвращает его класс Retriever."""
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as e:
        # ModuleNotFoundError может прилететь и из-за отсутствия зависимости внутри
        # модуля — не глотаем такое молча
        if e.name and e.name.split(".")[-1] != name:
            raise
        raise ValueError(
            f"неизвестный алгоритм '{name}'. Доступные: {', '.join(available())}"
        ) from None
    if not hasattr(module, "Retriever"):
        raise ValueError(f"в модуле '{name}' нет класса Retriever")
    return module.Retriever


def available() -> List[str]:
    """Список доступных алгоритмов = имена модулей в этой папке (кроме base)."""
    return sorted(
        m.name for m in pkgutil.iter_modules(__path__)
        if m.name != "base"
    )
