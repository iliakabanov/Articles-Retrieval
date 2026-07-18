# Articles-Retrieval — поиск статей справки под MAP@10

Retrieval-этап для RAG: по короткому разговорному вопросу пользователя вернуть
ранжированный топ-10 `article_id`. Метрика — **MAP@10**.

**Публичный тест: MAP@10: 0.6225.**

Полное описание подхода (обработка HTML, модели, валидация на `calibration.f`,
анализ ошибок) — в **[SOLUTION.md](SOLUTION.md)**.

## Кратко о подходе

- **Обработка HTML:** `html_to_text` (`src/data.py`) — убираем script/style, `<img>`
  заменяем его alt-текстом, собираем видимый текст, схлопываем пробелы. Для запросов
  дополнительно вычищаем маск-токены `<MONEY>/<ID>/…`.
- **Пайплайн:** `BM25` (лексика) + `RoSBERTa` + `e5-large` (семантика) → слияние
  **RRF** → **cross-encoder reranker** (`bge-reranker-v2-m3`, реранк по лучшему чанку).
- **Дообучение (главный вклад, +0.095 к тесту):** все три нейросети дообучены
  **LoRA-супервизно** на размеченных парах `запрос → правильная статья` из
  `calibration.f`. Валидация — честный сплит dev-400 / test-100.
- **Анализ ошибок** (`notebooks/03_hybrid_analysis.ipynb`, `04_multianswer.ipynb`):
  провалы AP=0 — в основном ранжирование (а не поиск); хаб-статьи; неоднозначность
  разметки. Решения — reranker/перевзвешивание, дообучение.

## Структура

```
src/            модули: config, metrics, data, text, retrievers/{bm25,dense,hybrid,rerank,...}
scripts/        run_eval, train_dense, train_reranker, build_answer_ft, eval_pipeline
notebooks/      01_eda, 02_baseline, 03_hybrid_analysis, 04_multianswer
data/candidate_data/   articles.f, calibration.f, test.f
answer.csv      отправленный результат
SOLUTION.md     описание подхода
```

## Установка

```bash
# torch — под вашу CUDA (пример для CUDA 12.x):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Воспроизведение answer.csv

Обучение на GPU (AMP) не воспроизводится битово, поэтому для **точного** совпадения
`answer.csv` используйте **приложенные дообученные веса** — инференс детерминирован
(фиксируем сиды + cuDNN).

### Вариант A — из готовых весов (рекомендуется, ~15–20 мин GPU)

Дообученные модели выложены на HuggingFace и **прописаны в скрипте по умолчанию** —
достаточно просто запустить (модели скачаются автоматически):

```bash
python scripts/build_answer_ft.py      # -> answer.csv
```

Используемые модели:
- `iliakabanov/russian-dense-retriever` — дообученная RoSBERTa;
- `iliakabanov/multillingual-dense-retriever` — дообученная e5-large;
- `iliakabanov/reranker-retriever` — дообученный bge-reranker-v2-m3.

#### Откуда брать модели

`build_answer_ft.py` (и `eval_pipeline.py`) выбирают источник моделей так:

| Способ | Команда / переменная | Модели |
|---|---|---|
| по умолчанию | `python scripts/build_answer_ft.py` | скачиваются с **HuggingFace** |
| флаг `--local` | `python scripts/build_answer_ft.py --local` | локальные из `.emb_cache/` (после Варианта B) |
| env-переменные | `ROS_FT=… E5_FT=… RR_FT=…` | точечно переопределяют любую модель (приоритетнее флага) |

- Для HF-скачивания **не** выставляйте `HF_HUB_OFFLINE=1` (иначе загрузка заблокируется).
- С `--local` можно ставить `HF_HUB_OFFLINE=1` — модели берутся с диска.
- Имена dense-моделей на HF намеренно прописаны в `src/retrievers/dense.py`
  (`MODEL_PRESETS`) с их префиксами `query:`/`search_query:` — если переименуете
  репозитории, обновите там.

### Вариант B — обучение с нуля

```bash
# 1. dense на dev-400 (эпоха 1); порядок: сначала dense, потом reranker
python scripts/train_dense.py --model intfloat/multilingual-e5-large --epochs 1 --batch_size 4
python scripts/train_dense.py --model ai-forever/ru-en-RoSBERTa      --epochs 1 --batch_size 4
# 2. reranker на dev-400 (эпоха 3)
python scripts/train_reranker.py --epochs 3 --batch_size 8
# 3. генерация ответа
python scripts/build_answer_ft.py
```

Модели сохранятся в `.emb_cache/*_ftsup` и `.emb_cache/bge-reranker-v2-m3_ft`,
дальше `build_answer_ft.py` соберёт `answer.csv`. Из-за недетерминизма GPU результат
может отличаться на пару тысячных от отправленного.

## Оценка на calibration

```bash
python scripts/run_eval.py --algo hybrid            # компоненты по отдельности
python scripts/eval_pipeline.py                     # финальный FT-пайплайн, dev-400/test-100
```

## Воспроизводимость

- Сиды (`SEED=42`) зафиксированы в обучении: `random`, `numpy`, `torch`,
  `torch.cuda`, генератор `DataLoader`; `cudnn.deterministic=True`.
- Сплит `calibration` детерминирован (`split_calibration`, `random_state=42`).
- Инференс/генерация (`build_answer_ft.py`) — с фиксированными сидами и cuDNN.
- **Данные:** положите выданные платформой `articles.f`, `calibration.f`, `test.f`
  в `data/candidate_data/` (в репозиторий не входят — не редистрибутируем задачу).
