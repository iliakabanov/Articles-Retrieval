# -*- coding: utf-8 -*-
"""LoRA-дообучение dense-модели СУПЕРВИЗНО на dev-сплите calibration.

Схема: calibration делится на dev(400)/test(100) фиксированным сидом. Обучаемся на
dev (пары query -> правильная статья + hard negatives), после КАЖДОЙ эпохи логируем:
  - train loss (MNRL),
  - train MAP@10 (на dev — на чём учились),
  - val   MAP@10 (на test — held-out 100).
Так видно переобучение: train-метрика растёт, а val — стоит/падает.

Требует torch/sentence-transformers/peft (torch_env).
    "<torch_env>/python.exe" scripts/train_dense.py --model intfloat/multilingual-e5-large --epochs 5 --batch_size 8
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMB_CACHE
from src.data import load_articles, load_calibration, build_truth, split_calibration
from src.metrics import mean_average_precision_at_k
from src.text import clean_text
from src.retrievers.dense import Retriever as Dense, resolve_prefixes
from src.retrievers.bm25 import Retriever as BM25


def main():
    ap = argparse.ArgumentParser(description="Супервизное LoRA-дообучение dense на dev calibration")
    ap.add_argument("--model", default="intfloat/multilingual-e5-large")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_seq", type=int, default=256)
    ap.add_argument("--neg", type=int, default=1, help="hard negatives на пару")
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.util import batch_to_device
    from peft import LoraConfig, get_peft_model

    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pre = resolve_prefixes(args.model)
    QP, PP = pre["query"], pre["passage"]
    out_path = args.out or str(EMB_CACHE / (args.model.replace("/", "_") + "_ftsup"))

    # ---- данные ----
    arts = load_articles()
    id2title = dict(zip(arts.article_id, arts.title))
    id2body = dict(zip(arts.article_id, arts.body_text))
    article_ids = arts["article_id"].to_numpy()
    doc_text = lambda aid: f"{id2title[aid]}. {id2body[aid][:600]}".strip()

    cal = load_calibration()
    truth = build_truth(cal)
    dev, test = split_calibration(cal, n_test=args.n_test, seed=SEED)
    print(f"dev: {len(dev)} | test: {len(test)} | модель: {args.model}")

    # hard negatives через BM25 (с фильтром дубликатов правильных статей)
    bm25 = BM25().fit(arts)
    dev_ranked = bm25.rank(dev, k=15)
    examples = []
    for _, r in dev.iterrows():
        qid = int(r["query_id"]); gts = truth[qid]
        gt_titles = {id2title[g] for g in gts}; gt_bodies = {id2body[g] for g in gts}
        negs = [d for d in dev_ranked[qid]
                if d not in gts and id2title[d] not in gt_titles and id2body[d] not in gt_bodies]
        for g in gts:                       # по паре на каждую правильную статью
            texts = [QP + clean_text(r["query_text"]), PP + doc_text(g)]
            texts += [PP + doc_text(n) for n in negs[:args.neg]]
            examples.append(InputExample(texts=texts))
    print(f"обучающих пар (из dev): {len(examples)}")

    # ---- модель + LoRA ----
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      bias="none", target_modules=["query", "key", "value"])
    model[0].auto_model = get_peft_model(model[0].auto_model, lora)
    model[0].auto_model.print_trainable_parameters()

    # ---- чанки корпуса (строятся один раз; кодируются каждую эпоху текущей моделью) ----
    helper = Dense(model=args.model, chunk_words=200, overlap=40)
    chunk_texts, seg_starts = [], []
    for t, b in zip(arts["title"], arts["body_text"]):
        seg_starts.append(len(chunk_texts))
        chunk_texts.extend(helper._chunks(t, b))
    seg_starts = np.asarray(seg_starts)

    def map_on(df, art_scores):
        preds = {}
        for row, (_, r) in enumerate(df.iterrows()):   # row = позиция в art_scores
            sc = art_scores[row]; kk = min(10, sc.shape[0])
            top = np.argpartition(-sc, kk - 1)[:kk]; top = top[np.argsort(-sc[top])]
            preds[int(r["query_id"])] = [int(article_ids[i]) for i in top]
        ids = set(preds)
        return mean_average_precision_at_k(preds, {q: truth[q] for q in ids}, 10)

    @torch.no_grad()
    def evaluate():
        model.eval()
        torch.cuda.empty_cache()          # освобождаем VRAM после backward-активаций
        print("      оценка: кодирую корпус...", flush=True)
        c = model.encode([PP + t for t in chunk_texts], batch_size=32,
                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        def scores(df):
            q = model.encode([QP + clean_text(t) for t in df["query_text"]], batch_size=32,
                             normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            return np.maximum.reduceat(q @ c.T, seg_starts, axis=1)
        return map_on(dev, scores(dev)), map_on(test, scores(test))

    # ---- ручной цикл обучения с пер-эпохным логом ----
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size,
                        collate_fn=model.smart_batching_collate)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    n_steps = len(loader)
    print("\nepoch |  train_loss | train_MAP@10 | val_MAP@10", flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        print(f"  эпоха {ep}: обучение ({n_steps} шагов)...", flush=True)
        run, nb = 0.0, 0
        for features, labels in loader:
            features = [batch_to_device(f, device) for f in features]
            labels = labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = loss_fn(features, labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        tr_map, va_map = evaluate()
        print(f"  {ep:>3} |   {run/nb:8.4f} |    {tr_map:.4f}    |   {va_map:.4f}", flush=True)

    # ---- сохранить дообученную модель ----
    model[0].auto_model = model[0].auto_model.merge_and_unload()
    model.save(out_path)
    print("\nсохранено:", out_path)


if __name__ == "__main__":
    main()
