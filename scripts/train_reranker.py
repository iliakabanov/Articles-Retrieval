# -*- coding: utf-8 -*-
"""LoRA-дообучение cross-encoder reranker'а на dev-400 (супервизно).

Кандидаты берём от FT-гибрида (top_n), документ = лучший dense-чанк (как на
инференсе). Позитив — gt-статья (label 1), негативы — прочие кандидаты (label 0).
После каждой эпохи логируем ЧИСТОЕ ранжирное качество reranker'а (кандидаты,
отсортированные только его скором → top-10 → MAP@10) на dev и test — так виден
вклад дообучения в само ранжирование, без blend.

Базовая (недообученная) строка печатается как "эпоха 0".
Требует torch/sentence-transformers/peft (torch_env). Прогон длинный (~30–40 мин):
per-epoch оценка гоняет cross-encoder по всем кандидатам.

    "<torch_env>/python.exe" -u scripts/train_reranker.py --epochs 3 --batch_size 8
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMB_CACHE
from src.data import load_articles, load_calibration, build_truth, split_calibration
from src.metrics import mean_average_precision_at_k
from src.text import clean_text
from src.retrievers.hybrid import Retriever as Hybrid

ROS_FT = str(EMB_CACHE / "ai-forever_ru-en-RoSBERTa_ftsup")
E5_FT = str(EMB_CACHE / "intfloat_multilingual-e5-large_ftsup")


def main():
    ap = argparse.ArgumentParser(description="LoRA-дообучение reranker'а на dev-400")
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--top_n", type=int, default=30, help="кандидатов от FT-гибрида")
    ap.add_argument("--full", action="store_true", help="учить на ВСЕХ 500 (финал, без held-out)")
    ap.add_argument("--out", default=str(EMB_CACHE / "bge-reranker-v2-m3_ft"))
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.util import batch_to_device
    from peft import LoraConfig, get_peft_model

    SEED = 42
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True     # детерминизм cuDNN
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arts = load_articles()
    cal = load_calibration(); truth = build_truth(cal)
    if args.full:
        train_df, test = cal, None
    else:
        train_df, test = split_calibration(cal, n_test=100, seed=SEED)
    q2text = {int(q): clean_text(t) for q, t in zip(cal.query_id, cal.query_text)}

    # ---- кандидаты FT-гибрида + лучший чанк документа ----
    print("строю кандидатов FT-гибрида + best_chunk...", flush=True)
    hyb = Hybrid(dense_models=f"{ROS_FT},{E5_FT}", w_bm25=1.5, w_dense=2, k_rrf=10).fit(arts)
    candidates = hyb.rank(cal, k=args.top_n)
    bc = hyb.dense_list[0].best_chunk_texts(cal, candidates)   # {(qid,aid): текст}
    for d in hyb.dense_list: d._model = None
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- обучающие примеры (позитив=gt, негативы=прочие кандидаты) ----
    examples = []
    for qid in train_df["query_id"].astype(int):
        for doc in candidates[qid]:
            lbl = 1.0 if doc in truth[qid] else 0.0
            examples.append(InputExample(texts=[q2text[qid], bc[(qid, doc)]], label=lbl))
    print(f"обучающих пар: {len(examples)} "
          f"(позитивов ~{sum(1 for e in examples if e.label==1)})", flush=True)

    # ---- модель + LoRA ----
    ce = CrossEncoder(args.model, num_labels=1, max_length=args.max_length, device=device)
    ce.model = get_peft_model(ce.model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="SEQ_CLS", target_modules=["query", "key", "value"]))
    ce.model.print_trainable_parameters()

    # ---- ЧИСТАЯ ранжирная оценка reranker'а (без blend) ----
    all_index = [(qid, doc) for qid in cal.query_id.astype(int) for doc in candidates[qid]]
    all_pairs = [(q2text[qid], bc[(qid, doc)]) for qid, doc in all_index]

    def pure_map():
        ce.model.eval()
        sc = ce.predict(all_pairs, batch_size=64, show_progress_bar=False)
        by_q = defaultdict(list)
        for (qid, doc), s in zip(all_index, sc):
            by_q[qid].append((doc, float(s)))
        preds = {q: [d for d, _ in sorted(by_q[q], key=lambda x: -x[1])][:10] for q in by_q}
        def sub(df):
            ids = set(df.query_id.astype(int))
            return mean_average_precision_at_k({q: preds[q] for q in ids},
                                               {q: truth[q] for q in ids}, 10)
        return sub(train_df), (sub(test) if test is not None else None)

    def line(ep, loss_s, tr, va):
        va_s = f"{va:.4f}" if va is not None else "  —   "
        print(f"  {ep} | {loss_s} |    {tr:.4f}    |   {va_s}", flush=True)

    print("\nepoch |  train_loss | pure_MAP train | pure_MAP test", flush=True)
    d0, t0 = pure_map()
    line("  0", "   (base) ", d0, t0)
    print("        (эпоха 0 = недообученный reranker)", flush=True)

    # ---- обучение (BCE) ----
    g = torch.Generator(); g.manual_seed(SEED)     # фиксируем порядок shuffle
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size,
                        collate_fn=ce.smart_batching_collate, generator=g)
    opt = torch.optim.AdamW([p for p in ce.model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    bce = torch.nn.BCEWithLogitsLoss()

    for ep in range(1, args.epochs + 1):
        ce.model.train()
        print(f"  эпоха {ep}: обучение ({len(loader)} шагов)...", flush=True)
        run, nb = 0.0, 0
        for features, labels in loader:
            features = batch_to_device(features, device); labels = labels.to(device).float()
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = ce.model(**features).logits.squeeze(-1)
                loss = bce(logits, labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item(); nb += 1
        print("      оценка: скорю кандидатов...", flush=True)
        dm, tm = pure_map()
        line(f"{ep:>3}", f"{run/nb:8.4f}", dm, tm)

    ce.model = ce.model.merge_and_unload()
    ce.save(args.out)
    print("\nсохранено:", args.out, flush=True)


if __name__ == "__main__":
    main()
