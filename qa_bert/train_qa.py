"""
train_qa.py — fine-tune a Spanish BERT for extractive QA on the BOE data. GPU.

    CUDA_VISIBLE_DEVICES=1 python train_qa.py \
        --model PlanTL-GOB-ES/roberta-base-bne-sqac \
        --train data/squad_train.json --dev data/squad_dev.json --out models/boe-qa

Standard SQuAD sliding-window fine-tune (max_len 384, doc_stride 128) so long BOE
documents are handled. Inference is CPU-only (see infer.py) — training here is the
only GPU step.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import numpy as np
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
                          TrainingArguments, Trainer, default_data_collator)

MAX_LEN, STRIDE = 384, 128
# BOE docs are long (median ~49k chars) but the teacher only read the first 16k,
# so all single-answer targets live there and administrative facts (location,
# procedure start, fieldwork) are stated near the top. Cap the context so each
# example yields ~a dozen windows instead of ~120 — otherwise training drowns in
# forced no-answer windows from the unlabeled tail. Must match infer.py's cap.
CAP_CHARS = 16000


def _cap_example(e: dict) -> dict:
    """Truncate an example's context to CAP_CHARS while keeping its answer inside.
    Answers past the cap (some species/location spans) get a window recentered on
    the span so their supervision is preserved rather than mislabeled no-answer."""
    ctx = e["context"]
    starts = e["answers"]["answer_start"]
    if not starts:                                   # unanswerable
        return {**e, "context": ctx[:CAP_CHARS]}
    s = starts[0]
    ans = e["answers"]["text"][0]
    if s + len(ans) <= CAP_CHARS:
        return {**e, "context": ctx[:CAP_CHARS]}
    lo = max(0, s - CAP_CHARS // 2)                  # recenter on the answer
    return {**e, "context": ctx[lo:lo + CAP_CHARS],
            "answers": {"text": [ans], "answer_start": [s - lo]}}


def _load(path):
    data = [_cap_example(e) for e in json.load(open(path))["data"]]
    return Dataset.from_list(data)


def _downsample_negatives(ds, neg_ratio: float, seed: int = 0):
    """Keep every window that contains an answer, plus `neg_ratio` no-answer (CLS)
    windows per positive. Long BOE docs make ~93% of windows no-answer, which biases
    the model toward abstaining (it under-answers project-location); this rebalances
    without dropping any real answer or shortening the context."""
    sp = np.array(ds["start_positions"]); ep = np.array(ds["end_positions"])
    is_neg = (sp == 0) & (ep == 0)
    pos_idx = np.where(~is_neg)[0]; neg_idx = np.where(is_neg)[0]
    keep_neg = int(min(len(neg_idx), len(pos_idx) * neg_ratio))
    rng = np.random.default_rng(seed)
    sel_neg = (rng.choice(neg_idx, size=keep_neg, replace=False)
               if keep_neg < len(neg_idx) else neg_idx)
    keep = np.sort(np.concatenate([pos_idx, sel_neg]))
    print(f"neg-downsample: {len(pos_idx)} pos + {len(sel_neg)}/{len(neg_idx)} neg "
          f"= {len(keep)} windows ({100*len(pos_idx)/max(1,len(keep)):.0f}% positive)",
          flush=True)
    return ds.select(keep.tolist())


def _prep_train(examples, tok):
    tok_ex = tok(examples["question"], examples["context"], truncation="only_second",
                 max_length=MAX_LEN, stride=STRIDE, return_overflowing_tokens=True,
                 return_offsets_mapping=True, padding="max_length")
    sample_map = tok_ex.pop("overflow_to_sample_mapping")
    offsets = tok_ex.pop("offset_mapping")
    starts, ends = [], []
    for i, off in enumerate(offsets):
        seq = tok_ex.sequence_ids(i)
        si = sample_map[i]
        ans = examples["answers"][si]
        if not ans["answer_start"]:               # unanswerable → point at [CLS]
            starts.append(0); ends.append(0); continue
        s_char = ans["answer_start"][0]; e_char = s_char + len(ans["text"][0])
        # context token range
        ctx0 = seq.index(1); ctx1 = len(seq) - 1 - seq[::-1].index(1)
        if not (off[ctx0][0] <= s_char and off[ctx1][1] >= e_char):
            starts.append(0); ends.append(0); continue   # answer not in this window
        ts = ctx0
        while ts <= ctx1 and off[ts][0] <= s_char: ts += 1
        te = ctx1
        while te >= ctx0 and off[te][1] >= e_char: te -= 1
        starts.append(ts - 1); ends.append(te + 1)
    tok_ex["start_positions"] = starts; tok_ex["end_positions"] = ends
    return tok_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mrm8488/roberta-base-bne-finetuned-sqac")
    ap.add_argument("--train", default="data/squad_train.json")
    ap.add_argument("--dev", default="data/squad_dev.json")
    ap.add_argument("--out", default="models/boe-qa")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay")
    ap.add_argument("--neg-ratio", type=float, default=0.0, dest="neg_ratio",
                    help="no-answer windows kept per positive window (0 = keep all)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForQuestionAnswering.from_pretrained(a.model)

    raw_train = _load(a.train)
    train = raw_train.map(lambda e: _prep_train(e, tok), batched=True,
                          remove_columns=raw_train.column_names)
    if a.neg_ratio > 0:
        train = _downsample_negatives(train, a.neg_ratio)
    eval_ds = None
    if a.dev and Path(a.dev).exists():
        raw_dev = _load(a.dev)
        eval_ds = raw_dev.map(lambda e: _prep_train(e, tok), batched=True,
                              remove_columns=raw_dev.column_names)

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, learning_rate=a.lr,
        num_train_epochs=a.epochs, weight_decay=a.weight_decay,
        fp16=True, logging_steps=50, save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        per_device_eval_batch_size=a.bs, report_to=[],
    )
    Trainer(model=model, args=args, train_dataset=train, eval_dataset=eval_ds,
            data_collator=default_data_collator, tokenizer=tok).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out)
    print("saved →", a.out)


if __name__ == "__main__":
    main()
