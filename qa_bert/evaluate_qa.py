"""
evaluate_qa.py — honest evaluation of the fine-tuned BOE QA model.

IMPORTANT: the dev labels are TEACHER-DISTILLED (qwen3:32b), not human gold, so
every number here measures *agreement with the teacher*, not correctness. The
split is document-level (see prepare_data.build), so dev documents were unseen in
training. Reported:

  - SQuAD Exact-Match and token-F1 (Spanish-aware normalization),
  - a no-answer confusion matrix (does the model abstain when the teacher did?),
  - a per-question-type breakdown (so the sparse fieldwork-date questions are
    visible rather than hidden in an average dominated by "no answer"),
  - a min_score threshold sweep to recalibrate the no-answer margin.

    python evaluate_qa.py --model models/boe-qa --dev data/squad_dev.json --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from infer import QA, _TRIVIAL

# Human-readable labels for the example-id suffix (see prepare_data.py ids).
TAG_LABEL = {
    "t0": "location (¿ubicación del proyecto?)",
    "t1": "fieldwork when (¿cuándo trabajo de campo?)",
    "t2": "fieldwork dates (¿entre qué fechas?)",
    "t3": "process start (¿cuándo se inició?)",
    "sp": "species span (extractor)",
    "loc": "location span (extractor)",
}

_ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del", "de"}


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)   # drop punctuation incl. ¿¡; keep accents
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _f1(pred: str, gold: str) -> float:
    p, g = _norm(pred).split(), _norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _em(pred: str, gold: str) -> float:
    return float(_norm(pred) == _norm(gold))


def _contains(pred: str, gold: str) -> bool:
    """Lenient: one normalized string contains the other. Credits a correct answer
    whose span boundary differs from the teacher's (e.g. 'Con fecha 14 de enero de
    2011...' vs the teacher's '14 de enero de 2011')."""
    p, g = _norm(pred), _norm(gold)
    return bool(p) and bool(g) and (g in p or p in g)


def _tag(ex_id: str) -> str:
    suf = str(ex_id).split("::")[-1]
    return suf if suf in TAG_LABEL else suf


def _score_at(rows: list, tau: float) -> dict:
    """Metrics over `rows` (each: gold_text|None, span_text, margin) at threshold tau.
    `contains` is the lenient content-match; `ans_recall` is how often the model
    answered when the teacher had an answer (the over-abstention signal)."""
    em = f1 = cm = 0.0
    tp = fp = fn = tn = 0
    for gold, span, margin in rows:
        pred = span if margin >= tau else ""
        gold_has, pred_has = bool(gold), bool(pred)
        if gold_has and pred_has:
            tp += 1
        elif gold_has and not pred_has:
            fn += 1
        elif not gold_has and pred_has:
            fp += 1
        else:
            tn += 1
        if not gold_has:                       # unanswerable: correct iff abstained
            em += float(not pred_has)
            f1 += float(not pred_has)
            cm += float(not pred_has)
        elif pred_has:
            em += _em(pred, gold)
            f1 += _f1(pred, gold)
            cm += float(_contains(pred, gold))
    n = len(rows) or 1
    ans_recall = tp / (tp + fn) if (tp + fn) else 0.0   # answered when teacher had one
    return {"n": len(rows), "EM": em / n, "F1": f1 / n, "contains": cm / n,
            "ans_recall": ans_recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/boe-qa")
    ap.add_argument("--dev", default="data/squad_dev.json")
    ap.add_argument("--device", default="cpu", help='"cpu" or "cuda:N"')
    ap.add_argument("--out", default="results/qa_eval.json")
    a = ap.parse_args()

    dev = json.load(open(a.dev))["data"]
    qa = QA(a.model, min_score=-1e9, device=a.device)   # never reject; we sweep tau ourselves

    # rows: (gold_text|None, predicted_span_text, margin) ; keep tag for breakdown
    by_tag = defaultdict(list)
    all_rows = []
    margins = []
    for i, ex in enumerate(dev, 1):
        gold_list = ex.get("answers", {}).get("text", [])
        gold = gold_list[0] if gold_list else None
        best, cls = qa._best(ex["question"], ex["context"])
        margin = (best["score"] - cls) if best["start"] is not None else float("-inf")
        span = best["answer"]
        # mirror the serving decoder's trivial-span guard so the calibrated
        # threshold and metrics match what /ask actually does
        if len(span.strip()) < 3 or span.strip().lower() in _TRIVIAL:
            span, margin = "", float("-inf")
        row = (gold, span, margin)
        all_rows.append(row)
        by_tag[_tag(ex["id"])].append(row)
        if margin != float("-inf"):
            margins.append(margin)
        if i % 50 == 0:
            print(f"  scored {i}/{len(dev)}", flush=True)

    # Threshold sweep over observed margins → pick tau maximizing overall F1.
    margins_sorted = sorted(set(round(m, 2) for m in margins))
    cand = [-1e9] + margins_sorted                      # -1e9 = accept everything
    sweep = [(tau, _score_at(all_rows, tau)) for tau in cand]
    best_tau, best_m = max(sweep, key=lambda t: t[1]["F1"])

    print("\n=== Threshold sweep (overall) — teacher-agreement ===")
    print(f"{'tau':>10} {'EM':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    # show a readable subset: the best plus a few around it
    shown = sorted({cand[0], best_tau} | set(margins_sorted[:: max(1, len(margins_sorted)//10 or 1)]))
    for tau in shown:
        m = _score_at(all_rows, tau)
        star = "  <== best F1" if abs(tau - best_tau) < 1e-9 else ""
        tau_s = "accept-all" if tau <= -1e8 else f"{tau:.2f}"
        print(f"{tau_s:>10} {m['EM']:.3f} {m['F1']:.3f} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} {m['tn']:>4}{star}")

    overall = _score_at(all_rows, best_tau)
    print(f"\n=== Overall @ best tau={best_tau:.2f} (teacher-agreement, {overall['n']} dev examples) ===")
    print(f"EM={overall['EM']:.3f}  F1={overall['F1']:.3f}  contains(lenient)={overall['contains']:.3f}")
    print(f"no-answer confusion: TP={overall['tp']} FP={overall['fp']} "
          f"FN={overall['fn']} TN={overall['tn']}  answer-recall={overall['ans_recall']:.3f}")

    print("\n=== Per question type @ best tau  (contains = lenient content match) ===")
    per_tag = {}
    for tag in sorted(by_tag):
        m = _score_at(by_tag[tag], best_tau)
        n_ans = sum(1 for g, _, _ in by_tag[tag] if g)
        per_tag[tag] = {**m, "answerable": n_ans, "label": TAG_LABEL.get(tag, tag)}
        print(f"  {TAG_LABEL.get(tag, tag):45s} n={m['n']:3d} ans={n_ans:3d} "
              f"EM={m['EM']:.3f} F1={m['F1']:.3f} contains={m['contains']:.3f} "
              f"ans-recall={m['ans_recall']:.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "note": "TEACHER-AGREEMENT, not human gold. Document-level held-out dev.",
        "model": a.model, "dev": a.dev, "n_dev": len(dev),
        "best_tau": best_tau, "overall": overall, "per_tag": per_tag,
        "sweep": [{"tau": t, **m} for t, m in sweep],
    }, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
