"""
infer.py — CPU inference for the fine-tuned Spanish BOE QA model.

    from infer import QA
    qa = QA("models/boe-qa")          # loads on CPU
    qa.answer("¿En qué provincia se ubica el proyecto?", doc_text)
    # -> {"answer": "Huesca", "score": 0.87, "start": 1234, "end": 1240}

Long documents are handled with the SQuAD sliding window; the best span across all
windows is returned. `score` below `min_score` → treated as "no answer".
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MAX_LEN, STRIDE = 384, 128
# Lone function words the no-answer-biased model sometimes emits as a low-margin
# span; treat these (and <3-char spans) as "no answer" rather than a real answer.
_TRIVIAL = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
            "al", "en", "y", "o", "a", "que", "se", "lo", "por", "con"}


class QA:
    def __init__(self, model_dir: str, min_score: float = 0.15, device: str = "cpu",
                 max_context_chars: int = 16000):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
        self.device = device                 # "cpu" in production; "cuda:N" for fast eval sweeps
        self.model.to(self.device).eval()
        if self.device == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))
        self.min_score = min_score
        # Cap the context to the first N chars before windowing. BOE docs run to
        # hundreds of kB; the single-answer facts sit near the top, so this keeps
        # CPU latency bounded and matches how the model was trained (train_qa.CAP_CHARS).
        self.max_context_chars = max_context_chars

    @torch.no_grad()
    def _best(self, question: str, context: str, n_best: int = 20) -> tuple:
        """Best span across all windows + the no-answer (CLS) score, BEFORE
        thresholding. Returns (best_dict, cls_score). `best["score"] - cls` is the
        answer's margin over "no answer"."""
        if self.max_context_chars:
            context = context[:self.max_context_chars]
        enc = self.tok(question, context, truncation="only_second", max_length=MAX_LEN,
                       stride=STRIDE, return_overflowing_tokens=True,
                       return_offsets_mapping=True, padding="max_length",
                       return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        sample_map = enc.pop("overflow_to_sample_mapping")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        best = {"answer": "", "score": float("-inf"), "start": None, "end": None}
        cls = float("-inf")   # "no answer" score; CLS logit sums are often negative,
                              # so seeding at 0.0 would clamp the no-answer margin and
                              # spuriously reject valid spans.
        for w in range(enc["input_ids"].shape[0]):
            seq = enc.sequence_ids(w)
            s_logits = out.start_logits[w].cpu().numpy(); e_logits = out.end_logits[w].cpu().numpy()
            cls = max(cls, float(s_logits[0] + e_logits[0]))   # "no answer" score
            s_idx = np.argsort(s_logits)[-n_best:]; e_idx = np.argsort(e_logits)[-n_best:]
            off = offsets[w]
            for si in s_idx:
                for ei in e_idx:
                    if seq[si] != 1 or seq[ei] != 1 or ei < si or ei - si > 40:
                        continue
                    sc = float(s_logits[si] + e_logits[ei])
                    if sc > best["score"]:
                        cs, ce = int(off[si][0]), int(off[ei][1])
                        best = {"answer": context[cs:ce], "score": sc, "start": cs, "end": ce}
        return best, cls

    def answer(self, question: str, context: str, n_best: int = 20) -> dict:
        best, cls = self._best(question, context, n_best)
        ans = best["answer"].strip()
        # reject: below the no-answer margin, or a trivially short / lone-function-word span
        if (best["start"] is None or best["score"] - cls < self.min_score
                or len(ans) < 3 or ans.lower() in _TRIVIAL):
            return {"answer": "", "score": 0.0, "start": None, "end": None}
        best["score"] = best["score"] - cls   # report the margin, not the raw logit sum
        return best
