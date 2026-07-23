# Spanish BERT extractive QA — training plan

Goal: a **Spanish BERT** that answers **free-text questions** about a BOE document
(the document is given directly — no retriever). Trained on the 600 docs (+ other
data). **GPU for training, CPU for inference.**

Status: scaffold built; **training is gated on the example-questions list** (needed
to generate the QA pairs). Everything else is ready.

---

## What kind of QA, and how it relates to the two buttons

- **Extractive span QA** (SQuAD-style): input = (question, document), output = a
  text span from the document + a confidence. Right tool for **single-answer**
  questions: *"¿En qué provincia?"*, *"¿Cuántas hectáreas?"*, *"¿Qué tipo de
  proyecto?"*, *"¿Es dato primario o secundario y por qué?"*.
- The **two list buttons stay on the deterministic extractors** (species → OTT,
  locations → NER): a span-QA model returns one span, so it is poor at *"list all
  species"*. The BERT QA **complements** them and *does* help location questions
  of the single-answer kind (*"¿en qué término municipal se ubica el proyecto?"*).

## Model

Start from a Spanish model already fine-tuned on Spanish extractive QA, then
domain-adapt on our BOE data:

- **Chosen (downloaded, cached, smoke-tested):**
  `mrm8488/roberta-base-bne-finetuned-sqac` — RoBERTa-BNE fine-tuned on SQAC (the
  Spanish QA Corpus), 499 MB. Answers Spanish factoid questions out of the box
  (e.g. *"¿En qué provincia…?"* → "Huesca", score 0.95) before any domain tuning.
  (The original `PlanTL-GOB-ES/roberta-base-bne-sqac` was removed from HF.)
- Alternatives: `JonatanGk/…-finetuned-sqac`, `dccuchile/…-finetuned-qa-sqac`.

Long docs (up to ~95k chars) → standard SQuAD **sliding window** (max_len 384,
doc_stride 128); best span across windows at inference.

## Training data — how we get (question, context, answer) triples

We have 639 docs but no QA pairs. Four complementary sources (assembled by
`prepare_data.py`):

1. **Reuse the triage evidence offsets (free, high quality).** The triage `/classify`
   returns evidence features as spans with char offsets and done/proposed/cited
   tags. Each theme maps to a question, e.g.
   `wind_farm` → *"¿Qué tipo de proyecto?"*, `conducted_surveys` →
   *"¿Qué trabajo de campo se realizó?"*, `reported_effort` →
   *"¿Cuál fue el esfuerzo de muestreo?"*. His matched span (with offset) is a
   ready-made answer span. **This is the concrete reuse of the triage system for us.**
2. **Our extractors (free).** Species (OTT) and location (NER) mentions already
   carry offsets → training spans for *"¿qué especie…?"* / *"¿qué lugar…?"* style
   questions (one example per mention).
3. **LLM teacher (for the open questions you provide).** For each doc × question,
   a local LLM (Qwen3 via Ollama / the deployed Qwen3-8B) extracts the verbatim
   answer span (or "no answer"); we locate its offset → a SQuAD example. Same
   teacher-distillation pattern already used for the thesis classifier.
4. **Public Spanish QA (optional warm-start).** SQAC / SQuAD-es to preserve
   general QA ability (already baked in if we start from the `-sqac` checkpoint).

SQuAD v2 format (with unanswerable questions) so the model can say "no answer"
when a doc doesn't contain it.

## What I need from you to proceed to training

- **The example-questions list** (drives sources 1 & 3). Drop it at
  `qa_bert/questions.txt` (one question per line; optional `theme` after a tab to
  map to the triage features).

## Pipeline / files

- `prepare_data.py` — build `data/squad_train.json` / `squad_dev.json` from the
  four sources above (triage reuse + extractors implemented; LLM-teacher reads
  `questions.txt` + an LLM endpoint).
- `train_qa.py` — HF `AutoModelForQuestionAnswering` fine-tune on **GPU**
  (`CUDA_VISIBLE_DEVICES`), sliding-window features, saves to `models/`.
- `infer.py` — load the trained model on **CPU**, `answer(question, doc)` with the
  sliding-window best-span decode. Wraps into a future `/ask` endpoint.

## Compute

Box has 3 idle A100 partitions (incl. 80 GB) + torch/cu128. RoBERTa-base QA
fine-tune ≈ minutes on GPU. Inference on CPU ≈ 0.1–1 s per (question, doc) with
windowing. No cost concern.
