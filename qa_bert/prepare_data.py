"""
prepare_data.py — build a SQuAD-v2 QA dataset from the 639 BOE docs.

Four sources (see PLAN.md):
  1. Paul's /classify evidence spans  → answers for themed questions  [implemented]
  2. our species/location extractors  → answers for entity questions  [implemented]
  3. LLM teacher over questions.txt   → answers for open questions     [runs if
                                                                        questions.txt
                                                                        + LLM endpoint]
  4. public Spanish QA (SQAC)         → warm-start, via the -sqac checkpoint

Output: data/squad_train.json, data/squad_dev.json (SQuAD-v2 schema).

This does NOT train — it only assembles data. Training is gated on questions.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
DOCS_DIR = Path(os.environ.get("BOE_DOCS", "/home/egaillac/BioMoQA-RAG/sample_revised"))
SERVICE = Path("/home/egaillac/MetaP/species_qa_service")

# Paul's evidence-feature key → a natural-language question. His matched span (with
# offsets) becomes the answer. Extend as his feature set grows.
THEME_QUESTIONS = {
    "wind_farm":            "¿Qué tipo de proyecto se evalúa?",
    "solar_pv":             "¿Qué tipo de proyecto se evalúa?",
    "power_line":           "¿Qué tipo de proyecto se evalúa?",
    "conducted_surveys":    "¿Qué trabajo de campo se realizó?",
    "reported_effort":      "¿Cuál fue el esfuerzo de muestreo?",
    "monitoring_plan":      "¿Qué plan de vigilancia se establece?",
    "seasonal_coverage":    "¿Qué cobertura estacional tiene el estudio?",
    "species_from_catalogue": "¿De qué catálogo proceden las especies citadas?",
    "protected_area":       "¿Qué espacio protegido se menciona?",
}


def _iter_docs() -> Iterable[dict]:
    for f in sorted(DOCS_DIR.glob("*.json")):
        yield json.load(open(f))


def _squad_qa(qid: str, question: str, text: str, answer: str, start: int) -> dict:
    return {
        "id": qid,
        "question": question,
        "context": text,
        "answers": {"text": [answer], "answer_start": [start]},
        "is_impossible": False,
    }


# ── Source 1: Paul's evidence spans ────────────────────────────────────────────

def from_paul(classify_result: dict) -> List[dict]:
    """Map a /classify result's `features` (span + offsets) to QA examples."""
    text = classify_result.get("text", "")
    out = []
    for feat, mentions in (classify_result.get("features") or {}).items():
        q = THEME_QUESTIONS.get(feat)
        if not q or not mentions:
            continue
        m = mentions[0]  # first mention is enough as the answer span
        s, e = m.get("start"), m.get("end")
        if s is None or e is None:
            continue
        out.append(_squad_qa(f"{classify_result.get('id')}::{feat}", q, text, text[s:e], s))
    return out


# ── Source 2: our extractors (species + locations) ─────────────────────────────

def from_extractors(doc: dict, species_ex, loc_ex) -> List[dict]:
    out = []
    text = doc.get("text", "")
    _id = doc.get("_id")
    sp = species_ex.extract(doc)
    if sp["n_species"]:
        top = sp["species"][0]
        for m in top["mentions"]:
            if m.get("start") is not None:
                out.append(_squad_qa(f"{_id}::sp", "¿Qué especie se menciona en el documento?",
                                     text, m["text"], m["start"]))
                break
    lc = loc_ex.extract({"_id": _id, "text": text})
    if lc["n_locations"]:
        m = lc["locations"][0]["mentions"][0]
        out.append(_squad_qa(f"{_id}::loc", "¿Qué localidad se menciona en el documento?",
                             text, m["text"], m["start"]))
    return out


# ── Source 3: LLM teacher over questions.txt (batched, one call per doc) ────────

import re

TEACHER_CTX = int(os.environ.get("TEACHER_CTX", "16000"))  # chars fed to the teacher


def teacher_batch(text: str, questions: List[str], base_url: str, model: str) -> Dict[int, Optional[tuple]]:
    """One LLM call for all questions on one doc → {q_index: (answer, start) or None}.

    Batching shares the (expensive) long-context read across all questions.
    """
    import requests
    ql = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
    prompt = (
        "Para cada PREGUNTA, extrae del DOCUMENTO el fragmento textual EXACTO "
        "(copiado literal, sin parafrasear) que la responde. Si la respuesta no "
        'está en el documento, usa null. Responde SOLO un objeto JSON '
        '{"0":"...","1":null,...}. /no_think\n\n'
        f"PREGUNTAS:\n{ql}\n\nDOCUMENTO:\n{text[:TEACHER_CTX]}\n\nJSON:"
    )
    try:
        r = requests.post(f"{base_url}/v1/chat/completions", timeout=300, json={
            "model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        })
        raw = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return {}
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}
    res: Dict[int, Optional[tuple]] = {}
    for i in range(len(questions)):
        v = obj.get(str(i), obj.get(i))
        if isinstance(v, str) and v.strip():
            span = v.strip().strip('"').strip()
            start = text.find(span)
            res[i] = (span, start) if start >= 0 else None
        else:
            res[i] = None
    return res


def from_teacher(doc: dict, questions: List[str], base_url: str, model: str) -> List[dict]:
    out, text, _id = [], doc.get("text", ""), doc.get("_id")
    res = teacher_batch(text, questions, base_url, model)
    for i, q in enumerate(questions):
        r = res.get(i)
        if r:
            out.append(_squad_qa(f"{_id}::t{i}", q, text, r[0], r[1]))
        else:  # SQuAD-v2 unanswerable example
            out.append({"id": f"{_id}::t{i}", "question": q, "context": text,
                        "answers": {"text": [], "answer_start": []}, "is_impossible": True})
    return out


# ── Assemble ───────────────────────────────────────────────────────────────────

def _load_questions() -> List[str]:
    qfile = HERE / "questions.txt"
    if not qfile.exists():
        return []
    out = []
    for line in qfile.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split("\t")[0].strip())
    return out


def run_teacher(model: str, base_url: str) -> None:
    """Slow step: teacher-label every doc, checkpointed to data/teacher.jsonl so
    the run is resumable. One batched LLM call per doc."""
    questions = _load_questions()
    if not questions:
        print("no questions.txt — nothing for the teacher to do"); return
    (HERE / "data").mkdir(exist_ok=True)
    ckpt = HERE / "data" / "teacher.jsonl"
    done = set()
    if ckpt.exists():
        for line in ckpt.open():
            try:
                done.add(json.loads(line)["doc"])
            except Exception:
                pass
    docs = list(_iter_docs())
    todo = [d for d in docs if d.get("_id") not in done]
    print(f"teacher: {len(done)} done, {len(todo)} to go, model={model}", flush=True)
    import time as _t
    with ckpt.open("a") as fh:
        for k, doc in enumerate(todo, 1):
            t0 = _t.time()
            exs = from_teacher(doc, questions, base_url, model)
            fh.write(json.dumps({"doc": doc.get("_id"), "examples": exs}, ensure_ascii=False) + "\n")
            fh.flush()
            n_ans = sum(1 for e in exs if not e.get("is_impossible"))
            print(f"  [{k}/{len(todo)}] {doc.get('_id')} {n_ans}/{len(questions)} answered ({_t.time()-t0:.0f}s)", flush=True)


def build(use_paul: bool, dev_frac: float = 0.1) -> None:
    sys.path.insert(0, str(SERVICE))
    from extractor import SpeciesExtractor
    from location_extractor import LocationExtractor
    species_ex, loc_ex = SpeciesExtractor(scan_text=True), LocationExtractor()

    examples: List[dict] = []
    for doc in _iter_docs():
        examples += from_extractors(doc, species_ex, loc_ex)
    # teacher examples from the checkpoint
    ckpt = HERE / "data" / "teacher.jsonl"
    if ckpt.exists():
        for line in ckpt.open():
            try:
                examples += json.loads(line)["examples"]
            except Exception:
                pass
    # Paul reuse: expects cached classify results at data/paul_classify.json
    paul_cache = HERE / "data" / "paul_classify.json"
    if use_paul and paul_cache.exists():
        for res in json.load(open(paul_cache)):
            examples += from_paul(res)

    # Split by DOCUMENT, not per-example. Every example id is "<doc_id>::<tag>",
    # so grouping on that prefix keeps all of a BOE document's windows/QAs on the
    # same side of the cut. A per-example split would put the same document's text
    # in both train and dev and make any dev metric optimistic.
    def _doc_of(ex: dict) -> str:
        return str(ex.get("id", "")).split("::")[0]

    by_doc: Dict[str, List[dict]] = {}
    for ex in examples:
        by_doc.setdefault(_doc_of(ex), []).append(ex)
    doc_ids = sorted(by_doc)
    random.seed(0); random.shuffle(doc_ids)
    n_dev_docs = int(len(doc_ids) * dev_frac)
    dev_docs = set(doc_ids[:n_dev_docs])
    dev = [ex for d in doc_ids if d in dev_docs for ex in by_doc[d]]
    train = [ex for d in doc_ids if d not in dev_docs for ex in by_doc[d]]
    random.shuffle(train); random.shuffle(dev)

    (HERE / "data").mkdir(exist_ok=True)
    for name, subset in [("squad_dev.json", dev), ("squad_train.json", train)]:
        json.dump({"version": "boe-v2", "data": subset}, open(HERE / "data" / name, "w"),
                  ensure_ascii=False, indent=1)
    ans = sum(1 for e in examples if e.get("answers", {}).get("text"))
    print(f"train={len(train)} dev={len(dev)} "
          f"docs train/dev={len(doc_ids)-n_dev_docs}/{n_dev_docs} "
          f"(answerable overall={ans}/{len(examples)}) → {HERE/'data'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", action="store_true", help="run the LLM teacher pass (slow)")
    ap.add_argument("--build", action="store_true", help="assemble squad_{train,dev}.json")
    ap.add_argument("--no-paul", action="store_true")
    ap.add_argument("--model", default=os.environ.get("TEACHER_LLM_MODEL", "qwen3:32b"))
    ap.add_argument("--url", default=os.environ.get("TEACHER_LLM_URL", "http://localhost:11434"))
    a = ap.parse_args()
    if a.teacher:
        run_teacher(a.model, a.url)
    if a.build or not a.teacher:
        build(use_paul=not a.no_paul)
