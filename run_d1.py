"""D1: Naturalistic generalization — HotpotQA + MuSiQue validation.

Tests whether P1 and P3 behavioral findings hold on real multi-hop QA datasets:
  1. Does the P1 model ranking (Tier I immune, Tier III prone) hold on natural text?
  2. Does prose-collapse (P3) appear when the passages are naturally written prose?

Protocol:
  HotpotQA (2-hop): fullwiki split, 200 questions
    - Condition A: supporting facts as bullet points (same as P3 bullets)
    - Condition B: full context paragraphs (naturally written prose, as-is)
  MuSiQue (2-hop + 3-hop): 200 questions
    - Condition A: supporting paragraphs as bullet points
    - Condition B: supporting paragraphs concatenated as prose block

Models: same 6 focal models from P3 + Llama-4-Scout and Llama-4-Maverick (from P5)

Key questions:
  Q1: Does Llama collapse on naturally written multi-hop prose? (validates P3)
  Q2: Does the Tier I/II/III ranking from P1 predict naturalistic accuracy? (validates P1)
  Q3: Is Llama-4 still format-robust on naturalistic text? (extends P5)

Output: results/d1_hotpotqa.jsonl, results/d1_musique.jsonl
  {dataset, question_id, model, condition, answer, prediction, correct}

Usage:
    python run_d1.py [--n-hotpot N] [--n-musique N] [--dry-run]
"""

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client, MODELS

RESULTS_DIR = Path(__file__).parent / "results"
CONCURRENCY = 3
SEEDS       = [0]  # single seed for naturalistic (questions are already diverse)

# Focal models from P3 + Llama-4 from P5
D1_MODELS = [
    "deepseek-r1",       # Tier I ceiling
    "qwen3-32b",         # Tier I immune
    "llama3-70b",        # Tier II/III, large Llama
    "gemma3-12b",        # Tier II immune
    "llama3-8b",         # Tier III prone
    "mixtral-8x7b",      # Tier III MoE
    "llama4-scout",      # Llama 4 — format-robust per P5
    "llama4-maverick",   # Llama 4 — prose-better per P5
]

_write_lock = threading.Lock()


def _append(path: Path, record: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _invoke(client, model_name: str, prompt: str) -> str:
    model_id = MODELS.get(model_name, model_name)
    return client.invoke(model_id, prompt, max_tokens=64, temperature=0.0)


def _score(pred: str, ans: str) -> int:
    pred = pred.strip().lower()
    ans  = ans.strip().lower()
    if ans in pred or pred.startswith(ans):
        return 1
    # Handle "yes"/"no" answers
    if ans in ("yes", "no") and pred.startswith(ans):
        return 1
    # Partial match: answer is multi-word, prediction starts with it
    ans_words = ans.split()
    if len(ans_words) >= 2 and pred.startswith(ans_words[0]):
        # Check at least 2 words match
        pred_words = pred.split()
        matches = sum(1 for w in ans_words if w in pred_words)
        if matches >= max(1, len(ans_words) // 2):
            return 1
    return 0


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            keys.add(f"{r['question_id']}|{r['model']}|{r['condition']}")
        except Exception:
            pass
    return keys


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _bullet_prompt(passages: list[str], question: str) -> str:
    ctx = "\n".join(f"- {p.strip()}" for p in passages if p.strip())
    return (f"Read the following passages and answer the question with a single "
            f"word or short phrase. Do not explain.\n\n"
            f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")


def _prose_prompt(passages: list[str], question: str) -> str:
    # Concatenate naturally written paragraphs as-is (already prose)
    ctx = " ".join(p.strip() for p in passages if p.strip())
    return (f"Read the following passage and answer the question with a single "
            f"word or short phrase. Do not explain.\n\n"
            f"Passage:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def load_hotpotqa(n: int = 200):
    from datasets import load_dataset
    print(f"Loading HotpotQA fullwiki (n={n})...")
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
    items = []
    for i, ex in enumerate(ds):
        if len(items) >= n:
            break
        # Extract supporting sentences only (the relevant passages)
        sup_facts = ex.get("supporting_facts", {})
        titles    = sup_facts.get("title", [])
        sent_ids  = sup_facts.get("sent_id", [])
        # Build a lookup: title -> list of sentences
        context   = ex.get("context", {})
        ctx_titles  = context.get("title", [])
        ctx_sents   = context.get("sentences", [])
        title2sents = {t: s for t, s in zip(ctx_titles, ctx_sents)}
        # Collect supporting sentences grouped by paragraph
        sup_paras = {}
        for t, sid in zip(titles, sent_ids):
            sents = title2sents.get(t, [])
            if sid < len(sents):
                sup_paras.setdefault(t, []).append(sents[sid])
        # Each supporting paragraph = one "passage"
        passages = [" ".join(sents) for sents in sup_paras.values() if sents]
        if len(passages) < 2:
            continue  # need at least 2 supporting passages for multi-hop
        answer = ex["answer"]
        if answer.lower() in ("yes", "no"):
            pass  # keep yes/no questions
        items.append({
            "id": f"hotpot_{ex['id']}",
            "question": ex["question"],
            "answer": answer,
            "passages": passages,
            "n_hops": 2,
        })
    print(f"  Loaded {len(items)} HotpotQA questions")
    return items[:n]


def run_hotpotqa(client, n: int = 200, dry_run: bool = False) -> None:
    out_path = RESULTS_DIR / "d1_hotpotqa.jsonl"
    done = _done_keys(out_path)
    questions = load_hotpotqa(n)

    tasks = []
    for q in questions:
        for mn in D1_MODELS:
            for cond in ["A", "B"]:
                key = f"{q['id']}|{mn}|{cond}"
                if key not in done:
                    tasks.append((q, mn, cond))

    total = len(done) + len(tasks)
    print(f"\n[D1-HotpotQA] {len(tasks)} calls remaining (of {total} total)")
    if dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]

    def call(q, mn, cond):
        prompt = (_bullet_prompt(q["passages"], q["question"]) if cond == "A"
                  else _prose_prompt(q["passages"], q["question"]))
        try:
            pred = _invoke(client, mn, prompt)
        except Exception as e:
            print(f"  ERR {mn}: {e}"); return
        correct = _score(pred, q["answer"])
        rec = {"dataset": "hotpotqa", "question_id": q["id"], "model": mn,
               "condition": cond, "answer": q["answer"],
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        if counter[0] % 50 == 0:
            sym = "✓" if correct else "✗"
            print(f"  [{counter[0]}/{total}] {mn:<20} cond={cond}  {sym}  Q: {q['question'][:40]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, q, mn, cond) for q, mn, cond in tasks]
        for f in as_completed(futs):
            f.result()

    _print_summary(out_path, "HotpotQA")


# ---------------------------------------------------------------------------
# MuSiQue
# ---------------------------------------------------------------------------

def load_musique(n: int = 200):
    from datasets import load_dataset
    print(f"Loading MuSiQue (n={n})...")
    ds = load_dataset("bdsaglam/musique", split="validation")
    items = []
    for ex in ds:
        if len(items) >= n:
            break
        # Only use answerable questions
        if not ex.get("answerable", True):
            continue
        # Supporting paragraphs only
        paras = ex.get("paragraphs", [])
        passages = [p["paragraph_text"] for p in paras if p.get("is_supporting", False)]
        if len(passages) < 2:
            continue
        items.append({
            "id": f"musique_{ex['id']}",
            "question": ex["question"],
            "answer": ex["answer"],
            "passages": passages,
            "n_hops": len(passages),
        })
    print(f"  Loaded {len(items)} MuSiQue questions")
    return items[:n]


def run_musique(client, n: int = 200, dry_run: bool = False) -> None:
    out_path = RESULTS_DIR / "d1_musique.jsonl"
    done = _done_keys(out_path)
    questions = load_musique(n)

    tasks = []
    for q in questions:
        for mn in D1_MODELS:
            for cond in ["A", "B"]:
                key = f"{q['id']}|{mn}|{cond}"
                if key not in done:
                    tasks.append((q, mn, cond))

    total = len(done) + len(tasks)
    print(f"\n[D1-MuSiQue] {len(tasks)} calls remaining (of {total} total)")
    if dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]

    def call(q, mn, cond):
        prompt = (_bullet_prompt(q["passages"], q["question"]) if cond == "A"
                  else _prose_prompt(q["passages"], q["question"]))
        try:
            pred = _invoke(client, mn, prompt)
        except Exception as e:
            print(f"  ERR {mn}: {e}"); return
        correct = _score(pred, q["answer"])
        rec = {"dataset": "musique", "question_id": q["id"], "model": mn,
               "condition": cond, "answer": q["answer"], "n_hops": q["n_hops"],
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        if counter[0] % 50 == 0:
            sym = "✓" if correct else "✗"
            print(f"  [{counter[0]}/{total}] {mn:<20} cond={cond}  {sym}  Q: {q['question'][:40]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, q, mn, cond) for q, mn, cond in tasks]
        for f in as_completed(futs):
            f.result()

    _print_summary(out_path, "MuSiQue")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(path: Path, dataset_name: str) -> None:
    from collections import defaultdict
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][r["condition"]].append(r["correct"])

    print(f"\n=== D1 {dataset_name} — Bullets (A) vs Prose (B) ===")
    print(f"{'Model':<22} {'Cond A':>8} {'Cond B':>8}  {'A−B':>8}  Format verdict")
    print("-" * 72)
    for mn in D1_MODELS:
        if mn not in acc: continue
        a_v = acc[mn]["A"]; b_v = acc[mn]["B"]
        if not a_v or not b_v: continue
        a = sum(a_v)/len(a_v); b = sum(b_v)/len(b_v)
        diff = a - b
        verdict = ("COLLAPSES" if diff > 0.15 else
                   "slight drop" if diff > 0.05 else
                   "FORMAT-ROBUST" if diff > -0.05 else "prose better")
        print(f"{mn:<22} {a:>8.3f} {b:>8.3f}  {diff:>+8.3f}  {verdict}")

    print(f"\nKey: Does Llama collapse on naturalistic prose? "
          f"Does model ranking match P1 synthetic results?")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hotpot",   type=int, default=200)
    parser.add_argument("--n-musique",  type=int, default=200)
    parser.add_argument("--dataset",    default="all", choices=["hotpotqa", "musique", "all"])
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = get_client(getattr(args, "backend", None))
    if not args.dry_run:

    if args.dataset in ("hotpotqa", "all"):
        run_hotpotqa(client, n=args.n_hotpot, dry_run=args.dry_run)

    if args.dataset in ("musique", "all"):
        run_musique(client, n=args.n_musique, dry_run=args.dry_run)

    print("\nD1 complete.")


if __name__ == "__main__":
    main()
