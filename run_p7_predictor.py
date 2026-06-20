"""P7: Validating the format-invariance probe as a PREDICTOR.

The paper reframes the synthetic bullets-vs-prose gap (delta = A - B) not as a
deployment vulnerability but as a *precision instrument* for integration-circuit
brittleness. A genuine instrument must PREDICT behaviour it was not defined on.

This experiment tests that. For each model we measure accuracy under the
canonical bullets baseline (A) plus three HELD-OUT format renderings of the
same N=3 facts that were never used to define the probe:

  numbered  : facts as an enumerated "1. 2. 3." list
  json      : facts as a JSON array of strings
  runon     : facts concatenated into one un-punctuated run-on line
              (a different fusion from the discourse-connector prose of cond B)

Validation claim: a model's canonical probe score (A - B, from P3/P5/P6)
should rank-correlate with its held-out gaps (A - numbered, A - json,
A - runon). If it does, the cheap bullets-vs-prose probe predicts format
brittleness in general -> it is a validated diagnostic, not a one-off curiosity.
Analysis (correlation) lives in analyze_p7.py once data lands.

Output: results/p7_predictor.jsonl
  {question_id, model, condition, seed, answer, prediction, correct}

Usage:
    python run_p7_predictor.py [--dry-run]
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client, MODELS
from generate import build_dataset
from run_p5 import prompt_a, _score, _append, _done_keys, _relevant, _BULLET_FILLERS

RESULTS_DIR = Path(__file__).parent / "results"
CONCURRENCY = 3
SEEDS       = [0]   # one seed: 100 trials/cell is ample for a 10-model rank correlation

# Span the behavioural spectrum: known collapsers, robust models, intermediate.
P7_MODELS = [
    "llama3-8b", "llama3-70b", "llama3.3-70b",   # collapse on prose
    "mistral-7b",                                  # intermediate
    "gemma3-12b", "qwen3-32b", "deepseek-r1",      # robust
    "mixtral-8x7b",                                # N-scaling-fragile but format-robust
    "llama4-scout", "llama4-maverick",             # cross-generation
]

# Held-out format conditions (canonical A=bullets comes from prompt_a in run_p5)
HELDOUT_CONDS = ["numbered", "json", "runon"]

_QUESTION_TAIL = "\n\nQuestion: {q}\n\nAnswer:"
_INSTR_LIST = ("Read the following passages and answer the question with a single "
               "word or short phrase. Do not explain.\n\n")
_INSTR_ONE  = ("Read the following passage and answer the question with a single "
               "word or short phrase. Do not explain.\n\n")


def _ctx_items(q: dict, seed: int) -> list[str]:
    """Relevant passages + 4 bullet-style fillers, shuffled (same pool as cond A)."""
    rng = random.Random(seed)
    rel = _relevant(q)
    fillers = rng.sample(_BULLET_FILLERS, k=4)
    items = rel + fillers
    rng.shuffle(items)
    return items


def prompt_numbered(q: dict, seed: int) -> str:
    items = _ctx_items(q, seed)
    ctx = "\n".join(f"{i+1}. {p}" for i, p in enumerate(items))
    return _INSTR_LIST + f"Passages:\n{ctx}" + _QUESTION_TAIL.format(q=q["question"])


def prompt_json(q: dict, seed: int) -> str:
    items = _ctx_items(q, seed)
    ctx = json.dumps(items, ensure_ascii=False)
    return _INSTR_LIST + f"Passages (JSON):\n{ctx}" + _QUESTION_TAIL.format(q=q["question"])


def prompt_runon(q: dict, seed: int) -> str:
    # Fuse facts into one un-punctuated run-on line: a DIFFERENT fusion style
    # than cond B's discourse-connector prose, so it is genuinely held out.
    items = _ctx_items(q, seed)
    ctx = " ".join(p.rstrip(".").strip() for p in items)
    return _INSTR_ONE + f"Passage:\n{ctx}" + _QUESTION_TAIL.format(q=q["question"])


PROMPT_FNS = {
    "A": prompt_a,
    "numbered": prompt_numbered,
    "json": prompt_json,
    "runon": prompt_runon,
}

_write_lock = threading.Lock()


def _invoke(client, model_name: str, prompt: str) -> str:
    model_id = MODELS.get(model_name, model_name)
    return client.invoke(model_id, prompt, max_tokens=32, temperature=0.0)


def run_p7(client, dry_run: bool = False) -> None:
    out_path = RESULTS_DIR / "p7_predictor.jsonl"
    done = _done_keys(out_path)

    questions = [q for q in build_dataset(n_questions=100, n_levels=[3], seeds=SEEDS)
                 if q["n"] == 3]

    conds = ["A"] + HELDOUT_CONDS
    tasks = []
    for q in questions:
        for mn in P7_MODELS:
            for cond in conds:
                for seed in SEEDS:
                    key = f"{q['id']}|{mn}|{cond}|{seed}"
                    if key not in done:
                        tasks.append((q, mn, cond, seed))

    total = len(done) + len(tasks)
    print(f"\n[P7] {len(tasks)} calls remaining (of {total} total)")
    print(f"     Models: {P7_MODELS}")
    print(f"     Conditions: {conds}  |  Seeds: {SEEDS}")
    if dry_run:
        print("[DRY RUN] no calls made."); return

    counter = [len(done)]

    def call(q, mn, cond, seed):
        prompt = PROMPT_FNS[cond](q, seed)
        try:
            pred = _invoke(client, mn, prompt)
        except Exception as e:
            print(f"  ERR {mn} {cond}: {e}"); return
        correct = _score(pred, q["answer"])
        rec = {"question_id": q["id"], "model": mn, "condition": cond,
               "seed": seed, "answer": q["answer"],
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        if counter[0] % 50 == 0 or counter[0] <= 3:
            sym = "✓" if correct else "✗"
            print(f"  [{counter[0]}/{total}] {mn:<16} {cond:<9} {sym} -> {pred[:18]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, q, mn, cond, seed) for q, mn, cond, seed in tasks]
        for f in as_completed(futs):
            f.result()

    _summary(out_path)


def _summary(path: Path) -> None:
    from collections import defaultdict
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][r["condition"]].append(r["correct"])
    print("\n=== P7 Held-out Format Gaps (baseline A=bullets) ===")
    print(f"{'Model':<16} {'A':>6} {'num':>6} {'json':>6} {'runon':>6} "
          f"{'gap_num':>8} {'gap_json':>9} {'gap_run':>8}")
    print("-" * 74)
    for mn in P7_MODELS:
        if mn not in acc or not acc[mn]["A"]:
            continue
        def m(c):
            v = acc[mn].get(c, [])
            return sum(v) / len(v) if v else float("nan")
        a = m("A")
        print(f"{mn:<16} {a:>6.3f} {m('numbered'):>6.3f} {m('json'):>6.3f} "
              f"{m('runon'):>6.3f} {a-m('numbered'):>+8.3f} {a-m('json'):>+9.3f} "
              f"{a-m('runon'):>+8.3f}")
    print("\nNext: analyze_p7.py correlates each model's canonical probe (A-B from")
    print("P3/P5/P6) against these held-out gaps to validate the probe as a predictor.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client(getattr(args, "backend", None))
    if not args.dry_run:
    run_p7(client, dry_run=args.dry_run)
    print("\nP7 complete.")


if __name__ == "__main__":
    main()
