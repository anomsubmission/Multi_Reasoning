"""P5: Llama-4 prose test (cross-generation format robustness).

Re-runs P3 (bullets vs prose, N=3) on Llama-3.3-70B and Llama-4 models to
determine whether prose-collapse is present in the newer Llama generations.

Key question: Are N-scaling improvement (P1) and format-robustness (P3) unified
by a single generational training change, or are they independent mechanisms?

If Llama-4 is prose-immune → same training change fixes both → unified story.
If Llama-4 still collapses → orthogonality holds across generations → richer story.

Output: results/p5_results.jsonl
  {question_id, model, condition, seed, answer, prediction, correct}

Usage:
    python run_p5.py [--dry-run]
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

RESULTS_DIR = Path(__file__).parent / "results"
CONCURRENCY = 3
SEEDS       = [0, 1]

# Only new-generation models — P3 already has llama3-8b, llama3-70b
P5_MODELS = [
    "llama3.3-70b",     # Llama 3.3 generation (not in P3)
    "llama4-scout",     # Llama 4 generation (17B MoE)
    "llama4-maverick",  # Llama 4 generation (17B MoE, stronger)
]

_write_lock = threading.Lock()


def _append(path: Path, record: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _invoke(client, model_name: str, prompt: str) -> str:
    model_id = MODELS.get(model_name, model_name)
    return client.invoke(model_id, prompt, max_tokens=32, temperature=0.0)


def _score(pred: str, ans: str) -> int:
    import re
    pred = re.sub(r'\\\\?_', '_', pred.strip().lower())
    ans  = ans.strip().lower()
    if ans in pred or pred.startswith(ans):
        return 1
    if '_' in ans:
        stem = ans.split('_')[0]
        if len(stem) >= 4 and pred.startswith(stem):
            return 1
    return 0


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            keys.add(f"{r['question_id']}|{r['model']}|{r['condition']}|{r['seed']}")
        except Exception:
            pass
    return keys


# ---------------------------------------------------------------------------
# Prompt builders (exact same logic as P3 in run_p2_p3_p4.py)
# ---------------------------------------------------------------------------

_CONNECTORS = [
    "Furthermore, ", "In addition, ", "It is also known that ",
    "Moreover, ", "Additionally, ", "Notably, ",
]

_PROSE_FILLERS = [
    "The context provided above contains all necessary information to answer the question.",
    "Background information has been carefully selected from reliable sources.",
    "The passages reflect the current state of knowledge on this topic.",
    "All relevant facts are present in the provided context.",
]

_BULLET_FILLERS = [
    "The sky was overcast that morning.",
    "A gentle breeze drifted through the valley.",
    "Somewhere in the distance a bell was ringing.",
    "The old library had not been opened in years.",
    "Rain was expected by the afternoon.",
    "The market closed early on holidays.",
    "Several birds perched on the fence post.",
    "The road curved sharply near the river.",
    "Thick fog made the journey difficult.",
    "A flag hung motionless above the gate.",
]


def _relevant(q: dict) -> list[str]:
    return [p for p in q["passages"]
            if q["answer"] in p or "Anyone" in p
            or q["question"].split()[-1].rstrip("?") in p]


def _make_coherent_passage(facts: list[str], rng: random.Random) -> str:
    facts = [f.rstrip(".") for f in facts]
    result = facts[0] + "."
    for i, fact in enumerate(facts[1:]):
        conn = _CONNECTORS[i % len(_CONNECTORS)]
        result += " " + conn + fact.lower() + "."
    while len(result.split()) < 90:
        result += " " + rng.choice(_PROSE_FILLERS)
    return result


def prompt_a(q: dict, seed: int) -> str:
    rng = random.Random(seed)
    rel = _relevant(q)
    fillers = rng.sample(_BULLET_FILLERS, k=4)
    all_p = rel + fillers
    rng.shuffle(all_p)
    ctx = "\n".join(f"- {p}" for p in all_p)
    return (f"Read the following passages and answer the question with a single "
            f"word or short phrase. Do not explain.\n\n"
            f"Passages:\n{ctx}\n\nQuestion: {q['question']}\n\nAnswer:")


def prompt_b(q: dict, seed: int) -> str:
    rng = random.Random(seed * 999)
    rel = _relevant(q)
    coherent = _make_coherent_passage(rel, rng)
    return (f"Read the following passage and answer the question with a single "
            f"word or short phrase. Do not explain.\n\n"
            f"Passage:\n{coherent}\n\nQuestion: {q['question']}\n\nAnswer:")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_p5(client, dry_run: bool = False) -> None:
    out_path = RESULTS_DIR / "p5_results.jsonl"
    done = _done_keys(out_path)

    questions = [q for q in build_dataset(n_questions=100, n_levels=[3], seeds=SEEDS)
                 if q["n"] == 3]

    tasks = []
    for q in questions:
        for mn in P5_MODELS:
            for cond in ["A", "B"]:
                for seed in SEEDS:
                    key = f"{q['id']}|{mn}|{cond}|{seed}"
                    if key not in done:
                        tasks.append((q, mn, cond, seed))

    total = len(done) + len(tasks)
    print(f"\n[P5] {len(tasks)} calls remaining (of {total} total)")
    print(f"     Models: {P5_MODELS}")
    print(f"     Conditions: A=bullets, B=prose  |  Seeds: {SEEDS}")
    if dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]

    def call(q, mn, cond, seed):
        prompt = prompt_a(q, seed) if cond == "A" else prompt_b(q, seed)
        try:
            pred = _invoke(client, mn, prompt)
        except Exception as e:
            print(f"  ERR {mn}: {e}"); return
        correct = _score(pred, q["answer"])
        rec = {"question_id": q["id"], "model": mn, "condition": cond,
               "seed": seed, "answer": q["answer"],
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        sym = "✓" if correct else "✗"
        if counter[0] % 20 == 0 or counter[0] <= 5:
            print(f"  [{counter[0]}/{total}] {mn:<20} cond={cond}  {sym}  -> {pred[:25]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, q, mn, cond, seed) for q, mn, cond, seed in tasks]
        for f in as_completed(futs):
            f.result()

    _print_p5_summary(out_path)


def _print_p5_summary(path: Path) -> None:
    from collections import defaultdict
    import math

    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    # Load P3 results for Llama-3 comparison baseline
    p3_path = RESULTS_DIR / "p3_results.jsonl"
    p3_records = []
    if p3_path.exists():
        p3_records = [json.loads(l) for l in p3_path.read_text().splitlines() if l.strip()]

    # Combine: P3 Llama models + P5 new models
    all_records = p3_records + records
    llama_models = ["llama3-8b", "llama3-70b", "llama3.3-70b", "llama4-scout", "llama4-maverick"]

    acc = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        if r["model"] in llama_models:
            acc[r["model"]][r["condition"]].append(r["correct"])

    print("\n=== P5 Summary — Llama cross-generation prose-collapse ===")
    print(f"{'Model':<24} {'Cond A':>8} {'Cond B':>8}  {'A−B':>8}  Gen  Interpretation")
    print("-" * 80)
    for mn in llama_models:
        if mn not in acc: continue
        a_vals = acc[mn]["A"]
        b_vals = acc[mn]["B"]
        if not a_vals or not b_vals: continue
        a = sum(a_vals) / len(a_vals)
        b = sum(b_vals) / len(b_vals)
        diff = a - b
        gen = ("Llama 3.1" if "llama3-8" in mn or "llama3-70" in mn
               else "Llama 3.3" if "3.3" in mn
               else "Llama 4")
        interp = ("COLLAPSES (A>>B)" if diff > 0.20
                  else "slight drop" if diff > 0.05
                  else "FORMAT-ROBUST" if diff > -0.05
                  else "prose better")
        print(f"{mn:<24} {a:>8.3f} {b:>8.3f}  {diff:>+8.3f}  {gen:<10}  {interp}")

    print("\nKey question: Is Llama-4 format-robust (A≈B)? If yes, generational fix unified P1+P3.")
    print("   If Llama-4 still collapses on prose: prose-collapse and N-scaling have independent mechanisms.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = get_client(getattr(args, "backend", None))
    if not args.dry_run:

    run_p5(client, dry_run=args.dry_run)
    print("\nP5 complete.")


if __name__ == "__main__":
    main()
