"""P1: N-scaling degradation — accuracy vs hop depth N for all models.

Measures accuracy at N=1,2,3,4 hops for every model in the registry.
Token count is held constant across N to isolate hop depth from context length.

Usage:
    python run_p1.py [--models llama3-8b gemma3-12b] [--n-questions 100] [--dry-run]

Results: results/p1_results.jsonl
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client, MODELS
from generate import build_dataset

RESULTS_DIR = Path(__file__).parent / "results"
N_LEVELS    = [1, 2, 3, 4]
SEEDS       = [0, 1, 2]
CONCURRENCY = 8

_write_lock = threading.Lock()


def _build_prompt(passages: list[str], question: str) -> str:
    ctx = "\n".join(f"- {p}" for p in passages)
    return (
        "Read the following passages and answer the question with a single "
        "word or short phrase. Do not explain.\n\n"
        f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:"
    )


def _score(pred: str, ans: str) -> int:
    import re
    pred = re.sub(r"\\\\?_", "_", pred.strip().lower())
    ans  = ans.strip().lower()
    if ans in pred or pred.startswith(ans):
        return 1
    if "_" in ans:
        stem = ans.split("_")[0]
        if len(stem) >= 4 and pred.startswith(stem):
            return 1
    return 0


def run(model_subset=None, n_questions=100, dry_run=False, backend=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "p1_results.jsonl"

    model_names = model_subset or list(MODELS.keys())

    done_ids: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_ids.add(f"{r['question_id']}|{r['model']}")
            except Exception:
                pass
    if done_ids:
        print(f"Resuming: {len(done_ids)} calls already done.")

    questions = build_dataset(n_questions=n_questions, n_levels=N_LEVELS, seeds=SEEDS)
    todo = [(q, mn) for q in questions for mn in model_names
            if f"{q['id']}|{mn}" not in done_ids]
    total = len(questions) * len(model_names)

    print(f"Models: {len(model_names)}  Questions: {len(questions)}  "
          f"Total: {total}  Remaining: {len(todo)}")

    if dry_run:
        for q, mn in todo[:3]:
            print(f"  [{mn}] {q['id']}  N={q['n']}")
            print("  " + _build_prompt(q["passages"], q["question"])[:200] + "...")
        return

    client = get_client(backend)
    completed = [len(done_ids)]

    def call_one(q, mn):
        prompt = _build_prompt(q["passages"], q["question"])
        try:
            pred = client.invoke(mn, prompt, max_tokens=32, temperature=0.0)
        except Exception as e:
            print(f"  ERR [{mn}] {q['id']}: {e}"); return None
        rec = {"question_id": q["id"], "model": mn, "n": q["n"], "seed": q["seed"],
               "answer": q["answer"], "prediction": pred, "correct": _score(pred, q["answer"])}
        with _write_lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            completed[0] += 1
        sym = "✓" if rec["correct"] else "✗"
        print(f"  [{completed[0]}/{total}] {mn:<22} {q['id']}  N={q['n']}  {sym}")
        return rec

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call_one, q, mn) for q, mn in todo]
        for f in as_completed(futs):
            f.result()

    _print_summary(out_path, model_names)


def _print_summary(path: Path, model_names: list) -> None:
    from collections import defaultdict
    import math
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    buckets = defaultdict(lambda: defaultdict(list))
    for r in records:
        buckets[r["model"]][r["n"]].append(r["correct"])
    print("\n=== P1: Accuracy by model × hop depth ===")
    print(f"{'Model':<24} {'N=1':>6} {'N=2':>6} {'N=3':>6} {'N=4':>6}  {'Drop':>7}")
    print("-" * 60)
    for mn in model_names:
        if mn not in buckets: continue
        accs = {n: (sum(v)/len(v) if v else float("nan"))
                for n, v in buckets[mn].items()}
        drop = accs.get(1, float("nan")) - accs.get(4, float("nan"))
        cols = [f"{accs.get(n, float('nan')):.3f}" for n in N_LEVELS]
        print(f"{mn:<24} {'  '.join(cols)}  {drop:>+7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", default=None, help="openai|anthropic|hf (or set LLM_BACKEND)")
    args = ap.parse_args()
    run(model_subset=args.models, n_questions=args.n_questions,
        dry_run=args.dry_run, backend=args.backend)
