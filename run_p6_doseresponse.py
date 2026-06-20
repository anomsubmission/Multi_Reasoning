"""P6: Format-gap dose-response across additional model families.

Extends the P3/P5 bullets-vs-prose (N=3) protocol to families and scales NOT
yet measured, so the synthetic-format effect can be read as a cross-family
*spectrum* rather than a Llama-only binary. Uses the EXACT prompt construction,
scoring, and dataset of P3/P5 so the new numbers are directly comparable.

Models added here (none appear in P3 or P5):
  - Gemma scale ladder endpoints: gemma3-4b, gemma3-27b
  - Mistral recipe diversity:      mistral-7b, ministral-8b
  - Amazon Nova family (new):      nova-micro, nova-lite, nova-pro

Combined with P3 (llama3-8b/70b, mixtral, qwen3, deepseek, gemma3-12b) and P5
(llama3.3-70b, llama4-scout, llama4-maverick) this yields a 16-model spectrum
of the format gap delta = A(bullets) - B(prose) at N=3.

Output: results/p6_doseresponse.jsonl
  {question_id, model, condition, seed, answer, prediction, correct}

Usage:
    python run_p6_doseresponse.py [--dry-run]
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
# Reuse the EXACT P3/P5 prompt + scoring logic so results are comparable.
from run_p5 import prompt_a, prompt_b, _score, _append, _done_keys

RESULTS_DIR = Path(__file__).parent / "results"
CONCURRENCY = 3
SEEDS       = [0, 1]

# Families/scales not present in P3 or P5 -> genuinely new spectrum points.
P6_MODELS = [
    "gemma3-4b",    # Gemma small end (P3 has 12b only)
    "gemma3-27b",   # Gemma large end
    "mistral-7b",   # Mistral dense 7B (paper's intermediate point)
    "ministral-8b", # Mistral newer 8B recipe
    "nova-micro",   # Amazon Nova family — entirely new family for the format test
    "nova-lite",
    "nova-pro",
]

_write_lock = threading.Lock()


def _invoke(client, model_name: str, prompt: str) -> str:
    model_id = MODELS.get(model_name, model_name)
    return client.invoke(model_id, prompt, max_tokens=32, temperature=0.0)


def run_p6(client, dry_run: bool = False) -> None:
    out_path = RESULTS_DIR / "p6_doseresponse.jsonl"
    done = _done_keys(out_path)

    questions = [q for q in build_dataset(n_questions=100, n_levels=[3], seeds=SEEDS)
                 if q["n"] == 3]

    tasks = []
    for q in questions:
        for mn in P6_MODELS:
            for cond in ["A", "B"]:
                for seed in SEEDS:
                    key = f"{q['id']}|{mn}|{cond}|{seed}"
                    if key not in done:
                        tasks.append((q, mn, cond, seed))

    total = len(done) + len(tasks)
    print(f"\n[P6] {len(tasks)} calls remaining (of {total} total)")
    print(f"     Models: {P6_MODELS}")
    print(f"     Conditions: A=bullets, B=prose  |  Seeds: {SEEDS}")
    if dry_run:
        print("[DRY RUN] no calls made."); return

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
        if counter[0] % 40 == 0 or counter[0] <= 3:
            sym = "✓" if correct else "✗"
            print(f"  [{counter[0]}/{total}] {mn:<16} cond={cond} {sym} -> {pred[:20]!r}")

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
    print("\n=== P6 Dose-Response Summary (A=bullets, B=prose, N=3) ===")
    print(f"{'Model':<16} {'A':>7} {'B':>7} {'A-B':>8}  Verdict")
    print("-" * 50)
    for mn in P6_MODELS:
        if mn not in acc or not acc[mn]["A"] or not acc[mn]["B"]:
            continue
        a = sum(acc[mn]["A"]) / len(acc[mn]["A"])
        b = sum(acc[mn]["B"]) / len(acc[mn]["B"])
        d = a - b
        v = ("COLLAPSES" if d > 0.20 else "slight drop" if d > 0.05
             else "robust" if d > -0.05 else "prose better")
        print(f"{mn:<16} {a:>7.3f} {b:>7.3f} {d:>+8.3f}  {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_client(getattr(args, "backend", None))
    if not args.dry_run:
    run_p6(client, dry_run=args.dry_run)
    print("\nP6 complete.")


if __name__ == "__main__":
    main()
