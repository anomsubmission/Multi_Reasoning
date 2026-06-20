"""P3-extra: run the identical P3 format contrast (bullets A vs fused prose B at
N=3) for the models NOT in the original 6-model focal set, so we can compute the
N-scaling-vs-format-gap correlation over the full 16-model set instead of 6.

Protocol is byte-for-byte the same as run_p2_p3_p4.run_p3:
  - questions: generate.build_dataset(n_questions=100, n_levels=[3], seeds=[0,1])
  - cond A: all passages as bullet points
  - cond B: relevant facts woven into one coherent paragraph (padded)
  - greedy decoding, normalized-EM scoring
Output: results/p3_extra.jsonl   (resumable)

Usage:
    python run_p3_extra.py [--dry-run] [--models m1,m2]
"""
import argparse, json, random, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client, MODELS
from generate import build_dataset
from run_p2_p3_p4 import _make_coherent_passage, _score, _done_keys, SEEDS

RESULTS_DIR = Path(__file__).parent / "results"
CONCURRENCY = 8

FOCAL = {"deepseek-r1", "qwen3-32b", "llama3-70b", "gemma3-12b", "llama3-8b", "mixtral-8x7b"}
# Everything in the registry that already appears in P1 but not in the P3 focal set.
EXTRA_MODELS = [m for m in MODELS if m not in FOCAL]

_write_lock = threading.Lock()

def _append(path: Path, record: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def prompt_a(all_passages, question):
    ctx = "\n".join(f"- {p}" for p in all_passages)
    return ("Read the following passages and answer the question with a single "
            "word or short phrase. Do not explain.\n\n"
            f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")


def prompt_b(passages_relevant, question, seed):
    rng = random.Random(seed * 999)
    coherent = _make_coherent_passage(passages_relevant, rng)
    return ("Read the following passage and answer the question with a single "
            "word or short phrase. Do not explain.\n\n"
            f"Passage:\n{coherent}\n\nQuestion: {question}\n\nAnswer:")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--outfile", default="p3_extra.jsonl")
    ap.add_argument("--backend", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",") if args.models else EXTRA_MODELS
    models = [m for m in models if m in MODELS]

    client = get_client(getattr(args, "backend", None))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.outfile
    done = _done_keys(out_path, ["question_id", "model", "condition", "seed"])

    questions = [q for q in build_dataset(n_questions=args.nq, n_levels=[3], seeds=seeds)
                 if q["n"] == 3]

    tasks = []
    for q in questions:
        relevant = [p for p in q["passages"]
                    if q["answer"] in p or "Anyone" in p or
                    q["question"].split()[-1].rstrip("?") in p]
        for mn in models:
            for cond in ["A", "B"]:
                for seed in seeds:
                    key = f"{q['id']}|{mn}|{cond}|{seed}"
                    if key not in done:
                        tasks.append((q, mn, cond, seed, relevant))

    print(f"[P3-extra] {len(tasks)} calls remaining ({len(done)} already done)")
    if args.dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]
    total = len(done) + len(tasks)

    def call(q, mn, cond, seed, relevant):
        if cond == "A":
            prompt = prompt_a(q["passages"], q["question"])
        else:
            prompt = prompt_b(relevant, q["question"], seed)
        try:
            pred = client.invoke(mn, prompt, max_tokens=32, temperature=0.0)
        except Exception as e:
            print(f"  ERR {mn} {cond}: {str(e)[:80]}"); return
        correct = _score(pred, q["answer"])
        _append(out_path, {"question_id": q["id"], "model": mn, "condition": cond,
                           "seed": seed, "answer": q["answer"],
                           "prediction": pred, "correct": correct})
        with _write_lock:
            counter[0] += 1
            if counter[0] % 50 == 0:
                print(f"  [{counter[0]}/{total}]")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, *t) for t in tasks]
        for f in as_completed(futs):
            f.result()

    print("[P3-extra] complete.")


if __name__ == "__main__":
    main()
