"""Phase 1 continuation: P2 (prior conflict), P3 (density/token), P4 (hop ordering).

All three share the same 6-model focus set and LLM client.
Run all three in sequence; results go to results/p2_*, results/p3_*, results/p4_*.

Usage:
    python run_p2_p3_p4.py [--experiment p2|p3|p4|all] [--dry-run]
"""

import argparse
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import get_client, MODELS

RESULTS_DIR = Path(__file__).parent / "results"
CONCURRENCY = 3  # reduced to avoid ServiceUnavailableException throttling
SEEDS       = [0, 1]

# 6 focal models: ceiling, strong-immune, large-dense, mid-immune, small-prone, MoE-prone
FOCAL_MODELS = [
    "deepseek-r1",      # Tier I ceiling
    "qwen3-32b",        # Tier I, no reasoning traces
    "llama3-70b",       # Tier I/II, large dense
    "gemma3-12b",       # Tier II, CLIFF_NAI immune
    "llama3-8b",        # Tier III, worst dense
    "mixtral-8x7b",     # Tier III, MoE anomaly
]

_write_lock = threading.Lock()

def _append(path: Path, record: dict) -> None:
    with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def _invoke(client, model_name: str, prompt: str,
            max_tokens: int = 64) -> str:
    model_id = MODELS.get(model_name, model_name)
    return client.invoke(model_id, prompt, max_tokens=max_tokens, temperature=0.0)


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


def _done_keys(path: Path, key_fields: list[str]) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            keys.add("|".join(str(r[f]) for f in key_fields))
        except Exception:
            pass
    return keys


# ---------------------------------------------------------------------------
# P2: Prior Conflict
# ---------------------------------------------------------------------------

# 120 factual questions with knowable answers — selected to span topics where
# smaller LLMs are likely to have wrong priors.
P2_PROBE_QUESTIONS = [
    # Easy warm-up (models should get these right)
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("How many bones does an adult human body have?", "206"),
    ("What planet is closest to the Sun?", "Mercury"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("In what year did World War II end?", "1945"),
    ("How many sides does a hexagon have?", "6"),
    ("What is the square root of 144?", "12"),
    # Commonly confused / tricky facts
    ("What is the largest desert in the world by area?", "Antarctic"),
    ("What is the longest river in the world?", "Nile"),
    ("What country has the most natural lakes?", "Canada"),
    ("What is the deepest lake in the world?", "Baikal"),
    ("What is the oldest known writing system?", "Sumerian"),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen"),
    ("How many moons does Mars have?", "2"),
    ("What is the smallest planet in the solar system?", "Mercury"),
    ("What metal is liquid at room temperature?", "mercury"),
    ("How many hearts does an octopus have?", "3"),
    ("What is the fastest land animal?", "cheetah"),
    ("What country invented paper?", "China"),
    ("What is the currency of Switzerland?", "franc"),
    ("What is the capital of New Zealand?", "Wellington"),
    ("What is the hardest known natural mineral?", "diamond"),
    # Plausible-but-wrong traps (models often confuse these)
    ("What year was the Eiffel Tower completed?", "1889"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("How many strings does a standard guitar have?", "6"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("How many valence electrons does oxygen have?", "6"),
    ("What is the atomic number of carbon?", "6"),
    ("What is the freezing point of water in Fahrenheit?", "32"),
    ("How many teeth does an adult human have?", "32"),
    ("What is the largest organ in the human body?", "skin"),
    ("How many bones are in the human spine?", "33"),
    ("What is the most spoken language in the world by native speakers?", "Mandarin"),
    ("What is the speed of sound in air at sea level in m/s (approx)?", "343"),
    ("What is the capital of South Africa?", "Pretoria"),
    ("How many time zones does China use?", "1"),
    ("What is the largest ocean?", "Pacific"),
    # Numbers / quantities that models often hallucinate
    ("How many days are in a leap year?", "366"),
    ("How many keys does a standard piano have?", "88"),
    ("How many players are on a basketball team on the court?", "5"),
    ("What is the boiling point of water in Fahrenheit?", "212"),
    ("How many continents are there?", "7"),
    ("How many bones does an adult human hand have?", "27"),
    ("What is the chemical formula for table salt?", "NaCl"),
    ("How many chromosomes do dogs have?", "78"),
    ("What is the largest country by land area?", "Russia"),
    ("How many chambers does a fish heart have?", "2"),
]


def run_p2_probe(client, dry_run: bool = False) -> dict:
    """Step 1: probe each focal model's baseline answers with no passages.
    Returns {model: [(question, expected_ans, model_ans, is_wrong), ...]}
    """
    out_path = RESULTS_DIR / "p2_probe.jsonl"
    done = _done_keys(out_path, ["model", "question"])
    results: dict[str, list] = {m: [] for m in FOCAL_MODELS}

    prompt_tmpl = ("Answer the following question with a short phrase or number. "
                   "Do not explain.\n\nQuestion: {q}\n\nAnswer:")

    tasks = [(mn, q, ans) for mn in FOCAL_MODELS
             for q, ans in P2_PROBE_QUESTIONS
             if f"{mn}|{q}" not in done]

    print(f"\n[P2-probe] {len(tasks)} calls remaining")
    if dry_run:
        print("[DRY RUN]"); return {}

    counter = [len(done)]
    total = len(done) + len(tasks)

    def call(mn, q, ans):
        prompt = prompt_tmpl.format(q=q)
        try:
            pred = _invoke(client, mn, prompt, max_tokens=32)
        except Exception as e:
            print(f"  ERR {mn}: {e}"); return
        correct = _score(pred, ans)
        rec = {"model": mn, "question": q, "expected": ans,
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        sym = "✓" if correct else "✗"
        print(f"  [{counter[0]}/{total}] {mn:<18} {sym}  Q: {q[:40]!r}  -> {pred[:30]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, mn, q, ans) for mn, q, ans in tasks]
        for f in as_completed(futs):
            f.result()

    # Load all probe results and identify wrong-prior anchors
    anchors: dict[str, list] = {m: [] for m in FOCAL_MODELS}
    for line in out_path.read_text().splitlines():
        r = json.loads(line)
        if r["model"] in anchors and not r["correct"]:
            anchors[r["model"]].append((r["question"], r["expected"], r["prediction"]))

    for mn in FOCAL_MODELS:
        print(f"  {mn}: {len(anchors[mn])} wrong-prior anchors")
    return anchors


def run_p2_main(client, anchors: dict, dry_run: bool = False) -> None:
    """Step 2: flood wrong-prior questions with N={1,3,5,10} confirming passages."""
    out_path = RESULTS_DIR / "p2_results.jsonl"
    done = _done_keys(out_path, ["model", "question", "n_passages", "seed"])
    N_LEVELS = [0, 1, 3, 5, 10]  # 0 = no passages (control arm)

    rng = random.Random(42)

    def make_passages(correct_ans: str, n: int, seed: int) -> list[str]:
        """Generate n passages all stating the correct answer."""
        r = random.Random(seed * 1000 + n)
        templates = [
            f"The correct answer is {correct_ans}.",
            f"Studies confirm that the answer is {correct_ans}.",
            f"According to experts, {correct_ans} is the correct answer.",
            f"Research shows the answer is {correct_ans}.",
            f"The established fact is that {correct_ans}.",
            f"It is well established that the answer is {correct_ans}.",
            f"The verified answer is {correct_ans}.",
            f"Documentation confirms {correct_ans} is correct.",
            f"The answer, confirmed by multiple sources, is {correct_ans}.",
            f"All sources agree: the answer is {correct_ans}.",
        ]
        chosen = r.sample(templates, k=min(n, len(templates)))
        if n > len(templates):
            chosen += r.choices(templates, k=n - len(templates))
        return chosen[:n]

    def build_prompt(q: str, correct_ans: str, n: int, seed: int) -> str:
        if n == 0:
            return (f"Answer the following question with a short phrase or number. "
                    f"Do not explain.\n\nQuestion: {q}\n\nAnswer:")
        passages = make_passages(correct_ans, n, seed)
        ctx = "\n".join(f"- {p}" for p in passages)
        return (f"Read the following passages and answer the question with a short "
                f"phrase or number. Do not explain.\n\n"
                f"Passages:\n{ctx}\n\nQuestion: {q}\n\nAnswer:")

    # Build task list
    tasks = []
    for mn in FOCAL_MODELS:
        anchor_list = anchors.get(mn, [])
        if not anchor_list:
            # Load from file
            anchor_list = []
            if (RESULTS_DIR / "p2_probe.jsonl").exists():
                for line in (RESULTS_DIR / "p2_probe.jsonl").read_text().splitlines():
                    r = json.loads(line)
                    if r["model"] == mn and not r["correct"]:
                        anchor_list.append((r["question"], r["expected"], r["prediction"]))
        for q, ans, _ in anchor_list[:30]:
            for n in N_LEVELS:
                for seed in SEEDS:
                    key = f"{mn}|{q}|{n}|{seed}"
                    if key not in done:
                        tasks.append((mn, q, ans, n, seed))

    print(f"\n[P2-main] {len(tasks)} calls remaining")
    if dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]
    total = len(done) + len(tasks)

    def call(mn, q, ans, n, seed):
        prompt = build_prompt(q, ans, n, seed)
        try:
            pred = _invoke(client, mn, prompt, max_tokens=48)
        except Exception as e:
            print(f"  ERR {mn}: {e}"); return
        correct = _score(pred, ans)
        rec = {"model": mn, "question": q, "expected": ans,
               "n_passages": n, "seed": seed,
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        sym = "✓" if correct else "✗"
        print(f"  [{counter[0]}/{total}] {mn:<18} N={n:>2}  {sym}  -> {pred[:28]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, mn, q, ans, n, seed) for mn, q, ans, n, seed in tasks]
        for f in as_completed(futs):
            f.result()

    _print_p2_summary(out_path)


def _print_p2_summary(path: Path) -> None:
    from collections import defaultdict
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # EU per model per N
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][r["n_passages"]].append(r["correct"])

    print("\n=== P2 Summary — Evidence Utilization (acc_with_passages − acc_no_passages) ===")
    print(f"{'Model':<22} {'N=0':>6} {'N=1':>6} {'N=3':>6} {'N=5':>6} {'N=10':>6}  EU@N=10")
    print("-" * 70)
    for mn in FOCAL_MODELS:
        if mn not in acc: continue
        def a(n): return sum(acc[mn][n])/len(acc[mn][n]) if acc[mn][n] else float("nan")
        base = a(0)
        row  = [a(n) for n in [0, 1, 3, 5, 10]]
        eu10 = row[-1] - base
        cols = [f"{v:.3f}" if not math.isnan(v) else "  -  " for v in row]
        print(f"{mn:<22} {'  '.join(cols)}  {eu10:+.3f}")

    print("\nHypothesis: EU stays positive for Tier I; crosses zero for Tier III at N≥5.")


# ---------------------------------------------------------------------------
# P3: Density vs Token Count
# ---------------------------------------------------------------------------

def _make_coherent_passage(passages: list[str], rng: random.Random) -> str:
    """Weave N facts into a single coherent narrative paragraph."""
    connectors = [
        "Furthermore, ", "In addition, ", "It is also known that ",
        "Moreover, ", "Additionally, ", "Notably, "
    ]
    facts = [p.rstrip(".") for p in passages]
    result = facts[0] + "."
    for i, fact in enumerate(facts[1:]):
        conn = connectors[i % len(connectors)]
        result += " " + conn + fact.lower() + "."
    # Pad with neutral sentences to match token count target (~450 tokens / 4 chars ≈ 112 words)
    fillers = [
        "The context provided above contains all necessary information to answer the question.",
        "Background information has been carefully selected from reliable sources.",
        "The passages reflect the current state of knowledge on this topic.",
        "All relevant facts are present in the provided context.",
    ]
    while len(result.split()) < 90:
        result += " " + rng.choice(fillers)
    return result


def run_p3(client, dry_run: bool = False) -> None:
    """P3: Condition A (3 separate dense passages) vs Condition B (1 coherent paragraph)."""
    from generate import build_dataset

    out_path = RESULTS_DIR / "p3_results.jsonl"
    done = _done_keys(out_path, ["question_id", "model", "condition", "seed"])

    questions = [q for q in build_dataset(n_questions=100, n_levels=[3], seeds=SEEDS)
                 if q["n"] == 3]

    def prompt_a(passages_relevant: list[str], all_passages: list[str], question: str) -> str:
        # Dense: all passages as separate bullet points (shuffled, including filler)
        ctx = "\n".join(f"- {p}" for p in all_passages)
        return (f"Read the following passages and answer the question with a single "
                f"word or short phrase. Do not explain.\n\n"
                f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")

    def prompt_b(passages_relevant: list[str], question: str, seed: int) -> str:
        # Coherent: weave relevant facts into one paragraph, pad to same token count
        rng = random.Random(seed * 999)
        coherent = _make_coherent_passage(passages_relevant, rng)
        return (f"Read the following passage and answer the question with a single "
                f"word or short phrase. Do not explain.\n\n"
                f"Passage:\n{coherent}\n\nQuestion: {question}\n\nAnswer:")

    tasks = []
    for q in questions:
        relevant = [p for p in q["passages"]
                    if q["answer"] in p or "Anyone" in p or
                    q["question"].split()[-1].rstrip("?") in p]
        for mn in FOCAL_MODELS:
            for cond in ["A", "B"]:
                for seed in SEEDS:
                    key = f"{q['id']}|{mn}|{cond}|{seed}"
                    if key not in done:
                        tasks.append((q, mn, cond, seed, relevant))

    print(f"\n[P3] {len(tasks)} calls remaining")
    if dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]
    total = len(done) + len(tasks)

    def call(q, mn, cond, seed, relevant):
        if cond == "A":
            prompt = prompt_a(relevant, q["passages"], q["question"])
        else:
            prompt = prompt_b(relevant, q["question"], seed)
        try:
            pred = _invoke(client, mn, prompt, max_tokens=32)
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
        print(f"  [{counter[0]}/{total}] {mn:<18} cond={cond}  {sym}  -> {pred[:28]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, q, mn, cond, seed, relevant)
                for q, mn, cond, seed, relevant in tasks]
        for f in as_completed(futs):
            f.result()

    _print_p3_summary(out_path)


def _print_p3_summary(path: Path) -> None:
    from collections import defaultdict
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][r["condition"]].append(r["correct"])

    print("\n=== P3 Summary — Dense (A) vs. Coherent (B) at N=3 ===")
    print(f"{'Model':<22} {'Cond A':>8} {'Cond B':>8}  {'A−B':>8}  Interpretation")
    print("-" * 72)
    for mn in FOCAL_MODELS:
        if mn not in acc: continue
        a = sum(acc[mn]["A"])/len(acc[mn]["A"]) if acc[mn]["A"] else float("nan")
        b = sum(acc[mn]["B"])/len(acc[mn]["B"]) if acc[mn]["B"] else float("nan")
        diff = a - b
        interp = ("structural" if diff < -0.05 else
                  ("density" if diff > 0.05 else "no difference"))
        print(f"{mn:<22} {a:>8.3f} {b:>8.3f}  {diff:>+8.3f}  {interp}")

    print("\nHypothesis: Tier III models fail A >> B (structural boundary crossing).")
    print("  If A-B < -0.05: failure is structural; if A≈B and both low: density; if both high: no confound.")


# ---------------------------------------------------------------------------
# P4: Hop Ordering Decomposition
# ---------------------------------------------------------------------------

def run_p4(client, dry_run: bool = False) -> None:
    """P4: 4 passage orderings on N=2 questions to separate Type A vs Type B failure."""
    from generate import build_dataset

    out_path = RESULTS_DIR / "p4_results.jsonl"
    done = _done_keys(out_path, ["question_id", "model", "ordering", "seed"])

    questions = [q for q in build_dataset(n_questions=100, n_levels=[2], seeds=SEEDS)
                 if q["n"] == 2]

    FILLERS = [
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

    def get_relevant(q: dict) -> tuple[str, str]:
        """Return (hop1_passage, hop2_passage) for an N=2 question."""
        hop1, hop2 = None, None
        for p in q["passages"]:
            if "Anyone" in p:
                hop2 = p  # chaining rule: Anyone who X also Y
            elif q["answer"] not in p and hop1 is None:
                # hop1 is the anchor fact about the character
                name = q["question"].split()[2]  # "What does NAME ..."
                if name in p:
                    hop1 = p
        # fallback: pick any two relevant passages
        relevant = [p for p in q["passages"]
                    if q["answer"] in p or "Anyone" in p or
                    q["question"].split()[2] in p]
        if hop1 is None and len(relevant) >= 1:
            hop1 = relevant[0]
        if hop2 is None and len(relevant) >= 2:
            hop2 = relevant[1]
        return hop1 or "", hop2 or ""

    def make_ordered(hop1: str, hop2: str, order: str, rng: random.Random) -> list[str]:
        filler_pool = rng.sample(FILLERS, 6)
        f = filler_pool
        orderings = {
            "forward":   [hop1, hop2] + f[:4],           # natural order, filler at end
            "reverse":   [hop2, hop1] + f[:4],            # reversed, filler at end
            "buried":    [f[0], hop1, f[1], hop2, f[2], f[3]],  # hops interleaved with filler
            "split_far": [hop1, f[0], f[1], f[2], f[3], hop2],  # hops separated by long filler
        }
        return orderings[order]

    def build_prompt(passages: list[str], question: str) -> str:
        ctx = "\n".join(f"- {p}" for p in passages)
        return (f"Read the following passages and answer the question with a single "
                f"word or short phrase. Do not explain.\n\n"
                f"Passages:\n{ctx}\n\nQuestion: {question}\n\nAnswer:")

    ORDERINGS = ["forward", "reverse", "buried", "split_far"]

    tasks = []
    for q in questions:
        hop1, hop2 = get_relevant(q)
        if not hop1 or not hop2:
            continue
        for mn in FOCAL_MODELS:
            for order in ORDERINGS:
                for seed in SEEDS:
                    key = f"{q['id']}|{mn}|{order}|{seed}"
                    if key not in done:
                        tasks.append((q, mn, order, seed, hop1, hop2))

    print(f"\n[P4] {len(tasks)} calls remaining")
    if dry_run:
        print("[DRY RUN]"); return

    counter = [len(done)]
    total = len(done) + len(tasks)

    def call(q, mn, order, seed, hop1, hop2):
        rng = random.Random(seed * 7919 + hash(q["id"]) % 1000)
        passages = make_ordered(hop1, hop2, order, rng)
        prompt = build_prompt(passages, q["question"])
        try:
            pred = _invoke(client, mn, prompt, max_tokens=32)
        except Exception as e:
            print(f"  ERR {mn}: {e}"); return
        correct = _score(pred, q["answer"])
        rec = {"question_id": q["id"], "model": mn, "ordering": order,
               "seed": seed, "answer": q["answer"],
               "prediction": pred, "correct": correct}
        _append(out_path, rec)
        with _write_lock:
            counter[0] += 1
        sym = "✓" if correct else "✗"
        print(f"  [{counter[0]}/{total}] {mn:<18} ord={order:<10}  {sym}  -> {pred[:25]!r}")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futs = [pool.submit(call, q, mn, order, seed, hop1, hop2)
                for q, mn, order, seed, hop1, hop2 in tasks]
        for f in as_completed(futs):
            f.result()

    _print_p4_summary(out_path)


def _print_p4_summary(path: Path) -> None:
    from collections import defaultdict
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][r["ordering"]].append(r["correct"])

    ORDERINGS = ["forward", "reverse", "buried", "split_far"]
    print("\n=== P4 Summary — Ordering Effects on N=2 Questions ===")
    print(f"{'Model':<22} {'forward':>8} {'reverse':>8} {'buried':>8} {'split_far':>10}  Type")
    print("-" * 75)
    for mn in FOCAL_MODELS:
        if mn not in acc: continue
        vals = {o: (sum(acc[mn][o])/len(acc[mn][o]) if acc[mn][o] else float("nan"))
                for o in ORDERINGS}
        fwd_rev_gap = vals["forward"] - vals["reverse"]
        fwd_split_gap = vals["forward"] - vals["split_far"]
        # Type A: fails reverse (can't even handle reordering)
        # Type B: handles forward/reverse but fails split_far
        if fwd_rev_gap > 0.05:
            typ = "A (order-sensitive)"
        elif fwd_split_gap > 0.05:
            typ = "B (distance-sensitive)"
        else:
            typ = "robust"
        cols = [f"{vals[o]:>8.3f}" for o in ORDERINGS]
        print(f"{mn:<22} {''.join(cols)}  {typ}")

    print("\nHypothesis: Tier III shows Type A (fails reverse); "
          "Tier II shows Type B (fails split_far); Tier I is robust.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["p2", "p2_probe", "p3", "p4", "all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    client = get_client(getattr(args, "backend", None))
    if not args.dry_run:

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_all = args.experiment == "all"

    if run_all or args.experiment in ("p2", "p2_probe"):
        anchors = run_p2_probe(client, dry_run=args.dry_run)
        if run_all or args.experiment == "p2":
            run_p2_main(client, anchors, dry_run=args.dry_run)

    if run_all or args.experiment == "p3":
        run_p3(client, dry_run=args.dry_run)

    if run_all or args.experiment == "p4":
        run_p4(client, dry_run=args.dry_run)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
