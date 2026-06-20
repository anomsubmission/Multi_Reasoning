"""Phase 2 — M3: Block Masking Knockout.

Zero-ablates each block range and measures accuracy drop.
Identifies which transformer block range drives integration capacity.

Block partitioning: n_layers split into 4 equal groups.
  Llama-8B (32 layers):  [0-7], [8-15], [16-23], [24-31]
  Gemma3-4b (34 layers): [0-8], [9-17], [18-25], [26-33]

Key question: does the critical range differ between Condition A and B?
  Same range → prose-collapse disrupts the format fed to the same circuit.
  Different range → prose recruits a different (failing) circuit.

Output:
  results/phase2/m3_block_mask.jsonl
  results/phase2/m3_block_mask_summary.txt

Usage:
    python probe_m3_block_mask.py [--n-examples 20] [--seed 0]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    PROBE_MODELS, build_prompt_a, build_prompt_b,
    get_arch_info, get_questions, load_model, score,
)

RESULTS_DIR = Path(__file__).parent / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_block_ranges(n_layers: int, n_groups: int = 4) -> list[tuple[int, int, str]]:
    """Partition n_layers into n_groups blocks. Returns [(start, end, label), ...]."""
    size = math.ceil(n_layers / n_groups)
    result = []
    for i in range(n_groups):
        start = i * size
        end = min(start + size, n_layers)
        if start >= n_layers:
            break
        result.append((start, end, f"B{start}-{end-1}"))
    return result


@contextmanager
def ablated_block(layers, start: int, end: int):
    """
    Context manager: zero-ablate transformer layers [start, end).
    Uses PyTorch forward hooks to intercept output and replace it with input.
    Works regardless of whether forward() returns a tensor or tuple.
    """
    handles = []

    def make_hook(layer_module):
        def hook(module, args, output):
            # args[0] is the input hidden_states to this layer
            hs_in = args[0] if args else None
            if hs_in is None:
                return output
            if isinstance(output, tuple):
                return (hs_in,) + output[1:]
            # Scalar tensor return (transformers 5.x LlamaDecoderLayer)
            return hs_in
        return hook

    for idx in range(start, end):
        handle = layers[idx].register_forward_hook(make_hook(layers[idx]))
        handles.append(handle)

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def run_block_mask(model, tok, prompt: str, answer: str,
                   blocks: list[tuple[int, int, str]],
                   arch_layers) -> dict:
    """
    Baseline forward pass + one ablated pass per block range.

    Returns:
        baseline_correct, baseline_pred,
        ablations: [{label, start, end, correct, pred, acc_drop}]
    """
    device = next(model.parameters()).device
    inp = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inp)
    pred_id = out.logits[0, -1].argmax().item()
    baseline_pred = tok.decode([pred_id]).strip()
    baseline_correct = score(baseline_pred, answer)

    ablations = []
    for start, end, label in blocks:
        with ablated_block(arch_layers, start, end):
            with torch.no_grad():
                out_abl = model(**inp)
        abl_id = out_abl.logits[0, -1].argmax().item()
        abl_pred = tok.decode([abl_id]).strip()
        abl_correct = score(abl_pred, answer)
        ablations.append({
            "label": label,
            "start": start,
            "end": end,
            "correct": abl_correct,
            "pred": abl_pred,
            "acc_drop": round(float(baseline_correct) - float(abl_correct), 4),
        })

    return {
        "baseline_correct": baseline_correct,
        "baseline_pred": baseline_pred,
        "ablations": ablations,
    }


def run_m3(n_examples: int = 20, seed: int = 0, models: list = None):
    out_path = RESULTS_DIR / "m3_block_mask.jsonl"
    summary_path = RESULTS_DIR / "m3_block_mask_summary.txt"

    questions = get_questions(n=n_examples, seed=seed, n_hops=3)
    print(f"Using {len(questions)} N=3 questions")

    done_keys: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_keys.add((r["model"], r["condition"], r["question_id"]))
            except Exception:
                pass

    all_results = []
    model_subset = {k: v for k, v in PROBE_MODELS.items()
                    if models is None or k in models}

    for model_name, model_id in model_subset.items():
        model, tok = load_model(model_name, model_id)
        arch = get_arch_info(model)
        arch_layers = arch["layers"]
        n_layers = arch["n_layers"]
        blocks = get_block_ranges(n_layers)
        print(f"  Block partition: {[(l, s, e) for s, e, l in blocks]}")

        for q in questions:
            for cond, prompt_fn in [("A", build_prompt_a), ("B", build_prompt_b)]:
                if (model_name, cond, q["id"]) in done_keys:
                    continue
                prompt = prompt_fn(q, seed=seed, tok=tok)
                print(f"  {model_name} cond={cond} q={q['id']} ...", end=" ", flush=True)

                res = run_block_mask(model, tok, prompt, q["answer"], blocks, arch_layers)
                sym = "✓" if res["baseline_correct"] else "✗"
                abl_str = " ".join(f"{a['label']}:{a['acc_drop']:+.2f}"
                                   for a in res["ablations"])
                print(f"{sym} | {abl_str}")

                record = {
                    "model": model_name,
                    "question_id": q["id"],
                    "answer": q["answer"],
                    "condition": cond,
                    "n_layers": n_layers,
                    **res,
                }
                all_results.append(record)
                done_keys.add((model_name, cond, q["id"]))
                with out_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc; gc.collect()

    all_results_full = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    _write_summary(all_results_full, summary_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {summary_path}")


def _write_summary(results: list, path: Path):
    all_labels = sorted(
        {a["label"] for r in results for a in r["ablations"]},
        key=lambda x: int(x[1:].split("-")[0]),
    )
    baseline_acc = defaultdict(list)
    drops = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r["model"], r["condition"])
        baseline_acc[key].append(int(r["baseline_correct"]))
        for a in r["ablations"]:
            drops[key][a["label"]].append(a["acc_drop"])

    col_w = 9
    lines = ["=== M3 Block Masking Knockout Summary ===\n"]
    lines.append("acc_drop = baseline_acc − ablated_acc  (positive = block was helping)")
    lines.append("")
    header = (f"{'Model:Cond':<28} {'Baseline':>9} " +
               " ".join(f"{l:>{col_w}}" for l in all_labels))
    lines.append(header)
    lines.append("-" * len(header))

    for model_name in PROBE_MODELS:
        for cond in ["A", "B"]:
            key = (model_name, cond)
            ba = (sum(baseline_acc[key]) / len(baseline_acc[key])
                  if baseline_acc[key] else float("nan"))
            drop_vals = []
            for lbl in all_labels:
                ds = drops[key][lbl]
                drop_vals.append(f"{sum(ds)/len(ds):>+{col_w}.3f}" if ds else f"{'---':>{col_w}}")
            lines.append(f"{model_name+':'+cond:<28} {ba:>9.3f} " + " ".join(drop_vals))

    # Highlight most critical block per condition
    lines.append("\nMost critical block per condition (highest mean acc_drop):")
    for model_name in PROBE_MODELS:
        for cond in ["A", "B"]:
            key = (model_name, cond)
            best_lbl, best_drop = None, -1.0
            for lbl in all_labels:
                ds = drops[key][lbl]
                if ds:
                    mean_d = sum(ds) / len(ds)
                    if mean_d > best_drop:
                        best_drop = mean_d
                        best_lbl = lbl
            lines.append(f"  {model_name}:{cond}  → {best_lbl}  (mean drop={best_drop:+.3f})")

    lines.extend([
        "\nHypothesis:",
        "  Llama-8B: blocks [8-15] or [16-23] show largest drop (mid-depth integration circuit)",
        "  Gemma-4b: more distributed, smaller drops (robust)",
        "  Key diagnostic: if critical block shifts between cond A and B,",
        "    prose-collapse uses a different (failing) circuit — novel mechanistic finding.",
        "    If same block, format disrupts input to the same circuit.",
    ])

    text = "\n".join(lines)
    path.write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all)")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    run_m3(n_examples=args.n_examples, seed=args.seed, models=models)
