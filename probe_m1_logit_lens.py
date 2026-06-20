"""Phase 2 — M1: Logit Lens across conditions and layers.

At which transformer layer does the correct answer first emerge in the model's
hidden-state representation? Compares Condition A (bullet passages) vs Condition B
(coherent prose) for N=3 multi-hop questions.

Models (local, MPS):
  - llama3-8b  (shows prose-collapse in P3)
  - gemma3-4b  (control: no prose-collapse)

Output:
  results/phase2/m1_logit_lens.jsonl   — per-example, per-layer records
  results/phase2/m1_logit_lens_summary.txt

Usage:
    python probe_m1_logit_lens.py [--n-examples 20] [--seed 0] [--layer-stride 2]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    PROBE_MODELS, build_prompt_a, build_prompt_b,
    get_arch_info, get_questions, load_model, score, target_token_ids,
)

RESULTS_DIR = Path(__file__).parent / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def logit_lens_analysis(model, tok, prompt: str, answer: str,
                        layer_stride: int = 2) -> dict:
    """
    Runs logit lens: at every `layer_stride`-th layer, decode hidden states via
    lm_head and record target-token rank and probability.

    Returns:
        predicted: str             — model's actual next token
        pred_correct: bool
        target_first_layer: int|None — first layer where target reaches rank 0
        layers: list of {layer, rank, prob, top1}
    """
    arch = get_arch_info(model)
    lm_head = arch["lm_head"]
    final_norm = arch["final_norm"]
    n_layers = arch["n_layers"]
    t_ids = target_token_ids(tok, answer)

    inp = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    # Actual prediction
    pred_id = out.logits[0, -1].argmax().item()
    pred_str = tok.decode([pred_id]).strip()
    pred_correct = score(pred_str, answer)

    layer_results = []
    first_rank0 = None

    for layer_idx in range(0, n_layers, layer_stride):
        h = out.hidden_states[layer_idx + 1]   # +1: index 0 is embedding
        # Align devices for sharded (device_map="auto") models; no-op on single device.
        fn_dev = next(final_norm.parameters()).device
        lm_dev = next(lm_head.parameters()).device
        h_norm = final_norm(h.to(fn_dev))
        logits = lm_head(h_norm.to(lm_dev))[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)

        sorted_ids = torch.argsort(logits, descending=True)
        rank = None
        best_prob = 0.0
        for i, sid in enumerate(sorted_ids[:500].cpu()):
            if sid.item() in t_ids:
                rank = i
                best_prob = probs[sid.item()].item()
                break
        if rank is None:
            valid = [tid for tid in t_ids if tid < probs.shape[0]]
            best_prob = max((probs[tid].item() for tid in valid), default=0.0)

        top1_str = tok.decode([sorted_ids[0].item()]).strip()

        if rank == 0 and first_rank0 is None:
            first_rank0 = layer_idx

        layer_results.append({
            "layer": layer_idx,
            "rank": rank,
            "prob": round(best_prob, 6),
            "top1": top1_str,
        })

    return {
        "predicted": pred_str,
        "pred_correct": pred_correct,
        "target_first_layer": first_rank0,
        "layers": layer_results,
    }


def run_m1(n_examples: int = 20, seed: int = 0, layer_stride: int = 2,
           models: list = None):
    out_path = RESULTS_DIR / "m1_logit_lens.jsonl"
    summary_path = RESULTS_DIR / "m1_logit_lens_summary.txt"

    questions = get_questions(n=n_examples, seed=seed, n_hops=3)
    print(f"Using {len(questions)} N=3 questions")

    # Resume: skip already-done (model, condition, question) triples
    done_keys: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_keys.add((r["model"], r["condition"], r["question_id"]))
            except Exception:
                pass

    model_subset = {k: v for k, v in PROBE_MODELS.items()
                    if models is None or k in models}
    all_results = []

    for model_name, model_id in model_subset.items():
        model, tok = load_model(model_name, model_id)

        for q in questions:
            for cond, prompt_fn in [("A", build_prompt_a), ("B", build_prompt_b)]:
                if (model_name, cond, q["id"]) in done_keys:
                    continue
                prompt = prompt_fn(q, seed=seed, tok=tok)
                print(f"  {model_name} cond={cond} q={q['id']} ...", end=" ", flush=True)

                res = logit_lens_analysis(model, tok, prompt, q["answer"],
                                          layer_stride=layer_stride)
                sym = "✓" if res["pred_correct"] else "✗"
                print(f"{sym} first_rank0_layer={res['target_first_layer']} "
                      f"pred={res['predicted']!r}")

                record = {
                    "model": model_name,
                    "question_id": q["id"],
                    "answer": q["answer"],
                    "condition": cond,
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

    # Re-read full file for summary (includes prior runs)
    all_results_full = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    _write_summary(all_results_full, summary_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {summary_path}")


def _write_summary(results: list, path: Path):
    acc = defaultdict(list)
    first_layer = defaultdict(list)

    for r in results:
        key = (r["model"], r["condition"])
        acc[key].append(int(r["pred_correct"]))
        if r["target_first_layer"] is not None:
            first_layer[key].append(r["target_first_layer"])

    lines = ["=== M1 Logit Lens Summary ===\n"]
    lines.append(f"{'Model':<20} {'Cond':>6} {'Acc':>8} {'MeanFL':>10} "
                 f"{'%HasFL':>8} {'N':>4}")
    lines.append("-" * 60)
    for model_name in PROBE_MODELS:
        for cond in ["A", "B"]:
            key = (model_name, cond)
            n = len(acc[key])
            if not n:
                continue
            a = sum(acc[key]) / n
            fl_vals = first_layer[key]
            fl_mean = sum(fl_vals) / len(fl_vals) if fl_vals else float("nan")
            pct_fl = len(fl_vals) / n
            fl_str = f"{fl_mean:.1f}" if fl_vals else " None "
            lines.append(f"{model_name:<20} {cond:>6} {a:>8.3f} {fl_str:>10} "
                         f"{pct_fl:>8.2%} {n:>4}")

    # Layer-by-layer probability profile
    layer_probs = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["model"], r["condition"])
        for l in r["layers"]:
            layer_probs[key][l["layer"]].append(l["prob"])

    header_layers = sorted({l["layer"] for r in results for l in r["layers"]})
    lines.append("\n--- Layer-by-Layer Correct-Answer Probability (mean across examples) ---")
    lines.append(f"{'Model:Cond':<28} " + " ".join(f"L{l:02d}" for l in header_layers))
    for model_name in PROBE_MODELS:
        for cond in ["A", "B"]:
            key = (model_name, cond)
            vals = []
            for lyr in header_layers:
                ps = layer_probs[key][lyr]
                vals.append(f"{sum(ps)/len(ps):.3f}" if ps else "  -  ")
            lines.append(f"{model_name+':'+cond:<28} " + " ".join(vals))

    lines.extend([
        "\nInterpretation:",
        "  MeanFL   = mean layer where answer first reaches rank 0 (lower = earlier emergence)",
        "  %HasFL   = fraction of examples where answer ever reaches rank 0",
        "  Hypothesis: Bullets (A) → lower MeanFL, higher %HasFL.",
        "              Prose (B) → higher MeanFL or None in Llama (late/absent emergence).",
        "              Gemma: A vs B difference should be small (prose-robust).",
    ])

    text = "\n".join(lines)
    path.write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-stride", type=int, default=2)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: all)")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    run_m1(n_examples=args.n_examples, seed=args.seed, layer_stride=args.layer_stride,
           models=models)
