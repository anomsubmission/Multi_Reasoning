"""Phase 2 — M5: Causal test of the attention "re-read" hypothesis.

Motivation
----------
M2 (attention routing) showed a *correlational* signature: the format-robust
control model Gemma-3-4B has a low prose relevance-ratio through layers 8-12
(like Llama) but then *recovers sharply at ~layer 17* (0.18 -> 0.77), whereas
Llama-3.1-8B never recovers and collapses on prose. We hypothesised that
Gemma-3's interleaved full-attention layers stage a global "re-read" of the
context around layer 17 that Llama's more uniform attention cannot perform.

M5 turns that correlational claim into a *causal* one. We knock out the
self-attention sub-layer at a single layer L (zeroing its output while leaving
the residual stream and MLP intact) and sweep L across the network, measuring
the resulting accuracy drop separately for bullets (A) and prose (B).

Causal predictions
------------------
  * Gemma-3-4B: knocking out attention at its re-read layer(s) (~L17, and the
    other global-attention layers) should hurt PROSE (B) disproportionately
    more than BULLETS (A) -- i.e. a sharp positive (B_drop - A_drop) peak at the
    re-read layer. This would show the re-read is *causally necessary* for
    Gemma's format robustness.
  * Llama-3.1-8B: having no such re-read, its (B_drop - A_drop) profile should
    be diffuse with no single protective layer (specificity control).

This complements M3 (whole-block ablation, necessity of the B8-15 integration
circuit) by isolating the *attention* contribution layer-by-layer, and it is
the causal counterpart to the correlational M2 relevance-ratio recovery.

Method
------
Single-forward (greedy next-token) scoring identical to M3, with prefix-aware
matching from probe_utils.score. For each model we record baseline accuracy
(no intervention) and, for each swept layer L, the accuracy under attention
knockout at L. Robust forward-hook on layer.self_attn (handles both tuple and
tensor returns, mirroring M3's ablated_block).

Models (local, MPS): gemma3-4b (control, has re-read), llama3-8b (collapses),
optionally mistral-7b (intermediate) and qwen2.5-7b (robust, second family).

Output:
  results/phase2/m5_reread.jsonl
  results/phase2/m5_reread_summary.txt

Usage:
  python probe_m5_reread.py --models gemma3-4b,llama3-8b --n-examples 30
  python probe_m5_reread.py --models gemma3-4b --layers 13,15,17,19 --n-examples 40
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

# --- CPU throttle: keep the laptop usable during local serial runs ----------
# Cap intra-op threads BEFORE importing torch so the default thread pool is
# small. Overridable via CLIFF_THREADS (the throttled runner sets this).
_THREADS = os.environ.get("CLIFF_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import torch

try:
    torch.set_num_threads(int(_THREADS))
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    PROBE_MODELS, build_prompt_a, build_prompt_b,
    get_arch_info, get_questions, load_model, score,
)

RESULTS_DIR = Path(__file__).parent / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Extra models usable for the second-family mechanistic check (cached locally).
EXTRA_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}
ALL_MODELS = {**PROBE_MODELS, **EXTRA_MODELS}

# Gemma-3 interleaved global ("full") attention layers follow a 1-global-every-6
# sliding-window pattern; for a 34-layer model these fall at 5,11,17,23,29.
# Annotated in the summary so the re-read layers are easy to spot.
GEMMA_GLOBAL_LAYERS = {5, 11, 17, 23, 29}


@contextmanager
def attn_knockout(layer_modules):
    """Zero the self-attention sub-layer output for one or more decoder layers.

    Accepts a single layer module or a list of them. Leaves the residual add and
    MLP intact -- only the information injected by *these layers'* attention is
    removed. Robust to tuple/tensor returns across transformers versions.
    """
    if not isinstance(layer_modules, (list, tuple)):
        layer_modules = [layer_modules]

    def hook(module, args, output):
        if isinstance(output, tuple):
            z = torch.zeros_like(output[0])
            return (z,) + tuple(output[1:])
        return torch.zeros_like(output)

    handles = []
    for lm in layer_modules:
        attn = getattr(lm, "self_attn", None) or getattr(lm, "attn", None)
        if attn is None:
            raise ValueError("decoder layer has no self_attn/attn submodule")
        handles.append(attn.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def detect_global_layers(model, n_layers: int):
    """Return (global_idx, local_idx) using config.layer_types when available.

    Gemma-3 interleaves sliding-window and full ("global") attention; the full
    layers are the hypothesised "re-read" machinery. Falls back to the known
    1-global-every-6 pattern when layer_types is absent.
    """
    cfg = getattr(model, "config", None)
    layer_types = getattr(cfg, "layer_types", None) if cfg is not None else None
    # Gemma3 may nest text config under .text_config
    if layer_types is None and cfg is not None:
        tcfg = getattr(cfg, "text_config", None)
        layer_types = getattr(tcfg, "layer_types", None) if tcfg is not None else None
    if layer_types and len(layer_types) == n_layers:
        glob = [i for i, t in enumerate(layer_types) if "full" in str(t).lower()]
        loc = [i for i, t in enumerate(layer_types) if i not in set(glob)]
        return glob, loc
    # Fallback: assume the 1-every-6 global pattern (5,11,17,...)
    glob = [i for i in range(n_layers) if (i + 1) % 6 == 0]
    loc = [i for i in range(n_layers) if i not in set(glob)]
    return glob, loc


def matched_local_control(global_idx, local_idx):
    """Pick a size-matched, evenly spaced subset of local layers to control for
    'number of attention layers removed'."""
    k = len(global_idx)
    if k == 0 or not local_idx:
        return []
    step = max(1, len(local_idx) // k)
    picked = local_idx[::step][:k]
    return picked


def predict_correct(model, tok, prompt: str, answer: str) -> bool:
    device = next(model.parameters()).device
    inp = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp)
    pred_id = out.logits[0, -1].argmax().item()
    pred = tok.decode([pred_id]).strip()
    return score(pred, answer)


def run_m5(models, n_examples=30, seed=0, mode="both", layers_arg=None, stride=1):
    out_path = RESULTS_DIR / "m5_reread.jsonl"
    summary_path = RESULTS_DIR / "m5_reread_summary.txt"

    questions = get_questions(n=n_examples, seed=seed, n_hops=3)
    print(f"Using {len(questions)} N=3 questions")

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["model"], r["condition"], r["layer"], r["question_id"]))
            except Exception:
                pass
        print(f"Resuming: {len(done)} records already written")

    model_subset = {k: ALL_MODELS[k] for k in models if k in ALL_MODELS}
    if not model_subset:
        print(f"No valid models in {models}. Available: {list(ALL_MODELS)}")
        return

    for model_name, model_id in model_subset.items():
        model, tok = load_model(model_name, model_id)
        arch = get_arch_info(model)
        arch_layers = arch["layers"]
        n_layers = arch["n_layers"]

        # Build the list of interventions to run for this model.
        # Each is (label, layer_indices_or_None). label "-1" == baseline.
        interventions = [("-1", None)]

        if mode in ("single", "both"):
            if layers_arg:
                sweep = [int(x) for x in layers_arg]
            else:
                sweep = list(range(0, n_layers, stride))
            for L in sweep:
                interventions.append((str(L), [L]))

        glob, loc = detect_global_layers(model, n_layers)
        if mode in ("sets", "both"):
            ctrl = matched_local_control(glob, loc)
            if glob:
                interventions.append(("set:global", glob))
            if ctrl:
                interventions.append(("set:local-ctl", ctrl))
        print(f"  {model_name}: {n_layers} layers; global={glob} "
              f"local-ctl={matched_local_control(glob, loc)}")
        print(f"  interventions: {[lbl for lbl, _ in interventions]}")

        for q in questions:
            prompts = {
                "A": build_prompt_a(q, seed=seed, tok=tok),
                "B": build_prompt_b(q, seed=seed, tok=tok),
            }
            for label, idxs in interventions:
                for cond in ["A", "B"]:
                    if (model_name, cond, label, q["id"]) in done:
                        continue
                    if idxs is None:
                        correct = predict_correct(model, tok, prompts[cond], q["answer"])
                    else:
                        mods = [arch_layers[i] for i in idxs]
                        with attn_knockout(mods):
                            correct = predict_correct(model, tok, prompts[cond], q["answer"])
                    rec = {
                        "model": model_name, "question_id": q["id"],
                        "answer": q["answer"], "condition": cond,
                        "layer": label, "set_layers": idxs, "n_layers": n_layers,
                        "global_layers": glob,
                        "correct": int(correct),
                    }
                    with out_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    done.add((model_name, cond, label, q["id"]))
            print(f"  {model_name} q={q['id']} done")

        del model
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        import gc; gc.collect()

    write_summary(out_path, summary_path)
    print(f"\nResults: {out_path}\nSummary: {summary_path}")


def _is_int_label(lbl) -> bool:
    try:
        int(lbl)
        return True
    except (ValueError, TypeError):
        return False


def write_summary(out_path: Path, summary_path: Path):
    recs = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    # acc[model][cond][label] -> list of correctness
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nlayers, glob_layers = {}, {}
    for r in recs:
        acc[r["model"]][r["condition"]][str(r["layer"])].append(r["correct"])
        nlayers[r["model"]] = r["n_layers"]
        glob_layers[r["model"]] = set(r.get("global_layers") or [])

    lines = ["=== M5 Causal Re-read Knockout Summary ===\n"]
    lines.append("Knock out self-attention output; measure accuracy drop vs baseline,")
    lines.append("separately for bullets (A) and prose (B).")
    lines.append("diff = (B_drop - A_drop): POSITIVE = intervention hurts prose more.")
    lines.append("Decisive test (set mode): does removing the GLOBAL ('re-read') layers")
    lines.append("collapse prose (large +diff) while a size-matched LOCAL set does not?\n")

    for model_name in acc:
        nl = nlayers[model_name]
        gl = glob_layers.get(model_name, set())

        def mean(cond, lbl):
            v = acc[model_name][cond].get(lbl, [])
            return sum(v) / len(v) if v else float("nan")
        baseA, baseB = mean("A", "-1"), mean("B", "-1")
        lines.append(f"\n--- {model_name} ({nl} layers; global={sorted(gl)}) ---")
        lines.append(f"baseline: A={baseA:.3f}  B={baseB:.3f}  "
                     f"(n={len(acc[model_name]['A'].get('-1', []))})")

        # Set-mode rows first (the decisive contrast)
        set_labels = [l for l in acc[model_name]["A"] if l.startswith("set:")]
        if set_labels:
            lines.append(f"{'intervention':>16} {'A_acc':>7} {'B_acc':>7} "
                         f"{'A_drop':>7} {'B_drop':>7} {'diff':>7}")
            for lbl in sorted(set_labels):
                aA, aB = mean("A", lbl), mean("B", lbl)
                dA, dB = baseA - aA, baseB - aB
                lines.append(f"{lbl:>16} {aA:>7.3f} {aB:>7.3f} {dA:>+7.3f} "
                             f"{dB:>+7.3f} {dB - dA:>+7.3f}")

        # Single-layer sweep profile
        int_labels = sorted((l for l in acc[model_name]["A"] if _is_int_label(l)),
                            key=lambda x: int(x))
        if int_labels:
            lines.append(f"\n{'Layer':>6} {'A_acc':>7} {'B_acc':>7} {'A_drop':>7} "
                         f"{'B_drop':>7} {'diff':>7}  flag")
            best_diff, best_L = -9.9, None
            for lbl in int_labels:
                L = int(lbl)
                aA, aB = mean("A", lbl), mean("B", lbl)
                dA, dB = baseA - aA, baseB - aB
                diff = dB - dA
                flag = "  <-global" if L in gl else ""
                lines.append(f"{L:>6} {aA:>7.3f} {aB:>7.3f} {dA:>+7.3f} "
                             f"{dB:>+7.3f} {diff:>+7.3f}{flag}")
                if diff > best_diff:
                    best_diff, best_L = diff, L
            lines.append(f"  => largest prose-protective layer: L{best_L} "
                         f"(diff={best_diff:+.3f})")

    lines.extend([
        "\nInterpretation key:",
        "  Gemma: set:global should show a large +diff (prose collapses when the",
        "    re-read layers are removed) while set:local-ctl shows little; in the",
        "    single sweep a +diff peak should sit on/near a global layer (~L17).",
        "  Llama: no global re-read to remove -> set:global and set:local-ctl",
        "    behave similarly and the single-sweep diff profile is diffuse.",
        "  This is the causal counterpart to the M2 relevance-ratio recovery.",
    ])
    text = "\n".join(lines)
    summary_path.write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma3-4b,llama3-8b",
                    help="comma-separated; available: " + ",".join(ALL_MODELS))
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="both", choices=["single", "sets", "both"],
                    help="single=per-layer sweep; sets=global-vs-local knockout; both")
    ap.add_argument("--layers", default=None,
                    help="comma-separated explicit layers for single mode (default: full sweep)")
    ap.add_argument("--stride", type=int, default=1, help="single-mode layer sweep stride")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    layers = [x.strip() for x in args.layers.split(",")] if args.layers else None
    run_m5(models, n_examples=args.n_examples, seed=args.seed, mode=args.mode,
           layers_arg=layers, stride=args.stride)
