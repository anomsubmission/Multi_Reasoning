"""Repair experiment: data-efficiency of circuit-targeted LoRA on Llama-3.1-8B.

Hypothesis: A small number of prose-format training examples (K) fine-tuned onto
the B8-15 integration circuit is sufficient to repair prose-collapse, making the
fix practical rather than merely theoretically possible.

Protocol:
  For K in {5, 10, 25, 50, 100}:
    For lora_config in {"targeted" (B8-15), "full" (all layers)}:
      1. Load fresh Llama-3.1-8B
      2. Apply LoRA to specified layers (rank=16)
      3. Train on K prose (condition B) examples, 5 epochs
      4. Evaluate on 100 held-out questions: bullets (A) and prose (B) at N=1,2,3

Output: results/repair/repair_results.jsonl
  {k_train, lora_config, condition, n_hops, question_id, correct}

Summary: results/repair/repair_summary.txt

Usage (local GPU/MPS):
    python run_repair.py [--device mps|cuda|cpu] [--k-values "5,10,25,50,100"]

Usage (cloud — passed via --device cuda --k-values ...):
    python run_repair.py --device cuda --n-eval 100 --n-epochs 5
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    PROBE_MODELS, load_model, build_prompt_a, build_prompt_b,
    get_questions, score as probe_score,
)
from probe_m4_lora import (
    ProbeDataset, apply_lora, train_lora,
    _target_modules_for_layers,
)

RESULTS_DIR = Path(__file__).parent / "results" / "repair"

TARGET_MODEL = "llama3-8b"
TARGET_MODEL_ID = PROBE_MODELS[TARGET_MODEL]

# Layer configs: name → (start, end_exclusive)
REPAIR_CONFIGS = {
    "targeted": (8, 16),   # B8-15 — the mechanistically identified locus
    "full":     (0, 32),   # all layers — ceiling control
}

K_VALUES_DEFAULT = [5, 10, 25, 50, 100]


# ---------------------------------------------------------------------------
# Evaluation (identical to M4 evaluate but returns list)
# ---------------------------------------------------------------------------

def evaluate_config(model, tok, eval_questions, conditions, n_hops_list,
                    k_train, lora_config, seed, device, done_keys, out_path):
    results = []
    model.eval()
    for q in eval_questions:
        if q["n"] not in n_hops_list:
            continue
        for cond in conditions:
            key = f"{k_train}|{lora_config}|{cond}|{q['n']}|{q['id']}"
            if key in done_keys:
                continue
            prompt = (build_prompt_b(q, seed=seed, tok=tok)
                      if cond == "B"
                      else build_prompt_a(q, seed=seed, tok=tok))
            inp = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **inp,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            pred_ids = gen[0][inp["input_ids"].shape[1]:]
            pred = tok.decode(pred_ids, skip_special_tokens=True).strip()
            correct = probe_score(pred, q["answer"])
            rec = {
                "k_train": k_train,
                "lora_config": lora_config,
                "condition": cond,
                "n_hops": q["n"],
                "question_id": q["id"],
                "seed": seed,
                "answer": q["answer"],
                "prediction": pred,
                "correct": int(correct),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(path: Path):
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        acc[(r["k_train"], r["lora_config"])][r["condition"]][r["n_hops"]].append(r["correct"])

    print("\n=== Repair Experiment Summary ===")
    print(f"{'K':>5}  {'Config':<10}  {'Cond':>4}  {'N=1':>6}  {'N=2':>6}  {'N=3':>6}")
    print("-" * 55)
    for (k, cfg) in sorted(acc.keys()):
        for cond in ["A", "B"]:
            cells = []
            for n in [1, 2, 3]:
                v = acc[(k, cfg)][cond].get(n, [])
                cells.append(f"{sum(v)/len(v):.3f}" if v else "  -  ")
            print(f"{k:>5}  {cfg:<10}  {cond:>4}  {'  '.join(cells)}")
        print()

    # Recovery column: prose (B) accuracy at N=3 by K
    print("\nProse (B) N=3 accuracy by training set size:")
    print(f"{'K':>5}  {'targeted (B8-15)':>17}  {'full':>6}")
    print("-" * 35)
    for k in sorted(set(k for k, _ in acc.keys())):
        def a(cfg):
            v = acc[(k, cfg)]["B"].get(3, [])
            return f"{sum(v)/len(v):.3f}" if v else "  -  "
        print(f"{k:>5}  {a('targeted'):>17}  {a('full'):>6}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",   default="mps")
    ap.add_argument("--k-values", default=",".join(map(str, K_VALUES_DEFAULT)))
    ap.add_argument("--n-eval",   type=int, default=100)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--rank",     type=int, default=16)
    ap.add_argument("--configs",  default="targeted,full")
    args = ap.parse_args()

    if not _HAS_PEFT:
        print("ERROR: pip install peft")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "repair_results.jsonl"

    k_values = [int(x) for x in args.k_values.split(",")]
    configs_to_run = [c.strip() for c in args.configs.split(",")]

    done_keys = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_keys.add(f"{r['k_train']}|{r['lora_config']}|{r['condition']}|{r['n_hops']}|{r['question_id']}")
            except Exception:
                pass
        print(f"Resuming: {len(done_keys)} records already done")

    rng = random.Random(args.seed)

    # Generate a large pool; eval set is fixed, train sets are drawn from remainder
    max_k = max(k_values)
    total_needed = max_k + args.n_eval
    all_q_n3 = get_questions(n=total_needed + 50, seed=args.seed, n_hops=3)
    rng.shuffle(all_q_n3)
    eval_q_n3 = all_q_n3[:args.n_eval]
    pool_q    = all_q_n3[args.n_eval:]   # training draws come from here

    # Also evaluate N=1 and N=2 for generalization profile
    eval_q_n1 = get_questions(n=50, seed=args.seed + 1, n_hops=1)
    eval_q_n2 = get_questions(n=50, seed=args.seed + 2, n_hops=2)
    eval_all  = eval_q_n3 + eval_q_n1 + eval_q_n2

    print(f"Eval pool: {len(eval_all)} questions (N=1:{len(eval_q_n1)}, N=2:{len(eval_q_n2)}, N=3:{len(eval_q_n3)})")
    print(f"K values: {k_values}")
    print(f"Configs: {configs_to_run}")

    tok = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Baseline (K=0, no LoRA) — run once
    if "baseline" not in done_keys:
        print("\n=== baseline (K=0, no LoRA) ===")
        model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_ID, torch_dtype=torch.float16,
            device_map=args.device, attn_implementation="eager")
        model.eval()
        recs = evaluate_config(model, tok, eval_all, ["A", "B"], [1, 2, 3],
                               k_train=0, lora_config="baseline",
                               seed=args.seed, device=args.device,
                               done_keys=done_keys, out_path=out_path)
        for r in recs:
            done_keys.add(f"{r['k_train']}|{r['lora_config']}|{r['condition']}|{r['n_hops']}|{r['question_id']}")
        del model
        _free(args.device)

    # K-sweep
    for k in k_values:
        train_q = pool_q[:k]
        for cfg in configs_to_run:
            already = sum(
                1 for r in [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
                if r.get("k_train") == k and r.get("lora_config") == cfg
            )
            if already >= args.n_eval * 2:  # both conditions evaluated
                print(f"K={k} {cfg}: already complete ({already} records), skipping")
                continue

            print(f"\n=== K={k}  config={cfg} ===")
            model = AutoModelForCausalLM.from_pretrained(
                TARGET_MODEL_ID, torch_dtype=torch.float16,
                device_map=args.device, attn_implementation="eager")
            model.eval()

            layer_start, layer_end = REPAIR_CONFIGS[cfg]
            model = apply_lora(model, layer_start, layer_end, rank=args.rank)
            model.print_trainable_parameters()

            print(f"  Training on K={k} prose examples for {args.n_epochs} epochs...")
            train_lora(model, tok, train_q, n_epochs=args.n_epochs,
                       device=args.device, seed=args.seed)

            recs = evaluate_config(model, tok, eval_all, ["A", "B"], [1, 2, 3],
                                   k_train=k, lora_config=cfg,
                                   seed=args.seed, device=args.device,
                                   done_keys=done_keys, out_path=out_path)
            for r in recs:
                done_keys.add(f"{r['k_train']}|{r['lora_config']}|{r['condition']}|{r['n_hops']}|{r['question_id']}")
            del model
            _free(args.device)

    print_summary(out_path)
    print(f"\nRepair experiment complete. Results: {out_path}")


def _free(device):
    import gc; gc.collect()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
