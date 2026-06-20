"""M4: Targeted LoRA recovery — causal closure for B8-15.

M3 established that block-masking layers 8-15 causes catastrophic accuracy drops
in BOTH bullet (A) and prose (B) conditions for Llama-3.1-8B.

M4 tests causal sufficiency: if LoRA adapters trained only on B8-15 recover prose
accuracy while leaving other layers frozen, then B8-15 is causally sufficient for
format-robust multi-evidence integration, not merely necessary.

Protocol:
  1. Generate training examples: N=3 multi-hop questions, prose condition B
  2. Train LoRA on ONLY layers 8-15 of Llama-3.1-8B
  3. Evaluate on held-out set: bullets (A) and prose (B) at N=1,2,3,4
  4. Compare to baseline (no LoRA) and full-layer LoRA (control)

LoRA configurations tested:
  - "targeted":  only B8-15 (layers 8,9,10,11,12,13,14,15)
  - "full":      all layers (B0-31) — control to verify LoRA itself can recover
  - "early":     B0-7 — ablation: upstream of critical block should NOT recover
  - "late":      B16-31 — ablation: downstream of critical block should NOT recover
  - "baseline":  no LoRA, original model weights

Results: results/phase2/m4_lora_recovery.jsonl
  {config, condition, n_hops, question_id, seed, answer, prediction, correct}

Usage:
    python probe_m4_lora.py [--n-train N] [--n-eval N] [--n-epochs N] [--seed N] [--device DEVICE]
    python probe_m4_lora.py --configs baseline,targeted  # subset of configs

Requirements: pip install peft
"""

import argparse
import gc
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    PROBE_MODELS, load_model, get_arch_info, build_prompt_a, build_prompt_b,
    get_questions, score as probe_score,
)

RESULTS_DIR = Path(__file__).parent / "results" / "phase2"

# Default target: Llama (collapser). Can override via --model-name.
TARGET_MODEL = "llama3-8b"
TARGET_MODEL_ID = PROBE_MODELS[TARGET_MODEL]

def get_lora_configs(n_layers: int) -> dict:
    """Layer ranges scaled to model depth. Mirrors the B8-15 proportional locus."""
    # For 32-layer model: targeted=8-15 (25-50% depth)
    # For 80-layer model: 25-50% → layers 20-39  (20 layers, same proportion)
    q = n_layers // 4
    return {
        "targeted": (q, q * 2),        # second quartile — integration circuit locus
        "full":     (0, n_layers),
        "early":    (0, q),
        "late":     (q * 2, n_layers),
    }

# Legacy constant kept for backwards compat (32-layer default)
LORA_CONFIGS = get_lora_configs(32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProbeDataset(Dataset):
    def __init__(self, questions, tok, device, seed=0, condition="B"):
        self.items = []
        for q in questions:
            prompt = (build_prompt_b(q, seed=seed, tok=tok)
                      if condition == "B"
                      else build_prompt_a(q, seed=seed, tok=tok))
            # Append answer immediately after prompt for supervised training
            full = prompt + " " + q["answer"]
            enc = tok(full, return_tensors="pt", truncation=True,
                      max_length=512, padding="max_length")
            # Build labels: -100 for prompt tokens and padding tokens
            prompt_len = len(tok(prompt, return_tensors="pt").input_ids[0])
            labels = enc.input_ids[0].clone()
            labels[:prompt_len] = -100
            # Mask padding positions
            labels[enc.attention_mask[0] == 0] = -100
            self.items.append({
                "input_ids": enc.input_ids[0],
                "attention_mask": enc.attention_mask[0],
                "labels": labels,
                "answer": q["answer"],
                "question_id": q["id"],
            })
        self.device = device

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "input_ids": it["input_ids"].to(self.device),
            "attention_mask": it["attention_mask"].to(self.device),
            "labels": it["labels"].to(self.device),
        }


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def _target_modules_for_layers(start: int, end: int) -> list[str]:
    """Return PEFT target_modules patterns for layers [start, end)."""
    modules = []
    for i in range(start, end):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            modules.append(f"model.layers.{i}.self_attn.{proj}")
    return modules


def apply_lora(model, layer_start: int, layer_end: int, rank: int = 16, alpha: int = 32):
    """Attach LoRA adapters to attention projections in layers [layer_start, layer_end)."""
    if not _HAS_PEFT:
        raise ImportError("peft is required. Run: pip install peft")
    target = _target_modules_for_layers(layer_start, layer_end)
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_cfg)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_lora(model, tok, train_questions, n_epochs: int, device: str, seed: int,
               lr: float = 3e-4, batch_size: int = 1) -> None:
    """Train LoRA adapters on prose (condition B) training examples."""
    dataset = ProbeDataset(train_questions, tok, device, seed=seed, condition="B")
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in loader:
            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += out.loss.item()
        avg = total_loss / len(loader)
        print(f"    epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")
    model.eval()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tok, eval_questions: list[dict], conditions: list[str],
             n_hops_list: list[int], config_name: str, seed: int,
             device: str, done_keys: set, out_path: Path) -> list[dict]:
    results = []
    model.eval()
    for q in eval_questions:
        if q["n"] not in n_hops_list:
            continue
        for cond in conditions:
            key = f"{config_name}|{cond}|{q['n']}|{q['id']}|{seed}"
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
                "config": config_name,
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
# Summary
# ---------------------------------------------------------------------------

def _print_m4_summary(path: Path) -> None:
    from collections import defaultdict
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # acc[config][condition][n_hops] → [correct, ...]
    for r in records:
        acc[r["config"]][r["condition"]][r["n_hops"]].append(r["correct"])

    configs = ["baseline", "targeted", "early", "late", "full"]
    print("\n=== M4 Summary — LoRA Recovery (Llama-3.1-8B) ===")
    print(f"{'Config':<12} {'Cond':>5}  {'N=1':>6} {'N=2':>6} {'N=3':>6} {'N=4':>6}  Interpretation")
    print("-" * 75)
    for cfg in configs:
        if cfg not in acc: continue
        for cond in ["A", "B"]:
            if cond not in acc[cfg]: continue
            vals = [acc[cfg][cond].get(n, []) for n in [1, 2, 3, 4]]
            cells = [(sum(v)/len(v) if v else float("nan")) for v in vals]
            row_str = "  ".join(f"{v:.3f}" if not math.isnan(v) else "  -  " for v in cells)
            n3 = cells[2]
            # Interpretation only on prose (B) at N=3
            if cond == "B":
                base_b = next(
                    (sum(acc["baseline"]["B"].get(3, []))/len(acc["baseline"]["B"].get(3, [1]))
                     if acc["baseline"]["B"].get(3) else float("nan")
                     for _ in [None]), float("nan")
                )
                if not math.isnan(n3) and not math.isnan(base_b):
                    delta = n3 - base_b
                    interp = (f"RECOVERED +{delta:.2f}" if delta > 0.10
                              else f"partial  +{delta:.2f}" if delta > 0.03
                              else f"no change {delta:+.2f}")
                else:
                    interp = ""
            else:
                interp = ""
            print(f"{cfg:<12} {cond:>5}  {row_str}  {interp}")

    print("\nPredictions:")
    print("  targeted RECOVERS prose (B) → B8-15 is causally sufficient for format-robust integration")
    print("  early/late DO NOT recover prose → only B8-15 circuit matters")
    print("  full RECOVERS prose (control) → LoRA training works at all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train",    type=int,  default=100,
                        help="Number of training examples (prose condition B)")
    parser.add_argument("--n-eval",     type=int,  default=50,
                        help="Number of held-out evaluation questions")
    parser.add_argument("--n-epochs",   type=int,  default=3,
                        help="Training epochs per LoRA config")
    parser.add_argument("--seed",       type=int,  default=0)
    parser.add_argument("--device",     type=str,  default="mps")
    parser.add_argument("--configs",    type=str,  default="baseline,targeted,early,late,full",
                        help="Comma-separated subset of configs to run")
    parser.add_argument("--rank",       type=int,  default=16, help="LoRA rank")
    parser.add_argument("--model-name", type=str,  default=TARGET_MODEL,
                        help=f"Model key from PROBE_MODELS (default: {TARGET_MODEL})")
    args = parser.parse_args()

    if not _HAS_PEFT:
        print("ERROR: peft not installed. Run: pip install peft")
        return

    model_name = args.model_name
    if model_name not in PROBE_MODELS:
        print(f"ERROR: unknown model '{model_name}'. Choose from: {list(PROBE_MODELS)}")
        return
    model_id = PROBE_MODELS[model_name]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Per-model output file so runs don't clobber each other
    out_path = RESULTS_DIR / f"m4_lora_{model_name}.jsonl"

    configs_to_run = [c.strip() for c in args.configs.split(",")]

    done_keys: set = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done_keys.add(f"{r['config']}|{r['condition']}|{r['n_hops']}|{r['question_id']}|{r['seed']}")
            except Exception:
                pass

    # Generate train/eval split from different seeds to avoid overlap
    rng = random.Random(args.seed)
    all_questions = get_questions(n=args.n_train + args.n_eval, seed=args.seed, n_hops=3)
    # Also include N=1,2,4 questions for the eval-only multi-hop profile
    all_questions_multi = []
    for n in [1, 2, 3, 4]:
        all_questions_multi += get_questions(n=args.n_eval // 2, seed=args.seed + 1, n_hops=n)

    rng.shuffle(all_questions)
    train_qs = all_questions[:args.n_train]
    eval_qs  = all_questions[args.n_train:args.n_train + args.n_eval]
    # De-duplicate eval set
    eval_ids = {q["id"] for q in eval_qs}
    eval_qs_multi = [q for q in all_questions_multi if q["id"] not in eval_ids][:args.n_eval]

    print(f"Train: {len(train_qs)} questions (prose B, N=3)")
    print(f"Eval:  {len(eval_qs)} N=3 questions + {len(eval_qs_multi)} multi-N questions")

    N_HOPS_EVAL = [1, 2, 3, 4]

    # Load tokenizer once
    print(f"\nLoading tokenizer for {model_id} ({model_name})...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for cfg_name in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Config: {cfg_name}  model: {model_name}")
        print(f"{'='*60}")

        # Fresh model load for each config (LoRA modifies model in place)
        print(f"Loading {model_name} ({model_id})...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()

        if cfg_name != "baseline":
            # Determine layer count from loaded model
            from probe_utils import get_arch_info
            _arch = get_arch_info(model)
            _lora_cfgs = get_lora_configs(_arch["n_layers"])
            layer_start, layer_end = _lora_cfgs[cfg_name]
            print(f"  Applying LoRA to layers {layer_start}-{layer_end-1}  "
                  f"(rank={args.rank}, n_layers={_arch['n_layers']})")
            model = apply_lora(model, layer_start, layer_end, rank=args.rank)
            model.print_trainable_parameters()

            print(f"  Training on {len(train_qs)} prose examples for {args.n_epochs} epochs...")
            train_lora(model, tok, train_qs, n_epochs=args.n_epochs,
                       device=args.device, seed=args.seed)

        print(f"  Evaluating...")
        all_eval = eval_qs + eval_qs_multi
        recs = evaluate(
            model, tok, all_eval,
            conditions=["A", "B"],
            n_hops_list=N_HOPS_EVAL,
            config_name=cfg_name,
            seed=args.seed,
            device=args.device,
            done_keys=done_keys,
            out_path=out_path,
        )
        # Update done_keys
        for r in recs:
            done_keys.add(f"{r['config']}|{r['condition']}|{r['n_hops']}|{r['question_id']}|{r['seed']}")

        n_correct_A = sum(r["correct"] for r in recs if r["condition"] == "A" and r["n_hops"] == 3)
        n_correct_B = sum(r["correct"] for r in recs if r["condition"] == "B" and r["n_hops"] == 3)
        n_A = sum(1 for r in recs if r["condition"] == "A" and r["n_hops"] == 3)
        n_B = sum(1 for r in recs if r["condition"] == "B" and r["n_hops"] == 3)

        print(f"  N=3 eval: A={n_correct_A}/{n_A} ({n_correct_A/max(n_A,1):.3f}), "
              f"B={n_correct_B}/{n_B} ({n_correct_B/max(n_B,1):.3f})")

        # Free memory before next config
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    _print_m4_summary(out_path)
    print(f"\nM4 complete. Results: {out_path}")


if __name__ == "__main__":
    main()
