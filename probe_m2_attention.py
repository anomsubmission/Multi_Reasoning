"""Phase 2 — M2: Attention Diffusion Analysis.

For each transformer block, measures the fraction of last-token attention directed
at relevant (hop-chain) tokens vs. filler tokens. Compares bullets (A) vs prose (B).

Models (local, MPS):
  - llama3-8b  (shows prose-collapse)
  - gemma3-4b  (control)

Output:
  results/phase2/m2_attention.jsonl
  results/phase2/m2_attention_summary.txt

Usage:
    python probe_m2_attention.py [--n-examples 20] [--seed 0]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    PROBE_MODELS, FILLER_SENTS, _apply_chat_template,
    get_arch_info, get_questions, load_model, relevant_passages, score,
    build_prompt_a,
)

# Padding sentences used in coherent prose (condition B)
_PROSE_PADDING = [
    "The context provided above contains all necessary information to answer the question.",
    "Background information has been carefully selected from reliable sources.",
    "All relevant facts are present in the provided context.",
    "The passages reflect the current state of knowledge on this topic.",
]

_COHERENT_CONNECTORS = [
    "Furthermore, ", "In addition, ", "It is also known that ",
    "Moreover, ", "Additionally, ",
]


def _build_prompt_b_with_fillers(q: dict, seed: int = 0, tok=None):
    """Build prose prompt and return (prompt, rel_passages, prose_filler_sents)."""
    import random
    rng = random.Random(seed * 999)
    rel = relevant_passages(q)
    facts = [p.rstrip(".") for p in rel]
    result = facts[0] + "."
    for i, fact in enumerate(facts[1:]):
        conn = _COHERENT_CONNECTORS[i % len(_COHERENT_CONNECTORS)]
        result += " " + conn + fact[0].lower() + fact[1:] + "."
    filler_pool = _PROSE_PADDING
    added_fillers = []
    while len(result.split()) < 90:
        f = rng.choice(filler_pool)
        result += " " + f
        added_fillers.append(f)
    text = (
        "Read the following passage and answer the question with a single "
        "word or short phrase. Do not explain.\n\n"
        f"Passage:\n{result}\n\nQuestion: {q['question']}\n\nAnswer:"
    )
    prompt = _apply_chat_template(tok, text) if tok is not None else text
    return prompt, rel, list(set(added_fillers))

RESULTS_DIR = Path(__file__).parent / "results" / "phase2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_token_spans(tok, prompt: str, passages: list[str]) -> set:
    """Return token indices that correspond to any passage in `passages`."""
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    token_set = set()
    for passage in passages:
        passage = passage.strip()
        char_start = prompt.find(passage)
        if char_start == -1:
            continue
        char_end = char_start + len(passage)
        for tok_idx, (s, e) in enumerate(offsets):
            if s >= char_start and e <= char_end and e > 0:
                token_set.add(tok_idx)
    return token_set


def attention_analysis(model, tok, prompt: str, rel_passages: list[str],
                       filler_passages: list[str], answer: str) -> dict:
    """
    Per-layer attention from the last token to relevant vs filler spans.

    Returns:
        predicted, pred_correct,
        blocks: [{layer, relevant_attn, filler_attn, relevance_ratio,
                  n_relevant_tokens, n_filler_tokens}]
    """
    device = next(model.parameters()).device
    inp = tok(prompt, return_tensors="pt").to(device)
    seq_len = inp["input_ids"].shape[1]
    last_idx = seq_len - 1

    rel_set = find_token_spans(tok, prompt, rel_passages)
    fil_set = find_token_spans(tok, prompt, filler_passages)

    with torch.no_grad():
        out = model(**inp, output_attentions=True)

    pred_id = out.logits[0, -1].argmax().item()
    pred_str = tok.decode([pred_id]).strip()
    pred_correct = score(pred_str, answer)

    block_results = []
    for layer_idx, attn in enumerate(out.attentions):
        # attn: [1, n_heads, seq_len, seq_len]
        avg_attn = attn[0].mean(dim=0)     # [seq_len, seq_len]
        last_row = avg_attn[last_idx].cpu().float()

        rel_ids = [i for i in rel_set if i < seq_len]
        fil_ids = [i for i in fil_set if i < seq_len]

        rel_attn = last_row[rel_ids].sum().item() if rel_ids else 0.0
        fil_attn = last_row[fil_ids].sum().item() if fil_ids else 0.0
        total = rel_attn + fil_attn
        ratio = rel_attn / total if total > 1e-9 else 0.0

        block_results.append({
            "layer": layer_idx,
            "relevant_attn": round(rel_attn, 6),
            "filler_attn": round(fil_attn, 6),
            "relevance_ratio": round(ratio, 6),
            "n_relevant_tokens": len(rel_ids),
            "n_filler_tokens": len(fil_ids),
        })

    return {
        "predicted": pred_str,
        "pred_correct": pred_correct,
        "blocks": block_results,
    }


def run_m2(n_examples: int = 20, seed: int = 0, models: list = None):
    out_path = RESULTS_DIR / "m2_attention.jsonl"
    summary_path = RESULTS_DIR / "m2_attention_summary.txt"

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

    import random
    all_results = []

    model_subset = {k: v for k, v in PROBE_MODELS.items()
                    if models is None or k in models}

    for model_name, model_id in model_subset.items():
        model, tok = load_model(model_name, model_id)

        for q in questions:
            rel = relevant_passages(q)
            rng = random.Random(seed)
            fillers = rng.sample(FILLER_SENTS, k=4)

            if (model_name, "A", q["id"]) not in done_keys:
                prompt_a = build_prompt_a(q, seed=seed, tok=tok)
                print(f"  {model_name} cond=A q={q['id']} ...", end=" ", flush=True)
                res_a = attention_analysis(model, tok, prompt_a, rel, fillers, q["answer"])
                print(f"{'✓' if res_a['pred_correct'] else '✗'} pred={res_a['predicted']!r}")
                rec_a = {
                    "model": model_name, "question_id": q["id"], "answer": q["answer"],
                    "condition": "A", **{k: v for k, v in res_a.items()},
                }
                all_results.append(rec_a)
                done_keys.add((model_name, "A", q["id"]))
                with out_path.open("a") as f:
                    f.write(json.dumps(rec_a) + "\n")

            if (model_name, "B", q["id"]) not in done_keys:
                prompt_b, rel_b, prose_fillers = _build_prompt_b_with_fillers(q, seed=seed, tok=tok)
                print(f"  {model_name} cond=B q={q['id']} ...", end=" ", flush=True)
                res_b = attention_analysis(model, tok, prompt_b, rel_b, prose_fillers, q["answer"])
                print(f"{'✓' if res_b['pred_correct'] else '✗'} pred={res_b['predicted']!r}")
                rec_b = {
                    "model": model_name, "question_id": q["id"], "answer": q["answer"],
                    "condition": "B", **{k: v for k, v in res_b.items()},
                }
                all_results.append(rec_b)
                done_keys.add((model_name, "B", q["id"]))
                with out_path.open("a") as f:
                    f.write(json.dumps(rec_b) + "\n")

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
    layer_ratios = defaultdict(lambda: defaultdict(list))
    acc = defaultdict(list)

    for r in results:
        key = (r["model"], r["condition"])
        acc[key].append(int(r["pred_correct"]))
        for b in r["blocks"]:
            layer_ratios[key][b["layer"]].append(b["relevance_ratio"])

    all_layers = sorted({b["layer"] for r in results for b in r["blocks"]})
    lines = ["=== M2 Attention Diffusion Summary ===\n"]

    lines.append("Accuracy:")
    for model_name in PROBE_MODELS:
        for cond in ["A", "B"]:
            key = (model_name, cond)
            a = sum(acc[key]) / len(acc[key]) if acc[key] else float("nan")
            lines.append(f"  {model_name}:{cond}  acc={a:.3f}  (n={len(acc[key])})")

    lines.append("\nRelevance Ratio by Layer (relevant_attn / (rel+fil)):")
    lines.append("Higher = more attention on hop-relevant tokens = better signal routing")
    lines.append(f"\n{'Model:Cond':<28} " + " ".join(f"B{l:02d}" for l in all_layers))
    for model_name in PROBE_MODELS:
        for cond in ["A", "B"]:
            key = (model_name, cond)
            vals = []
            for lyr in all_layers:
                ps = layer_ratios[key][lyr]
                vals.append(f"{sum(ps)/len(ps):.3f}" if ps else "  -  ")
            lines.append(f"{model_name+':'+cond:<28} " + " ".join(vals))

    lines.extend([
        "\nHypothesis:",
        "  Llama bullets (A): relevance_ratio rises and stays elevated in layers 16-31",
        "  Llama prose (B):   relevance_ratio stays low (filler-like sentences dominate)",
        "  Gemma: ratio similar for A and B (robust to format)",
        "  The block range where A-B diverges most is the mechanistic locus",
        "  of the prose-collapse phenomenon.",
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
    run_m2(n_examples=args.n_examples, seed=args.seed, models=models)
