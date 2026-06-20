"""Cloud probe runner: M1 (logit lens) + M2 (attention routing) + M3 (block masking)
for a single large model loaded in 4-bit across multiple GPUs (device_map=auto).

Reuses the exact analysis logic from the local probe scripts so results are
directly comparable to the Llama-3.1-8B / Gemma-3-4B runs in the paper.

Usage:
    python cloud_probe.py --model-id unsloth/Meta-Llama-3.1-70B-Instruct \
        --model-name llama3.1-70b --n-examples 30 --layer-stride 4 \
        --out-dir ./out
"""
import argparse, json, random, sys, time, traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).parent))
from probe_utils import (
    build_prompt_a, build_prompt_b, get_questions, relevant_passages, score,
    get_arch_info, FILLER_SENTS,
)
from probe_m1_logit_lens import logit_lens_analysis
from probe_m2_attention import attention_analysis, _build_prompt_b_with_fillers
from probe_m3_block_mask import get_block_ranges, run_block_mask


def load_4bit(model_id: str):
    print(f"[load] {model_id} (device_map=auto, eager attn)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    common = dict(device_map="auto", attn_implementation="eager",
                  torch_dtype=torch.float16)
    if "4bit" in model_id.lower():
        # Pre-quantized checkpoint: quantization config travels with the model.
        print("[load] pre-quantized 4-bit checkpoint", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **common)
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, **common)
    model.eval()
    info = get_arch_info(model)
    print(f"[load] layers={info['n_layers']} hidden={info['hidden_size']}", flush=True)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--n-examples", type=int, default=30)
    ap.add_argument("--layer-stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="./out")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    f_m1 = (out / f"{args.model_name}_m1.jsonl").open("a")
    f_m2 = (out / f"{args.model_name}_m2.jsonl").open("a")
    f_m3 = (out / f"{args.model_name}_m3.jsonl").open("a")
    status = out / f"{args.model_name}_status.json"

    def log_status(**kw):
        status.write_text(json.dumps({"model": args.model_name, "ts": time.time(), **kw}))

    log_status(stage="loading")
    t0 = time.time()
    model, tok = load_4bit(args.model_id)
    arch = get_arch_info(model)
    layers, n_layers = arch["layers"], arch["n_layers"]
    blocks = get_block_ranges(n_layers)
    log_status(stage="loaded", n_layers=n_layers, load_s=round(time.time()-t0))

    questions = get_questions(n=args.n_examples, seed=args.seed, n_hops=3)
    print(f"[run] {len(questions)} questions, {n_layers} layers, blocks={[b[2] for b in blocks]}", flush=True)

    done = 0
    for q in questions:
        try:
            pa = build_prompt_a(q, seed=args.seed, tok=tok)
            pb = build_prompt_b(q, seed=args.seed, tok=tok)

            # M1 logit lens
            for cond, prompt in [("A", pa), ("B", pb)]:
                r = logit_lens_analysis(model, tok, prompt, q["answer"], args.layer_stride)
                f_m1.write(json.dumps({"model": args.model_name, "question_id": q["id"],
                                       "answer": q["answer"], "condition": cond, **r}) + "\n")
            f_m1.flush()

            # M2 attention routing
            rel = relevant_passages(q)
            rng = random.Random(args.seed); fillers = rng.sample(FILLER_SENTS, k=4)
            ra = attention_analysis(model, tok, pa, rel, fillers, q["answer"])
            f_m2.write(json.dumps({"model": args.model_name, "question_id": q["id"],
                                   "answer": q["answer"], "condition": "A", **ra}) + "\n")
            pb2, relb, prose_fillers = _build_prompt_b_with_fillers(q, seed=args.seed, tok=tok)
            rb = attention_analysis(model, tok, pb2, relb, prose_fillers, q["answer"])
            f_m2.write(json.dumps({"model": args.model_name, "question_id": q["id"],
                                   "answer": q["answer"], "condition": "B", **rb}) + "\n")
            f_m2.flush()

            # M3 block masking
            for cond, prompt in [("A", pa), ("B", pb)]:
                r = run_block_mask(model, tok, prompt, q["answer"], blocks, layers)
                f_m3.write(json.dumps({"model": args.model_name, "question_id": q["id"],
                                       "answer": q["answer"], "condition": cond,
                                       "n_layers": n_layers, **r}) + "\n")
            f_m3.flush()

            done += 1
            if done % 5 == 0:
                log_status(stage="running", done=done, total=len(questions))
                print(f"[run] {done}/{len(questions)}", flush=True)
        except Exception as e:
            print(f"[err] q={q['id']}: {e}\n{traceback.format_exc()}", flush=True)

    for f in (f_m1, f_m2, f_m3): f.close()
    log_status(stage="done", done=done, total=len(questions), elapsed_s=round(time.time()-t0))
    print(f"[done] {done}/{len(questions)} in {round(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    main()
