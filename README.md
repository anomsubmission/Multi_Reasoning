# Two Orthogonal Failure Modes of LLM Multi-Evidence Integration

Code for reproducing the experiments in the paper. Works with any LLM provider
or locally loaded model — no cloud account required.

## Setup

```bash
pip install -r requirements.txt
```

## Choose a backend

Set two environment variables before running any experiment script:

| Backend | Variables to set |
|---------|-----------------|
| OpenAI API | `LLM_BACKEND=openai` + `OPENAI_API_KEY=sk-...` |
| Any OpenAI-compatible endpoint (Together, Fireworks, vLLM, Ollama) | `LLM_BACKEND=openai` + `OPENAI_API_KEY=...` + `OPENAI_BASE_URL=https://...` |
| Anthropic API | `LLM_BACKEND=anthropic` + `ANTHROPIC_API_KEY=sk-ant-...` |
| Local HuggingFace model | `LLM_BACKEND=hf` (no key needed) |

Example with Together AI:
```bash
export LLM_BACKEND=openai
export OPENAI_API_KEY=<your-together-key>
export OPENAI_BASE_URL=https://api.together.xyz/v1
```

## Run all behavioral experiments

```bash
bash run_experiments.sh
```

Or run individual experiments:

```bash
python run_p1.py                        # N-scaling degradation (all models)
python run_p2_p3_p4.py --experiment p3  # format sensitivity only
python run_p5.py                        # cross-generation (Llama-3.3, Llama-4)
python run_p6_doseresponse.py           # format gap spectrum (remaining families)
python run_p7_predictor.py              # held-out format predictor validation
python run_d1.py                        # naturalistic validation (HotpotQA, MuSiQue)
```

All scripts are **resume-safe**: re-running skips completed calls and appends
to the existing results file.

## Run mechanistic probes (local, open-weight models only)

The mechanistic probes (M1–M5) require direct access to model weights and run
locally. They are independent of the LLM_BACKEND setting.

```bash
python probe_m1_logit_lens.py  --models llama3-8b,gemma3-4b
python probe_m2_attention.py   --models llama3-8b,gemma3-4b
python probe_m3_block_mask.py  --models llama3-8b,gemma3-4b
python probe_m4_lora.py        --model llama3-8b
python probe_m5_reread.py      --models gemma3-4b,llama3-8b
python run_repair.py           # LoRA K-sweep
```

## Analyze results

```bash
python analyze_spectrum.py  # format-gap spectrum + orthogonality
python analyze_p7.py        # predictor validation correlations
```

## Results

Results are written to `results/` as `.jsonl` files (one JSON record per line).
The `results/phase2/` subdirectory holds mechanistic probe outputs.

## Specifying models

Pass any model name your backend accepts with `--models`:

```bash
# Use paper short names (resolved via MODELS dict in llm_client.py)
python run_p1.py --models llama3-8b gemma3-12b

# Or pass full model IDs directly
python run_p1.py --models meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-32B-Instruct
```

## Paper

> *Two Orthogonal Failure Modes of LLM Multi-Evidence Integration*
