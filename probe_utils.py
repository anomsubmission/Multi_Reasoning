"""Shared utilities for Phase 2 mechanistic probe scripts.

Provides architecture-agnostic access to:
  - model layer count
  - lm_head and final layer norm
  - transformer layer list (for block masking)
  - prompt building for bullets (A) and prose (B) conditions
  - scoring
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from generate import build_dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

PROBE_MODELS = {
    "llama3-8b":       "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "gemma3-4b":       "google/gemma-3-4b-it",
    "mistral-7b":      "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma3-12b":      "google/gemma-3-12b-it",
    "llama3-70b-4bit": "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
}

FILLER_SENTS = [
    "A gentle breeze drifted through the valley.",
    "The sky was overcast that morning.",
    "Somewhere in the distance a bell was ringing.",
    "The old library had not been opened in years.",
    "Rain was expected by the afternoon.",
    "The market closed early on holidays.",
    "Several birds perched on the fence post.",
    "The road curved sharply near the river.",
    "Thick fog made the journey difficult.",
    "A flag hung motionless above the gate.",
]

COHERENT_CONNECTORS = [
    "Furthermore, ", "In addition, ", "It is also known that ",
    "Moreover, ", "Additionally, ",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Gemma3 must use bfloat16 (float16 produces NaN logits on MPS)
_MODEL_DTYPE = {
    "gemma3-4b": torch.bfloat16,
}
_DEFAULT_DTYPE = torch.float16


def _default_device():
    if torch.cuda.is_available(): return "auto"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_model(model_name: str, model_id: str, device: str = None):
    print(f"Loading {model_name} ({model_id})...")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = _MODEL_DTYPE.get(model_name, _DEFAULT_DTYPE)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device is not None else _default_device(),
        attn_implementation="eager",
    )
    model.eval()
    info = get_arch_info(model)
    print(f"  Layers: {info['n_layers']}, Hidden: {info['hidden_size']}")
    return model, tok


# ---------------------------------------------------------------------------
# Architecture helpers (Llama-style and Gemma3-style)
# ---------------------------------------------------------------------------

def get_arch_info(model) -> dict:
    """Return {'n_layers', 'hidden_size', 'lm_head', 'final_norm', 'layers'}."""
    # Gemma3ForConditionalGeneration (multimodal wrapper)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lang = model.model.language_model
        return {
            "n_layers": len(lang.layers),
            "hidden_size": lang.config.hidden_size if hasattr(lang, "config") else lang.norm.weight.shape[0],
            "lm_head": model.lm_head,
            "final_norm": lang.norm,
            "layers": lang.layers,
        }
    # LlamaForCausalLM / standard decoder-only
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return {
            "n_layers": len(model.model.layers),
            "hidden_size": model.config.hidden_size,
            "lm_head": model.lm_head,
            "final_norm": model.model.norm,
            "layers": model.model.layers,
        }
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


# ---------------------------------------------------------------------------
# Question utilities
# ---------------------------------------------------------------------------

def get_questions(n: int = 20, seed: int = 0, n_hops: int = 3) -> list[dict]:
    return [
        q for q in build_dataset(n_questions=n, n_levels=[n_hops], seeds=[seed])
        if q["n"] == n_hops
    ]


def relevant_passages(q: dict) -> list[str]:
    """Extract the hop-relevant passages (not filler)."""
    relevant = []
    subj = q["question"].split()[2] if len(q["question"].split()) > 2 else ""
    for p in q["passages"]:
        if "Anyone" in p or q["answer"] in p:
            relevant.append(p)
        elif subj and subj in p:
            relevant.append(p)
    return relevant


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _apply_chat_template(tok, user_text: str) -> str:
    """Apply chat template if the tokenizer supports it, otherwise return raw text."""
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        msgs = [{"role": "user", "content": user_text}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return user_text


def build_prompt_a(q: dict, n_filler: int = 4, seed: int = 0, tok=None) -> str:
    """Condition A: relevant + filler as shuffled bullet points."""
    rng = random.Random(seed)
    rel = relevant_passages(q)
    fillers = rng.sample(FILLER_SENTS, k=min(n_filler, len(FILLER_SENTS)))
    all_p = rel + fillers
    rng.shuffle(all_p)
    ctx = "\n".join(f"- {p}" for p in all_p)
    text = (
        "Read the following passages and answer the question with a single "
        "word or short phrase. Do not explain.\n\n"
        f"Passages:\n{ctx}\n\nQuestion: {q['question']}\n\nAnswer:"
    )
    return _apply_chat_template(tok, text) if tok is not None else text


def build_prompt_b(q: dict, seed: int = 0, tok=None) -> str:
    """Condition B: relevant facts woven into coherent prose, padded to ~300 tokens."""
    rng = random.Random(seed * 999)
    rel = relevant_passages(q)
    facts = [p.rstrip(".") for p in rel]
    result = facts[0] + "."
    for i, fact in enumerate(facts[1:]):
        conn = COHERENT_CONNECTORS[i % len(COHERENT_CONNECTORS)]
        result += " " + conn + fact[0].lower() + fact[1:] + "."
    fillers = [
        "The context provided above contains all necessary information to answer the question.",
        "Background information has been carefully selected from reliable sources.",
        "All relevant facts are present in the provided context.",
        "The passages reflect the current state of knowledge on this topic.",
    ]
    while len(result.split()) < 90:
        result += " " + rng.choice(fillers)
    text = (
        "Read the following passage and answer the question with a single "
        "word or short phrase. Do not explain.\n\n"
        f"Passage:\n{result}\n\nQuestion: {q['question']}\n\nAnswer:"
    )
    return _apply_chat_template(tok, text) if tok is not None else text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(pred: str, ans: str) -> bool:
    pred_l = pred.strip().lower()
    ans_l = ans.strip().lower()
    if ans_l in pred_l or pred_l.startswith(ans_l):
        return True
    if "_" in ans_l:
        stem = ans_l.split("_")[0]
        if len(stem) >= 4 and (pred_l.startswith(stem) or stem.startswith(pred_l)):
            return True
    # Partial first token: pred may be a prefix of ans (e.g. "Wol" for "Wolij_loc")
    if len(pred_l) >= 3 and ans_l.startswith(pred_l):
        return True
    return False


def target_token_ids(tok, answer: str) -> set:
    """Return token ids that correspond to the answer string."""
    ids = set()
    for prefix in [" " + answer, answer, " " + answer.split("_")[0]]:
        ids.update(tok.encode(prefix, add_special_tokens=False))
    return ids
