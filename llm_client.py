"""Provider-agnostic LLM client for the multi-evidence integration experiments.

Supports three backends selectable via --backend (or LLM_BACKEND env var):

  openai     — OpenAI API or any OpenAI-compatible endpoint (Together, Fireworks,
               local vLLM, Ollama with --openai-compat, etc.)
  anthropic  — Anthropic API (Claude models)
  hf         — Local HuggingFace model loaded directly (no API key needed)

The client exposes a single method:
    client.invoke(model_name, prompt, max_tokens=64, temperature=0.0) -> str

All experiment scripts import this module for LLM inference.
work unchanged regardless of which backend you choose.

Quick-start examples
--------------------
# OpenAI / OpenAI-compatible:
export LLM_BACKEND=openai
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional; default is OpenAI
python run_p1.py --models gpt-4o-mini

# Anthropic:
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python run_p1.py --models claude-3-haiku-20240307

# Local HuggingFace (no API key):
export LLM_BACKEND=hf
python run_p1.py --models meta-llama/Llama-3.1-8B-Instruct

# Together AI (OpenAI-compatible):
export LLM_BACKEND=openai
export OPENAI_API_KEY=<together-key>
export OPENAI_BASE_URL=https://api.together.xyz/v1
python run_p1.py --models meta-llama/Llama-3.1-8B-Instruct-Turbo

Adding new models
-----------------
No model registry is required. Pass any model name your backend accepts
directly to --models. The MODELS dict below maps the paper's short names
to common public equivalents; extend it freely.
"""

import os
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Model name map: paper short name -> (openai-compat name, anthropic name, hf id)
# The experiment scripts accept any name; this map lets you use the same short
# names as the paper without changing every script.
# ---------------------------------------------------------------------------
MODELS: dict[str, str] = {
    # Llama family (HuggingFace / Together / Fireworks IDs)
    "llama3-8b":       "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-70b":      "meta-llama/Llama-3.1-70B-Instruct",
    "llama3.3-70b":    "meta-llama/Llama-3.3-70B-Instruct",
    "llama4-scout":    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Gemma
    "gemma3-4b":       "google/gemma-3-4b-it",
    "gemma3-12b":      "google/gemma-3-12b-it",
    "gemma3-27b":      "google/gemma-3-27b-it",
    # Mistral
    "mistral-7b":      "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "ministral-8b":    "mistralai/Ministral-8B-Instruct-2410",
    # Qwen
    "qwen3-32b":       "Qwen/Qwen3-32B-Instruct",
    # DeepSeek
    "deepseek-r1":     "deepseek-ai/DeepSeek-R1",
}


def _resolve(model_name: str) -> str:
    """Resolve a short name to a full model ID; pass through if not in map."""
    return MODELS.get(model_name, model_name)


# ---------------------------------------------------------------------------
# Backend: OpenAI / OpenAI-compatible
# ---------------------------------------------------------------------------

class _OpenAIClient:
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", None),
        )

    def invoke(self, model_name: str, prompt: str,
               max_tokens: int = 64, temperature: float = 0.0,
               retries: int = 8) -> str:
        model_id = _resolve(model_name)
        wait = 2.0
        for attempt in range(retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                name = type(e).__name__
                if "RateLimitError" in name or "ServiceUnavailable" in name:
                    if attempt < retries - 1:
                        time.sleep(wait); wait = min(wait * 2, 60.0); continue
                raise
        raise RuntimeError(f"invoke failed after {retries} retries")

    def check_alive(self) -> bool:
        try:
            self.invoke("gpt-3.5-turbo", "hi", max_tokens=1)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Backend: Anthropic
# ---------------------------------------------------------------------------

class _AnthropicClient:
    def __init__(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        )

    def invoke(self, model_name: str, prompt: str,
               max_tokens: int = 64, temperature: float = 0.0,
               retries: int = 8) -> str:
        model_id = _resolve(model_name)
        wait = 2.0
        for attempt in range(retries):
            try:
                msg = self._client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()
            except Exception as e:
                name = type(e).__name__
                if "RateLimitError" in name or "OverloadedError" in name:
                    if attempt < retries - 1:
                        time.sleep(wait); wait = min(wait * 2, 60.0); continue
                raise
        raise RuntimeError(f"invoke failed after {retries} retries")

    def check_alive(self) -> bool:
        return True  # key validity checked on first real call


# ---------------------------------------------------------------------------
# Backend: local HuggingFace
# ---------------------------------------------------------------------------

class _HFClient:
    def __init__(self):
        self._loaded: dict = {}

    def _load(self, model_id: str):
        if model_id in self._loaded:
            return self._loaded[model_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[hf] loading {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        device_map = "auto" if self._has_cuda() else ("mps" if self._has_mps() else "cpu")
        dtype = torch.bfloat16 if "gemma" in model_id.lower() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map,
            attn_implementation="eager",
        )
        model.eval()
        self._loaded[model_id] = (model, tok)
        return model, tok

    @staticmethod
    def _has_cuda():
        import torch; return torch.cuda.is_available()

    @staticmethod
    def _has_mps():
        import torch
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def invoke(self, model_name: str, prompt: str,
               max_tokens: int = 64, temperature: float = 0.0,
               retries: int = 1) -> str:
        import torch
        model_id = _resolve(model_name)
        model, tok = self._load(model_id)
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def check_alive(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(backend: str = None):
    """Return a client for the requested backend.

    backend is resolved from (in order):
      1. the backend argument
      2. the LLM_BACKEND environment variable
      3. "openai" as default
    """
    backend = backend or os.environ.get("LLM_BACKEND", "openai")
    if backend == "openai":
        return _OpenAIClient()
    if backend == "anthropic":
        return _AnthropicClient()
    if backend == "hf":
        return _HFClient()
    raise ValueError(f"Unknown backend '{backend}'. Choose: openai, anthropic, hf")
