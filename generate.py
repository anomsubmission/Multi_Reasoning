"""Synthetic multi-hop question generator for the ICL saturation pilot (P1).

Each question requires integrating N independent character-attribute facts to
answer a simple relational query.  Characters and attributes are randomly
generated so the model has no pretraining prior on them.

Output schema (per question):
  {
    "id": str,              # unique id, e.g. "q0042_n3_s1"
    "n": int,               # number of passages required
    "seed": int,
    "passages": [str, ...], # N relevant + padding to hold total tokens constant
    "question": str,
    "answer": str,          # single correct token / short string
    "total_tokens": int     # approximate; filled in by caller
  }
"""

import hashlib
import json
import random
import string
from pathlib import Path

# ---------------------------------------------------------------------------
# Random name / attribute pools
# ---------------------------------------------------------------------------

_CONSONANTS = "bcdfghjklmnprstvwxz"
_VOWELS     = "aeiou"

def _fake_name(rng: random.Random) -> str:
    """2-syllable CV-CVC nonsense name, title-cased."""
    s = (rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
         + rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
         + rng.choice(_CONSONANTS))
    return s.capitalize()

_ATTRIBUTES = [
    ("carries",        "carry",       "item"),
    ("lives in",       "live in",     "location"),
    ("owns",           "own",         "object"),
    ("works as",       "work as",     "profession"),
    ("is allergic to", "be allergic to", "substance"),
    ("speaks",         "speak",       "language"),
    ("trains at",      "train at",    "place"),
    ("collects",       "collect",     "thing"),
    ("fears",          "fear",        "creature"),
    ("studies",        "study",       "subject"),
]

_FILLERS = [
    "The sky was overcast that morning.",
    "A gentle breeze drifted through the valley.",
    "Somewhere in the distance a bell was ringing.",
    "The old library had not been opened in years.",
    "Rain was expected by the afternoon.",
    "The market closed early on holidays.",
    "Several birds perched on the fence post.",
    "The road curved sharply near the river.",
    "Thick fog made the journey difficult.",
    "A flag hung motionless above the gate.",
]


def _unique_names(rng: random.Random, n: int) -> list[str]:
    seen, names = set(), []
    while len(names) < n:
        name = _fake_name(rng)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _unique_values(rng: random.Random, n: int, tag: str) -> list[str]:
    seen, vals = set(), []
    while len(vals) < n:
        v = _fake_name(rng) + "_" + tag[:3]
        if v not in seen:
            seen.add(v)
            vals.append(v)
    return vals


# ---------------------------------------------------------------------------
# Question templates
# The "answer" always resolves through N-1 hops:
#   passage 1:  Zofrik lives in Birvot_loc
#   passage 2:  Whoever lives in Birvot_loc carries Wumpal_ite
#   question:   What does Zofrik carry?
#   answer:     Wumpal_ite
# For N=1 we just state the fact directly.
# ---------------------------------------------------------------------------

def _build_chain(rng: random.Random, n: int) -> tuple[list[str], str, str]:
    """Return (passages, question, answer)."""
    attrs = rng.sample(_ATTRIBUTES, k=min(n, len(_ATTRIBUTES)))
    # attrs[i] = (conjugated_verb, infinitive_verb, tag)
    # n characters, n attribute-values
    names  = _unique_names(rng, n + 1)   # extra name for red-herrings
    anchor = names[0]

    # Build a chain: anchor -> attr[0] -> attr[1] -> ... -> attr[n-1]
    chain_vals = []
    for conj, inf, tag in attrs[:n]:
        chain_vals.append(_unique_values(rng, 1, tag)[0])

    passages = []
    if n == 1:
        conj, inf, tag = attrs[0]
        val            = chain_vals[0]
        passages.append(f"{anchor} {conj} {val}.")
        question = f"What does {anchor} {inf}?"
        answer   = val
    else:
        conj0, inf0, _ = attrs[0]
        passages.append(f"{anchor} {conj0} {chain_vals[0]}.")
        for i in range(1, n):
            conj_i, inf_i, _ = attrs[i]
            conj_prev, _, _  = attrs[i-1]
            passages.append(
                f"Anyone who {conj_prev} {chain_vals[i-1]} "
                f"also {conj_i} {chain_vals[i]}."
            )
        _, inf_last, _ = attrs[n-1]
        answer         = chain_vals[n-1]
        question       = f"What does {anchor} {inf_last}?"

    return passages, question, answer


def _filler_passage(rng: random.Random, target_tokens: int,
                    existing_tokens: int) -> str:
    """Build a filler passage of approximately (target_tokens - existing_tokens) tokens.
    Rough rule: 1 token ≈ 4 chars."""
    needed_chars = max(0, (target_tokens - existing_tokens) * 4)
    out = []
    while sum(len(s) + 1 for s in out) < needed_chars:
        out.append(rng.choice(_FILLERS))
    return " ".join(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TARGET_TOKENS = 300   # total passage tokens held constant across N levels


def generate_question(n: int, seed: int, q_idx: int) -> dict:
    rng = random.Random(seed * 10_000 + q_idx * 100 + n)

    passages, question, answer = _build_chain(rng, n)

    # Count rough tokens in relevant passages
    relevant_text = " ".join(passages)
    relevant_tok  = len(relevant_text) // 4

    # Pad with filler to reach TARGET_TOKENS
    filler = _filler_passage(rng, TARGET_TOKENS, relevant_tok)
    if filler:
        filler_sents = [s.strip().rstrip(".") + "." for s in filler.split(". ") if s.strip()]
        all_passages = passages[:]
        for j, fs in enumerate(filler_sents):
            pos = rng.randint(0, len(all_passages))
            all_passages.insert(pos, fs)
    else:
        all_passages = passages[:]

    rng.shuffle(all_passages)

    uid = f"q{q_idx:04d}_n{n}_s{seed}"
    return {
        "id":       uid,
        "n":        n,
        "seed":     seed,
        "passages": all_passages,
        "question": question,
        "answer":   answer,
    }


def build_dataset(n_questions: int = 100,
                  n_levels: list = None,
                  seeds: list = None,
                  out_path: Path = None) -> list[dict]:
    if n_levels is None:
        n_levels = [1, 2, 3, 4]
    if seeds is None:
        seeds = [0, 1, 2]

    records = []
    for seed in seeds:
        for n in n_levels:
            for q_idx in range(n_questions):
                records.append(generate_question(n, seed, q_idx))

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(records, indent=2))
        print(f"Wrote {len(records)} questions to {out_path}")

    return records


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/p1_questions.json")
    build_dataset(out_path=out)
