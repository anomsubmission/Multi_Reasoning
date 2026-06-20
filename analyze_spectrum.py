"""analyze_spectrum.py — cross-family format-gap spectrum + orthogonality.

Combines:
  * N-scaling drop per model  (P1: acc@N=1 - acc@N=4)
  * format gap delta per model (P3 + P5 + P6: acc(A bullets) - acc(B prose) @ N=3)

Prints the full cross-family spectrum (dose-response) and recomputes the
orthogonality correlation between the two failure modes over every model that
has both measurements. This is the authoritative source for the r value the
paper should cite (reconciling the RESEARCH_PLAN's r=0.029 vs the draft's
r=-0.11).

Usage: python analyze_spectrum.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).parent / "results"

# Normalise name variants across experiment files.
ALIAS = {"llama3-8b-instruct": "llama3-8b"}


def load(fname):
    p = RES / fname
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def norm(m):
    return ALIAS.get(m, m)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs); vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    # --- N-scaling drop from P1 ---
    p1 = load("p1_results.jsonl")
    by = defaultdict(lambda: defaultdict(list))
    for r in p1:
        by[norm(r["model"])][r["n"]].append(r["correct"])
    nscale_drop = {}
    for m, nn in by.items():
        if nn.get(1) and nn.get(4):
            nscale_drop[m] = mean(nn[1]) - mean(nn[4])

    # --- format gap from P3 + P5 + P6, with the CORRECTED scaled Maverick ---
    # p5_results.jsonl contains an early, contaminated llama4-maverick run whose
    # bullets baseline was depressed (see paper footnote). The uniform-protocol
    # re-run lives in p3_maverick_scaled.jsonl; use it and drop the stale one.
    fmt = defaultdict(lambda: defaultdict(list))
    for r in load("p3_results.jsonl") + load("p5_results.jsonl") + load("p6_doseresponse.jsonl"):
        if norm(r["model"]) == "llama4-maverick":
            continue  # replaced by the scaled re-run below
        fmt[norm(r["model"])][r["condition"]].append(r["correct"])
    for r in load("p3_maverick_scaled.jsonl"):
        fmt[norm(r["model"])][r["condition"]].append(r["correct"])
    fmt_delta = {}
    for m, cc in fmt.items():
        if cc.get("A") and cc.get("B"):
            fmt_delta[m] = mean(cc["A"]) - mean(cc["B"])

    # --- spectrum table ---
    print("=== Cross-family format-gap spectrum (A=bullets, B=prose, N=3) ===")
    print(f"{'Model':<16} {'fmt A-B':>8} {'Nscale drop':>12}")
    print("-" * 40)
    for m in sorted(fmt_delta, key=lambda x: -fmt_delta[x]):
        d = nscale_drop.get(m, float("nan"))
        print(f"{m:<16} {fmt_delta[m]:>+8.3f} {d:>+12.3f}")

    # --- orthogonality over models with BOTH measures ---
    both = [(m, fmt_delta[m], nscale_drop[m]) for m in fmt_delta if m in nscale_drop]
    print(f"\n=== Orthogonality (models with both measures: {len(both)}) ===")
    if len(both) >= 3:
        xs = [b[1] for b in both]; ys = [b[2] for b in both]
        r = pearson(xs, ys); rho = spearman(xs, ys)
        print(f"  Pearson  r   = {r:+.3f}   (R^2 = {r*r:.3f})")
        print(f"  Spearman rho = {rho:+.3f}")
        print(f"  models: {sorted(m for m, _, _ in both)}")
        print("\n  -> This is the authoritative r for the paper's orthogonality claim.")
    else:
        print("  Not enough overlapping models yet (P6 may still be running).")


if __name__ == "__main__":
    main()
