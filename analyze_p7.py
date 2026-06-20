"""analyze_p7.py — validate the format probe as a predictor.

Computes each model's canonical bullets-vs-prose probe score (delta = A - B)
from P3/P5/P6, and correlates it against the held-out format gaps measured in
P7 (A - numbered, A - json, A - runon). A strong positive correlation means a
model's cheap bullets-vs-prose probe predicts its brittleness under format
perturbations it was never measured on -> the probe is a validated diagnostic.

Usage: python analyze_p7.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).parent / "results"


def load(fname):
    p = RES / fname
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def acc_by_model_cond(records):
    acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        acc[r["model"]][r["condition"]].append(r["correct"])
    return acc


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
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
    # Canonical probe delta (A - B) from all bullets-vs-prose runs.
    canon = acc_by_model_cond(load("p3_results.jsonl")
                              + load("p5_results.jsonl")
                              + load("p6_doseresponse.jsonl"))
    canon_delta = {}
    for m, cc in canon.items():
        if cc.get("A") and cc.get("B"):
            canon_delta[m] = mean(cc["A"]) - mean(cc["B"])

    # Held-out gaps from P7.
    p7 = acc_by_model_cond(load("p7_predictor.jsonl"))
    heldout = ["numbered", "json", "runon"]

    rows = []
    for m, cc in p7.items():
        if not cc.get("A") or m not in canon_delta:
            continue
        a = mean(cc["A"])
        gaps = {h: (a - mean(cc[h]) if cc.get(h) else float("nan")) for h in heldout}
        rows.append((m, canon_delta[m], a, gaps))

    rows.sort(key=lambda r: -r[1])
    print("=== P7 predictor validation ===")
    print(f"{'Model':<16} {'probe(A-B)':>10} {'A':>6} "
          + " ".join(f"{'gap_'+h:>9}" for h in heldout))
    print("-" * 60)
    for m, probe, a, gaps in rows:
        print(f"{m:<16} {probe:>+10.3f} {a:>6.3f} "
              + " ".join(f"{gaps[h]:>+9.3f}" for h in heldout))

    print(f"\nN models with both probe and held-out data: {len(rows)}")
    if len(rows) >= 3:
        probe_vals = [r[1] for r in rows]
        print("\nCorrelation of canonical probe (A-B) with held-out gaps:")
        for h in heldout:
            ys = [r[3][h] for r in rows]
            pairs = [(x, y) for x, y in zip(probe_vals, ys) if not math.isnan(y)]
            if len(pairs) >= 3:
                xs2, ys2 = zip(*pairs)
                print(f"  vs gap_{h:<9} Pearson r={pearson(list(xs2), list(ys2)):+.3f}  "
                      f"Spearman rho={spearman(list(xs2), list(ys2)):+.3f}  (n={len(pairs)})")
        # Mean held-out gap as a single robustness index
        mean_gap = []
        for r in rows:
            gs = [r[3][h] for h in heldout if not math.isnan(r[3][h])]
            mean_gap.append(mean(gs) if gs else float("nan"))
        pairs = [(x, y) for x, y in zip(probe_vals, mean_gap) if not math.isnan(y)]
        if len(pairs) >= 3:
            xs2, ys2 = zip(*pairs)
            print(f"  vs MEAN held-out gap   Pearson r={pearson(list(xs2), list(ys2)):+.3f}  "
                  f"Spearman rho={spearman(list(xs2), list(ys2)):+.3f}  (n={len(pairs)})")
        print("\nInterpretation: strong positive r => the bullets-vs-prose probe predicts")
        print("format brittleness on held-out perturbations -> validated diagnostic.")


if __name__ == "__main__":
    main()
