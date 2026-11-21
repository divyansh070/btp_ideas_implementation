"""Validation gate (guide §4): every check returns a row; training refuses to start if any row fails."""
from collections import Counter

import numpy as np

from .features import FEAT_NAMES, Scaler, make_split, window_idx


def run_gate(all_cells, kept, prep, pbt_labels, cfg):
    g = cfg["gate"]
    rows = []

    def add(check, value, passed, detail=""):
        rows.append(dict(check=check, value=value, passed=bool(passed), detail=detail))

    # 1. counts per chemistry vs expected (PBT's post-filter split, see document.md Phase 0 audit)
    got = Counter(c["chem"] for c in kept)
    exp = cfg["expected_counts"]
    for chem in sorted(set(exp) | set(got)):
        add(f"count {chem}", got.get(chem, 0), got.get(chem, 0) == exp.get(chem, -1), f"expected {exp.get(chem)}")

    # 2. EOL: zero fallbacks, and agreement with PBT's published labels
    bad = [c["name"] for c in kept if c["label_source"] not in ("threshold", "extrapolated")]
    add("EOL fallbacks", len(bad), not bad, ", ".join(bad[:5]))
    mism = [c["name"] for c in all_cells if c["name"] in pbt_labels
            and (c["eol"] is None or abs(c["eol"] - pbt_labels[c["name"]]) > 1)]  # PBT labels <=100-cycle cells too
    extra = [c["name"] for c in kept if c["name"] not in pbt_labels]
    add("EOL vs PBT labels (mismatch or dropped-but-labelled)", len(mism), not mism, ", ".join(mism[:5]))
    add("kept cells without a PBT label", len(extra), not extra, ", ".join(extra[:5]) or ("no PBT label files" if not pbt_labels else ""))

    # 3. per-cell corr(RUL, capacity) > 0.5, over cycles 1..min(n, EOL)
    corrs = {}
    for c in kept:
        n = min(c["n_cycles"], c["eol"])
        q = c["qmax"][:n]
        ok = np.isfinite(q)
        corrs[c["name"]] = np.corrcoef(c["eol"] - np.arange(1, n + 1)[ok], q[ok])[0, 1]
    low = sorted((v, k) for k, v in corrs.items() if not v > g["min_corr"])
    add(f"per-cell corr(RUL, capacity) > {g['min_corr']}", round(min(corrs.values()), 3), not low,
        "; ".join(f"{k}={v:.2f}" for v, k in low[:5]))

    # 4. SNL split three ways, no cell tagged twice
    snl = [c for c in kept if c["dataset"] == "SNL"]
    names = [c["name"] for c in kept]
    add("SNL tagged LFP/NMC/NCA", dict(Counter(c["chem"] for c in snl)),
        {c["chem"] for c in snl} <= {"LFP", "NMC", "NCA"}, "")
    add("no cell tagged twice", len(names) - len(set(names)), len(names) == len(set(names)))

    # 5. scaled feature ranges: fit on source-train + full target adaptation, check target val+test windows
    lo, hi = g["range"]
    p_lo, p_hi = g["percentiles"]
    for held in cfg["folds"]:
        worst_rng, worst_std = [], []
        for seed in cfg["seeds"]:
            sp = make_split(prep["chem"], held, seed, cfg["split"])
            fit = window_idx(prep, np.concatenate([sp["src_train"], sp["adapt"]]))
            tgt = window_idx(prep, np.concatenate([sp["val"], sp["test"]]))
            sc = Scaler().fit(prep["X"][fit], prep["D"][fit])
            xf = sc.transform(prep["X"][fit], prep["D"][fit])[0].reshape(-1, prep["X"].shape[-1])
            xt = sc.transform(prep["X"][tgt], prep["D"][tgt], clip=False)[0].reshape(-1, prep["X"].shape[-1])
            q = np.percentile(xt, [p_lo, p_hi], axis=0)
            out = np.flatnonzero((q[0] < lo) | (q[1] > hi))
            worst_rng.append((len(out), round(float(q.min()), 2), round(float(q.max()), 2), seed, [FEAT_NAMES[i] for i in out[:5]]))
            sd = xf.std(0)
            worst_std.append((float(sd.min()), [FEAT_NAMES[i] for i in np.flatnonzero(sd <= g["min_std"])]))
        n_out = max(r[0] for r in worst_rng)
        add(f"fold {held}: target features p{p_lo}-p{p_hi} in [{lo}, {hi}]", n_out, n_out == 0,
            "; ".join(f"seed {r[3]}: {r[0]} out, range [{r[1]}, {r[2]}], feats {r[4]}" for r in worst_rng))
        low_std = sorted({n for _, names in worst_std for n in names})
        add(f"fold {held}: min feature std > {g['min_std']}", round(min(v for v, _ in worst_std), 4), not low_std,
            ", ".join(low_std))

    # 6. splits cell-disjoint (asserted inside make_split; recorded here)
    add("splits cell-disjoint (all folds x seeds)", "ok", True)
    return rows
