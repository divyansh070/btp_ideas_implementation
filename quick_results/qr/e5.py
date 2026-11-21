"""E5: does a text description of the cell improve cross-chemistry RUL?

Same LOCO folds, splits (same seed -> same cells for every variant: paired), training and lstm_frac model as E2;
the only difference between variants is the extra input concatenated to z through Linear + LeakyReLU -> 32:
  A_none      no extra branch (= E2 lstm_frac)
  B_onehot    one-hot of the fold's source chemistries; held-out cells get all zeros (a label has no slot for them)
  C_text      PBT Llama prompt embedding, unit length
  D_shuffled  control: each cell gets another cell's embedding (fixed derangement per seed), unit length
  E_text_dyn  C + the 6 window dynamics (5 scaled + latent displacement)
lstm_frac converts with the TRUE EOL (oracle_eol=True). A and C are also run with the lstm_rul head (cycles directly,
no oracle) for comparison with E4's PBT numbers.
"""
import time

import numpy as np
import pandas as pd

from . import train as T
from .features import make_split, regime_sets, window_idx
from .metrics import chem_metrics

VARIANTS = ("A_none", "B_onehot", "C_text", "D_shuffled", "E_text_dyn")


def unit(m):
    return (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def derangement(n, seed):
    """perm[i] != i for all i: cell i reads the embedding of cell perm[i]."""
    order = np.random.default_rng([seed, 5]).permutation(n)
    perm = np.empty(n, int)
    perm[order] = np.roll(order, -1)
    return perm


def aux_for(variant, prep, held, seed):
    """(aux_cells [n_cells, d] or None, aux_dyn)."""
    if variant == "A_none":
        return None, False
    if variant == "B_onehot":
        src = sorted(set(prep["chem"]) - {held})
        m = np.zeros((len(prep["chem"]), len(src)), np.float32)
        for i, c in enumerate(prep["chem"]):
            if c in src:
                m[i, src.index(c)] = 1.0
        return m, False
    text = unit(prep["text"])
    if variant == "C_text":
        return text, False
    if variant == "D_shuffled":
        return text[derangement(len(text), seed)], False
    if variant == "E_text_dyn":
        return text, True
    raise ValueError(variant)


def _fit_score(prep, sets, test, variant, label, held, seed, tcfg):
    aux, dyn = aux_for(variant, prep, held, seed)
    model, sc = T.fit_single(prep, sets, label, tcfg, seed, aux_cells=aux, aux_dyn=dyn)
    p = T.to_cycles(prep, test, T.predict_single(model, sc, prep, test), label, tcfg["rul_scale"])
    return chem_metrics(prep["rul"][test], p, prep["cell_idx"][test])


def run_e5(prep, cfg, log=print):
    c, tcfg = cfg["e5"], cfg["e2"]
    budget, t0 = 60 * c["budget_minutes"], time.time()
    rows, t_frac, t_rul, done, notes = [], [], [], [], dict(rul_dropped_from_seed=None, seeds_reduced=False)
    rul_on = bool(c["rul_variants"])
    for seed in c["seeds"]:
        if done:  # budget guard: drop the non-oracle extra first, then whole seeds
            el, tf, tr = time.time() - t0, np.mean(t_frac), (np.mean(t_rul) if t_rul else 0.0)
            if rul_on and el + tf + tr > budget:
                rul_on, notes["rul_dropped_from_seed"] = False, seed
                log(f"E5: budget: dropping lstm_rul variants from seed {seed}")
            if el + tf > budget:
                notes["seeds_reduced"] = True
                log(f"E5: budget: stopping after seeds {done}")
                break
        splits = {h: make_split(prep["chem"], h, seed, cfg["split"]) for h in cfg["folds"]}
        for label, variants, times, on in (("frac", c["variants"], t_frac, True), ("rul", c["rul_variants"], t_rul, rul_on)):
            if not on:
                continue
            s0 = time.time()
            for held, sp in splits.items():
                test = window_idx(prep, sp["test"])
                for regime in c["regimes"]:
                    sets = regime_sets(sp, regime)
                    if sets is None:
                        continue
                    for v in variants:
                        rows.append(dict(fold=held, regime=regime, variant=v, label=label, model=f"lstm_{label}",
                                         seed=seed, oracle_eol=label == "frac",
                                         **_fit_score(prep, sets, test, v, label, held, seed, tcfg)))
            times.append(time.time() - s0)
            log(f"E5: seed {seed} {label} done in {times[-1]:.0f}s")
        done.append(seed)
    notes |= dict(seeds_done=done, seconds=round(time.time() - t0, 1),
                  seconds_per_seed_frac=round(float(np.mean(t_frac)), 1) if t_frac else None,
                  seconds_per_seed_rul=round(float(np.mean(t_rul)), 1) if t_rul else None)
    return rows, notes


def paired(df, x, y, label="frac", regime=None, folds=None, n_boot=10000, seed=0):
    """x vs y on identical (fold, seed, regime) cells: wins of x, mean RMSE(x) - RMSE(y), bootstrap 95 % CI."""
    d = df[df.label == label]
    if regime is not None:
        d = d[d.regime == regime]
    if folds is not None:
        d = d[d.fold.isin(folds)]
    p = d.pivot_table(index=["fold", "seed", "regime"], columns="variant", values="rmse")
    if x not in p or y not in p:
        return None
    diff = (p[x] - p[y]).dropna().to_numpy()
    if len(diff) == 0:
        return None
    boot = np.random.default_rng(seed).choice(diff, (n_boot, len(diff))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = "supported" if hi < 0 else "not supported at this scale"
    if lo > 0:
        verdict += f" ({x} worse: CI above 0)"
    return dict(comparison=f"{x} vs {y}", label=label, regime=regime or "all", folds="+".join(folds) if folds else "all",
                pairs=len(diff), wins=int((diff < 0).sum()), mean_rmse_diff=round(float(diff.mean()), 2),
                ci95_low=round(float(lo), 2), ci95_high=round(float(hi), 2), verdict=verdict)


def paired_table(df, cfg):
    c = cfg["e5"]
    out = []
    for x, y in (("C_text", "A_none"), ("C_text", "D_shuffled"), ("C_text", "B_onehot")):
        for regime in c["regimes"]:
            out.append(paired(df, x, y, regime=regime, n_boot=c["bootstrap"]))
        out.append(paired(df, x, y, regime="zero_shot", folds=c["focus_folds"], n_boot=c["bootstrap"]))
    for regime in c["regimes"]:
        out.append(paired(df, "C_text", "A_none", label="rul", regime=regime, n_boot=c["bootstrap"]))
    return pd.DataFrame([r for r in out if r])
