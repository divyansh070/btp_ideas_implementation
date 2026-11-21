"""Windows, features, labels, cell-level LOCO splits, scaler.

Window (guide §4): 30 consecutive cycles, 10 sampled evenly, stride 1 (capped per cell, see max_windows_per_cell).
Per sampled cycle: 18 stats (mean/std/min/max/var/median of V, I in C-rate, Qd) + 12 window-relative deltas
(mean/std/min/max of each signal minus the window's first sampled cycle) = 30 features, matching the guide's
"10 cycles x 30 features" (§6.1). Labels: RUL = EOL - last sampled cycle; fraction of life = RUL / EOL.
Dynamics (router, §6.4): 5 here + latent displacement computed inside the model = 6.
"""
import zlib

import numpy as np

DELTA_COLS = [s * 6 + j for s in range(3) for j in range(4)]  # mean,std,min,max of each signal
N_FEAT = 18 + len(DELTA_COLS)
FEAT_NAMES = [f"{g}_{st}" for g in ("V", "I_C", "Qd") for st in ("mean", "std", "min", "max", "var", "median")]
FEAT_NAMES += ["d_" + FEAT_NAMES[c] for c in DELTA_COLS]
DYN_NAMES = ("dQ_slope", "dQ_spread", "dQ_curvature", "dV_mean", "Q_ratio")


def cell_windows(cell, W=30, K=10, cap=None):
    n = min(cell["n_cycles"], cell["eol"])  # never use cycles past EOL (RUL would go negative)
    if n < W:
        return None
    starts = np.arange(n - W + 1)
    if cap and len(starts) > cap:  # ponytail: evenly spaced subsample keeps runtime < 1 h; stride-1 for all is cap=None
        starts = starts[np.round(np.linspace(0, len(starts) - 1, cap)).astype(int)]
    off = np.round(np.linspace(0, W - 1, K)).astype(int)
    idx = starts[:, None] + off
    s = cell["stats"][idx]                                         # [N, K, 18]
    x = np.concatenate([s, s[:, :, DELTA_COLS] - s[:, :1, DELTA_COLS]], axis=2)
    q = cell["qmax"][idx] / cell["qmax"][idx][:, :1]                # capacity relative to window start
    t = off / (W - 1)
    lin = np.polyfit(t, q.T, 1)
    quad = np.polyfit(t, q.T, 2)
    dyn = np.stack([lin[0], q.std(1), quad[0], s[:, -1, 0] - s[:, 0, 0], q[:, -1]], 1)
    end_cycle = idx[:, -1] + 1
    rul = (cell["eol"] - end_cycle).astype(np.float32)
    ok = np.isfinite(x).all((1, 2)) & np.isfinite(dyn).all(1)
    return dict(X=x[ok].astype(np.float32), D=dyn[ok].astype(np.float32), rul=rul[ok],
                frac=(rul[ok] / cell["eol"]).astype(np.float32), end_cycle=end_cycle[ok], n_nan=int((~ok).sum()))


def prepare(kept, text, W=30, K=10, cap=None):
    """All windows of all kept cells, stacked. text: {cell_name: vector}; a missing cell raises (PBT would use zeros)."""
    missing = [c["name"] for c in kept if c["name"] not in text]
    if missing:
        raise KeyError(f"{len(missing)} cells have no prompt embedding, e.g. {missing[:3]}")
    parts, cells = [], []
    for i, c in enumerate(kept):
        w = cell_windows(c, W, K, cap)
        if w is None or len(w["rul"]) == 0:
            raise ValueError(f"{c['name']}: no valid windows")
        w["cell_idx"] = np.full(len(w["rul"]), i)
        parts.append(w)
        cells.append({k: v for k, v in c.items() if k not in ("stats", "qmax")} | {"n_windows": len(w["rul"]), "n_nan_windows": w["n_nan"]})
    prep = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "D", "rul", "frac", "end_cycle", "cell_idx")}
    prep["eol"] = np.array([c["eol"] for c in cells], np.float32)[prep["cell_idx"]]
    prep["chem"] = np.array([c["chem"] for c in cells])
    prep["text"] = np.stack([text[c["name"]] for c in cells])
    prep["cells"] = cells
    return prep


# ---------------------------------------------------------------- splits (cell level)
def make_split(chem, held, seed, cfg):
    """Held-out chemistry: adapt/val/test 60/10/30, cell-disjoint. Source: every other chemistry, 80/20 train/val per chemistry."""
    rng = np.random.default_rng([seed, zlib.crc32(held.encode())])
    tgt = rng.permutation(np.flatnonzero(chem == held))
    n = len(tgt)
    n_test = max(1, round(cfg["test"] * n))
    n_val = max(1, round(cfg["val"] * n)) if n >= 3 else 0
    split = dict(test=tgt[:n_test], val=tgt[n_test:n_test + n_val], adapt=tgt[n_test + n_val:])
    src_tr, src_va = [], []
    for c in sorted(set(chem) - {held}):
        ids = rng.permutation(np.flatnonzero(chem == c))
        k = max(1, round(cfg["source_val"] * len(ids))) if len(ids) >= 3 else 0
        src_va += list(ids[:k])
        src_tr += list(ids[k:])
    split.update(src_train=np.array(src_tr, int), src_val=np.array(src_va, int))
    every = np.concatenate(list(split.values()))
    assert len(every) == len(set(every)) == len(chem), "cell assigned to two splits or left out"
    return split


def regime_sets(split, regime):
    """(train cells, val cells, scaler-fit cells) or None if the regime needs more adaptation cells than exist."""
    a, tv = split["adapt"], split["val"]
    if regime == "target_only":
        tr, va = a, tv
    elif regime == "zero_shot":  # no target cell anywhere; scaler is source-only by necessity
        tr, va = split["src_train"], split["src_val"]
    else:
        k = int(regime.split("_")[1])  # few_<k>
        if k > len(a):
            return None
        tr, va = np.concatenate([split["src_train"], a[:k]]), np.concatenate([split["src_val"], tv])
    # only with < 3 cells per chemistry (smoke test): no validation cell exists, select on train
    return tr, (va if len(va) else tr), tr


def window_idx(prep, cells):
    return np.flatnonzero(np.isin(prep["cell_idx"], cells))


class Scaler:
    """Min-max per feature, fit on the given windows only, then clipped to [-0.5, 1.5] (guide §4 rule 3)."""

    def fit(self, X, D):
        f = X.reshape(-1, X.shape[-1])
        self.xlo, self.xhi = f.min(0), f.max(0)
        self.dlo, self.dhi = D.min(0), D.max(0)
        return self

    def transform(self, X, D, clip=True):
        x = (X - self.xlo) / np.maximum(self.xhi - self.xlo, 1e-8)
        d = (D - self.dlo) / np.maximum(self.dhi - self.dlo, 1e-8)
        if clip:
            x, d = np.clip(x, -0.5, 1.5), np.clip(d, -0.5, 1.5)
        return x.astype(np.float32), d.astype(np.float32)
