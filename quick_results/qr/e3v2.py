"""E3 v2: MoE vs single head with the fixes A-D, protocol pre-registered in configs/e3v2.yaml.

Phase 1 (source chemistries only; also the zero-shot model), per (fold, seed):
  single_head          E2 architecture, grid over lr, 24 epochs
  single_head_matched  same, head width chosen so its parameter count matches the MoE's
  moe_mask / moe_bias  E3 MoE with fix A (Stage A length / L_spec grid, beta_init 0.2, beta lr x10), 24 epochs
  config per model selected on source-val MSE.
Phase 2 (few_2 / few_5), from the selected phase-1 model, encoder frozen for every model, k adaptation cells,
best epoch (epoch 0 = no update included) on target-val MSE:
  single heads         train the head
  MoE spawn            add an expert for the held-out chemistry, copied from the shared expert ("shared") or from
                       the expert with the highest mean unmasked router weight on the k cells ("nearest"); train the
                       new expert + router + beta, keep the existing experts and the shared expert frozen.
All models predict fraction of life; cycles use the TRUE EOL (oracle_eol=True).
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import train as T
from .features import make_split, regime_sets, window_idx
from .metrics import chem_metrics
from .models import MoE, SingleHead


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def matched_width(prep, n_experts=2):
    """Head width whose SingleHead parameter count is closest to the MoE's (same n_feat, text dim, dynamics)."""
    target = n_params(MoE(prep["X"].shape[-1], prep["text"].shape[1], prep["D"].shape[1], n_experts, 0.0))
    widths = range(16, 2049, 4)
    w = min(widths, key=lambda w: abs(n_params(SingleHead(prep["X"].shape[-1], width=w)) - target))
    return w, target, n_params(SingleHead(prep["X"].shape[-1], width=w)), n_params(SingleHead(prep["X"].shape[-1]))


def _single_val_mse(model, sc, prep, cells):
    idx = window_idx(prep, cells)
    return float(np.mean((T.predict_single(model, sc, prep, idx) - prep["frac"][idx]) ** 2))


def phase1(prep, sp, held, seed, cfg, width):
    """Selected phase-1 model per name, plus one row per tried config (val MSE + specialisation diagnostics)."""
    v, e2, e3 = cfg["e3v2"], cfg["e2"], cfg["e3"]
    sets = regime_sets(sp, "zero_shot")
    best, rows = {}, []
    for name, w in (("single_head", 64), ("single_head_matched", width)):
        for g in v["phase1"]["single_head_grid"]:
            c = e2 | {"epochs": v["phase1"]["epochs_total"], "lr": g["lr"], "width": w}
            model, sc = T.fit_single(prep, sets, "frac", c, seed)
            val = _single_val_mse(model, sc, prep, sets[1])
            rows.append(dict(model=name, config=g["id"], val_mse=val))
            if name not in best or val < best[name][1]:
                best[name] = ((model, sc), val, g["id"])
    for mode in ("mask", "bias"):
        name = f"moe_{mode}"
        for g in v["phase1"]["moe_grid"]:
            c = e3 | v["phase1"]["moe_fixed"] | dict(epochs_a=g["epochs_a"], epochs_b=v["phase1"]["epochs_total"] - g["epochs_a"])
            c["losses"] = dict(e3["losses"], spec=g["spec"])
            fit = T.fit_moe(prep, sets, held, mode, c, seed)
            sel = min(fit["history"], key=lambda h: h["val_mse"])
            rows.append(dict(model=name, config=g["id"], val_mse=sel["val_mse"], beta=sel["beta"],
                             max_expert_corr=sel["max_expert_corr"], gate_entropy_source=sel["gate_entropy_source"]))
            if name not in best or sel["val_mse"] < best[name][1]:
                best[name] = (fit, sel["val_mse"], g["id"])
    for r in rows:
        r["selected"] = best[r["model"]][2] == r["config"]
    return best, rows


def _freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def _train_select(model, params, loss_fn, tr_batches, val_fn, epochs, lr):
    """Adam on `params`; best epoch by val_fn() where epoch 0 (no update) is a candidate."""
    opt = torch.optim.Adam(params, lr=lr)
    best, state, best_ep = val_fn(), copy.deepcopy(model.state_dict()), 0
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr_batches():
            opt.zero_grad()
            loss_fn(b).backward()
            opt.step()
        v = val_fn()
        if v < best:
            best, state, best_ep = v, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(state)
    model.eval()
    return best_ep, best


def adapt_single(model, sc, prep, adapt, tval, cfg2, seed, batch):
    T.set_seed(seed)
    m = copy.deepcopy(model)
    _freeze(m.enc)
    idx = window_idx(prep, adapt)
    x, _ = sc.transform(prep["X"][idx], prep["D"][idx])
    x, y = T._t(x), T._t(prep["frac"][idx])
    ep, _ = _train_select(m, [p for p in m.parameters() if p.requires_grad],
                          lambda b: F.mse_loss(m(x[b]), y[b]), lambda: T._batches(len(y), batch, True),
                          lambda: _single_val_mse(m, sc, prep, tval), cfg2["epochs"], cfg2["lr"])
    return m, ep


def nearest_expert(fit, prep, adapt, cfg):
    """Expert with the highest mean unmasked router weight softmax(logits / tau) on the adaptation windows."""
    t = T._moe_tensors(prep, window_idx(prep, adapt), fit["sc"], fit["sets"], fit["chem_ids"])
    with torch.no_grad():
        logits = fit["model"](t["x"], t["d"], t["text"], t["match"], fit["tau"], "uniform")["logits"]
    return int(torch.softmax(logits / fit["tau"], 1).mean(0).argmax())


def adapt_moe(fit, prep, adapt, tval, held, init, cfg2, seed, cfg):
    T.set_seed(seed)
    src = "shared" if init == "shared" else nearest_expert(fit, prep, adapt, cfg)
    m = copy.deepcopy(fit["model"])
    m.add_expert(src)
    new = dict(fit, model=m, sets=fit["sets"] + [{held}])
    _freeze(m.enc)
    _freeze(m.shared)
    for e in m.experts[:-1]:
        _freeze(e)
    tr = T._moe_tensors(prep, window_idx(prep, adapt), fit["sc"], new["sets"], fit["chem_ids"])
    va = window_idx(prep, tval)

    def val():
        return float(np.mean((T.predict_moe(new, prep, va, cfg)[0] - prep["frac"][va]) ** 2))

    def loss(b):
        o = m(tr["x"][b], tr["d"][b], tr["text"][b], tr["match"][b], fit["tau"], fit["mode"])
        return F.mse_loss(o["pred"], tr["y"][b])

    ep, _ = _train_select(m, [p for p in m.parameters() if p.requires_grad], loss,
                          lambda: T._batches(len(tr["y"]), cfg["batch"], True), val, cfg2["epochs"], cfg2["lr"])
    return new, ep, src


def run_e3v2(prep, cfg, log=print):
    v = cfg["e3v2"]
    width, moe_n, matched_n, single_n = matched_width(prep)
    notes = dict(params=dict(moe=moe_n, single_head=single_n, single_head_matched=matched_n, matched_width=width),
                 seeds_reduced=False)
    rows, cfg_rows, t0, done, per_seed = [], [], time.time(), [], []
    for seed in v["seeds"]:
        if done and (time.time() - t0) + np.mean(per_seed) > 60 * v["budget_minutes"]:
            notes["seeds_reduced"] = True
            log(f"E3v2: budget: stopping after seeds {done}")
            break
        s0 = time.time()
        for held in v["folds"]:
            sp = make_split(prep["chem"], held, seed, cfg["split"])
            test = window_idx(prep, sp["test"])
            best, crow = phase1(prep, sp, held, seed, cfg, width)
            cfg_rows += [dict(fold=held, seed=seed, **r) for r in crow]

            def score(regime, model, pred, **extra):
                rows.append(dict(fold=held, regime=regime, model=model, seed=seed, oracle_eol=True,
                                 **extra, **chem_metrics(prep["rul"][test], pred * prep["eol"][test], prep["cell_idx"][test])))

            for name, (obj, _, cid) in best.items():  # zero-shot = phase-1 model
                if name.startswith("single"):
                    score("zero_shot", name, T.predict_single(*obj, prep, test), config=cid)
                else:
                    score("zero_shot", name, T.predict_moe(obj, prep, test, cfg["e3"])[0], config=cid)
            for regime in v["few_shot"]:
                k = int(regime.split("_")[1])
                if len(sp["adapt"]) < k:
                    continue
                adapt = sp["adapt"][:k]
                tval = sp["val"] if len(sp["val"]) else adapt  # only with < 3 cells per chemistry (smoke test)
                for name in ("single_head", "single_head_matched"):
                    (model, sc), _, cid = best[name]
                    m, ep = adapt_single(model, sc, prep, adapt, tval, v["phase2"], seed, cfg["e2"]["batch"])
                    score(regime, name, T.predict_single(m, sc, prep, test), config=cid, adapt_epoch=ep)
                for mode in ("mask", "bias"):
                    fit, _, cid = best[f"moe_{mode}"]
                    for init in v["phase2"]["spawn_init"]:
                        new, ep, src = adapt_moe(fit, prep, adapt, tval, held, init, v["phase2"], seed, cfg["e3"])
                        pred, g, _, _ = T.predict_moe(new, prep, test, cfg["e3"])
                        score(regime, f"moe_{mode}_spawn_{init}", pred, config=cid, adapt_epoch=ep,
                              spawned_from=str(src), new_expert_gate_test=float(g[:, -1].mean()))
            log(f"E3v2: seed {seed} fold {held} done ({time.time() - t0:.0f}s)")
        per_seed.append(time.time() - s0)
        done.append(seed)
    notes |= dict(seeds_done=done, seconds=round(time.time() - t0, 1))
    return rows, cfg_rows, notes


def paired_vs(df, cfg):
    """Every MoE variant vs single_head and vs single_head_matched, per regime, all folds and focus folds."""
    from .e5 import paired
    v = cfg["e3v2"]
    d = df.assign(variant=df.model, label="frac")
    out = []
    for regime in ["zero_shot"] + v["few_shot"]:
        moes = sorted(m for m in d[d.regime == regime].model.unique() if m.startswith("moe_"))
        for m in moes:
            for ref in ("single_head", "single_head_matched"):
                for folds in (None, v["focus_folds"]):
                    r = paired(d, m, ref, regime=regime, folds=folds, n_boot=v["bootstrap"])
                    if r:
                        r["verdict"] = ("MoE better" if r["ci95_high"] < 0 else
                                        "MoE worse (CI above 0)" if r["ci95_low"] > 0 else "not better at this scale")
                        out.append(r)
    return out
