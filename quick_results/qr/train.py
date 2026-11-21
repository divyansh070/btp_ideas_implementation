"""Training loops. Model selection = best epoch on the validation cells only; the blind test is never looked at."""
import copy
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .features import Scaler, window_idx
from .models import MoE, SingleHead

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _t(a, dtype=torch.float32):
    return torch.as_tensor(a, dtype=dtype, device=DEV)


def target(prep, idx, label, rul_scale):
    return prep["frac"][idx] if label == "frac" else prep["rul"][idx] / rul_scale


def to_cycles(prep, idx, out, label, rul_scale):
    """frac -> cycles uses the TRUE EOL of the cell: 'oracle EOL', marked in every results row."""
    return out * prep["eol"][idx] if label == "frac" else out * rul_scale


def _batches(n, bs, shuffle):
    order = torch.randperm(n, device=DEV) if shuffle else torch.arange(n, device=DEV)
    return [order[i:i + bs] for i in range(0, n, bs)]


# ---------------------------------------------------------------- single head (LSTM -> attention -> LayerNorm head)
def fit_single(prep, sets, label, cfg, seed):
    tr_c, va_c, fit_c = sets
    set_seed(seed)
    fit = window_idx(prep, fit_c)
    sc = Scaler().fit(prep["X"][fit], prep["D"][fit])
    data = {}
    for k, cells in (("tr", tr_c), ("va", va_c)):
        idx = window_idx(prep, cells)
        x, _ = sc.transform(prep["X"][idx], prep["D"][idx])
        data[k] = (_t(x), _t(target(prep, idx, label, cfg["rul_scale"])))
    model = SingleHead(prep["X"].shape[-1], "sigmoid" if label == "frac" else "softplus").to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state = math.inf, None
    for _ in range(cfg["epochs"]):
        model.train()
        x, y = data["tr"]
        for b in _batches(len(y), cfg["batch"], True):
            opt.zero_grad()
            F.mse_loss(model(x[b]), y[b]).backward()
            opt.step()
        v = _mse(model, *data["va"], cfg["batch"])
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, sc


@torch.no_grad()
def _mse(model, x, y, bs):
    model.eval()
    p = torch.cat([model(x[b]) for b in _batches(len(y), bs, False)])
    return F.mse_loss(p, y).item()


@torch.no_grad()
def predict_single(model, sc, prep, idx, bs=1024):
    model.eval()
    x, _ = sc.transform(prep["X"][idx], prep["D"][idx])
    x = _t(x)
    return torch.cat([model(x[b]) for b in _batches(len(x), bs, False)]).cpu().numpy()


# ---------------------------------------------------------------- LightGBM on the last sampled cycle's 30 features
def fit_lgbm(prep, sets, cfg, seed):
    import lightgbm as lgb
    tr, va = window_idx(prep, sets[0]), window_idx(prep, sets[1])
    m = lgb.LGBMRegressor(**cfg["params"], random_state=seed, verbose=-1)
    m.fit(prep["X"][tr, -1], prep["frac"][tr], eval_set=[(prep["X"][va, -1], prep["frac"][va])],
          callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False)])
    return m


# ---------------------------------------------------------------- MoE
def expert_sets(source_chems):
    """E=2, 'LFP vs rest' over the chemistries present in this fold's source; if LFP is held out, NMC vs rest."""
    s = sorted(source_chems)
    first = "LFP" if "LFP" in s else "NMC"
    return [{first}, set(s) - {first}]


def components(chem):
    return set(chem.split("-"))  # NMC-LCO -> {NMC, LCO}


def match_matrix(chems, sets):
    """1 where the cell's cathode components overlap the expert's chemistries (the category-bias / hard-mask rule)."""
    comp = [set().union(*map(components, s)) for s in sets]
    return np.array([[float(bool(components(c) & ce)) for ce in comp] for c in chems], np.float32)


def own_expert(chems, sets):
    return np.array([next((e for e, s in enumerate(sets) if c in s), -1) for c in chems])


def _mmd(z, grp):
    """Mean RBF-MMD over pairs of source-chemistry groups present in the batch (median-distance bandwidth)."""
    ids = [g for g in torch.unique(grp) if g >= 0 and (grp == g).sum() > 1]
    if len(ids) < 2:
        return z.new_zeros(())
    d = torch.cdist(z, z) ** 2
    k = torch.exp(-d / (d.detach().median() + 1e-8))
    tot = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = grp == ids[i], grp == ids[j]
            tot.append(k[a][:, a].mean() + k[b][:, b].mean() - 2 * k[a][:, b].mean())
    return torch.stack(tot).mean()


def _moe_tensors(prep, idx, sc, sets, chem_ids):
    x, d = sc.transform(prep["X"][idx], prep["D"][idx])
    ci = prep["cell_idx"][idx]
    chems = prep["chem"][ci]
    return dict(x=_t(x), d=_t(d), text=_t(prep["text"][ci]), match=_t(match_matrix(chems, sets)),
                own=_t(own_expert(chems, sets), torch.long), grp=_t([chem_ids.get(c, -1) for c in chems], torch.long),
                y=_t(prep["frac"][idx]))


def fit_moe(prep, sets_cells, held, mode, cfg, seed):
    """Stage A: encoder + each expert on its own chemistry, router uniform. Stage B: router on, five losses."""
    tr_c, va_c, fit_c = sets_cells
    set_seed(seed)
    fit = window_idx(prep, fit_c)
    sc = Scaler().fit(prep["X"][fit], prep["D"][fit])
    src = sorted(set(prep["chem"]) - {held})
    sets = expert_sets(src)
    chem_ids = {c: i for i, c in enumerate(src)}          # MMD groups: source chemistries only
    tr = _moe_tensors(prep, window_idx(prep, tr_c), sc, sets, chem_ids)
    va = _moe_tensors(prep, window_idx(prep, va_c), sc, sets, chem_ids)
    model = MoE(prep["X"].shape[-1], prep["text"].shape[1], prep["D"].shape[1] , len(sets), cfg["bias"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w, n_a, n_b = cfg["losses"], cfg["epochs_a"], cfg["epochs_b"]
    hist, best, best_state, best_tau = [], math.inf, None, cfg["tau"][0]
    norms = torch.cat([tr["text"], va["text"]]).norm(dim=1)   # raw (pre-normalisation) embedding norms of this run
    text_norm = dict(text_norm_raw_mean=norms.mean().item(), text_norm_raw_min=norms.min().item(),
                     text_norm_raw_max=norms.max().item())
    for ep in range(n_a + n_b):
        stage_a = ep < n_a
        p = 0.0 if stage_a else (ep - n_a) / max(n_b - 1, 1)  # Stage-B progress in [0, 1]
        tau = cfg["tau"][0] + (cfg["tau"][1] - cfg["tau"][0]) * p
        r = w["route"] * max(0.0, 1 - p / cfg["route_anneal_frac"])  # annealed to 0 by 40% of Stage B
        mmd_w = w["mmd"] * (2 / (1 + math.exp(-10 * p)) - 1)
        model.train()
        for b in _batches(len(tr["y"]), cfg["batch"], True):
            o = model(tr["x"][b], tr["d"][b], tr["text"][b], tr["match"][b], tau, "uniform" if stage_a else mode)
            y, own = tr["y"][b], tr["own"][b]
            has = own >= 0
            f_own = o["f"][has, own[has]]
            if stage_a:
                loss = F.mse_loss(o["fs"], y) + (F.mse_loss(f_own, y[has]) if has.any() else 0)
            else:
                loss = F.mse_loss(o["pred"], y)
                if w["mmd"]:
                    loss = loss + mmd_w * _mmd(o["z"], tr["grp"][b])
                if w["bal"]:
                    loss = loss + w["bal"] * (model.E * (o["g"].mean(0) ** 2).sum() - 1)
                if r and has.any():
                    loss = loss + r * F.cross_entropy(o["logits"][has] / tau, own[has])
                if w["spec"] and has.any():
                    loss = loss + w["spec"] * F.relu((f_own - y[has]) ** 2 - (o["pred"][has] - y[has]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if stage_a:
            continue
        o = _moe_eval(model, va, tau, mode, cfg["batch"])
        v = F.mse_loss(o["pred"], va["y"]).item()
        hist.append(dict(epoch=ep, tau=round(tau, 3), route_w=r, mmd_w=mmd_w, bias=cfg["bias"], val_mse=v,
                         **diagnostics(o), **_routing_split(o, va["own"] < 0, tau), **text_norm))
        if v < best:
            best, best_state, best_tau = v, copy.deepcopy(model.state_dict()), tau
    model.load_state_dict(best_state)
    return dict(model=model, sc=sc, sets=sets, chem_ids=chem_ids, tau=best_tau, mode=mode, history=hist)


@torch.no_grad()
def _moe_eval(model, t, tau, mode, bs):
    model.eval()
    outs = [model(t["x"][b], t["d"][b], t["text"][b], t["match"][b], tau, mode)
            for b in _batches(len(t["y"]), bs, False)]
    return {k: torch.cat([o[k] for o in outs]) for k in ("pred", "g", "f", "fs", "logits")} | {"beta": outs[0]["beta"]}


def _routing_split(o, held, tau):
    """Router logit gap |l0 - l1| (raw and /tau) and gate entropy, separately for source and held-out val windows."""
    out = {}
    for name, m in (("source", ~held), ("heldout", held)):
        if m.any():
            gap = (o["logits"][m, 0] - o["logits"][m, 1]).abs().mean().item() if o["logits"].shape[1] == 2 else float("nan")
            ent = -(o["g"][m] * torch.log(o["g"][m] + 1e-9)).sum(1).mean().item()
        else:  # zero-shot: no held-out cell in validation
            gap = ent = float("nan")
        out |= {f"logit_gap_{name}": gap, f"logit_gap_over_tau_{name}": gap / tau, f"gate_entropy_{name}": ent}
    return out


def diagnostics(o):
    g = o["g"]
    ent = -(g * torch.log(g + 1e-9)).sum(1).mean().item()
    preds = torch.cat([o["f"], o["fs"][:, None]], 1).T
    c = torch.corrcoef(preds)
    off = c[~torch.eye(len(c), dtype=bool, device=c.device)]
    d = {f"gate_mean_e{e}": g[:, e].mean().item() for e in range(g.shape[1])}
    return d | dict(gate_entropy=ent, beta=o["beta"].item(), max_expert_corr=torch.nan_to_num(off).max().item(),
                    val_pred_min=o["pred"].min().item(), val_pred_max=o["pred"].max().item())


@torch.no_grad()
def predict_moe(fitted, prep, idx, cfg):
    t = _moe_tensors(prep, idx, fitted["sc"], fitted["sets"], fitted["chem_ids"])
    o = _moe_eval(fitted["model"], t, fitted["tau"], fitted["mode"], cfg["batch"])
    return o["pred"].cpu().numpy(), o["g"].cpu().numpy(), t["match"].cpu().numpy(), o
