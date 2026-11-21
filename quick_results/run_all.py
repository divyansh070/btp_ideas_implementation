"""Quick-results proof of concept for Version 1: E0-E4 + REPORT.md.

    python run_all.py --data DATA_DIR --out OUT_DIR [--download] [--use-hust] [--e4] [--steps e0,e1,e2,e3,e4,report]

Each step skips itself if its results CSV already exists in OUT_DIR (delete the CSV to rerun it).
"""
import argparse
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qr import data as qd
from qr import plots
from qr.features import feature_constancy, make_split, prepare, regime_sets, window_idx
from qr.gate import run_gate
from qr.metrics import chem_metrics, md_table, summarise
from qr import train as T

HERE = Path(__file__).parent
STEPS = ["e0", "e1", "e2", "e3", "e4", "report"]


def load_cfg(path, use_hust=False, e4=False):
    cfg = yaml.safe_load(Path(path).read_text())
    if use_hust or cfg["data"]["use_hust"]:
        cfg["data"]["use_hust"] = True
        cfg["data"]["datasets"] = list(dict.fromkeys(cfg["data"]["datasets"] + ["HUST"]))
        for k, v in cfg["expected_counts_hust_extra"].items():
            cfg["expected_counts"][k] = cfg["expected_counts"].get(k, 0) + v
    cfg["e4"]["enabled"] = e4 or cfg["e4"]["enabled"]
    return cfg


def _timed(out, name, fn):
    t0 = time.time()
    r = fn()
    if r == "cached":  # keep the runtime of the run that produced the CSV
        return r
    rt = out / "runtime.json"
    d = json.loads(rt.read_text()) if rt.exists() else {}
    d[name] = round(time.time() - t0, 1)
    rt.write_text(json.dumps(d, indent=1))
    return r


# ---------------------------------------------------------------- E0
def e0(cfg, data_dir, out):
    if (out / "e0_gate.csv").exists() and (out / "prepared.pkl").exists():
        print("E0: cached")
        return "cached"
    ds = cfg["data"]["datasets"]
    cells, kept = qd.load_cells(data_dir, ds, cfg["data"]["min_cycles"])
    text = qd.load_text_embeddings(data_dir)
    prep = prepare(kept, text, cfg["data"]["window"], cfg["data"]["sampled"], cfg["data"]["max_windows_per_cell"])
    gate = run_gate(cells, kept, prep, qd.load_pbt_labels(data_dir, ds), cfg)
    raw = pd.DataFrame([dict(chem=c["chem"], dataset=c["dataset"], drop=c["drop"]) for c in cells])
    tab = raw.groupby(["chem", "dataset"]).agg(raw_cells=("drop", "size"), kept_cells=("drop", lambda s: (s == "").sum()))
    w = pd.DataFrame(prep["cells"]).groupby(["chem", "dataset"]).agg(
        windows=("n_windows", "sum"), nan_windows_dropped=("n_nan_windows", "sum"),
        eol_min=("eol", "min"), eol_median=("eol", "median"), eol_max=("eol", "max"))
    tab.join(w).reset_index().to_csv(out / "e0_cells.csv", index=False)
    pd.DataFrame([dict(name=c["name"], chem=c["chem"], n_cycles=c["n_cycles"], eol=c["eol"],
                       label_source=c["label_source"], drop=c["drop"]) for c in cells]).to_csv(out / "e0_cell_list.csv", index=False)
    with open(out / "prepared.pkl", "wb") as f:
        pickle.dump(prep, f)
    pd.DataFrame(feature_constancy(prep)).to_csv(out / "e0_feature_constancy.csv", index=False)
    pd.DataFrame(gate).to_csv(out / "e0_gate.csv", index=False)
    print(pd.DataFrame(gate)[["check", "value", "passed"]].to_string())


def _prep(out):
    with open(out / "prepared.pkl", "rb") as f:
        return pickle.load(f)


def _require_gate(cfg, out):
    g = pd.read_csv(out / "e0_gate.csv")
    bad = g[~g.passed]
    if len(bad) and not cfg["gate"]["allow_fail"]:
        raise SystemExit("Validation gate FAILED, refusing to train (set gate.allow_fail: true to override):\n"
                         + bad[["check", "value", "detail"]].to_string())


# ---------------------------------------------------------------- E1
def e1(cfg, out):
    if (out / "e1_life.csv").exists():
        print("E1: cached")
        return "cached"
    prep = _prep(out)
    plots.e1_figure(prep, out / "e1_fraction_of_life.png")
    wchem = prep["chem"][prep["cell_idx"]]
    rows = []
    for c in sorted(set(prep["chem"])):
        e = np.array([x["eol"] for x in prep["cells"] if x["chem"] == c])
        r, f = prep["rul"][wchem == c], prep["frac"][wchem == c]
        rows.append(dict(chem=c, cells=len(e), eol_min=int(e.min()), eol_median=int(np.median(e)), eol_max=int(e.max()),
                         rul_p95_cycles=round(float(np.percentile(r, 95))), frac_p95=round(float(np.percentile(f, 95)), 3)))
    pd.DataFrame(rows).to_csv(out / "e1_life.csv", index=False)


# ---------------------------------------------------------------- E2
def e2(cfg, out):
    if (out / "e2_raw.csv").exists():
        print("E2: cached")
        return "cached"
    _require_gate(cfg, out)
    prep, c = _prep(out), cfg["e2"]
    rows = []
    for seed in cfg["seeds"]:
        for held in cfg["folds"]:
            sp = make_split(prep["chem"], held, seed, cfg["split"])
            test = window_idx(prep, sp["test"])
            for regime in c["regimes"]:
                sets = regime_sets(sp, regime)
                if sets is None:
                    continue
                for m in c["models"]:
                    if m == "lgbm":
                        p = T.fit_lgbm(prep, sets, c["lgbm"], seed).predict(prep["X"][test, -1]) * prep["eol"][test]
                        label = "frac"
                    else:
                        label = m.split("_")[1]
                        model, sc = T.fit_single(prep, sets, label, c, seed)
                        p = T.to_cycles(prep, test, T.predict_single(model, sc, prep, test), label, c["rul_scale"])
                    rows.append(dict(fold=held, regime=regime, model=m, seed=seed, oracle_eol=label == "frac",
                                     n_target_train_cells=_n_target(sets[0], sp),
                                     **chem_metrics(prep["rul"][test], p, prep["cell_idx"][test])))
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "e2_raw.csv", index=False)


def _n_target(train_cells, sp):
    return int(np.isin(train_cells, np.concatenate([sp["adapt"], sp["val"], sp["test"]])).sum())


# ---------------------------------------------------------------- E3
def e3(cfg, out):
    if (out / "e3_raw.csv").exists():
        print("E3: cached")
        return "cached"
    _require_gate(cfg, out)
    prep, c = _prep(out), cfg["e3"]
    rows, gates, hist = [], [], []
    for seed in cfg["seeds"]:
        for held in cfg["folds"]:
            sp = make_split(prep["chem"], held, seed, cfg["split"])
            test = window_idx(prep, sp["test"])
            src_val = window_idx(prep, sp["src_val"])
            for regime in c["regimes"]:
                sets = regime_sets(sp, regime)
                if sets is None:
                    continue
                for v in c["variants"]:
                    fit = T.fit_moe(prep, sets, held, v, c, seed)
                    pred, g, match, _ = T.predict_moe(fit, prep, test, c)
                    sel = min(fit["history"], key=lambda h: h["val_mse"])
                    experts = " | ".join("+".join(sorted(s)) for s in fit["sets"])
                    has_match = bool(match.sum(1).max() > 0)
                    rows.append(dict(fold=held, regime=regime, model=f"moe_{v}", seed=seed, oracle_eol=True,
                                     experts=experts, held_out_has_matching_expert=has_match,
                                     selected_epoch=sel["epoch"], **{k: sel[k] for k in sel if k not in ("epoch",)},
                                     **chem_metrics(prep["rul"][test], pred * prep["eol"][test], prep["cell_idx"][test])))
                    gates.append(dict(fold=held, regime=regime, variant=v, seed=seed, set="test", chem=held, experts=experts,
                                      has_match=has_match, **{f"gate_e{e}": float(g[:, e].mean()) for e in range(g.shape[1])}))
                    gv, vchem = (T.predict_moe(fit, prep, src_val, c)[1], prep["chem"][prep["cell_idx"][src_val]]) \
                        if len(src_val) else (None, [])  # empty only in the smoke test
                    for ch in sorted(set(vchem)):
                        gates.append(dict(fold=held, regime=regime, variant=v, seed=seed, set="source_val", chem=ch,
                                          experts=experts, has_match=True,
                                          **{f"gate_e{e}": float(gv[vchem == ch, e].mean()) for e in range(gv.shape[1])}))
                    hist += [dict(fold=held, regime=regime, variant=v, seed=seed, **h) for h in fit["history"]]
                    print({k: rows[-1][k] for k in ("fold", "regime", "model", "seed", "rmse", "beta", "gate_entropy")}, flush=True)
    pd.DataFrame(gates).to_csv(out / "e3_gates.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "e3_history.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "e3_raw.csv", index=False)


# ---------------------------------------------------------------- E4
def e4(cfg, data_dir, out):
    status = out / "e4_status.txt"
    if status.exists():
        print("E4: cached")
        return "cached"
    if not cfg["e4"]["enabled"]:
        status.write_text("skipped: E4 disabled (set RUN_E4=True in the notebook or pass --e4)")
        return
    from qr.pbt_e4 import run_e4
    rows, reason = run_e4(data_dir, _prep(out), cfg, out / "e4_work")
    if rows:
        pd.DataFrame(rows).to_csv(out / "e4_raw.csv", index=False)
    status.write_text(reason)
    print("E4:", reason)


# ---------------------------------------------------------------- report
DIAGRAM = """```
  cycle data of one cell                               text: 10 ageing factors (PBT prompt)
        |                                                         |
  any 30 consecutive cycles, 10 sampled                 frozen Llama-3.1-8B last token
        |                                               (PBT's cached 4096-d vector)
  per cycle: 17 stats + 11 deltas = 28 features                   |
        |                                               linear + LeakyReLU -> 64
  scaler (fit: source train + target adaptation, clip [-0.5,1.5]) |
        |                                    6 dynamics (dQ slope/spread/curvature,
  shared encoder: LSTM(64) x2 -> attention   dV mean, Q ratio, latent displacement)
        |  -> z (128)                                             |
        +--------------------------+               router MLP -> logits / tau
        |                          |                + category bias (ours)  or  x 0/1 mask (PBT)
  experts f_1 (e.g. LFP), f_2 (rest)  shared f_s    -> softmax -> g
        |                          |                              |
        +---- (1-beta) * sum_e g_e f_e(z)  +  beta * f_s(z) <-----+
                         |
              fraction of life in [0,1]  x EOL (ORACLE: true EOL here)  ->  RUL in cycles
```"""


def report(cfg, out):
    L = ["# Quick results: Version 1 proof of concept", ""]
    rt = json.loads((out / "runtime.json").read_text()) if (out / "runtime.json").exists() else {}
    prep = _prep(out)
    L += ["## Setup", f"- Datasets: {', '.join(cfg['data']['datasets'])} (BatteryLife v9 processed pickles, PBT life-label rule).",
          f"- Kept cells: {len(prep['cells'])}; windows: {len(prep['rul'])} "
          f"(30-cycle windows, 10 sampled, stride 1, at most {cfg['data']['max_windows_per_cell']} per cell, evenly spaced).",
          f"- Seeds: {cfg['seeds']} (each seed redraws the cell splits and the model init).",
          f"- Leave-one-chemistry-out folds: {cfg['folds']}; HNEI (NMC-LCO) is always source. Held-out chemistry: "
          "60 % adaptation / 10 % validation / 30 % blind test, cell-disjoint; source 80/20 train/val.",
          "- Selection: best epoch on validation cells only (source val for zero-shot; source val + target val otherwise). "
          "The blind test is only scored.",
          f"- Runtime (s): {rt}; host: {platform.node()} / torch device {T.DEV}.", ""]
    rows = []
    for held in cfg["folds"]:
        sp = make_split(prep["chem"], held, cfg["seeds"][0], cfg["split"])
        rows.append(dict(fold=held, **{k: len(v) for k, v in sp.items()}))
    L += ["Cells per split (seed %d):" % cfg["seeds"][0], "", md_table(pd.DataFrame(rows)), ""]

    L += ["## E0: pipeline and validation gate", "", md_table(pd.read_csv(out / "e0_cells.csv")), ""]
    g = pd.read_csv(out / "e0_gate.csv")
    g["passed"] = g.passed.map({True: "PASS", False: "**FAIL**"})
    L += [md_table(g), "", f"Gate: **{'PASS' if (g.passed == 'PASS').all() else 'FAIL'}**"
          + ("" if (g.passed == "PASS").all() or not cfg["gate"]["allow_fail"] else " (training ran anyway: gate.allow_fail=true)"), ""]

    if (out / "e0_feature_constancy.csv").exists():
        fc = pd.read_csv(out / "e0_feature_constancy.csv").round(4)
        L += ["Feature constancy (raw, unscaled; `frac_cells_constant` = share of cells where the feature does not move "
              "inside the cell). Informational, not a gate check:", "", md_table(fc), ""]

    if (out / "e1_life.csv").exists():
        L += ["## E1: why fraction of life", "", "![E1](e1_fraction_of_life.png)", "",
              md_table(pd.read_csv(out / "e1_life.csv")), "",
              "RUL in cycles spans very different ranges per chemistry; fraction of life puts every chemistry on [0, 1]. "
              "Converting back to cycles needs EOL, which is unknown in service (open issue a).", ""]

    e2r = pd.read_csv(out / "e2_raw.csv") if (out / "e2_raw.csv").exists() else None
    if e2r is not None:
        s = summarise(e2r, ["fold", "regime", "model", "oracle_eol"])
        L += ["## E2: cross-chemistry gap (single-head models)", "",
              "Metrics on the held-out chemistry's blind-test cells only (per chemistry; R² never pooled). "
              "`oracle_eol=True`: fraction of life converted with the TRUE EOL. `lstm_rul` predicts cycles directly (no oracle).", "",
              md_table(s), ""]
        L += _gap_lines(e2r)

    if (out / "e3_raw.csv").exists():
        e3r = pd.read_csv(out / "e3_raw.csv")
        single = e2r[(e2r.model == "lstm_frac") & e2r.regime.isin(cfg["e3"]["regimes"])] if e2r is not None else pd.DataFrame()
        both = pd.concat([single.assign(model="single_head"), e3r], ignore_index=True)
        match = e3r.groupby("fold").held_out_has_matching_expert.first().to_dict()
        both["held_out_has_matching_expert"] = both.fold.map(match)
        L += ["## E3: mini Track B (MoE) vs single head", "", "All rows use oracle EOL.", "",
              md_table(summarise(both, ["fold", "held_out_has_matching_expert", "regime", "model"])), ""]
        diag = e3r.groupby(["fold", "regime", "model"])[["beta", "gate_entropy", "gate_mean_e0", "gate_mean_e1",
                                                         "max_expert_corr", "val_pred_min", "val_pred_max"]].mean().round(3)
        L += ["Diagnostics at the selected epoch (validation windows, mean over seeds):", "", md_table(diag.reset_index()), ""]
        gates = pd.read_csv(out / "e3_gates.csv")
        if plots.e3_gate_figure(gates[gates.set == "test"], out / "e3_gate_weights.png", cfg["e3"]["gate_plot_regime"]):
            L += ["![E3 gates](e3_gate_weights.png)", ""]
        gm = gates.groupby(["fold", "regime", "variant", "set", "chem"])[[c for c in gates if c.startswith("gate_e")]].mean().round(3)
        L += ["Mean gate weight per chemistry:", "", md_table(gm.reset_index()), ""]
        nomatch = [f for f, m in match.items() if not m]
        L += [f"**Folds where the held-out chemistry has no matching expert: {', '.join(nomatch) or 'none'}.** "
              "There the hard mask gives the shared expert alone; the additive bias keeps a blend.", ""]
        L += _moe_vs_single(both, nomatch)

    status = (out / "e4_status.txt").read_text() if (out / "e4_status.txt").exists() else "not run"
    L += ["## E4: released PBT, zero-shot", "", f"Status: {status}", ""]
    if (out / "e4_raw.csv").exists():
        e4r = pd.read_csv(out / "e4_raw.csv")
        L += [md_table(summarise(e4r, ["fold", "regime", "model", "test_cells_in_PBT_pretraining"])), "",
              "PBT reads cycles 1-100 and predicts total life; RUL = predicted life - current cycle, scored on windows "
              "ending at cycle >= 100. Not history-independent. Test cells that were in PBT's pretraining split are a leak.", ""]

    L += ["## How Version 1 works end to end", "", DIAGRAM, ""]
    L += ["## What this shows", "",
          "- The shared pipeline runs end to end on real SNL / CALCE / HNEI cells with the PBT life-label rule, "
          "and every gate check is reported with numbers (E0).",
          "- Whether a leave-one-chemistry-out gap exists for single-head models, measured as zero-shot vs few-shot vs "
          "target-only on the same blind-test cells (E2, see the gap lines above).",
          "- How the hard mask and the additive bias route an unseen chemistry, with gate weights logged (E3).", "",
          "## What this does NOT show", "",
          f"- Anything statistically solid: tiny N (held-out test sets of 3-6 cells, {len(cfg['seeds'])} seeds). Preliminary only.",
          "- Real-world RUL accuracy for fraction-of-life models: every `oracle_eol=True` number uses the TRUE EOL to "
          "convert to cycles. That is a leak in deployment (CLAUDE.md, open issue a).",
          "- That PBT has or lacks a gap: the released PBT was pretrained on LFP/NMC/NCA/LCO cells, possibly these very "
          "cells, so E4 is optimistic, and Track A (re-headed, fine-tuned PBT) was not run here.",
          "- The full Track B model: no Neural ODE, E=2, few epochs, no hyperparameter search, windows subsampled per cell.",
          "- Any conclusion beyond these folds: the MoE result above is reported as measured, whichever way it goes.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    print("wrote", out / "REPORT.md")


def _gap_lines(e2r):
    m = e2r[e2r.model == "lstm_frac"].groupby(["fold", "regime"]).rmse.mean().unstack()
    out = ["Gap, `lstm_frac` mean RMSE (cycles, oracle EOL):", ""]
    for fold, r in m.iterrows():
        parts = [f"{k} {r[k]:.0f}" for k in m.columns if pd.notna(r.get(k))]
        out.append(f"- {fold}: " + ", ".join(parts))
    return out + [""]


def _moe_vs_single(both, nomatch):
    m = both.groupby(["fold", "regime", "model"]).rmse.mean().unstack()
    out = []
    for v in [c for c in m.columns if c.startswith("moe_")]:
        if "single_head" not in m:
            break
        ok = m[[v, "single_head"]].dropna()
        wins = int((ok[v] < ok["single_head"]).sum())
        nm = ok[ok.index.get_level_values(0).isin(nomatch)]
        out.append(f"- `{v}` has lower mean RMSE than the single head in {wins} of {len(ok)} fold x regime cells "
                   f"({int((nm[v] < nm['single_head']).sum())} of {len(nm)} in no-matching-expert folds).")
    if out:
        out.append("- These are mean RMSE comparisons over a few seeds on 3-6 test cells; differences inside one std "
                   "(table above) are not evidence either way.")
    return out + [""]


# ---------------------------------------------------------------- main
def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs" / "quick.yaml"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--download", action="store_true", help="fetch data first (Colab only)")
    ap.add_argument("--use-hust", action="store_true")
    ap.add_argument("--e4", action="store_true")
    ap.add_argument("--allow-gate-fail", action="store_true", help="train even if a gate check fails (report shows FAIL)")
    a = ap.parse_args(argv)
    cfg = cfg or load_cfg(a.config, a.use_hust, a.e4)
    cfg["gate"]["allow_fail"] = cfg["gate"]["allow_fail"] or a.allow_gate_fail
    data_dir, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if a.download:
        qd.download(data_dir, cfg["data"]["datasets"], pbt_ckpt=cfg["e4"]["enabled"])
    steps = a.steps.split(",")
    for s in STEPS:
        if s not in steps:
            continue
        fn = {"e0": lambda: e0(cfg, data_dir, out), "e1": lambda: e1(cfg, out), "e2": lambda: e2(cfg, out),
              "e3": lambda: e3(cfg, out), "e4": lambda: e4(cfg, data_dir, out), "report": lambda: report(cfg, out)}[s]
        _timed(out, s, fn)


if __name__ == "__main__":
    main()
