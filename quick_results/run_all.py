"""Quick-results proof of concept for Version 1: E0-E4 + REPORT.md.

    python run_all.py --data DATA_DIR --out OUT_DIR [--download] [--use-hust] [--e4] [--steps e0,e1,e2,e3,e4,report]

Each step skips itself if its results CSV already exists in OUT_DIR (delete the CSV to rerun it).
"""
import argparse
import json
import pickle
import platform
import re
import subprocess
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
STEPS = ["e0", "e1", "e2", "e3", "e4", "e5", "report"]


def load_cfg(path, use_hust=False, e4=False):
    cfg = yaml.safe_load(Path(path).read_text())
    if use_hust or cfg["data"]["use_hust"]:
        cfg["data"]["use_hust"] = True
        cfg["data"]["datasets"] = list(dict.fromkeys(cfg["data"]["datasets"] + ["HUST"]))
        for k, v in cfg["expected_counts_hust_extra"].items():
            cfg["expected_counts"][k] = cfg["expected_counts"].get(k, 0) + v
    cfg["e4"]["enabled"] = e4 or cfg["e4"]["enabled"]
    if not cfg["e4"].get("pbt_repo"):
        cfg["e4"]["pbt_repo"] = str(HERE.parent / "third_party" / "PBT")  # gitignored; absent on Colab -> clone
    return cfg


def _cached(cfg, step, *paths):
    """A step is skipped only if all its outputs exist and it was not named in --force."""
    return step not in cfg["run"]["force"] and all(p.exists() for p in paths)


def _timed(out, name, fn):
    t0 = time.time()
    r = fn()
    rt = out / "runtime.json"
    d = json.loads(rt.read_text()) if rt.exists() else {}
    d[name] = "cached" if r == "cached" else round(time.time() - t0, 1)  # this invocation only
    rt.write_text(json.dumps(d, indent=1))
    return r


# ---------------------------------------------------------------- E0
def e0(cfg, data_dir, out):
    if _cached(cfg, "e0", out / "e0_gate.csv", out / "prepared.pkl"):
        print("E0: cached")
        return "cached"
    ds = cfg["data"]["datasets"]
    cells, kept = qd.load_cells(data_dir, ds, cfg["data"]["min_cycles"])
    text = qd.load_text_embeddings(data_dir)
    no_text = [c for c in kept if c["name"] not in text]
    if no_text and cfg["data"]["text_embedding_missing"] == "drop":
        # PBT only embedded cells in its own splits (e.g. SNL NCA 20-80 _a/_b); never zero-fill like PBT does
        print(f"WARNING: dropping {len(no_text)} cells with no prompt embedding: {[c['name'] for c in no_text]}", flush=True)
        for c in no_text:
            c["drop"] = "no_prompt_embedding"
        kept = [c for c in kept if c["drop"] == ""]
    prep = prepare(kept, text, cfg["data"]["window"], cfg["data"]["sampled"], cfg["data"]["max_windows_per_cell"])  # mode error: raises here
    gate = run_gate(cells, kept, prep, qd.load_pbt_labels(data_dir, ds), cfg)
    dropped_text = [c["name"] for c in cells if c["drop"] == "no_prompt_embedding"]
    gate.insert(0, dict(check="cells dropped: no prompt embedding", value=len(dropped_text), passed=True,
                        detail=", ".join(dropped_text)))
    pd.DataFrame([dict(name=c["name"], chem=c["chem"], dataset=c["dataset"], reason=c["drop"])
                  for c in cells if c["drop"]], columns=["name", "chem", "dataset", "reason"]).to_csv(out / "e0_dropped_cells.csv", index=False)
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
    if _cached(cfg, "e1", out / "e1_life.csv"):
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
    if _cached(cfg, "e2", out / "e2_raw.csv"):
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
    if _cached(cfg, "e3", out / "e3_raw.csv"):
        print("E3: cached")
        return "cached"
    _require_gate(cfg, out)
    prep, c = _prep(out), cfg["e3"]
    rows, gates, hist = [], [], []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
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
                                     code_commit=cfg["run"]["code_commit"], e3_started=started,
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
E4_ADAPTER = None  # tests inject a fake PBT adapter here
E4_FILES = ("e4_raw.csv", "e4_cells.csv", "e4_finetune_history.csv", "e4_notes.json")


def e4(cfg, data_dir, out):
    status = out / "e4_status.txt"
    if _cached(cfg, "e4", status):
        print("E4: cached")
        return "cached"
    for f in E4_FILES:  # never let a previous E4 result survive next to a new status
        (out / f).unlink(missing_ok=True)
    if not cfg["e4"]["enabled"]:
        status.write_text("skipped: E4 disabled (set RUN_E4=True in the notebook or pass --e4)")
        return
    _require_gate(cfg, out)
    from qr import pbt_e4
    prep, started, step = _prep(out), time.strftime("%Y-%m-%dT%H:%M:%S%z"), "build adapter"
    try:
        adapter, reason = (E4_ADAPTER, "") if E4_ADAPTER is not None else pbt_e4.build_adapter(
            data_dir, prep, cfg, out / "e4_work")
        if adapter is None:
            status.write_text(reason)
            print("E4:", reason)
            return
        step = "run"
        rows, cells, hist, notes = pbt_e4.run_e4(prep, cfg, adapter)
    except Exception as e:  # report the reason, never crash the whole run
        import traceback
        status.write_text(f"failed at step '{step}': {type(e).__name__}: {str(e)[:400]}")
        print(traceback.format_exc())
        return
    stamp = dict(code_commit=cfg["run"]["code_commit"], e4_started=started)
    pd.DataFrame([r | stamp for r in rows]).to_csv(out / "e4_raw.csv", index=False)
    pd.DataFrame(cells).to_csv(out / "e4_cells.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "e4_finetune_history.csv", index=False)
    (out / "e4_notes.json").write_text(json.dumps(notes | stamp | {"adapter": getattr(adapter, "info", {})}, indent=1))
    status.write_text(f"ran: {len(rows)} result rows, adapter {getattr(adapter, 'info', {})}")
    print("E4:", status.read_text())


# ---------------------------------------------------------------- E5
def e5(cfg, out):
    if _cached(cfg, "e5", out / "e5_raw.csv"):
        print("E5: cached")
        return "cached"
    _require_gate(cfg, out)
    from qr.e5 import run_e5
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows, notes = run_e5(_prep(out), cfg)
    stamp = dict(code_commit=cfg["run"]["code_commit"], e5_started=started)
    (out / "e5_notes.json").write_text(json.dumps(notes | stamp, indent=1))
    pd.DataFrame([r | stamp for r in rows]).to_csv(out / "e5_raw.csv", index=False)


# ---------------------------------------------------------------- report
DIAGRAM = """```
  cycle data of one cell                               text: 10 ageing factors (PBT prompt)
        |                                                         |
  any 30 consecutive cycles, 10 sampled                 frozen Llama-3.1-8B last token
        |                                               (PBT's cached 4096-d vector)
  per cycle: 17 stats + 8 deltas = 25 features                    |
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
          f"- Runtime (s): {rt}; host: {platform.node()} / torch device {T.DEV}; code commit "
          f"{cfg.get('run', {}).get('code_commit', 'unknown')}.", ""]
    if (out / "e0_dropped_cells.csv").exists():
        dr = pd.read_csv(out / "e0_dropped_cells.csv")
        by = "; ".join(f"{n} {r}" for r, n in dr.reason.value_counts().items()) if len(dr) else "none"
        L += [f"Dropped cells: {len(dr)} ({by}). Missing prompt embeddings policy: "
              f"`{cfg['data']['text_embedding_missing']}`.", "",
              md_table(dr) if len(dr) else "none", ""]
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
        cols = ["beta", "gate_entropy", "gate_entropy_source", "gate_entropy_heldout", "logit_gap_over_tau_source",
                "gate_mean_e0", "gate_mean_e1", "max_expert_corr", "val_pred_min", "val_pred_max", "text_norm_raw_mean"]
        diag = e3r.groupby(["fold", "regime", "model"])[[c for c in cols if c in e3r]].mean().round(3)
        L += ["Diagnostics at the selected epoch (validation windows, mean over seeds):", "", md_table(diag.reset_index()), ""]
        gates = pd.read_csv(out / "e3_gates.csv")
        if plots.e3_gate_figure(gates[gates.set == "test"], out / "e3_gate_weights.png", cfg["e3"]["gate_plot_regime"]):
            L += ["![E3 gates](e3_gate_weights.png)", ""]
        gm = gates.groupby(["fold", "regime", "variant", "set", "chem"])[[c for c in gates if c.startswith("gate_e")]].mean().round(3)
        L += ["Mean gate weight per chemistry:", "", md_table(gm.reset_index()), ""]
        nomatch = [f for f, m in match.items() if not m]
        L += [f"**Folds where the held-out chemistry has no matching expert: {', '.join(nomatch) or 'none'}.** "
              "There the hard mask gives the shared expert alone (prediction = f_shared, beta not applied); "
              "the additive bias keeps a blend.", "",
              "How `held_out_has_matching_expert` is decided (`qr/train.py` expert_sets / match_matrix): each fold has "
              "E = 2 experts built from that fold's source chemistries (LFP vs rest; NMC vs rest when LFP is held out). "
              "A chemistry's cathode components are its name split on '-' (NMC-LCO -> {NMC, LCO}). A cell matches an "
              "expert when its components overlap the union of that expert's components. HNEI (NMC-LCO) is always a "
              "source cell, so it puts NMC and LCO into the 'rest' expert: held-out NMC and LCO match it (partial "
              "match through NMC-LCO), while held-out NCA and LFP share no component with any expert.", "",
              md_table(_match_table(prep, cfg["folds"])), ""]
        L += _moe_vs_single(both, nomatch)
        L += _run1_vs_run2(summarise(both, ["fold", "regime", "model"]), e3r)

    L += _e4_section(cfg, out, e2r)
    L += _e5_section(cfg, out, e2r)

    L += ["## How Version 1 works end to end", "", DIAGRAM, ""]
    L += ["## What this shows", "",
          f"- The shared pipeline runs end to end on real {' / '.join(cfg['data']['datasets'])} cells with the PBT life-label "
          "rule; every gate check is reported with its number (E0 gate table).",
          "- Whether single-head models show a leave-one-chemistry-out gap: in the E2 table, compare the `zero_shot`, "
          "`few_2` / `few_5` and `target_only` rows of each fold (same blind-test cells), and the E2 'Gap' summary for `lstm_frac`.",
          "- How the hard mask and the additive bias route an unseen chemistry: E3 results table (MoE vs `single_head` per fold "
          "and regime, with `held_out_has_matching_expert`), the E3 diagnostics table and the mean-gate-weight-per-chemistry table.",
          *_pbt_shows(cfg, out, e2r), "",
          "## What this does NOT show", "",
          f"- Anything statistically solid: tiny N (held-out test sets of 3-6 cells, {len(cfg['seeds'])} seeds). Preliminary only.",
          "- Real-world RUL accuracy for fraction-of-life models: every `oracle_eol=True` number uses the TRUE EOL to "
          "convert to cycles. That is a leak in deployment (CLAUDE.md, open issue a).",
          "- An unbiased PBT number: the released PBT was pretrained on LFP, NMC, NCA and LCO cells (13 Li-ion datasets "
          "incl. SNL, CALCE, HNEI, HUST), often these very cells (E4 leakage table), so every E4 number is optimistic. "
          "Leave-one-chemistry-out for PBT is only honest after re-pretraining without the held-out chemistry.",
          "- The guide's Track A: E4 uses PBT's own task (total life from the first <= 100 cycles, RUL = life - cycle), "
          "not a re-headed windowed-RUL PBT (A3); E4b is our reimplementation of PBT's FT recipe, not finetune_model.py, "
          "with 10 epochs and no hyperparameter search.",
          "- The full Track B model: no Neural ODE, E=2, few epochs, no hyperparameter search, windows subsampled per cell.",
          "- Any conclusion beyond these folds: the MoE result above is reported as measured, whichever way it goes.",
          "- That text helps in general (E5): the embedding is PBT's fixed prompt per cell; with 4 folds x few seeds a "
          "CI that excludes 0 is evidence at this scale only, and the shuffled-text control (D) is what separates "
          "'the description is informative' from 'any extra per-cell vector helps'.", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    print("wrote", out / "REPORT.md")


def _e5_section(cfg, out, e2r):
    if not (out / "e5_raw.csv").exists():
        return []
    from qr.e5 import paired_table
    e5r = pd.read_csv(out / "e5_raw.csv")
    n = json.loads((out / "e5_notes.json").read_text())
    rt = json.loads((out / "runtime.json").read_text()).get("e5", "n/a") if (out / "runtime.json").exists() else "n/a"
    L = ["## E5: does a text description of the cell improve cross-chemistry RUL?", "",
         f"Code commit **{n['code_commit']}**, started {n['e5_started']}; runtime {rt} s; seeds {n['seeds_done']}"
         + (" (**reduced to fit the time budget**)" if n["seeds_reduced"] else "")
         + (f"; lstm_rul variants **dropped from seed {n['rul_dropped_from_seed']}** (budget)" if n["rul_dropped_from_seed"]
            is not None else "") + ".", "",
         "Same folds, splits, training and E2 lstm_frac model; only the extra input differs (Linear + LeakyReLU -> 32, "
         "concatenated to z). A none; B one-hot of source chemistries (held-out cells all zeros); C PBT Llama prompt "
         "embedding (unit length); D shuffled-text control (another cell's embedding, fixed derangement per seed); "
         "E text + 6 dynamics. Same seed = same cells for every variant (paired). `label=frac` rows use **oracle EOL**; "
         "`label=rul` rows (A, C only) predict cycles directly, comparable with E4's PBT numbers.", ""]
    if e2r is not None:
        a = e5r[(e5r.variant == "A_none") & (e5r.label == "frac")].merge(
            e2r[e2r.model == "lstm_frac"], on=["fold", "regime", "seed"], suffixes=("", "_e2"))
        if len(a):
            L += [f"Sanity: A_none reproduces E2 lstm_frac on the {len(a)} shared fold x regime x seed cells, "
                  f"max |RMSE difference| = {float((a.rmse - a.rmse_e2).abs().max()):.3g} cycles.", ""]
    L += [md_table(summarise(e5r, ["fold", "regime", "label", "variant", "oracle_eol"])), ""]
    pt = paired_table(e5r, cfg)
    if len(pt):
        L += ["Paired comparisons on RMSE (cycles): diff = RMSE(first) - RMSE(second), negative = first is better; "
              f"wins = pairs where the first is better; 95 % CI = bootstrap over pairs ({cfg['e5']['bootstrap']} draws). "
              "'supported' only if the whole CI is below 0.", "", md_table(pt), ""]
        L += [f"- {r.comparison} ({r.label}, {r.regime}, folds {r.folds}): {r.wins}/{r.pairs} wins, mean diff "
              f"{r.mean_rmse_diff:+.1f} [{r.ci95_low:+.1f}, {r.ci95_high:+.1f}] -> **{r.verdict}**." for r in pt.itertuples()]
    return L + [""]


def _e4_section(cfg, out, e2r):
    status = (out / "e4_status.txt").read_text() if (out / "e4_status.txt").exists() else "not run"
    rt = json.loads((out / "runtime.json").read_text()).get("e4", "n/a") if (out / "runtime.json").exists() else "n/a"
    L = ["## E4: released PBT on our LOCO protocol", "", f"Status: {status}", f"Runtime: {rt} s", ""]
    if not (out / "e4_raw.csv").exists():
        return L
    e4r = pd.read_csv(out / "e4_raw.csv")
    notes = json.loads((out / "e4_notes.json").read_text())
    L += [f"Code commit **{notes['code_commit']}**, started {notes['e4_started']}; adapter {notes['adapter']}. "
          f"E4a {notes['e4a_seconds']} s, E4b {notes['e4b_seconds']} s, E4b seeds {notes['e4b_seeds']}"
          + (" (**reduced to fit the time budget**)" if notes["e4b_seeds_reduced"] else "") + ".",
          f"Cells PBT could not use: {notes['skipped_cells'] or 'none'}.", "",
          "PBT predicts total cycle life from the first N <= 100 cycles. For a window ending at cycle c it gets the "
          "first min(c, 100) cycles and RUL = predicted life - c (no oracle EOL, no future cycles). "
          "Same blind-test cells as E2/E3. On the same 2 / 5 adaptation cells as E2 few_2 / few_5: `PBT_FT` = all "
          "parameters (lr 1e-5), `PBT_AT` = PBT's adapters (width 16) + head; best epoch by target-val MAPE, where "
          "epoch -1 (no update; for FT the pretrained model) is a candidate. `PBT_selected` = whichever of zero-shot / "
          "FT / AT has the lowest target-val MAPE, per fold x regime x seed (see `qr/pbt_e4.py`).", ""]
    cells = pd.read_csv(out / "e4_cells.csv")
    leak = (cells.drop_duplicates(["fold", "cell"]).groupby(["fold", "pbt_split"]).size().unstack(fill_value=0)
            .reindex(columns=["train", "val", "test", "none"], fill_value=0).reset_index())
    L += ["**Leakage.** Released PBT was pretrained on all four held-out chemistries (LFP, NMC, NCA, LCO are all in "
          "its 13 pretraining datasets). Blind-test cells by their split in PBT's own data_split_recorder "
          "(unique cells over seeds; train/val = PBT trained or selected on them):", "", md_table(leak), "",
          "`pbt_test_split_only` rows score only test cells that were in PBT's test split (never trained on).", ""]
    e4r["subset"] = e4r.subset.fillna("all_test_cells")
    regs = list(dict.fromkeys(["zero_shot"] + cfg["e4"]["finetune"]["regimes"] + ["target_only"]))
    ref = e2r[e2r.regime.isin(regs)].assign(subset="all_test_cells") if e2r is not None else pd.DataFrame()
    both = pd.concat([e4r, ref], ignore_index=True)
    both = both.sort_values(["fold", "regime"], key=lambda c: c.map({r: i for i, r in enumerate(regs)}) if c.name == "regime" else c)
    tab = summarise(both, ["fold", "regime", "model", "subset", "oracle_eol"])
    L += ["E4 next to E2 on the same folds, regimes and test cells:", "", md_table(tab), ""]
    if "chosen" in e4r:
        ch = e4r[e4r.model == "PBT_selected"].groupby(["fold", "regime", "chosen"]).size().unstack(fill_value=0)
        L += ["Which PBT variant target-val selected (count over seeds):", "", md_table(ch.reset_index()), ""]
    seen = (cells.drop_duplicates(["fold", "cell"]).assign(seen=lambda d: d.pbt_split.isin(["train", "val"]))
            .groupby("fold").seen.mean())
    L += ["**Track A gate (per fold): does the best PBT variant still leave a gap vs target-only?** PBT_selected "
          "(chosen on target val among zero-shot / FT / AT) with the most adaptation cells available, vs the "
          "**non-oracle** E2 target_only `lstm_rul` (main comparison; same no-oracle footing as PBT), mean RMSE ± std "
          "over seeds. target_only trains on the full 60 % adaptation set. The best oracle-EOL target_only model is "
          "shown for reference. Leakage = share of the fold's test cells that PBT trained or selected on.", ""]
    for fold in cfg["folds"]:
        L.append(f"- {fold}: " + _gate_verdict(e4r[e4r.fold == fold], ref[ref.fold == fold] if len(ref) else ref,
                                               float(seen.get(fold, np.nan))))
    return L + [""]


def _pbt_shows(cfg, out, e2r):
    """'What this shows' line on released PBT vs our models, computed from the E4 / E2 / E5 tables."""
    if not (out / "e4_raw.csv").exists() or e2r is None:
        return []
    e4r = pd.read_csv(out / "e4_raw.csv")
    zs = e4r[(e4r.model == "PBT_zero_shot") & (e4r.get("subset", "all_test_cells") == "all_test_cells")]
    ours = [e2r]
    if (out / "e5_raw.csv").exists():
        ours.append(pd.read_csv(out / "e5_raw.csv").assign(model=lambda d: d.variant + "/" + d.label))
    ours = pd.concat(ours, ignore_index=True).groupby(["fold", "regime", "model"]).rmse.mean().groupby("fold").min()
    pbt = zs.groupby("fold").rmse.mean()
    folds = [f for f in cfg["folds"] if f in pbt and f in ours]
    wins = [f for f in folds if pbt[f] < ours[f]]
    seen = ""
    if (out / "e4_cells.csv").exists():
        c = pd.read_csv(out / "e4_cells.csv").drop_duplicates(["fold", "cell"])
        seen = f" {c.pbt_split.isin(['train', 'val']).mean():.0%} of these test cells were in PBT's train/val split."
    return [f"- Released PBT zero-shot has a lower mean RMSE than every one of our models (E2/E5, any regime, incl. "
            f"target-only and oracle-EOL ones) on {len(wins)} of {len(folds)} folds ({', '.join(wins) or 'none'})."
            + (" These chemistries are all seen in PBT's pretraining, so this is largely leakage:" if wins else
               " All four chemistries are in PBT's pretraining data:") + seen +
            " PBT also gets more information than our window models: the first min(c, 100) cycles of the cell "
            "(early-life history) and the current cycle index c (RUL = predicted life - c)."]


def _gate_verdict(e4f, e2f, leak):
    lk = f"leakage {leak:.0%}" if np.isfinite(leak) else "leakage n/a"
    sel = e4f[e4f.model == "PBT_selected"]
    if sel.empty or e2f.empty or not (e2f.regime == "target_only").any():
        return f"{lk}; not decidable (no PBT_selected or no E2 target_only rows)."
    reg = max(set(sel.regime), key=lambda r: int(r.split("_")[1]))   # most adaptation cells
    sel = sel[sel.regime == reg]
    to = e2f[e2f.regime == "target_only"].groupby("model").rmse.agg(["mean", "std"]).fillna(0)
    if "lstm_rul" not in to.index:
        return f"{lk}; not decidable (no non-oracle lstm_rul target_only rows)."
    m, sd = sel.rmse.mean(), (sel.rmse.std() if len(sel) > 1 else 0.0)
    b, bs = to.loc["lstm_rul", "mean"], to.loc["lstm_rul", "std"]
    oracle = to.drop(index="lstm_rul")
    ref = (f"; oracle-EOL target_only {oracle['mean'].idxmin()} {oracle['mean'].min():.0f}" if len(oracle) else "")
    chosen = ", ".join(f"{k} {v}" for k, v in sel.chosen.value_counts().items()) if "chosen" in sel else "n/a"
    nums = (f"PBT_selected {reg} RMSE {m:.0f} ± {sd:.0f} (chosen: {chosen}) vs non-oracle target_only lstm_rul "
            f"{b:.0f} ± {bs:.0f}{ref}")
    if np.isfinite(leak) and leak > 0.5:
        return f"{lk}; {nums} -> **gate not testable on this fold: PBT pretrained on these cells**."
    verdict = ("gap remains (PBT worse by more than 1 std each)" if m - sd > b + bs else
               "no gap (PBT better by more than 1 std each)" if m + sd < b - bs else "within noise")
    return f"{lk}; {nums} -> **{verdict}**{' (single seed)' if len(sel) < 2 else ''}."


def _match_table(prep, folds):
    rows = []
    for held in folds:
        sets = T.expert_sets(sorted(set(prep["chem"]) - {held}))
        comps = [sorted(set().union(*map(T.components, st))) for st in sets]
        hit = [e for e, c in enumerate(comps) if T.components(held) & set(c)]
        rows.append(dict(fold=held, experts=" ; ".join("+".join(sorted(st)) for st in sets),
                         expert_components=" ; ".join("+".join(c) for c in comps),
                         held_out_components="+".join(sorted(T.components(held))),
                         matching_expert=", ".join(f"e{e}" for e in hit) or "none"))
    return pd.DataFrame(rows)


RUN1 = HERE / "outputs_run1" / "REPORT.md"


def _md_section_table(path, heading):
    """First markdown table after `heading` in a previous REPORT.md, as a DataFrame of strings."""
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(heading))
    rows = []
    for l in lines[i + 1:]:
        if l.startswith("|"):
            rows.append([c.strip() for c in l.strip("|").split("|")])
        elif rows:
            break
    return pd.DataFrame(rows[2:], columns=rows[0])


RUN1_COMMIT = "d7588be"  # run 1 predates commit stamping; inferred in document.md (25-feature diagram, no match table)


def _e3_provenance(e3r):
    if "code_commit" not in e3r:
        return None, None
    return ", ".join(sorted(set(e3r.code_commit.astype(str)))), ", ".join(sorted(set(e3r.e3_started.astype(str))))


def _run1_vs_run2(run2, e3r):
    commit, started = _e3_provenance(e3r)
    head = [f"E3 results in this report: code commit **{commit or 'not recorded'}**, started {started or 'not recorded'}.", ""]
    if not RUN1.exists():
        return head
    m1 = re.search(r"code commit ([0-9a-f]{7,})", RUN1.read_text())
    run1_commit = m1.group(1) if m1 else RUN1_COMMIT
    try:
        run1 = _md_section_table(RUN1, "## E3")
    except (StopIteration, ValueError) as e:
        return head + [f"Run-1 comparison skipped: could not read the E3 table in {RUN1.name} ({e}).", ""]
    keys, met = ["fold", "regime", "model"], ["rmse", "r2", "mape50"]
    m = run1[keys + met].merge(run2[keys + met], on=keys, how="outer", suffixes=(" run1", " this"))
    moe = m[m.model.str.startswith("moe_")]
    identical = len(moe) > 0 and all((moe[f"{c} run1"] == moe[f"{c} this"]).all() for c in met)
    refuse = (None if commit and commit != run1_commit and not identical else
              "E3 results carry no code-commit stamp, so they predate this check and may be cached from run 1"
              if not commit else
              f"E3 results come from commit {commit}, the same as run 1 ({run1_commit})"
              if commit == run1_commit else
              "every MoE number equals run 1 to the last digit, which means E3 was almost certainly skipped as cached")
    if refuse:
        return head + [f"**Not labelled as run 2: {refuse}. Rerun with `--force e3`.**", ""]
    m = m[keys + [f"{c} {r}" for c in met for r in ("run1", "this")]].fillna("n/a")
    m.columns = [c.replace(" this", " run2") for c in m.columns]
    return head + ["### E3 run 1 vs run 2", "",
                   f"Run 1: commit {run1_commit} (`outputs_run1/REPORT.md`). Run 2: commit {commit}. Run 2 differs from "
                   "run 1 only by the unit-length normalisation of the text embedding at the router input "
                   "(`qr/models.py`, MoE.forward); same prepared.pkl, config, seeds and splits. `single_head` rows come "
                   "from E2 and are identical by construction.", "", md_table(m), ""]


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
    ap.add_argument("--force", default="", help="comma-separated steps to rerun even if their outputs exist, e.g. e3")
    a = ap.parse_args(argv)
    cfg = cfg or load_cfg(a.config, a.use_hust, a.e4)
    cfg["gate"]["allow_fail"] = cfg["gate"]["allow_fail"] or a.allow_gate_fail
    data_dir, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    force = [f for f in a.force.split(",") if f]
    bad = set(force) - set(STEPS)
    if bad:
        raise SystemExit(f"--force: unknown step(s) {sorted(bad)}; choose from {STEPS}")
    cfg["run"] = dict(code_commit=commit, force=force, invoked=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if a.download:
        qd.download(data_dir, cfg["data"]["datasets"], pbt_ckpt=cfg["e4"]["enabled"])
    steps = a.steps.split(",")
    for s in STEPS:
        if s not in steps:
            continue
        fn = {"e0": lambda: e0(cfg, data_dir, out), "e1": lambda: e1(cfg, out), "e2": lambda: e2(cfg, out),
              "e3": lambda: e3(cfg, out), "e4": lambda: e4(cfg, data_dir, out), "e5": lambda: e5(cfg, out),
              "report": lambda: report(cfg, out)}[s]
        _timed(out, s, fn)


if __name__ == "__main__":
    main()
