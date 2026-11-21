"""Synthetic end-to-end smoke test: 6 fake cells, 3 chemistries, CPU, < 1 minute. No downloads."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
# macOS only: torch and lightgbm load two different libomp runtimes and segfault in one process. Colab (Linux) is
# unaffected, so LightGBM is exercised only off macOS.
MAC = sys.platform == "darwin"
sys.path.insert(0, str(ROOT))

import run_all  # noqa: E402
from qr import data as qd  # noqa: E402


def test_eol_rule_branches():
    c = np.arange(1, 301)
    assert qd.eol_pbt(1 - 0.001 * c) == (200, "threshold")         # crosses 0.80 at cycle 200
    eol, src = qd.eol_pbt(np.linspace(1.0, 0.81, 300))              # stops in (0.80, 0.825): extrapolate
    assert src == "extrapolated" and 300 < eol < 340
    assert qd.eol_pbt(np.linspace(1.0, 0.9, 300))[0] is None        # never reaches 0.825: dropped


def smoke_cfg():
    cfg = run_all.load_cfg(ROOT / "configs" / "quick.yaml")
    cfg["seeds"] = [0]
    cfg["data"]["datasets"] = ["SNL"]
    cfg["data"]["max_windows_per_cell"] = 30
    cfg["expected_counts"] = {"LFP": 2, "NMC": 2, "NCA": 2}
    cfg["folds"] = ["LFP", "NMC", "NCA"]
    cfg["gate"]["allow_fail"] = True       # 1 target test cell per fold: ranges are not meaningful here
    cfg["e2"].update(regimes=["target_only", "zero_shot", "few_1"], epochs=2,
                     models=["lstm_frac", "lstm_rul"] + ([] if MAC else ["lgbm"]))
    cfg["e3"].update(regimes=["zero_shot", "few_1"], epochs_a=1, epochs_b=2, gate_plot_regime="zero_shot")
    cfg["e5"].update(seeds=[0, 1], regimes=["zero_shot", "few_1"], bootstrap=200)
    return cfg


def test_end_to_end(tmp_path):
    t0 = time.time()
    data, out = tmp_path / "data", tmp_path / "out"
    qd.make_synthetic(data)
    run_all.main(["--data", str(data), "--out", str(out)], cfg=smoke_cfg())

    gate = pd.read_csv(out / "e0_gate.csv").set_index("check")
    for k in ["count LFP", "count NMC", "count NCA", "EOL fallbacks",
              "EOL vs PBT labels (mismatch or dropped-but-labelled)", "per-cell corr(RUL, capacity) > 0.5"]:
        assert gate.loc[k, "passed"], (k, gate.loc[k].to_dict())
    cells = pd.read_csv(out / "e0_cell_list.csv")
    assert set(cells.label_source) == {"threshold", "extrapolated"} and (cells["drop"].isna()).all()

    e2 = pd.read_csv(out / "e2_raw.csv")
    assert set(e2.model) == {"lstm_frac", "lstm_rul"} | (set() if MAC else {"lgbm"}) and np.isfinite(e2.rmse).all()
    assert e2[e2.model == "lstm_rul"].oracle_eol.eq(False).all() and e2[e2.model != "lstm_rul"].oracle_eol.all()
    e3 = pd.read_csv(out / "e3_raw.csv")
    assert set(e3.model) == {"moe_mask", "moe_bias"} and np.isfinite(e3.rmse).all()
    hist = pd.read_csv(out / "e3_history.csv")
    for col in ["route_w", "mmd_w", "bias", "tau", "logit_gap_source", "gate_entropy_source", "gate_entropy_heldout",
                "text_norm_raw_mean"]:
        assert col in hist, col
    last = hist.groupby(["fold", "regime", "variant", "seed"]).route_w.last()
    assert (last == 0).all() and (hist.groupby(["fold", "regime", "variant", "seed"]).route_w.first() > 0).all()
    # with sets LFP|rest, held-out NCA / LFP have no matching expert; NMC folds keep one via NMC-LCO only in real data
    assert not e3[e3.fold == "NCA"].held_out_has_matching_expert.any()
    e5 = pd.read_csv(out / "e5_raw.csv")
    assert set(e5.variant) == {"A_none", "B_onehot", "C_text", "D_shuffled", "E_text_dyn"} and set(e5.seed) == {0, 1}
    assert set(e5[e5.label == "rul"].variant) == {"A_none", "C_text"} and not e5[e5.label == "rul"].oracle_eol.any()
    assert e5.code_commit.notna().all() and e5.e5_started.notna().all() and np.isfinite(e5.rmse).all()
    a = e5[(e5.variant == "A_none") & (e5.label == "frac") & (e5.seed == 0)].set_index(["fold", "regime"]).rmse
    b = e2[(e2.model == "lstm_frac") & (e2.seed == 0)].set_index(["fold", "regime"]).rmse
    assert np.allclose(a, b.loc[a.index])                               # A_none is exactly E2 lstm_frac (paired)
    report = (out / "REPORT.md").read_text()
    for s in ["## E0", "## E1", "## E2", "## E3", "## E4", "## E5", "Paired comparisons", "C_text vs D_shuffled",
              "oracle", "What this does NOT show"]:
        assert s in report, s
    assert (out / "e1_fraction_of_life.png").exists() and (out / "e3_gate_weights.png").exists()
    assert time.time() - t0 < 60, f"smoke test too slow: {time.time() - t0:.0f}s"


@pytest.mark.skipif(MAC, reason="torch + lightgbm libomp clash segfaults on macOS; Colab/Linux runs it")
def test_lgbm(tmp_path):
    from qr.features import make_split, prepare, regime_sets, window_idx
    from qr.train import fit_lgbm
    qd.make_synthetic(tmp_path)
    _, kept = qd.load_cells(tmp_path, ["SNL"])
    prep = prepare(kept, qd.load_text_embeddings(tmp_path), cap=30)
    cfg = smoke_cfg()
    sp = make_split(prep["chem"], "NMC", 0, cfg["split"])
    test = window_idx(prep, sp["test"])
    p = fit_lgbm(prep, regime_sets(sp, "zero_shot"), cfg["e2"]["lgbm"], 0).predict(prep["X"][test, -1])
    assert np.isfinite(p).all()


def test_missing_prompt_embedding(tmp_path):
    import pickle
    data = tmp_path / "data"
    qd.make_synthetic(data)
    emb_file = data / "embeddings" / qd.EMB_FILES[0]
    emb = pickle.loads(emb_file.read_bytes())
    gone = "SNL_18650_NCA_35C_0-100_0.5-2C_b"
    del emb[gone]
    emb_file.write_bytes(pickle.dumps(emb))
    cfg = smoke_cfg()
    run_all.main(["--data", str(data), "--out", str(tmp_path / "out"), "--steps", "e0"], cfg=cfg)
    dropped = pd.read_csv(tmp_path / "out" / "e0_dropped_cells.csv")
    assert dropped.to_dict("records") == [dict(name=gone, chem="NCA", dataset="SNL", reason="no_prompt_embedding")]
    gate = pd.read_csv(tmp_path / "out" / "e0_gate.csv").set_index("check")
    assert int(gate.loc["cells dropped: no prompt embedding", "value"]) == 1
    assert not gate.loc["count NCA", "passed"]                       # 1 of 2 expected NCA cells left
    cfg["data"]["text_embedding_missing"] = "error"
    with pytest.raises(KeyError, match="no prompt embedding"):
        run_all.main(["--data", str(data), "--out", str(tmp_path / "out2"), "--steps", "e0"], cfg=cfg)


def _flaky_server(payload, cut_first=True):
    """Loopback HTTP server: first GET advertises the full length but sends half and drops (like Zenodo did);
    later GETs honour 'Range: bytes=N-' with 206. Returns (base_url, list of Range headers seen, server)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            seen.append(rng)
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            body = payload[start:]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if cut_first and len(seen) == 1:
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/f.zip", seen, srv


def test_fetch_resumes_and_verifies(tmp_path):
    import hashlib
    payload = np.random.default_rng(0).bytes(300_000)
    md5 = hashlib.md5(payload).hexdigest()
    url, seen, srv = _flaky_server(payload)
    try:
        dest = qd.fetch(url, tmp_path / "f.zip", size=None, md5=md5, retries=4, backoff=0, chunk=8192)  # size from Content-Length
        assert dest.read_bytes() == payload and not (tmp_path / "f.zip.tmp").exists()
        offset = int(seen[1].split("=")[1].rstrip("-"))
        assert seen[0] is None and 0 < offset <= 150_000                # second request resumed the partial .tmp
        n = len(seen)
        qd.fetch(url, dest, size=len(payload), md5=md5)                 # complete, size-verified: skipped
        assert len(seen) == n
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):  # wrong checksum is never renamed
            qd.fetch(url, tmp_path / "g.zip", size=len(payload), md5="0" * 32, retries=2, backoff=0)
        assert not (tmp_path / "g.zip").exists()
    finally:
        srv.shutdown()


def test_moe_mask_without_match_is_shared_only_and_beta_trains():
    import torch
    from qr.models import MoE
    torch.manual_seed(0)
    m = MoE(n_feat=25, d_text=16, n_dyn=5, n_experts=2, bias=2.0)
    x, d, txt = torch.randn(4, 10, 25), torch.randn(4, 5), torch.randn(4, 16)
    match = torch.tensor([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])     # rows 2-3: no matching specialist
    with torch.no_grad():
        m.beta_logit.fill_(0.3)                                         # beta != 0.5, so a (1-beta)/beta mix would show
    o = m(x, d, txt, match, 1.0, "mask")
    assert torch.equal(o["g"][2:], torch.zeros(2, 2))
    assert torch.allclose(o["pred"][2:], o["fs"][2:])                   # f_shared alone, not beta * f_shared
    blend = (1 - o["beta"]) * (o["g"][:2] * o["f"][:2]).sum(1) + o["beta"] * o["fs"][:2]
    assert torch.allclose(o["pred"][:2], blend)
    # beta: a trainable parameter with gradient, no clamp (sigmoid of an unbounded logit)
    assert m.beta_logit.requires_grad and any(p is m.beta_logit for p in m.parameters())
    o = m(x, d, txt, match, 1.0, "bias")
    torch.nn.functional.mse_loss(o["pred"], torch.rand(4)).backward()
    assert m.beta_logit.grad is not None and m.beta_logit.grad.abs().item() > 0
    # router input is the unit-length text embedding: rescaling the raw vector changes nothing
    with torch.no_grad():
        m.eval()
        a, b = m(x, d, txt, match, 1.0, "bias"), m(x, d, 100 * txt, match, 1.0, "bias")
    assert torch.allclose(a["logits"], b["logits"], atol=1e-5)


def test_cached_e3_is_marked_and_force_reruns(tmp_path, monkeypatch):
    import json
    data, out = tmp_path / "data", tmp_path / "out"
    qd.make_synthetic(data)
    cfg = smoke_cfg()
    args = ["--data", str(data), "--out", str(out), "--steps", "e0,e3,report"]
    run_all.main(args, cfg=cfg)
    first = pd.read_csv(out / "e3_raw.csv")
    assert first.code_commit.notna().all() and first.e3_started.notna().all()
    assert isinstance(json.loads((out / "runtime.json").read_text())["e3"], float)

    run_all.main(args, cfg=smoke_cfg())                                   # outputs exist: skipped
    rt = json.loads((out / "runtime.json").read_text())
    assert rt["e3"] == "cached" and rt["e0"] == "cached"
    assert pd.read_csv(out / "e3_raw.csv").e3_started.equals(first.e3_started)

    import time
    time.sleep(1.1)                                                       # e3_started has 1 s resolution
    run_all.main(args + ["--force", "e3"], cfg=smoke_cfg())              # --force reruns despite the CSV
    rt = json.loads((out / "runtime.json").read_text())
    assert isinstance(rt["e3"], float) and rt["e0"] == "cached"
    assert not pd.read_csv(out / "e3_raw.csv").e3_started.equals(first.e3_started)

    # a "run 1" report from the same commit must not be presented as run 1 vs run 2
    run1 = tmp_path / "run1" / "REPORT.md"
    run1.parent.mkdir()
    run1.write_text((out / "REPORT.md").read_text())
    monkeypatch.setattr(run_all, "RUN1", run1)
    run_all.main(["--data", str(data), "--out", str(out), "--steps", "report"], cfg=smoke_cfg())
    rep = (out / "REPORT.md").read_text()
    assert "Not labelled as run 2" in rep and "### E3 run 1 vs run 2" not in rep
    with pytest.raises(SystemExit):
        run_all.main(args + ["--force", "e9"], cfg=smoke_cfg())


class FakePBT:
    """Test double with PBTAdapter's interface: life = (1 + bias) * true EOL; fine-tuning removes the bias."""
    info = {"device": "cpu", "checkpoint": "fake"}

    def __init__(self, eol, missing=()):
        self.eol, self.missing, self.bias, self.finetunes = eol, set(missing), 0.3, 0

    def inputs(self, name):
        if name in self.missing:
            raise KeyError(f"{name}: no PBT prompt embedding")
        return name

    def life(self, name):
        return np.full(100, (1 + self.bias) * self.eol[name])

    def finetune(self, train, val, cfg, seed):
        self.bias, self.finetunes = 0.05, self.finetunes + 1
        return [dict(epoch=0, val_mape_life=0.05)]

    def reset(self):
        self.bias = 0.3

    def split_of(self, name):
        return "test" if name.endswith("_a") else "train"


def test_e4_with_fake_pbt(tmp_path, monkeypatch):
    data, out = tmp_path / "data", tmp_path / "out"
    qd.make_synthetic(data)
    cfg = smoke_cfg()
    cfg["e4"].update(enabled=True)
    cfg["e4"]["finetune"].update(regimes=["few_1"], epochs=1)
    run_all.main(["--data", str(data), "--out", str(out), "--steps", "e0,e2"], cfg=cfg)
    cells = pd.read_csv(out / "e0_cell_list.csv")
    fake = FakePBT(dict(zip(cells.name, cells.eol)), missing={"SNL_18650_NMC_15C_0-100_0.5-2C_b"})
    monkeypatch.setattr(run_all, "E4_ADAPTER", fake)
    run_all.main(["--data", str(data), "--out", str(out), "--steps", "e4,report", "--force", "e4"], cfg=cfg)
    r = pd.read_csv(out / "e4_raw.csv")
    assert set(r.model) == {"PBT_zero_shot", "PBT_finetuned"} and (~r.oracle_eol).all() and np.isfinite(r.rmse).all()
    assert r.code_commit.notna().all() and fake.finetunes > 0
    zs, ft = r[r.model == "PBT_zero_shot"], r[r.model == "PBT_finetuned"]
    assert ft.rmse.mean() < zs[zs.subset == "all_test_cells"].rmse.mean()   # fake fine-tune removes most of the bias
    assert set(zs.subset) <= {"all_test_cells", "pbt_test_split_only"}
    c = pd.read_csv(out / "e4_cells.csv")
    assert set(c.pbt_split) <= {"train", "val", "test", "none"} and not c[c.cell.str.contains("NMC_15C")].used.any()
    rep = (out / "REPORT.md").read_text()
    for s in ["## E4: released PBT on our LOCO protocol", "**Leakage.**", "Track A gate", "pbt_test_split_only",
              "no PBT prompt embedding"]:
        assert s in rep, s
    verdicts = [l for l in rep.split("Track A gate")[1].splitlines() if l.startswith("- ")][:len(cfg["folds"])]
    assert len(verdicts) == len(cfg["folds"])
    for f in cfg["folds"]:  # a verdict wherever PBT was fine-tuned; "not decidable" only where it could not be
        v = next(x for x in verdicts if x.startswith(f"- {f}: "))
        assert ("->" in v) == (f in set(ft.fold)), v


PBT_REPO = ROOT.parent / "third_party" / "PBT"


@pytest.mark.skipif(not (PBT_REPO / "models" / "PBT.py").exists(), reason="third_party/PBT not cloned")
def test_e4_real_pbt_code_with_tiny_fake_checkpoint(tmp_path):
    """Real PBT model/data code (pinned commit) with random tiny weights: import, no-tokenizer init, inputs, FT."""
    import json
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler
    from qr import pbt_e4
    from qr.features import prepare
    data, ckpt = tmp_path / "data", tmp_path / "ckpt"
    qd.make_synthetic(data)
    ckpt.mkdir()
    args = dict(task_name="battery_life_prediction", dataset="MIX_large", d_model=16, d_ff=16, min_d_ff=16, n_heads=2,
                e_layers=1, d_layers=1, d_llm=16, gate_d_ff=64, dk_factor=1, dropout=0.0, activation="gelu",
                charge_discharge_length=300, early_cycle_threshold=100, patch_len=10, stride=10, num_views=1,
                down_sample_ratio=0.75, topK=-1, use_dff_scale=False, d_ff_scale_factor=None, num_general_experts=1,
                num_experts=2, output_num=1, cathode_experts=7, anode_experts=6, format_experts=6,
                temperature_experts=10, LLM_path="unused", pca_path="missing.pkl")
    (ckpt / "args.json").write_text(json.dumps(args))
    joblib.dump(StandardScaler().fit(np.array([[150.0], [250.0]])), ckpt / "label_scaler")
    # build once to get a state dict of the right shape, save it the way load_checkpoint_in_model reads it
    for m in pbt_e4.OPTIONAL:
        try:
            __import__(m)
        except ImportError:
            sys.modules[m] = pbt_e4._Stub(m)
    sys.path.insert(0, str(PBT_REPO))
    (tmp_path / "p.pkl").write_bytes(__import__("pickle").dumps(None))
    with pbt_e4._cwd(PBT_REPO):
        from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
        from models import PBT
        PBT.AutoTokenizer = pbt_e4._NoTokenizer
        model = PBT.Model(BatteryLifeConfig(BatteryElectrochemicalConfig(args | {"pca_path": str(tmp_path / "p.pkl")}), None))
    torch.save(model.state_dict(), ckpt / "pytorch_model.bin")

    _, kept = qd.load_cells(data, ["SNL"])
    emb = qd.load_text_embeddings(data)
    prep = prepare(kept, emb, cap=20)
    ad = pbt_e4.PBTAdapter(PBT_REPO, ckpt, data, emb, tmp_path / "work", "cpu")
    name = kept[0]["name"]
    curves, attn, e, mask, eol = ad.inputs(name)
    assert curves.shape[1:] == (100, 3, 300) and attn.sum(1)[0] == 1 and mask.shape == (29,) and eol == kept[0]["eol"]
    life = ad.life(name)
    assert life.shape == (len(curves),) and np.isfinite(life).all()
    cfg = smoke_cfg()
    cfg.update(folds=["NMC"])
    cfg["e4"]["finetune"].update(regimes=["few_1"], epochs=1)
    rows, cells, hist, notes = pbt_e4.run_e4(prep, cfg, ad)
    assert {r["model"] for r in rows} == {"PBT_zero_shot", "PBT_finetuned"} and all(np.isfinite(r["rmse"]) for r in rows)
    assert len(hist) == 1 and notes["e4b_seeds"] == [0]
    ad.reset()
    assert np.allclose(ad.life(name), life, rtol=1e-2)                  # reset restores the pretrained weights



def test_e5_aux_inputs_paired_stats_and_budget(tmp_path):
    from qr import e5 as E5
    from qr.features import prepare
    p = E5.derangement(50, 3)
    assert (p != np.arange(50)).all() and sorted(p) == list(range(50)) and (p == E5.derangement(50, 3)).all()
    qd.make_synthetic(tmp_path)
    _, kept = qd.load_cells(tmp_path, ["SNL"])
    prep = prepare(kept, qd.load_text_embeddings(tmp_path), cap=20)
    oh, dyn = E5.aux_for("B_onehot", prep, "NCA", 0)
    assert not dyn and oh.shape[1] == 2 and (oh[prep["chem"] == "NCA"] == 0).all() and (oh.sum(1)[prep["chem"] != "NCA"] == 1).all()
    t, _ = E5.aux_for("C_text", prep, "NCA", 0)
    sh, _ = E5.aux_for("D_shuffled", prep, "NCA", 0)
    assert np.allclose(np.linalg.norm(t, axis=1), 1) and not np.allclose(t, sh) and np.allclose(sorted(map(tuple, t)), sorted(map(tuple, sh)))
    assert E5.aux_for("E_text_dyn", prep, "NCA", 0)[1]
    # paired stats: x better on every pair -> supported; noise around 0 -> not supported
    rows = [dict(fold=f, seed=s, regime="zero_shot", label="frac", variant=v, rmse=r)
            for f in "ab" for s in range(10) for v, r in (("C_text", 90 + s % 3), ("A_none", 100 + s % 2))]
    r = E5.paired(pd.DataFrame(rows), "C_text", "A_none", regime="zero_shot", n_boot=500)
    assert r["pairs"] == 20 and r["wins"] == 20 and r["ci95_high"] < 0 and r["verdict"] == "supported"
    rows = [dict(fold="a", seed=s, regime="zero_shot", label="frac", variant=v, rmse=100 + (s % 2 - 0.5) * (v == "C_text"))
            for s in range(10) for v in ("C_text", "A_none")]
    assert E5.paired(pd.DataFrame(rows), "C_text", "A_none", n_boot=500)["verdict"] == "not supported at this scale"
    # budget guard: drop lstm_rul first, then seeds
    cfg = smoke_cfg()
    cfg["folds"] = ["NCA"]
    cfg["e5"].update(seeds=[0, 1, 2], regimes=["zero_shot"], variants=["A_none"], budget_minutes=0)
    rows, notes = E5.run_e5(prep, cfg, log=lambda *_: None)
    assert notes["seeds_done"] == [0] and notes["seeds_reduced"] and notes["rul_dropped_from_seed"] == 1
