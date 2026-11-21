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
    # with sets LFP|rest, held-out NCA / LFP have no matching expert; NMC folds keep one via NMC-LCO only in real data
    assert not e3[e3.fold == "NCA"].held_out_has_matching_expert.any()
    report = (out / "REPORT.md").read_text()
    for s in ["## E0", "## E1", "## E2", "## E3", "## E4", "oracle", "What this does NOT show"]:
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
