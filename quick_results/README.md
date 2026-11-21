# quick_results: Version 1 proof of concept (Colab)

Self-contained code for a small, preliminary run of Version 1 (design guide Part 1). It does not import `src/` or
`third_party/PBT`. All downloads and experiments run on Google Colab (free T4); data and outputs are cached on Google Drive.

**Run it:** open [`quick_results_colab.ipynb`](https://colab.research.google.com/github/divyansh070/btp_ideas_implementation/blob/main/quick_results/quick_results_colab.ipynb),
choose a T4 runtime, run all cells. Flags in the first cell: `USE_HUST`, `RUN_E4`, `ALLOW_GATE_FAIL`.
Same thing from a shell: `python run_all.py --data DATA_DIR --out OUT_DIR --download [--use-hust] [--e4] [--allow-gate-fail]`.
Each step skips itself if its CSV already exists in `OUT_DIR`; delete the CSV to rerun one step. `OUT_DIR/REPORT.md` is the result.

**Data** (`qr/data.py`, fetched only inside Colab): BatteryLife v9 per-dataset zips from Zenodo 17958489 (SNL, CALCE, HNEI ≈ 0.25 GB;
plus HUST 1.39 GB, on by default in the notebook via `USE_HUST`), PBT's life labels (figshare 31148413), PBT's Llama prompt embeddings (Zenodo 17972780, 6 MB).
SNL is split into LFP / NMC / NCA by cell name; CALCE = LCO, HNEI = NMC-LCO (always a source, never held out).
Life labels follow PBT's rule (`process_scripts/extract_life_labels.py`, ported) and are cross-checked against PBT's JSONs.
The pickles come from Zenodo and are loaded with `pickle` (as PBT does); only load them inside Colab.

**E0, pipeline and validation gate.** PBT filters (< 100 cycles, last-cycle SOH ≥ 82.5 %, life ≤ 100), 30-cycle windows with 10
cycles sampled (stride 1, at most 150 per cell for runtime), 17 stats + 11 deltas per cycle (`Qd_min` dropped: ~0 every cycle by construction), RUL and fraction-of-life labels,
cell-level leave-one-chemistry-out splits (held-out chemistry 60 / 10 / 30), scaler fit on source-train + target-adaptation and clipped
to [-0.5, 1.5]. Gate checks: counts vs PBT's split, zero EOL fallbacks, agreement with PBT labels, per-cell corr(RUL, capacity) > 0.5,
SNL three-way tagging, scaled target features (1st-99th percentile) in [-0.2, 1.2], feature std > 0.01. Training refuses to start if a check
fails unless `ALLOW_GATE_FAIL`. Expect the std check to flag `Qd_min` (discharge capacity restarts at 0 every cycle, so it is constant).

**E1, why fraction of life.** Cycle life per chemistry and the RUL label in cycles vs fraction of life (one figure).

**E2, cross-chemistry gap.** Single-head window model (2-layer LSTM 64 → attention → LayerNorm head) predicting fraction of life
(`lstm_frac`) or RUL in cycles (`lstm_rul`, no oracle), and LightGBM on the last cycle's features. Folds LFP / NMC / NCA / LCO;
regimes target-only, zero-shot, few-shot 2 and 5 cells. RMSE (cycles), R² and MAPE on RUL ≥ 50 per held-out chemistry, mean ± std over seeds 0-2.

**E3, mini Track B.** Shared encoder → z (128) → E = 2 experts (LFP vs the rest of the fold's source; NMC vs rest when LFP is held out)
plus an always-on shared expert; prediction (1−β)Σ g·f + β·f_shared. Router: PBT text embedding (linear + LeakyReLU → 64) + 6 dynamics.
Routing: PBT-style hard mask vs our additive category bias, τ 2.0 → 0.5. Stage A (router uniform) then Stage B with L_RUL, L_MMD,
L_bal, L_route (annealed to 0 by 40 %), L_spec, each switchable in `configs/quick.yaml`. Logs gate weights per chemistry, β, entropy,
expert correlation; reports the folds where the held-out chemistry has no matching expert (LFP, NCA). The Neural ODE is omitted.

**E4 (optional), released PBT zero-shot.** Loads the Zenodo checkpoint and scores RUL = predicted life − current cycle on the same test
cells. It needs the gated Llama tokenizer (Colab secret `HF_TOKEN`); otherwise it skips and the reason goes into the report. Untested end to end.

**Caveats.** Fraction-of-life numbers are converted to cycles with the TRUE EOL ("oracle EOL"). Test sets are 3-6 cells. The released
PBT already saw these chemistries (possibly these cells). Model selection uses validation cells only.

**Local smoke test** (no downloads, < 1 minute on CPU): `python -m pytest -q tests/test_smoke.py` builds 6 synthetic cells (3 chemistries)
and runs E0-E3 + report. On macOS the LightGBM parts are skipped: torch and LightGBM load two different `libomp` runtimes and segfault in
one process. Colab (Linux) is unaffected. `requirements.txt` uses lower bounds so Colab keeps its preinstalled CUDA torch;
`requirements-lock-mac.txt` records the versions the smoke test passed with.
