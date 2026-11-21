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
cycles sampled (stride 1, at most 150 per cell for runtime), 17 stats + 8 deltas = 25 features per cycle (`Qd_min` dropped: ~0 every cycle by construction; deltas of `V_max`, `V_min`, `I_C_max` dropped: protocol-fixed, ~0 inside a window), RUL and fraction-of-life labels,
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

**E3 v2, MoE vs single head with fixes A-D** (`--steps e3v2`, protocol pre-registered in `configs/e3v2.yaml`).
Phase 1 on source chemistries (also the zero-shot model): single_head, single_head_matched (head width matched to the
MoE's parameter count), moe_mask / moe_bias with fix A (Stage A length / L_spec grid, beta starts at 0.2 with 10x lr),
24 epochs each, config chosen on source-val. Few-shot (2 / 5 cells): encoder frozen for every model; single heads tune
the head, the MoE spawns an expert for the held-out chemistry from the shared expert or the nearest expert and tunes it
+ router + beta; epoch chosen on target-val (epoch 0 = no update included). Seeds 0-4, paired bootstrap CIs vs both
single heads, LFP + NCA reported separately, specialisation table (target corr < 0.8, beta < 0.4).

**E4 (optional, `RUN_E4`), released PBT on our LOCO protocol.** Loads the Zenodo checkpoint (17972780) with PBT's
own code (`third_party/PBT` if present, else a clone pinned to 74f03a7) and PBT's cached prompt embeddings. No Llama
download or token is needed: PBT's tokenizer is never used and is stubbed. E4a: zero-shot, PBT's own input (first
min(c, 100) cycles, 300 points each) for every blind-test window, RUL = predicted life − window end cycle, no oracle EOL.
Scored on all test cells and on only those in PBT's own test split; a leakage table shows how many test cells PBT trained on.
E4b: PBT's FT recipe (all parameters, Adam, lr 1e-4, MSE on scaled life, batch 32, 10 epochs, best epoch on target val)
on the same 2 / 5 adaptation cells as E2, our loop rather than `finetune_model.py` (see `qr/pbt_e4.py`). Seeds are
dropped if E4 would exceed `budget_minutes`. The report adds a per-fold "Track A gate" verdict vs E2 target-only.
Failures write their reason to `e4_status.txt` instead of crashing.

**E5, does a text description of the cell help?** E2's `lstm_frac` model with one extra input concatenated to z
(Linear + LeakyReLU → 32): A none, B one-hot of source chemistries (held-out = zeros), C PBT Llama prompt embedding
(unit length), D shuffled-text control (another cell's embedding, fixed derangement per seed), E text + 6 dynamics.
A and C are also run with `lstm_rul` (no oracle) for comparison with E4. Same folds and paired splits, zero-shot and
few-5, seeds 0-4. The report gives per-variant tables, paired wins and bootstrap 95 % CIs for C vs A / D / B ("supported"
only if the CI is below 0), and the LFP + NCA zero-shot subset. Budget 70 min: lstm_rul is dropped first, then seeds.

**Caveats.** Fraction-of-life numbers are converted to cycles with the TRUE EOL ("oracle EOL"). Test sets are 3-6 cells. The released
PBT already saw these chemistries (often these cells: see the E4 leakage table). Model selection uses validation cells only.

**Local smoke test** (no downloads, < 1 minute on CPU): `python -m pytest -q tests/test_smoke.py` builds 6 synthetic cells (3 chemistries)
and runs E0-E3 + report. On macOS the LightGBM parts are skipped: torch and LightGBM load two different `libomp` runtimes and segfault in
one process. Colab (Linux) is unaffected. `requirements.txt` uses lower bounds so Colab keeps its preinstalled CUDA torch;
`requirements-lock-mac.txt` records the versions the smoke test passed with.
