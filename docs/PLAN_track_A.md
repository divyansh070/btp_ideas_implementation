# Plan: shared data pipeline + Track A (A1–A5)

Source of truth: design guide Part 1 (§3–§6.8). Repo facts: `document.md` entry "PBT repo audit" (PBT commit 74f03a7).
Each task is ≤ 1 day and ends with a **Done-check**: a command that must pass, or a number that must be met.
Nothing trains before the validation gate (P8) passes, except the A2 reproduction, which uses PBT's own pipeline and data.

## 0. Access, downloads, decisions (start day 1, in parallel)

| Item | Access | Size | Status |
|---|---|---|---|
| Llama-3.1-8B-Instruct (HF `meta-llama/Llama-3.1-8B-Instruct`) | **Gated, manual approval.** Accept the Llama 3.1 licence on the HF page, then put a read token in `.env` | 16.07 GB weights; we only need tokenizer + config (~10 MB) unless we re-embed | request now |
| PBT checkpoints + prompt embeddings (Zenodo 17972780) | open, CC-BY-4.0 | 214 MB | download in A1.1 |
| BatteryLife v9 processed data (Zenodo 17958489) | open, CC-BY-4.0 | our 8 datasets: 8.01 GB (MATR 4.86 GB, HUST 1.39, RWTH 1.06, XJTU 0.45, SNL 0.11, CALCE 0.09, HNEI 0.04, UL_PUR 0.01) | **needs your OK (> 1 GB)** |
| PBT life labels (figshare 31148413, `Life labels.zip`) | open | < 1 MB | download in P2 |
| SNL, UL-PUR, HNEI raw (guide §4 rule 4: "need an access request") | only if we decide to use raw data instead of BatteryLife-processed | — | **decision D2** |

Decisions needed from you (see document.md → Known issues):
- **D1** Which table does the gate's "cell counts match" use: the guide's (MATR 180, SNL 61, XJTU ~55) or PBT's post-filter split (MATR 169, HUST 77, SNL 50, XJTU 23, RWTH 48, UL-PUR 2, CALCE 13, HNEI 14)? Proposal: gate on PBT's post-filter counts, and report raw counts next to them.
- **D2** Use BatteryLife-processed pickles (same input as PBT, open) or raw sources? Proposal: BatteryLife, so Track A and PBT see identical data.
- **D3** EOL at inference (open issue a). Proposal: Track A predicts RUL in cycles directly (A3), so no EOL is needed. Decide before any fraction-of-life training (Track B).
- **D4** Definition of a "measurable gap" for the week-6 gate (see the gate section below).

## Shared data pipeline (P1–P8)

Code lives in `src/data/`; configs in `configs/`. Each task adds one small runnable check (`assert` self-check or `tests/test_*.py`).

- **P1 Reader → standard schema.** One reader for the BatteryLife pickle format (`cycle_data` list of per-cycle dicts) plus a per-dataset metadata table (chemistry, nominal capacity overrides: RWTH 1.85 Ah, SNL NCA 20-80 3.2 Ah, as in PBT). Output columns: cell_id, cycle_number, voltage, current (C-rate), discharge_capacity (Ah), temperature (NaN if missing), cycle_life, chemistry.
  Done-check: reader loads every cell of the 8 datasets; schema test passes; per-dataset raw cell counts printed.
- **P2 EOL labels, fail loudly.** Compute EOL with PBT's rule (SOH = max Qd / nominal / SOC interval; drop if SOH never ≤ 82.5%; linear-extrapolate to 80% for cells in (80%, 82.5%)). Raise on a missing cycle_life, never fall back to the last cycle. Cross-check against PBT's figshare labels.
  Done-check: 0 fallbacks; our EOL equals PBT's label for every cell both keep (report any mismatch by cell).
- **P3 Filters + count report.** Drop cells with < 100 cycles; apply the 82.5% rule.
  Done-check: `results/counts.csv` (raw / after filter / PBT split) matches the table chosen in D1.
- **P4 Chemistry tagging; SNL split three ways.** Tag chemistry from the cell name (SNL_18650_{LFP,NMC,NCA}_…).
  Done-check: SNL → LFP / NMC / NCA with no cell tagged twice; under PBT's filters: 9 / 21 / 20.
- **P5 Windows, features, labels.** 30 consecutive cycles, 10 sampled evenly, stride 1; 18 stats (mean, std, min, max, var, median × 3 signals) + window deltas; store absolute RUL and per-window EOL.
  Done-check: test on a synthetic cell with a known linear fade (exact RUL, window count = n_cycles − 29).
- **P6 LOCO split + scaler.** For held-out chemistry k: k's cells split 60 / 10 / 30 (adaptation / val / blind test), cell-disjoint and seeded; 5 / 10 / 20-cell nested subsets of the adaptation set. Fit the scaler on source-train + target-adaptation windows only, then clip to [−0.5, 1.5].
  Done-check: asserts that no cell appears in two splits and no blind-test cell enters the scaler fit; seeds logged.
- **P7 Ageing-factor text per cell.** Reuse PBT's prompt strings (`Prompts/Mapping_helper.py`) so our cell ids map to PBT's cached Llama embeddings.
  Done-check: every kept cell has a non-zero 4096-d embedding (PBT silently substitutes zeros, `data_loader.py:918`; we raise instead).
- **P8 Validation gate (hard stop before any training).** `python -m src.data.gate` checks:
  1. cell counts match D1;
  2. zero EOL fallbacks;
  3. per-cell corr(RUL, capacity) > 0.5;
  4. every scaled target feature in [−0.2, 1.2], no feature with std < 0.01;
  5. SNL split three ways, no cell tagged twice.

  Done-check: exit code 0 and `results/gate_report.json` committed with the numbers. Log in document.md.

## Track A

- **A1 Setup (1 day).**
  1. Download Zenodo 17972780; record files, `args.json` (confirm el2 / dl10 / GE5 / E5, head general-expert count) and whether `MIX_large_pca_0.99_Llama.pkl` and `label_scaler` ship with it.
  2. With HF access: fetch only the Llama tokenizer + config (`allow_patterns`), not the weights.
  3. Make `third_party/PBT` importable on the Mac: stub the unused `batteryml` import, run with `--mixed_precision no`, `wandb` offline. All patches are logged, and kept as a patch file outside third_party.
  4. Check the embedding cache covers all cells of the 8 datasets.

  Done-check: one forward pass of the released checkpoint on one HNEI test cell on MPS/CPU returns a finite cycle life. The cache check reports 0 missing cells.
- **A2 Reproduce (1–2 days).** Target: **HNEI, PBT-TL test MAPE 0.033 ± 0.002**, first 20 cycles as input, 3 seeds (Supplementary Table 2, arXiv:2512.16334 v4, p. 40). Why HNEI first: 40 MB of data, 9 / 3 / 2 cells (the paper's 9 training cells match PBT's recorded split), and it fits on this Mac.
  1. Zero-shot reference: `evaluate_model.py --eval_cycle_min 20 --eval_cycle_max 20` on the pretrained checkpoint.
  2. Fine-tune (FT, lr ∈ {1e-5, 2.5e-5, 5e-5, 1e-4}, choose on val) with seeds 2021 / 42 / 2024.

  Done-check: mean test MAPE@20 over 3 seeds is in **[0.023, 0.043]** (±0.010). This band is proposed and needs your OK: HNEI's test set is only 2 cells, so MAPE is coarse. If we miss the band, stop: nothing downstream is interpretable.
  Second confirmation (GPU, after the 4.86 GB download): **MATR 0.077 ± 0.006** (102 training cells, 33 test). Done if within ±0.015.
- **A3 Re-head to windowed RUL (2 tasks × 1 day).**
  1. Feed PBT the window's 10 sampled cycles as its cycle sequence (300-point curves, PBT's representation, not our 18 features). Replace the regression head; label = RUL in cycles (per D3).
     Done-check: a training run on the source chemistries reduces val loss. Val predictions are not collapsed (min ≠ max), logged per epoch.
  2. Convert-baseline (predict cycle life − current cycle) on the same windows, for reference.
     Done-check: both numbers in one table on the validation split.
- **A4 Specialise (1 day per method).** FT (lr 1e-5 to 1e-4) vs adapter tuning (`--finetune_method AT`, N 1–12, width 1–128), chosen on validation only.
  Done-check: per-fold config table in `configs/` with the val metric that selected it; blind test untouched.
- **A5 Held-out test (1 day per fold, 4 folds: LFP / NMC / NCA / LCO).** Zero-shot, then 5 / 10 / 20 target cells, then the full 60%. Metrics per chemistry (never pooled R²): per-cell RMSE, R², MAPE, plus floor MAPE (RUL ≥ 50 cycles). Mean ± std over seeds.
  Done-check: `results/trackA_loco.csv`, with every row labelled **"optimistic: chemistry seen in PBT pretraining"**. NCA = SNL-NCA + UL-PUR (UL-PUR alone cannot be evaluated); the fold cell counts are stated.
  Seeds: the guide's metrics section says 10. PBT uses 3. Proposal: 3 for A2 (matches the paper), 10 for A5.

## Week-6 gate (guide §5)

Build Track B only if fine-tuned PBT leaves a measurable gap on a held-out chemistry. **D4 proposal:** gap = PBT few-shot (20 cells) floor-MAPE on held-out chemistry k is ≥ 20% worse than PBT fine-tuned with k's full 60% adaptation set, or ≥ 20% worse than its in-distribution (seen-chemistry) result, in ≥ 1 fold, beyond 2 std over seeds. Otherwise, report the PBT result + the LOCO protocol.

## Leak / risk register
- True EOL at test time = leak. Track A avoids it by predicting RUL in cycles (D3); flag in document.md if it ever happens.
- Released PBT saw LFP / NMC / NCA / LCO in pretraining, so LOCO is optimistic (guide §5 caveat). Options for later: re-pretrain without k (GPU, days), or add Na-ion / Zn-ion.
- An unseen category in PBT gets an all-zero mask for that factor, leaving general experts + the other factors (audit answer b). This matters only for truly new chemistries.
- MATR-scale runs need a CUDA GPU (RAM estimate ~3.7 GB for the MATR train set alone; M1 has 8 GB).
