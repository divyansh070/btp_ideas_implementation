# Development log — Version 1 (Track A → Track B)

## Decisions
- 2026-11-20 — Log entries cannot contain their own hash; "Commit:" gives the exact commit subject (`git log --grep '<subject>'` finds it).
- 2026-11-20 — Slide deck read from `~/Downloads/PBT_to_Our_Model.pptx` (no .pptx in docs/); not copied into the repo pending user confirmation.

## Known issues / open questions
- (a) How is EOL obtained at inference to turn predicted fraction-of-life back into cycles? True EOL at test time is a leak (guide §3, §6.9, Part 4 item 6). Options: estimate, separate EOL predictor, or predict RUL in cycles directly. **Must be decided before any fraction-of-life training.** Not blocking Track A (A3 re-head can predict RUL in cycles).
- (b) How does PBT create experts for a category absent from pretraining (e.g. Zn anode, Na cathode)? Paper silent (PBT doc §9) → answer from repo.
- (c) Number of general experts per layer? Paper silent → answer from repo.
- (d) Meaning of the "2 / 10" encoder/decoder layers hyperparameter (Supp. Table 4); guess is intra-cycle / inter-cycle stacks → answer from repo.
- (e) Are specialist weights renormalised after the hard mask? Paper does not mention it → answer from repo.
- (f) Is a softmax applied to the gate? Equations show a raw linear layer → answer from repo.
- Docs: the pptx named in CLAUDE.md (`docs/*.pptx`) is not in docs/.

## Log

### 2026-11-20 23:47 — Read design docs; Version 1 summary and open questions
- Changed: document.md (Decisions, Known issues, this entry)
- Why: brief step 1; CLAUDE.md "Source of truth". Read both PDFs in docs/ in full (PyMuPDF) and the deck (python-pptx, 14 slides).
- Verified by: `python3` extraction → PBT doc 10 pages, design guide 14 pages, deck 14 slides; all text read.
- Summary of Version 1 (design guide Part 1):
  1. Aim: one model predicting RUL across chemistries, incl. unseen ones; evaluated leave-one-chemistry-out (LOCO) over LFP / NMC / NCA / LCO.
  2. Our task ≠ PBT's: input is a 30-cycle window anywhere in life (10 cycles sampled), output RUL in cycles; Track B label = fraction of life (RUL / EOL).
  3. Shared pipeline: one reader per dataset → one schema (cell_id, cycle_number, V, I in C-rate, Q_dis Ah, T °C, cycle_life, chemistry); PBT filters (<100 cycles drop; SOH never ≤82.5% drop; linear-extrapolate 82.5→80%); windows stride 1; 18 stats features + window deltas; text template of 10 ageing factors per cell.
  4. Validation gate before any training: counts match table, zero EOL fallbacks, per-cell RUL–capacity corr > 0.5, scaled features in [−0.2, 1.2] with std ≥ 0.01, SNL split into LFP/NMC/NCA with no cell tagged twice.
  5. Datasets: MATR, HUST, SNL-LFP (LFP); XJTU, RWTH, SNL-NMC (NMC); UL-PUR, SNL-NCA (NCA); CALCE (LCO); HNEI (NMC-LCO).
  6. Track A = benchmark released PBT: A1 setup (repo, Zenodo weights, Llama-3.1-8B, cache text embeddings), A2 reproduce a PBT-reported number, A3 re-head to windowed RUL, A4 full FT vs adapters (choose on val), A5 LOCO with 5/10/20 cells then full 60%.
  7. LOCO split of held-out chemistry: 60% adaptation / 10% val / 30% blind test, cell-disjoint; zero-shot + few-shot regimes.
  8. Caveat: released PBT saw LFP/NMC/NCA/LCO, so LOCO on it is optimistic unless re-pretrained or Na/Zn-ion added.
  9. Gate ~week 6: build Track B only if fine-tuned PBT leaves a measurable gap on a held-out chemistry; else PBT + protocol is the contribution.
  10. Track B: LSTM→attention→Neural ODE encoder (z ∈ R^128), E=2 small experts (LFP vs rest) + always-on shared expert, router on Llama text embedding + 6 dynamics features + additive category bias (not a hard mask), softmax with τ 2.0→0.5.
  11. Output (1−β)Σg_e f_e(z) + β f_shared(z); two-stage training (10+10 epochs), five losses (RUL, MMD, balance, annealed route CE → 0, spec).
  12. Metrics per chemistry (never pooled R²), per-cell RMSE/R²/MAPE, 10 seeds, floor MAPE for RUL ≥ 50 cycles; selection on validation only.
- Assumptions / risks: deck taken from ~/Downloads (see Decisions). Guide says SNL/UL-PUR/HNEI need access requests and Llama is gated: start day one.
- Commit: "docs: Version 1 summary and open questions (document.md)"
