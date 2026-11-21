# Development log — Version 1 (Track A → Track B)

## Decisions
- 2026-11-20 — Log entries cannot contain their own hash; "Commit:" gives the exact commit subject (`git log --grep '<subject>'` finds it).
- 2026-11-20 — Slide deck read from `~/Downloads/PBT_to_Our_Model.pptx` (no .pptx in docs/); not copied into the repo pending user confirmation.

## Known issues / open questions
- (a) How is EOL obtained at inference to turn predicted fraction-of-life back into cycles? True EOL at test time is a leak (guide §3, §6.9, Part 4 item 6). Options: estimate, separate EOL predictor, or predict RUL in cycles directly. **Must be decided before any fraction-of-life training.** Not blocking Track A (A3 re-head can predict RUL in cycles).
- (b) How does PBT create experts for a category absent from pretraining (e.g. Zn anode, Na cathode)? Paper silent (PBT doc §9) → answer from repo. **ANSWERED 2026-11-21 (see "PBT repo audit"): it does not; that factor's mask is all-zero.**
- (c) Number of general experts per layer? Paper silent → answer from repo. **ANSWERED: 5 per MoE layer (released ckpt GE5); 1 in output head.**
- (d) Meaning of the "2 / 10" encoder/decoder layers hyperparameter (Supp. Table 4); guess is intra-cycle / inter-cycle stacks → answer from repo. **ANSWERED: e_layers=2 intra-cycle MoE-MLP layers, d_layers=10 inter-cycle transformer layers (+1 CyclePatch MoE tokeniser).**
- (e) Are specialist weights renormalised after the hard mask? Paper does not mention it → answer from repo. **ANSWERED: yes, divided by their sum (+1e-9).**
- (f) Is a softmax applied to the gate? Equations show a raw linear layer → answer from repo. **ANSWERED: yes, softmax over all specialists before the mask.**
- Dataset counts: PBT's recorded split (after its filters) gives MATR 169, HUST 77, SNL 50, XJTU 23, RWTH 48, UL-PUR 2, CALCE 13, HNEI 14. The guide's table (RUL-QMoE) says MATR 180, SNL 61, XJTU ~55. **Which table does the validation gate's "counts match" check against? Needs user decision.**
- SNL under PBT's split = LFP 9 / NMC 21 / NCA 20; SNL-LFP has 0 validation cells in PBT's split.
- Design guide says SNL, UL-PUR, HNEI need access requests, but processed versions are open (CC-BY-4.0) in BatteryLife v9 (Zenodo 17958489). Confirm whether we use BatteryLife-processed data or raw sources.
- PBT doc §4.7 says the output head is "five general linear layers plus five gated"; code builds 5 gated (top-2) + **1** general (`models/PBT.py:561-565`, `:397`). Re-check against the released args.json once downloaded.
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

### 2026-11-20 23:50 — FAILED: full clone of PBT stalled
- Changed: nothing (third_party/PBT removed by git on abort)
- Why: brief step 2
- Verified by: `git clone https://github.com/Ruifeng-Tan/PBT third_party/PBT` → ~8 MB of 57 MB after several minutes; stopped at user's request to pause.
- Fix: next session used `git clone --depth 1` → 1.8 s.
- Commit: none (no repo change)

### 2026-11-21 16:17 — PBT repo audit (answers b–f)
- Changed: document.md only. third_party/PBT cloned (gitignored) at commit 74f03a73f04febe66877e08bec030d148e043fa0 (2026-10-21), `--depth 1`.
- Why: brief step 2; guide §5 A1 + caveat "check the repository".
- Verified by: reading the files cited below; split counts via `python3 -I` AST parse of `data_provider/data_split_recorder.py`; sizes via Zenodo/figshare/HF public APIs.
- Answers (paths relative to third_party/PBT):
  - (b) Absent category → **no new experts are created.** Masks come from fixed dicts in `data_provider/gate_masker.py:25-64` (MIX_large, used for pretraining and for every non-MIX_all fine-tune: `finetune_model.py:615-626`). `data_provider/data_loader.py:786-789,801-804,817-820,833-837` start each factor's mask at zeros and set 1s only if the value is in the dict, so an unknown value (e.g. `MnO2`, `zinc metal`, `coin`, `Unknown`) leaves that factor's specialists at weight 0. Only the other factors' specialists + general experts remain. A cell missing from `gate_data/*.json` entirely raises (`data_loader.py:790-795`; commented alternatives: all-ones = let gate choose, all-zeros = general only). If no specialist survives in a layer, the specialist output is 0 (`models/PBT.py:184-187`), so the layer = general experts (+ residual).
  - (c) **5 general experts per MoE layer** (`--num_general_experts`, `models/PBT.py:533-535,543-545,553-555`; released ckpt name `..._E5_GE5_...` in `scripts/finetune_PBT.sh:5`, decoded by `run_main.py:307,322`). Output head: 1 general linear (`models/PBT.py:563-565`) + `num_experts`=5 gated linears, top-2 hard-coded (`models/PBT.py:397`), no hard mask on the head.
  - (d) **e_layers = intra-cycle MoE-MLP layers, d_layers = inter-cycle transformer layers** (`models/PBT.py:501,539-547,550-558`). Released ckpt: `el2_dl10` (`scripts/finetune_PBT.sh:5`). Plus one CyclePatch tokeniser MoE layer (`models/PBT.py:529-537`), so 13 MoE layers. NB `scripts/PBT.sh:31` defaults to d_layers=5, not the released 10.
  - (e) **Yes, renormalised**: softmax → ×mask → optional top-K (off, topK=-1) → divide by sum+1e-9 (`models/PBT.py:156-172, 244-259, 333-348`). So the specialists form a convex combination; general experts are added with weight 1 (`models/PBT.py:70-71,127-128`).
  - (f) **Yes, softmax** over all specialists (cathode+anode+format+temperature concatenated: 7+6+6+10=29 in MIX_large, `gate_masker.py:64`) *before* the mask (`models/PBT.py:157,245,334`). Gate path: ê = LeakyReLU(Linear(4096→512)) shared (`models/PBT.py:522`); per-layer `Linear(512→29, bias=False)` (`:40,:86`).
  - Extra: guide + load-balance losses exist (`models/PBT.py:194-209`) but are off in the released model (`run_main.py:184,186,553`; ckpt name `guideFalse_LBFalse`). This matches the paper.
- Data format: one pickle per cell, `root_path/<DATASET>/<cell>.pkl`, dict with `nominal_capacity_in_Ah`, `SOC_interval`, `cycle_data` (list of per-cycle dicts with `current_in_A`, `voltage_in_V`, `charge_capacity_in_Ah`, `discharge_capacity_in_Ah`, `time_in_s`) (`data_loader.py:372,1049-1061`). Life labels: `root_path/Life labels/<PREFIX>_labels.json` {file_name: eol} (`data_loader.py:1004-1025`), computed by `process_scripts/extract_life_labels.py:32-63` (SOH = maxQd/nominal/SOC_interval; ≥0.825 at end → dropped; (0.8,0.825) → linear extrapolation to 0.80; else first cycle ≤0.8). Cells with no label are **silently skipped** (`data_loader.py:779-781`); cells with eol ≤100 are dropped (`:1115`). Splits are fixed file lists in `data_provider/data_split_recorder.py` (MATR line 8, HUST 4, SNL 13, RWTH 17, UL_PUR 29, CALCE 33, HNEI 37, XJTU 58), one split per single dataset (only MIX/ZN/CALB/NA have per-seed variants).
- Preprocessed data / loaders: **no raw-data loaders** for MATR/HUST/SNL/XJTU/RWTH/UL-PUR/CALCE/HNEI. It reads BatteryLife-processed pickles: BatteryLife v9, Zenodo 17958489 (open, CC-BY-4.0, 31.09 GB total; our 8 datasets = MATR 4.86 + HUST 1.39 + RWTH 1.06 + XJTU 0.45 + SNL 0.11 + CALCE 0.09 + HNEI 0.04 + UL_PUR 0.01 = 8.01 GB). Life labels + corrected ZN/CALB: figshare 31148413 (CC-BY-4.0, ~1 GB, mostly ZN-coin). Split counts (train/val/test): MATR 102/34/33, HUST 47/15/15, SNL 30/10/10, XJTU 15/4/4, RWTH 30/9/9, UL_PUR 2/0/0, CALCE 9/2/2, HNEI 9/3/2.
- Text prompt: hand-written per-condition strings in `Prompts/<DS>_protocol_prompt.py` (template `Prompts/Protocol Prompt Template.py`), selected by cell name in `Prompts/Mapping_helper.py:1529 do_mapping`; prefixed with a task-description sentence (80%, or 90% for CALB) in `data_loader.py:1079-1105`. Covers specs (format, cathode, anode, electrolyte, manufacturer, nominal capacity), formation/"working history", temperature, charge/discharge protocol, SOC range.
- Llama embeddings: `generate_aging_condition_pretraining.ipynb` cells 5–9. Chat template (system "You are a helpful assistant.") → tokenizer → drop BOS → `get_input_embeddings` → forward with causal mask → **last-valid-token** hidden state (4096-d). Model loaded with `load_in_4bit=True` (bitsandbytes, CUDA-only). Cached as dict {cell_name: vec} in `root_path/{training,validation,testing}_DKP_embed_all_Llama.pkl`, read at `data_loader.py:631-645`. Pre-computed set on Zenodo 17972780 (`aging_condition_prompt_embeddings.zip`, 6 MB). A cell missing from the cache gets a **zero vector silently** (`data_loader.py:918-919`). NB the notebook's committed `prompt_mode` is `'specification_reordered'`, not `'full'`.
- Fine-tuning / adapters: `sh scripts/finetune_PBT.sh` → `accelerate launch --mixed_precision bf16 finetune_model.py`. Reads ckpt hparams from `args_path/args.json` and overwrites CLI args (`finetune_model.py:518-561`); loads weights with `load_checkpoint_in_model` (`:752`) and `label_scaler` from args_path. `--finetune_method FT` = all params; `AT` / `AT_reverse` / `AT_nCP` = bottleneck adapters (LayerNorm→Linear→GELU→Linear, `layers/Adapters.py:10-12`) + regression head, width `--adapter_size`, depth `--adapter_layers`. Pretraining: `sh scripts/PBT.sh` → `run_main.py`. Eval: `scripts/evaluate_model.sh`.
- Requirements (`requirements.txt`, pipreqs-generated): torch 2.4.1, transformers 4.44.2, accelerate 1.1.1, peft 0.12.0, datasets 3.0.0, evaluate 0.4.3, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, scikit_learn 1.8.0, matplotlib 3.10.8, joblib 1.4.2, tqdm 4.66.5, reformer_pytorch 1.4.4, BatteryML 0.0.1. numpy/sklearn/scipy pins need **Python ≥ 3.11**. No CUDA version pinned. Missing from requirements but imported: `wandb` (run_main/finetune/evaluate/utils/tools), `denseweight` (data_loader:10), `huggingface_hub`; notebook also needs `bitsandbytes`, `seaborn`. `BatteryML` is **not on PyPI** (404) and is only imported for a docstring (`data_loader.py:25`).
- Model init needs, even with cached embeddings: Llama **tokenizer + config** from `LLM_path` with `local_files_only=True` (`models/PBT.py:487-493`, `finetune_model.py:595`) and a PCA pickle `pca_path` (`models/PBT.py:519`). So the gated HF access is still required (small files only), and `MIX_large_pca_0.99_Llama.pkl` must ship with the ckpt or be regenerated → check once the ckpt is downloaded.
- Assumptions / risks: line numbers valid for commit 74f03a7 only. Zenodo/HF sizes are from public API metadata, not downloads.
- Commit: "docs: PBT repo audit, answers to open questions b-f (document.md)"
