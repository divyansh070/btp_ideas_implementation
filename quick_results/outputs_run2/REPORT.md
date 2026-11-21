# Quick results: Version 1 proof of concept

## Setup
- Datasets: SNL, CALCE, HNEI, HUST (BatteryLife v9 processed pickles, PBT life-label rule).
- Kept cells: 154; windows: 22969 (30-cycle windows, 10 sampled, stride 1, at most 150 per cell, evenly spaced).
- Seeds: [0, 1, 2] (each seed redraws the cell splits and the model init).
- Leave-one-chemistry-out folds: ['LFP', 'NMC', 'NCA', 'LCO']; HNEI (NMC-LCO) is always source. Held-out chemistry: 60 % adaptation / 10 % validation / 30 % blind test, cell-disjoint; source 80/20 train/val.
- Selection: best epoch on validation cells only (source val for zero-shot; source val + target val otherwise). The blind test is only scored.
- Runtime (s): {'e0': 272.2, 'e1': 1.0, 'e2': 512.3, 'e3': 1548.2, 'e4': 0.0, 'report': 6.8}; host: 3dc8137e9959 / torch device cuda; code commit 7c7387b.

Dropped cells: 11 (9 dropped: last-cycle SOH >= 0.825; 2 no_prompt_embedding). Missing prompt embeddings policy: `drop`.

| name | chem | dataset | reason |
|---|---|---|---|
| SNL_18650_LFP_15C_0-100_0.5-2C_b | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-0.5C_a | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-1C_a | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-1C_b | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-1C_c | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-1C_d | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-2C_a | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_25C_0-100_0.5-2C_b | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_LFP_35C_0-100_0.5-1C_a | LFP | SNL | dropped: last-cycle SOH >= 0.825 |
| SNL_18650_NCA_25C_20-80_0.5-0.5C_a | NCA | SNL | no_prompt_embedding |
| SNL_18650_NCA_25C_20-80_0.5-0.5C_b | NCA | SNL | no_prompt_embedding |

Cells per split (seed 0):

| fold | test | val | adapt | src_train | src_val |
|---|---|---|---|---|---|
| LFP | 26 | 9 | 51 | 54 | 14 |
| NMC | 6 | 2 | 13 | 106 | 27 |
| NCA | 6 | 2 | 12 | 107 | 27 |
| LCO | 4 | 1 | 8 | 113 | 28 |

## E0: pipeline and validation gate

| chem | dataset | raw_cells | kept_cells | windows | nan_windows_dropped | eol_min | eol_median | eol_max |
|---|---|---|---|---|---|---|---|---|
| LCO | CALCE | 13 | 13 | 1950 | 0 | 448 | 534.0 | 828 |
| LFP | HUST | 77 | 77 | 11550 | 0 | 1164 | 1911.0 | 2768 |
| LFP | SNL | 18 | 9 | 1350 | 0 | 2325 | 3162.0 | 3842 |
| NCA | SNL | 22 | 20 | 3000 | 0 | 180 | 334.0 | 840 |
| NMC | SNL | 21 | 21 | 3019 | 0 | 143 | 468.0 | 639 |
| NMC-LCO | HNEI | 14 | 14 | 2100 | 0 | 218 | 251.0 | 266 |

| check | value | passed | detail |
|---|---|---|---|
| cells dropped: no prompt embedding | 2 | PASS | SNL_18650_NCA_25C_20-80_0.5-0.5C_a, SNL_18650_NCA_25C_20-80_0.5-0.5C_b |
| count LCO | 13 | PASS | expected 13 |
| count LFP | 86 | PASS | expected 86 |
| count NCA | 20 | PASS | expected 20 |
| count NMC | 21 | PASS | expected 21 |
| count NMC-LCO | 14 | PASS | expected 14 |
| EOL fallbacks | 0 | PASS | nan |
| EOL vs PBT labels (mismatch or dropped-but-labelled) | 0 | PASS | nan |
| kept cells without a PBT label | 0 | PASS | nan |
| per-cell corr(RUL, capacity) > 0.5 | 0.86 | PASS | nan |
| SNL tagged LFP/NMC/NCA | {'LFP': 9, 'NCA': 20, 'NMC': 21} | PASS | nan |
| no cell tagged twice | 0 | PASS | nan |
| fold LFP: target features p1-p99 in [-0.2, 1.2] | 0 | PASS | seed 0: 0 out, range [0.0, 1.0], feats []; seed 1: 0 out, range [0.0, 1.0], feats []; seed 2: 0 out, range [0.0, 1.0], feats [] |
| fold LFP: min feature std > 0.01 | 0.013 | PASS | nan |
| fold NMC: target features p1-p99 in [-0.2, 1.2] | 0 | PASS | seed 0: 0 out, range [0.0, 0.98], feats []; seed 1: 0 out, range [0.0, 1.0], feats []; seed 2: 0 out, range [0.0, 1.0], feats [] |
| fold NMC: min feature std > 0.01 | 0.0128 | PASS | nan |
| fold NCA: target features p1-p99 in [-0.2, 1.2] | 0 | PASS | seed 0: 0 out, range [0.0, 0.98], feats []; seed 1: 0 out, range [0.0, 0.98], feats []; seed 2: 0 out, range [0.0, 0.98], feats [] |
| fold NCA: min feature std > 0.01 | 0.0113 | PASS | nan |
| fold LCO: target features p1-p99 in [-0.2, 1.2] | 0 | PASS | seed 0: 0 out, range [0.0, 1.02], feats []; seed 1: 0 out, range [0.0, 0.98], feats []; seed 2: 0 out, range [0.0, 0.98], feats [] |
| fold LCO: min feature std > 0.01 | 0.0116 | PASS | nan |
| splits cell-disjoint (all folds x seeds) | ok | PASS | nan |

Gate: **PASS**

Feature constancy (raw, unscaled; `frac_cells_constant` = share of cells where the feature does not move inside the cell). Informational, not a gate check:

| feature | raw_mean | raw_global_std | median_within_cell_std | frac_cells_constant | globally_constant |
|---|---|---|---|---|---|
| V_mean | 3.5194 | 0.2452 | 0.0066 | 0.1688 | False |
| V_std | 0.2985 | 0.0757 | 0.0128 | 0.0 | False |
| V_min | 2.2025 | 0.3582 | 0.0116 | 0.4156 | False |
| V_max | 3.8694 | 0.3081 | 0.0005 | 0.9675 | False |
| V_var | 0.0948 | 0.0498 | 0.0068 | 0.0 | False |
| V_median | 3.575 | 0.2508 | 0.0117 | 0.1039 | False |
| I_C_mean | -0.0516 | 0.1115 | 0.0049 | 0.0 | False |
| I_C_std | 1.8493 | 1.0746 | 0.0404 | 0.013 | False |
| I_C_min | -2.9252 | 1.6613 | 0.0003 | 0.7013 | False |
| I_C_max | 2.7955 | 2.2208 | 0.0003 | 0.8636 | False |
| I_C_var | 4.5747 | 4.0809 | 0.1346 | 0.026 | False |
| I_C_median | 0.1379 | 0.4507 | 0.08 | 0.0649 | False |
| Qd_mean | 0.3214 | 0.1254 | 0.0252 | 0.0 | False |
| Qd_std | 0.4973 | 0.2058 | 0.0313 | 0.0 | False |
| Qd_max | 1.6037 | 0.7148 | 0.0867 | 0.0 | False |
| Qd_var | 0.2896 | 0.2441 | 0.0227 | 0.0 | False |
| Qd_median | 0.012 | 0.0456 | 0.0064 | 0.3701 | False |
| d_V_mean | 0.0005 | 0.0189 | 0.0024 | 0.0 | False |
| d_V_std | 0.0006 | 0.0098 | 0.0032 | 0.0 | False |
| d_I_C_mean | 0.0002 | 0.0201 | 0.0043 | 0.0 | False |
| d_I_C_std | 0.0014 | 0.0467 | 0.0072 | 0.0 | False |
| d_I_C_min | -0.0019 | 0.1511 | 0.0003 | 0.013 | False |
| d_Qd_mean | -0.0024 | 0.0471 | 0.0044 | 0.0 | False |
| d_Qd_std | -0.0029 | 0.0265 | 0.0026 | 0.0 | False |
| d_Qd_max | -0.0093 | 0.0199 | 0.0034 | 0.0 | False |

## E1: why fraction of life

![E1](e1_fraction_of_life.png)

| chem | cells | eol_min | eol_median | eol_max | rul_p95_cycles | frac_p95 |
|---|---|---|---|---|---|---|
| LCO | 13 | 448 | 534 | 828 | 537 | 0.901 |
| LFP | 86 | 1164 | 1978 | 3842 | 2232 | 0.939 |
| NCA | 20 | 180 | 334 | 840 | 476 | 0.868 |
| NMC | 21 | 143 | 468 | 639 | 482 | 0.884 |
| NMC-LCO | 14 | 218 | 251 | 266 | 210 | 0.837 |

RUL in cycles spans very different ranges per chemistry; fraction of life puts every chemistry on [0, 1]. Converting back to cycles needs EOL, which is unknown in service (open issue a).

## E2: cross-chemistry gap (single-head models)

Metrics on the held-out chemistry's blind-test cells only (per chemistry; R² never pooled). `oracle_eol=True`: fraction of life converted with the TRUE EOL. `lstm_rul` predicts cycles directly (no oracle).

| fold | regime | model | oracle_eol | rmse | r2 | r2_cell_mean | mape50 | seeds |
|---|---|---|---|---|---|---|---|---|
| LFP | target_only | lstm_frac | True | 103 ± 10 | 0.973 ± 0.0027 | 0.973 ± 0.0018 | 0.108 ± 0.0094 | 3 |
| LFP | target_only | lstm_rul | False | 259 ± 57 | 0.829 ± 0.057 | 0.752 ± 0.074 | 0.286 ± 0.092 | 3 |
| LFP | target_only | lgbm | True | 156 ± 91 | 0.93 ± 0.071 | 0.937 ± 0.043 | 0.0909 ± 0.013 | 3 |
| LFP | zero_shot | lstm_frac | True | 727 ± 94 | -0.383 ± 0.48 | -0.58 ± 0.64 | 1.56 ± 0.53 | 3 |
| LFP | zero_shot | lstm_rul | False | 818 ± 1.5e+02 | -0.726 ± 0.59 | -0.894 ± 0.68 | 0.788 ± 0.12 | 3 |
| LFP | zero_shot | lgbm | True | 416 ± 59 | 0.563 ± 0.078 | 0.546 ± 0.063 | 0.408 ± 0.076 | 3 |
| LFP | few_2 | lstm_frac | True | 451 ± 1.2e+02 | 0.476 ± 0.23 | 0.487 ± 0.08 | 0.524 ± 0.18 | 3 |
| LFP | few_2 | lstm_rul | False | 618 ± 99 | 0.0314 ± 0.21 | -0.133 ± 0.18 | 1.03 ± 0.19 | 3 |
| LFP | few_2 | lgbm | True | 337 ± 76 | 0.708 ± 0.1 | 0.722 ± 0.084 | 0.299 ± 0.07 | 3 |
| LFP | few_5 | lstm_frac | True | 358 ± 72 | 0.673 ± 0.098 | 0.622 ± 0.076 | 0.492 ± 0.065 | 3 |
| LFP | few_5 | lstm_rul | False | 588 ± 1.7e+02 | 0.0967 ± 0.41 | -0.107 ± 0.24 | 0.882 ± 0.29 | 3 |
| LFP | few_5 | lgbm | True | 289 ± 1e+02 | 0.778 ± 0.13 | 0.822 ± 0.081 | 0.201 ± 0.025 | 3 |
| NMC | target_only | lstm_frac | True | 48.1 ± 5.7 | 0.879 ± 0.049 | 0.853 ± 0.032 | 0.184 ± 0.011 | 3 |
| NMC | target_only | lstm_rul | False | 147 ± 11 | -0.0717 ± 0.11 | -5.64 ± 1.4 | 0.78 ± 0.043 | 3 |
| NMC | target_only | lgbm | True | 42.1 ± 4.6 | 0.908 ± 0.033 | 0.816 ± 0.13 | 0.173 ± 0.035 | 3 |
| NMC | zero_shot | lstm_frac | True | 129 ± 34 | 0.191 ± 0.27 | -0.185 ± 0.4 | 0.499 ± 0.078 | 3 |
| NMC | zero_shot | lstm_rul | False | 159 ± 27 | -0.229 ± 0.12 | -0.933 ± 0.31 | 0.49 ± 0.033 | 3 |
| NMC | zero_shot | lgbm | True | 154 ± 30 | -0.149 ± 0.16 | -0.314 ± 0.16 | 0.515 ± 0.04 | 3 |
| NMC | few_2 | lstm_frac | True | 128 ± 24 | 0.206 ± 0.098 | 0.02 ± 0.19 | 0.441 ± 0.051 | 3 |
| NMC | few_2 | lstm_rul | False | 152 ± 26 | -0.117 ± 0.12 | -1.32 ± 0.49 | 0.52 ± 0.03 | 3 |
| NMC | few_2 | lgbm | True | 116 ± 49 | 0.298 ± 0.43 | 0.227 ± 0.42 | 0.396 ± 0.086 | 3 |
| NMC | few_5 | lstm_frac | True | 86.8 ± 1.6 | 0.621 ± 0.09 | 0.452 ± 0.15 | 0.355 ± 0.056 | 3 |
| NMC | few_5 | lstm_rul | False | 149 ± 26 | -0.0769 ± 0.12 | -1.98 ± 0.83 | 0.577 ± 0.035 | 3 |
| NMC | few_5 | lgbm | True | 77.3 ± 37 | 0.685 ± 0.21 | 0.601 ± 0.14 | 0.311 ± 0.075 | 3 |
| NCA | target_only | lstm_frac | True | 88.9 ± 11 | 0.404 ± 0.27 | 0.212 ± 0.23 | 0.393 ± 0.032 | 3 |
| NCA | target_only | lstm_rul | False | 128 ± 32 | -0.106 ± 0.17 | -0.802 ± 0.25 | 0.605 ± 0.081 | 3 |
| NCA | target_only | lgbm | True | 53.1 ± 8.9 | 0.742 ± 0.23 | 0.698 ± 0.21 | 0.233 ± 0.056 | 3 |
| NCA | zero_shot | lstm_frac | True | 158 ± 37 | -0.767 ± 0.7 | -1.67 ± 0.18 | 0.915 ± 0.11 | 3 |
| NCA | zero_shot | lstm_rul | False | 119 ± 32 | 0.0612 ± 0.078 | -0.583 ± 0.49 | 0.589 ± 0.21 | 3 |
| NCA | zero_shot | lgbm | True | 141 ± 25 | -0.415 ± 0.52 | -1.26 ± 0.15 | 0.815 ± 0.12 | 3 |
| NCA | few_2 | lstm_frac | True | 117 ± 40 | 0.109 ± 0.096 | -0.412 ± 0.51 | 0.619 ± 0.16 | 3 |
| NCA | few_2 | lstm_rul | False | 110 ± 34 | 0.205 ± 0.11 | -0.212 ± 0.31 | 0.483 ± 0.1 | 3 |
| NCA | few_2 | lgbm | True | 88.2 ± 4.1 | 0.402 ± 0.32 | 0.158 ± 0.059 | 0.443 ± 0.0093 | 3 |
| NCA | few_5 | lstm_frac | True | 95.9 ± 20 | 0.345 ± 0.25 | 0.299 ± 0.22 | 0.438 ± 0.054 | 3 |
| NCA | few_5 | lstm_rul | False | 133 ± 22 | -0.29 ± 0.57 | -0.74 ± 0.47 | 0.599 ± 0.12 | 3 |
| NCA | few_5 | lgbm | True | 66.2 ± 4.7 | 0.633 ± 0.28 | 0.546 ± 0.22 | 0.293 ± 0.061 | 3 |
| LCO | target_only | lstm_frac | True | 93.6 ± 12 | 0.71 ± 0.053 | 0.727 ± 0.059 | 0.329 ± 0.028 | 3 |
| LCO | target_only | lstm_rul | False | 170 ± 12 | 0.0504 ± 0.072 | -0.00379 ± 0.059 | 0.624 ± 0.033 | 3 |
| LCO | target_only | lgbm | True | 104 ± 50 | 0.609 ± 0.33 | 0.573 ± 0.38 | 0.405 ± 0.22 | 3 |
| LCO | zero_shot | lstm_frac | True | 289 ± 7.4 | -1.76 ± 0.22 | -2.06 ± 0.079 | 0.783 ± 0.019 | 3 |
| LCO | zero_shot | lstm_rul | False | 223 ± 24 | -0.655 ± 0.32 | -0.722 ± 0.34 | 0.554 ± 0.024 | 3 |
| LCO | zero_shot | lgbm | True | 206 ± 22 | -0.416 ± 0.28 | -0.455 ± 0.27 | 0.67 ± 0.077 | 3 |
| LCO | few_2 | lstm_frac | True | 174 ± 14 | -0.00931 ± 0.23 | -0.169 ± 0.32 | 0.597 ± 0.062 | 3 |
| LCO | few_2 | lstm_rul | False | 225 ± 7.2 | -0.678 ± 0.13 | -0.741 ± 0.11 | 0.548 ± 0.011 | 3 |
| LCO | few_2 | lgbm | True | 158 ± 21 | 0.166 ± 0.18 | 0.063 ± 0.29 | 0.454 ± 0.091 | 3 |
| LCO | few_5 | lstm_frac | True | 167 ± 56 | 0.0379 ± 0.53 | -0.045 ± 0.65 | 0.558 ± 0.094 | 3 |
| LCO | few_5 | lstm_rul | False | 201 ± 21 | -0.346 ± 0.25 | -0.378 ± 0.24 | 0.551 ± 0.017 | 3 |
| LCO | few_5 | lgbm | True | 136 ± 33 | 0.379 ± 0.24 | 0.347 ± 0.21 | 0.458 ± 0.16 | 3 |

Gap, `lstm_frac` mean RMSE (cycles, oracle EOL):

- LCO: few_2 174, few_5 167, target_only 94, zero_shot 289
- LFP: few_2 451, few_5 358, target_only 103, zero_shot 727
- NCA: few_2 117, few_5 96, target_only 89, zero_shot 158
- NMC: few_2 128, few_5 87, target_only 48, zero_shot 129

## E3: mini Track B (MoE) vs single head

All rows use oracle EOL.

| fold | held_out_has_matching_expert | regime | model | rmse | r2 | r2_cell_mean | mape50 | seeds |
|---|---|---|---|---|---|---|---|---|
| LFP | False | zero_shot | single_head | 727 ± 94 | -0.383 ± 0.48 | -0.58 ± 0.64 | 1.56 ± 0.53 | 3 |
| LFP | False | few_2 | single_head | 451 ± 1.2e+02 | 0.476 ± 0.23 | 0.487 ± 0.08 | 0.524 ± 0.18 | 3 |
| LFP | False | few_5 | single_head | 358 ± 72 | 0.673 ± 0.098 | 0.622 ± 0.076 | 0.492 ± 0.065 | 3 |
| NMC | True | zero_shot | single_head | 129 ± 34 | 0.191 ± 0.27 | -0.185 ± 0.4 | 0.499 ± 0.078 | 3 |
| NMC | True | few_2 | single_head | 128 ± 24 | 0.206 ± 0.098 | 0.02 ± 0.19 | 0.441 ± 0.051 | 3 |
| NMC | True | few_5 | single_head | 86.8 ± 1.6 | 0.621 ± 0.09 | 0.452 ± 0.15 | 0.355 ± 0.056 | 3 |
| NCA | False | zero_shot | single_head | 158 ± 37 | -0.767 ± 0.7 | -1.67 ± 0.18 | 0.915 ± 0.11 | 3 |
| NCA | False | few_2 | single_head | 117 ± 40 | 0.109 ± 0.096 | -0.412 ± 0.51 | 0.619 ± 0.16 | 3 |
| NCA | False | few_5 | single_head | 95.9 ± 20 | 0.345 ± 0.25 | 0.299 ± 0.22 | 0.438 ± 0.054 | 3 |
| LCO | True | zero_shot | single_head | 289 ± 7.4 | -1.76 ± 0.22 | -2.06 ± 0.079 | 0.783 ± 0.019 | 3 |
| LCO | True | few_2 | single_head | 174 ± 14 | -0.00931 ± 0.23 | -0.169 ± 0.32 | 0.597 ± 0.062 | 3 |
| LCO | True | few_5 | single_head | 167 ± 56 | 0.0379 ± 0.53 | -0.045 ± 0.65 | 0.558 ± 0.094 | 3 |
| LFP | False | zero_shot | moe_mask | 840 ± 65 | -0.786 ± 0.1 | -1.13 ± 0.11 | 1.87 ± 0.13 | 3 |
| LFP | False | zero_shot | moe_bias | 760 ± 98 | -0.5 ± 0.48 | -0.744 ± 0.68 | 1.6 ± 0.52 | 3 |
| LFP | False | few_2 | moe_mask | 422 ± 1.2e+02 | 0.541 ± 0.2 | 0.536 ± 0.088 | 0.558 ± 0.06 | 3 |
| LFP | False | few_2 | moe_bias | 429 ± 1.4e+02 | 0.519 ± 0.27 | 0.533 ± 0.13 | 0.497 ± 0.067 | 3 |
| LFP | False | few_5 | moe_mask | 416 ± 1.3e+02 | 0.548 ± 0.23 | 0.576 ± 0.09 | 0.447 ± 0.015 | 3 |
| LFP | False | few_5 | moe_bias | 447 ± 1.7e+02 | 0.468 ± 0.36 | 0.519 ± 0.16 | 0.462 ± 0.067 | 3 |
| NMC | True | zero_shot | moe_mask | 141 ± 25 | 0.0313 ± 0.13 | -0.29 ± 0.33 | 0.516 ± 0.09 | 3 |
| NMC | True | zero_shot | moe_bias | 146 ± 28 | -0.0344 ± 0.15 | -0.379 ± 0.35 | 0.526 ± 0.09 | 3 |
| NMC | True | few_2 | moe_mask | 134 ± 22 | 0.131 ± 0.081 | -0.0987 ± 0.19 | 0.479 ± 0.072 | 3 |
| NMC | True | few_2 | moe_bias | 136 ± 23 | 0.101 ± 0.097 | -0.166 ± 0.21 | 0.478 ± 0.074 | 3 |
| NMC | True | few_5 | moe_mask | 116 ± 37 | 0.345 ± 0.26 | 0.145 ± 0.36 | 0.408 ± 0.087 | 3 |
| NMC | True | few_5 | moe_bias | 120 ± 38 | 0.298 ± 0.29 | 0.101 ± 0.37 | 0.425 ± 0.083 | 3 |
| NCA | False | zero_shot | moe_mask | 165 ± 30 | -0.976 ± 0.84 | -1.96 ± 0.32 | 0.939 ± 0.033 | 3 |
| NCA | False | zero_shot | moe_bias | 155 ± 31 | -0.757 ± 0.81 | -1.67 ± 0.29 | 0.887 ± 0.062 | 3 |
| NCA | False | few_2 | moe_mask | 104 ± 22 | 0.202 ± 0.39 | -0.0455 ± 0.16 | 0.518 ± 0.081 | 3 |
| NCA | False | few_2 | moe_bias | 125 ± 24 | -0.101 ± 0.36 | -0.716 ± 0.23 | 0.69 ± 0.11 | 3 |
| NCA | False | few_5 | moe_mask | 72.2 ± 10 | 0.608 ± 0.18 | 0.463 ± 0.12 | 0.33 ± 0.018 | 3 |
| NCA | False | few_5 | moe_bias | 80.9 ± 9.6 | 0.524 ± 0.18 | 0.226 ± 0.083 | 0.388 ± 0.017 | 3 |
| LCO | True | zero_shot | moe_mask | 280 ± 7.2 | -1.59 ± 0.31 | -1.86 ± 0.2 | 0.734 ± 0.047 | 3 |
| LCO | True | zero_shot | moe_bias | 281 ± 4.6 | -1.62 ± 0.26 | -1.89 ± 0.14 | 0.74 ± 0.035 | 3 |
| LCO | True | few_2 | moe_mask | 161 ± 6.1 | 0.138 ± 0.12 | 0.0539 ± 0.088 | 0.607 ± 0.069 | 3 |
| LCO | True | few_2 | moe_bias | 174 ± 14 | -0.00148 ± 0.074 | -0.113 ± 0.14 | 0.591 ± 0.029 | 3 |
| LCO | True | few_5 | moe_mask | 152 ± 34 | 0.22 ± 0.28 | 0.15 ± 0.27 | 0.521 ± 0.14 | 3 |
| LCO | True | few_5 | moe_bias | 163 ± 19 | 0.12 ± 0.14 | 0.0393 ± 0.15 | 0.561 ± 0.082 | 3 |

Diagnostics at the selected epoch (validation windows, mean over seeds):

| fold | regime | model | beta | gate_entropy | gate_entropy_source | gate_entropy_heldout | logit_gap_over_tau_source | gate_mean_e0 | gate_mean_e1 | max_expert_corr | val_pred_min | val_pred_max | text_norm_raw_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LCO | few_2 | moe_bias | 0.494 | 0.274 | 0.278 | 0.17 | 1.523 | 0.517 | 0.483 | 0.891 | 0.099 | 0.887 | 148.199 |
| LCO | few_2 | moe_mask | 0.485 | 0.0 | 0.0 | 0.0 | 2.251 | 0.589 | 0.411 | 0.88 | 0.085 | 0.897 | 148.199 |
| LCO | few_5 | moe_bias | 0.483 | 0.305 | 0.305 | 0.29 | 1.294 | 0.5 | 0.5 | 0.926 | 0.085 | 0.886 | 148.2 |
| LCO | few_5 | moe_mask | 0.491 | 0.0 | 0.0 | 0.0 | 0.925 | 0.589 | 0.411 | 0.901 | 0.083 | 0.902 | 148.2 |
| LCO | zero_shot | moe_bias | 0.469 | 0.213 | 0.213 | nan | 1.47 | 0.578 | 0.422 | 0.892 | 0.084 | 0.892 | 148.198 |
| LCO | zero_shot | moe_mask | 0.468 | 0.0 | 0.0 | nan | 1.596 | 0.61 | 0.39 | 0.888 | 0.084 | 0.893 | 148.198 |
| LFP | few_2 | moe_bias | 0.477 | 0.361 | 0.317 | 0.427 | 1.44 | 0.47 | 0.53 | 0.812 | 0.126 | 0.856 | 148.256 |
| LFP | few_2 | moe_mask | 0.481 | 0.083 | 0.138 | 0.0 | 2.484 | 0.225 | 0.378 | 0.725 | 0.134 | 0.854 | 148.256 |
| LFP | few_5 | moe_bias | 0.467 | 0.405 | 0.391 | 0.426 | 1.167 | 0.486 | 0.514 | 0.847 | 0.124 | 0.85 | 148.252 |
| LFP | few_5 | moe_mask | 0.481 | 0.126 | 0.208 | 0.0 | 1.85 | 0.162 | 0.442 | 0.812 | 0.132 | 0.833 | 148.252 |
| LFP | zero_shot | moe_bias | 0.472 | 0.334 | 0.334 | nan | 1.319 | 0.392 | 0.608 | 0.922 | 0.155 | 0.841 | 148.275 |
| LFP | zero_shot | moe_mask | 0.467 | 0.061 | 0.061 | nan | 3.582 | 0.476 | 0.524 | 0.828 | 0.16 | 0.852 | 148.275 |
| NCA | few_2 | moe_bias | 0.48 | 0.261 | 0.258 | 0.303 | 1.347 | 0.53 | 0.47 | 0.922 | 0.09 | 0.891 | 148.195 |
| NCA | few_2 | moe_mask | 0.449 | 0.0 | 0.0 | 0.0 | 2.664 | 0.589 | 0.342 | 0.878 | 0.09 | 0.889 | 148.195 |
| NCA | few_5 | moe_bias | 0.5 | 0.332 | 0.318 | 0.52 | 1.295 | 0.531 | 0.469 | 0.902 | 0.124 | 0.89 | 148.196 |
| NCA | few_5 | moe_mask | 0.452 | 0.0 | 0.0 | 0.0 | 1.678 | 0.589 | 0.342 | 0.873 | 0.119 | 0.899 | 148.196 |
| NCA | zero_shot | moe_bias | 0.483 | 0.213 | 0.213 | nan | 1.95 | 0.595 | 0.405 | 0.875 | 0.091 | 0.881 | 148.192 |
| NCA | zero_shot | moe_mask | 0.476 | 0.0 | 0.0 | nan | 2.434 | 0.633 | 0.367 | 0.847 | 0.09 | 0.895 | 148.192 |
| NMC | few_2 | moe_bias | 0.493 | 0.276 | 0.28 | 0.212 | 1.08 | 0.553 | 0.447 | 0.947 | 0.099 | 0.897 | 148.204 |
| NMC | few_2 | moe_mask | 0.495 | 0.0 | 0.0 | 0.0 | 1.122 | 0.586 | 0.414 | 0.929 | 0.096 | 0.896 | 148.204 |
| NMC | few_5 | moe_bias | 0.488 | 0.113 | 0.109 | 0.161 | 2.226 | 0.594 | 0.406 | 0.863 | 0.082 | 0.898 | 148.204 |
| NMC | few_5 | moe_mask | 0.483 | 0.0 | 0.0 | 0.0 | 1.282 | 0.586 | 0.414 | 0.873 | 0.085 | 0.902 | 148.204 |
| NMC | zero_shot | moe_bias | 0.489 | 0.227 | 0.227 | nan | 1.716 | 0.589 | 0.411 | 0.934 | 0.091 | 0.898 | 148.205 |
| NMC | zero_shot | moe_mask | 0.475 | 0.0 | 0.0 | nan | 2.975 | 0.63 | 0.37 | 0.926 | 0.088 | 0.902 | 148.205 |

![E3 gates](e3_gate_weights.png)

Mean gate weight per chemistry:

| fold | regime | variant | set | chem | gate_e0 | gate_e1 |
|---|---|---|---|---|---|---|
| LCO | few_2 | bias | source_val | LFP | 0.854 | 0.146 |
| LCO | few_2 | bias | source_val | NCA | 0.025 | 0.975 |
| LCO | few_2 | bias | source_val | NMC | 0.021 | 0.979 |
| LCO | few_2 | bias | source_val | NMC-LCO | 0.056 | 0.944 |
| LCO | few_2 | bias | test | LCO | 0.054 | 0.946 |
| LCO | few_2 | mask | source_val | LFP | 1.0 | 0.0 |
| LCO | few_2 | mask | source_val | NCA | 0.0 | 1.0 |
| LCO | few_2 | mask | source_val | NMC | 0.0 | 1.0 |
| LCO | few_2 | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| LCO | few_2 | mask | test | LCO | 0.0 | 1.0 |
| LCO | few_5 | bias | source_val | LFP | 0.827 | 0.173 |
| LCO | few_5 | bias | source_val | NCA | 0.014 | 0.986 |
| LCO | few_5 | bias | source_val | NMC | 0.019 | 0.981 |
| LCO | few_5 | bias | source_val | NMC-LCO | 0.043 | 0.957 |
| LCO | few_5 | bias | test | LCO | 0.116 | 0.884 |
| LCO | few_5 | mask | source_val | LFP | 1.0 | 0.0 |
| LCO | few_5 | mask | source_val | NCA | 0.0 | 1.0 |
| LCO | few_5 | mask | source_val | NMC | 0.0 | 1.0 |
| LCO | few_5 | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| LCO | few_5 | mask | test | LCO | 0.0 | 1.0 |
| LCO | zero_shot | bias | source_val | LFP | 0.917 | 0.083 |
| LCO | zero_shot | bias | source_val | NCA | 0.028 | 0.972 |
| LCO | zero_shot | bias | source_val | NMC | 0.03 | 0.97 |
| LCO | zero_shot | bias | source_val | NMC-LCO | 0.095 | 0.905 |
| LCO | zero_shot | bias | test | LCO | 0.041 | 0.959 |
| LCO | zero_shot | mask | source_val | LFP | 1.0 | 0.0 |
| LCO | zero_shot | mask | source_val | NCA | 0.0 | 1.0 |
| LCO | zero_shot | mask | source_val | NMC | 0.0 | 1.0 |
| LCO | zero_shot | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| LCO | zero_shot | mask | test | LCO | 0.0 | 1.0 |
| LFP | few_2 | bias | source_val | LCO | 0.027 | 0.973 |
| LFP | few_2 | bias | source_val | NCA | 0.078 | 0.922 |
| LFP | few_2 | bias | source_val | NMC | 0.705 | 0.295 |
| LFP | few_2 | bias | source_val | NMC-LCO | 0.673 | 0.327 |
| LFP | few_2 | bias | test | LFP | 0.619 | 0.381 |
| LFP | few_2 | mask | source_val | LCO | 0.0 | 1.0 |
| LFP | few_2 | mask | source_val | NCA | 0.0 | 1.0 |
| LFP | few_2 | mask | source_val | NMC | 0.808 | 0.192 |
| LFP | few_2 | mask | source_val | NMC-LCO | 0.703 | 0.297 |
| LFP | few_2 | mask | test | LFP | 0.0 | 0.0 |
| LFP | few_5 | bias | source_val | LCO | 0.058 | 0.942 |
| LFP | few_5 | bias | source_val | NCA | 0.162 | 0.838 |
| LFP | few_5 | bias | source_val | NMC | 0.699 | 0.301 |
| LFP | few_5 | bias | source_val | NMC-LCO | 0.636 | 0.364 |
| LFP | few_5 | bias | test | LFP | 0.634 | 0.366 |
| LFP | few_5 | mask | source_val | LCO | 0.0 | 1.0 |
| LFP | few_5 | mask | source_val | NCA | 0.0 | 1.0 |
| LFP | few_5 | mask | source_val | NMC | 0.604 | 0.396 |
| LFP | few_5 | mask | source_val | NMC-LCO | 0.475 | 0.525 |
| LFP | few_5 | mask | test | LFP | 0.0 | 0.0 |
| LFP | zero_shot | bias | source_val | LCO | 0.053 | 0.947 |
| LFP | zero_shot | bias | source_val | NCA | 0.106 | 0.894 |
| LFP | zero_shot | bias | source_val | NMC | 0.7 | 0.3 |
| LFP | zero_shot | bias | source_val | NMC-LCO | 0.732 | 0.268 |
| LFP | zero_shot | bias | test | LFP | 0.719 | 0.281 |
| LFP | zero_shot | mask | source_val | LCO | 0.0 | 1.0 |
| LFP | zero_shot | mask | source_val | NCA | 0.0 | 1.0 |
| LFP | zero_shot | mask | source_val | NMC | 0.977 | 0.023 |
| LFP | zero_shot | mask | source_val | NMC-LCO | 0.968 | 0.032 |
| LFP | zero_shot | mask | test | LFP | 0.0 | 0.0 |
| NCA | few_2 | bias | source_val | LCO | 0.07 | 0.93 |
| NCA | few_2 | bias | source_val | LFP | 0.87 | 0.13 |
| NCA | few_2 | bias | source_val | NMC | 0.009 | 0.991 |
| NCA | few_2 | bias | source_val | NMC-LCO | 0.008 | 0.992 |
| NCA | few_2 | bias | test | NCA | 0.1 | 0.9 |
| NCA | few_2 | mask | source_val | LCO | 0.0 | 1.0 |
| NCA | few_2 | mask | source_val | LFP | 1.0 | 0.0 |
| NCA | few_2 | mask | source_val | NMC | 0.0 | 1.0 |
| NCA | few_2 | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| NCA | few_2 | mask | test | NCA | 0.0 | 0.0 |
| NCA | few_5 | bias | source_val | LCO | 0.035 | 0.965 |
| NCA | few_5 | bias | source_val | LFP | 0.83 | 0.17 |
| NCA | few_5 | bias | source_val | NMC | 0.057 | 0.943 |
| NCA | few_5 | bias | source_val | NMC-LCO | 0.092 | 0.908 |
| NCA | few_5 | bias | test | NCA | 0.311 | 0.689 |
| NCA | few_5 | mask | source_val | LCO | 0.0 | 1.0 |
| NCA | few_5 | mask | source_val | LFP | 1.0 | 0.0 |
| NCA | few_5 | mask | source_val | NMC | 0.0 | 1.0 |
| NCA | few_5 | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| NCA | few_5 | mask | test | NCA | 0.0 | 0.0 |
| NCA | zero_shot | bias | source_val | LCO | 0.041 | 0.959 |
| NCA | zero_shot | bias | source_val | LFP | 0.91 | 0.09 |
| NCA | zero_shot | bias | source_val | NMC | 0.033 | 0.967 |
| NCA | zero_shot | bias | source_val | NMC-LCO | 0.086 | 0.914 |
| NCA | zero_shot | bias | test | NCA | 0.199 | 0.801 |
| NCA | zero_shot | mask | source_val | LCO | 0.0 | 1.0 |
| NCA | zero_shot | mask | source_val | LFP | 1.0 | 0.0 |
| NCA | zero_shot | mask | source_val | NMC | 0.0 | 1.0 |
| NCA | zero_shot | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| NCA | zero_shot | mask | test | NCA | 0.0 | 0.0 |
| NMC | few_2 | bias | source_val | LCO | 0.067 | 0.933 |
| NMC | few_2 | bias | source_val | LFP | 0.888 | 0.112 |
| NMC | few_2 | bias | source_val | NCA | 0.057 | 0.943 |
| NMC | few_2 | bias | source_val | NMC-LCO | 0.131 | 0.869 |
| NMC | few_2 | bias | test | NMC | 0.056 | 0.944 |
| NMC | few_2 | mask | source_val | LCO | 0.0 | 1.0 |
| NMC | few_2 | mask | source_val | LFP | 1.0 | 0.0 |
| NMC | few_2 | mask | source_val | NCA | 0.0 | 1.0 |
| NMC | few_2 | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| NMC | few_2 | mask | test | NMC | 0.0 | 1.0 |
| NMC | few_5 | bias | source_val | LCO | 0.041 | 0.959 |
| NMC | few_5 | bias | source_val | LFP | 0.982 | 0.018 |
| NMC | few_5 | bias | source_val | NCA | 0.035 | 0.965 |
| NMC | few_5 | bias | source_val | NMC-LCO | 0.062 | 0.938 |
| NMC | few_5 | bias | test | NMC | 0.04 | 0.96 |
| NMC | few_5 | mask | source_val | LCO | 0.0 | 1.0 |
| NMC | few_5 | mask | source_val | LFP | 1.0 | 0.0 |
| NMC | few_5 | mask | source_val | NCA | 0.0 | 1.0 |
| NMC | few_5 | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| NMC | few_5 | mask | test | NMC | 0.0 | 1.0 |
| NMC | zero_shot | bias | source_val | LCO | 0.058 | 0.942 |
| NMC | zero_shot | bias | source_val | LFP | 0.903 | 0.097 |
| NMC | zero_shot | bias | source_val | NCA | 0.037 | 0.963 |
| NMC | zero_shot | bias | source_val | NMC-LCO | 0.072 | 0.928 |
| NMC | zero_shot | bias | test | NMC | 0.04 | 0.96 |
| NMC | zero_shot | mask | source_val | LCO | 0.0 | 1.0 |
| NMC | zero_shot | mask | source_val | LFP | 1.0 | 0.0 |
| NMC | zero_shot | mask | source_val | NCA | 0.0 | 1.0 |
| NMC | zero_shot | mask | source_val | NMC-LCO | 0.0 | 1.0 |
| NMC | zero_shot | mask | test | NMC | 0.0 | 1.0 |

**Folds where the held-out chemistry has no matching expert: LFP, NCA.** There the hard mask gives the shared expert alone (prediction = f_shared, beta not applied); the additive bias keeps a blend.

How `held_out_has_matching_expert` is decided (`qr/train.py` expert_sets / match_matrix): each fold has E = 2 experts built from that fold's source chemistries (LFP vs rest; NMC vs rest when LFP is held out). A chemistry's cathode components are its name split on '-' (NMC-LCO -> {NMC, LCO}). A cell matches an expert when its components overlap the union of that expert's components. HNEI (NMC-LCO) is always a source cell, so it puts NMC and LCO into the 'rest' expert: held-out NMC and LCO match it (partial match through NMC-LCO), while held-out NCA and LFP share no component with any expert.

| fold | experts | expert_components | held_out_components | matching_expert |
|---|---|---|---|---|
| LFP | NMC ; LCO+NCA+NMC-LCO | NMC ; LCO+NCA+NMC | LFP | none |
| NMC | LFP ; LCO+NCA+NMC-LCO | LFP ; LCO+NCA+NMC | NMC | e1 |
| NCA | LFP ; LCO+NMC+NMC-LCO | LFP ; LCO+NMC | NCA | none |
| LCO | LFP ; NCA+NMC+NMC-LCO | LFP ; LCO+NCA+NMC | LCO | e1 |

- `moe_bias` has lower mean RMSE than the single head in 5 of 12 fold x regime cells (3 of 6 in no-matching-expert folds).
- `moe_mask` has lower mean RMSE than the single head in 6 of 12 fold x regime cells (3 of 6 in no-matching-expert folds).
- These are mean RMSE comparisons over a few seeds on 3-6 test cells; differences inside one std (table above) are not evidence either way.

E3 results in this report: code commit **7c7387b**, started 2026-11-21T15:58:50+0000.

### E3 run 1 vs run 2

Run 1: commit d7588be (`outputs_run1/REPORT.md`). Run 2: commit 7c7387b. Run 2 differs from run 1 only by the unit-length normalisation of the text embedding at the router input (`qr/models.py`, MoE.forward); same prepared.pkl, config, seeds and splits. `single_head` rows come from E2 and are identical by construction.

| fold | regime | model | rmse run1 | rmse run2 | r2 run1 | r2 run2 | mape50 run1 | mape50 run2 |
|---|---|---|---|---|---|---|---|---|
| LCO | few_2 | moe_bias | 167 ± 2.6 | 174 ± 14 | 0.0765 ± 0.047 | -0.00148 ± 0.074 | 0.661 ± 0.066 | 0.591 ± 0.029 |
| LCO | few_2 | moe_mask | 171 ± 7.6 | 161 ± 6.1 | 0.0298 ± 0.011 | 0.138 ± 0.12 | 0.671 ± 0.12 | 0.607 ± 0.069 |
| LCO | few_2 | single_head | 174 ± 14 | 174 ± 14 | -0.00931 ± 0.23 | -0.00931 ± 0.23 | 0.597 ± 0.062 | 0.597 ± 0.062 |
| LCO | few_5 | moe_bias | 163 ± 9.5 | 163 ± 19 | 0.12 ± 0.054 | 0.12 ± 0.14 | 0.577 ± 0.075 | 0.561 ± 0.082 |
| LCO | few_5 | moe_mask | 165 ± 26 | 152 ± 34 | 0.0953 ± 0.21 | 0.22 ± 0.28 | 0.583 ± 0.071 | 0.521 ± 0.14 |
| LCO | few_5 | single_head | 167 ± 56 | 167 ± 56 | 0.0379 ± 0.53 | 0.0379 ± 0.53 | 0.558 ± 0.094 | 0.558 ± 0.094 |
| LCO | zero_shot | moe_bias | 278 ± 5.8 | 281 ± 4.6 | -1.56 ± 0.27 | -1.62 ± 0.26 | 0.726 ± 0.035 | 0.74 ± 0.035 |
| LCO | zero_shot | moe_mask | 276 ± 13 | 280 ± 7.2 | -1.54 ± 0.41 | -1.59 ± 0.31 | 0.721 ± 0.067 | 0.734 ± 0.047 |
| LCO | zero_shot | single_head | 289 ± 7.4 | 289 ± 7.4 | -1.76 ± 0.22 | -1.76 ± 0.22 | 0.783 ± 0.019 | 0.783 ± 0.019 |
| LFP | few_2 | moe_bias | 418 ± 1.3e+02 | 429 ± 1.4e+02 | 0.547 ± 0.23 | 0.519 ± 0.27 | 0.521 ± 0.048 | 0.497 ± 0.067 |
| LFP | few_2 | moe_mask | 406 ± 1.2e+02 | 422 ± 1.2e+02 | 0.574 ± 0.21 | 0.541 ± 0.2 | 0.568 ± 0.035 | 0.558 ± 0.06 |
| LFP | few_2 | single_head | 451 ± 1.2e+02 | 451 ± 1.2e+02 | 0.476 ± 0.23 | 0.476 ± 0.23 | 0.524 ± 0.18 | 0.524 ± 0.18 |
| LFP | few_5 | moe_bias | 424 ± 1.6e+02 | 447 ± 1.7e+02 | 0.521 ± 0.31 | 0.468 ± 0.36 | 0.476 ± 0.07 | 0.462 ± 0.067 |
| LFP | few_5 | moe_mask | 420 ± 1.4e+02 | 416 ± 1.3e+02 | 0.54 ± 0.25 | 0.548 ± 0.23 | 0.474 ± 0.068 | 0.447 ± 0.015 |
| LFP | few_5 | single_head | 358 ± 72 | 358 ± 72 | 0.673 ± 0.098 | 0.673 ± 0.098 | 0.492 ± 0.065 | 0.492 ± 0.065 |
| LFP | zero_shot | moe_bias | 751 ± 1.3e+02 | 760 ± 98 | -0.49 ± 0.6 | -0.5 ± 0.48 | 1.64 ± 0.59 | 1.6 ± 0.52 |
| LFP | zero_shot | moe_mask | 839 ± 20 | 840 ± 65 | -0.791 ± 0.12 | -0.786 ± 0.1 | 1.89 ± 0.19 | 1.87 ± 0.13 |
| LFP | zero_shot | single_head | 727 ± 94 | 727 ± 94 | -0.383 ± 0.48 | -0.383 ± 0.48 | 1.56 ± 0.53 | 1.56 ± 0.53 |
| NCA | few_2 | moe_bias | 122 ± 19 | 125 ± 24 | -0.0751 ± 0.41 | -0.101 ± 0.36 | 0.685 ± 0.086 | 0.69 ± 0.11 |
| NCA | few_2 | moe_mask | 115 ± 42 | 104 ± 22 | 0.0841 ± 0.4 | 0.202 ± 0.39 | 0.551 ± 0.19 | 0.518 ± 0.081 |
| NCA | few_2 | single_head | 117 ± 40 | 117 ± 40 | 0.109 ± 0.096 | 0.109 ± 0.096 | 0.619 ± 0.16 | 0.619 ± 0.16 |
| NCA | few_5 | moe_bias | 85.3 ± 18 | 80.9 ± 9.6 | 0.434 ± 0.3 | 0.524 ± 0.18 | 0.422 ± 0.14 | 0.388 ± 0.017 |
| NCA | few_5 | moe_mask | 78.3 ± 15 | 72.2 ± 10 | 0.574 ± 0.1 | 0.608 ± 0.18 | 0.316 ± 0.034 | 0.33 ± 0.018 |
| NCA | few_5 | single_head | 95.9 ± 20 | 95.9 ± 20 | 0.345 ± 0.25 | 0.345 ± 0.25 | 0.438 ± 0.054 | 0.438 ± 0.054 |
| NCA | zero_shot | moe_bias | 157 ± 29 | 155 ± 31 | -0.797 ± 0.79 | -0.757 ± 0.81 | 0.908 ± 0.053 | 0.887 ± 0.062 |
| NCA | zero_shot | moe_mask | 162 ± 38 | 165 ± 30 | -0.92 ± 0.92 | -0.976 ± 0.84 | 0.96 ± 0.13 | 0.939 ± 0.033 |
| NCA | zero_shot | single_head | 158 ± 37 | 158 ± 37 | -0.767 ± 0.7 | -0.767 ± 0.7 | 0.915 ± 0.11 | 0.915 ± 0.11 |
| NMC | few_2 | moe_bias | 130 ± 24 | 136 ± 23 | 0.183 ± 0.1 | 0.101 ± 0.097 | 0.465 ± 0.066 | 0.478 ± 0.074 |
| NMC | few_2 | moe_mask | 135 ± 21 | 134 ± 22 | 0.114 ± 0.058 | 0.131 ± 0.081 | 0.482 ± 0.069 | 0.479 ± 0.072 |
| NMC | few_2 | single_head | 128 ± 24 | 128 ± 24 | 0.206 ± 0.098 | 0.206 ± 0.098 | 0.441 ± 0.051 | 0.441 ± 0.051 |
| NMC | few_5 | moe_bias | 122 ± 33 | 120 ± 38 | 0.266 ± 0.25 | 0.298 ± 0.29 | 0.445 ± 0.074 | 0.425 ± 0.083 |
| NMC | few_5 | moe_mask | 105 ± 27 | 116 ± 37 | 0.462 ± 0.14 | 0.345 ± 0.26 | 0.386 ± 0.059 | 0.408 ± 0.087 |
| NMC | few_5 | single_head | 86.8 ± 1.6 | 86.8 ± 1.6 | 0.621 ± 0.09 | 0.621 ± 0.09 | 0.355 ± 0.056 | 0.355 ± 0.056 |
| NMC | zero_shot | moe_bias | 146 ± 27 | 146 ± 28 | -0.0337 ± 0.13 | -0.0344 ± 0.15 | 0.519 ± 0.096 | 0.526 ± 0.09 |
| NMC | zero_shot | moe_mask | 138 ± 28 | 141 ± 25 | 0.0738 ± 0.17 | 0.0313 ± 0.13 | 0.505 ± 0.1 | 0.516 ± 0.09 |
| NMC | zero_shot | single_head | 129 ± 34 | 129 ± 34 | 0.191 ± 0.27 | 0.191 ± 0.27 | 0.499 ± 0.078 | 0.499 ± 0.078 |

## E4: released PBT, zero-shot

Status: skipped: E4 disabled (set RUN_E4=True in the notebook or pass --e4)

## How Version 1 works end to end

```
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
```

## What this shows

- The shared pipeline runs end to end on real SNL / CALCE / HNEI / HUST cells with the PBT life-label rule; every gate check is reported with its number (E0 gate table).
- Whether single-head models show a leave-one-chemistry-out gap: in the E2 table, compare the `zero_shot`, `few_2` / `few_5` and `target_only` rows of each fold (same blind-test cells), and the E2 'Gap' summary for `lstm_frac`.
- How the hard mask and the additive bias route an unseen chemistry: E3 results table (MoE vs `single_head` per fold and regime, with `held_out_has_matching_expert`), the E3 diagnostics table and the mean-gate-weight-per-chemistry table.

## What this does NOT show

- Anything statistically solid: tiny N (held-out test sets of 3-6 cells, 3 seeds). Preliminary only.
- Real-world RUL accuracy for fraction-of-life models: every `oracle_eol=True` number uses the TRUE EOL to convert to cycles. That is a leak in deployment (CLAUDE.md, open issue a).
- That PBT has or lacks a gap: the released PBT was pretrained on LFP/NMC/NCA/LCO cells, possibly these very cells, so E4 is optimistic, and Track A (re-headed, fine-tuned PBT) was not run here.
- The full Track B model: no Neural ODE, E=2, few epochs, no hyperparameter search, windows subsampled per cell.
- Any conclusion beyond these folds: the MoE result above is reported as measured, whichever way it goes.
