"""E4 (optional): released PBT checkpoint, zero-shot on our held-out test cells.

PBT predicts total cycle life from the first 100 cycles; RUL = predicted life - window end cycle, scored only on
windows ending at cycle >= 100 (PBT reads cycles 1-100, so earlier windows would see the future).
Any failure returns (rows=[], reason) so run_all can write the reason into the report instead of crashing.
Untested end to end: it needs the gated Llama tokenizer (third_party/PBT/models/PBT.py:487-493, local_files_only).
"""
import json
import os
import pickle
import subprocess
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np

from .features import make_split, window_idx
from .metrics import chem_metrics

PBT_REPO = "https://github.com/Ruifeng-Tan/PBT"
PBT_COMMIT = "74f03a73f04febe66877e08bec030d148e043fa0"  # audited in document.md
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"


def run_e4(root, prep, cfg, work):
    step = "start"
    try:
        root, work = Path(root), Path(work)
        step = "find checkpoint"
        cands = sorted(Path(root, "pbt_ckpt").rglob("args.json"))
        ckpts = [p.parent for p in cands if json.loads(p.read_text()).get("dataset") == "MIX_large"] or [p.parent for p in cands]
        if not ckpts:
            return [], "skipped: no args.json found in the PBT checkpoint zip"
        ckpt = ckpts[0]
        step = "Llama tokenizer"
        token = os.environ.get("HF_TOKEN")
        if not token:
            return [], ("skipped: PBT's model constructor loads the gated Llama-3.1-8B-Instruct tokenizer "
                        "(models/PBT.py:487, local_files_only=True). Set HF_TOKEN (approved access) to run E4.")
        from huggingface_hub import snapshot_download
        tok = snapshot_download(LLAMA, token=token, local_dir=str(work / "llama_tok"),
                                allow_patterns=["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json"])
        step = "clone PBT"
        repo = work / "PBT"
        if not repo.exists():
            subprocess.run(["git", "clone", "-q", PBT_REPO, str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "-q", PBT_COMMIT], check=True)
        step = "import PBT"
        sys.modules.setdefault("batteryml", types.ModuleType("batteryml"))          # only used in a docstring
        sys.modules.setdefault("batteryml.data", types.ModuleType("batteryml.data"))
        sys.modules.setdefault("batteryml.data.battery_data", types.SimpleNamespace(BatteryData=object))
        sys.path.insert(0, str(repo))
        cwd = os.getcwd()
        os.chdir(repo)  # PBT opens ./gate_data/*.json relative to the repo
        try:
            import joblib
            import torch
            from accelerate import load_checkpoint_in_model
            from transformers import AutoConfig
            from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
            from models import PBT
            from data_provider.data_loader import Dataset_PBT
            from data_provider.gate_masker import gate_masker as gm
            from data_provider.data_split_recorder import split_recorder

            step = "build model"
            args = json.loads((ckpt / "args.json").read_text())
            args["LLM_path"] = tok
            pca = list(ckpt.glob("*pca*.pkl"))
            if pca:
                args["pca_path"] = str(pca[0])
            elif not Path(args.get("pca_path", "")).exists():
                args["pca_path"] = str(work / "pca_unused.pkl")   # loaded but never used in forward (models/PBT.py:519)
                pickle.dump(None, open(args["pca_path"], "wb"))
            model = PBT.Model(BatteryLifeConfig(BatteryElectrochemicalConfig(args), AutoConfig.from_pretrained(tok)))
            load_checkpoint_in_model(model, str(ckpt))
            model = model.to(torch.bfloat16).eval()   # CPU bf16: avoids the T4's missing native bf16
            ls = joblib.load(ckpt / "label_scaler")
            std, mean = float(np.sqrt(ls.var_[-1])), float(ls.mean_[-1])

            step = "build inputs"
            ds = Dataset_PBT.__new__(Dataset_PBT)
            ds.__dict__.update(args=Namespace(wo_DKPrompt=False), root_path=str(root), dataset="quick", seq_len=1,
                               early_cycle_threshold=100, charge_discharge_len=300, eval_cycle_min=100, eval_cycle_max=100,
                               need_keys=['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s'],
                               ZN_coin_charge_first_file_names=[])
            gate = {k: json.load(open(f"gate_data/{k}.json")) for k in ("cathodes", "anodes", "formats", "temperatures")}

            def mask(fn):
                parts = []
                for key, n_exp, table, val in (
                        ("cathode_experts", None, gm.MIX_large_cathodes2mask, "_".join(gate["cathodes"][fn])),
                        ("anode_experts", None, gm.MIX_large_anode2mask, _anode(gate["anodes"][fn][0])),
                        ("format_experts", None, gm.MIX_large_format2mask, gate["formats"][fn][0]),
                        ("temperature_experts", None, gm.MIX_large_temperature2mask, gate["temperatures"][fn])):
                    m = np.zeros(args[key])
                    if val in table:
                        m[table[val]] = 1
                    parts.append(m)
                return np.concatenate(parts)  # same order as data_loader.py:900

            emb = {}
            for f in Path(root, "embeddings").glob("*_DKP_embed_all_Llama.pkl"):
                emb.update(pickle.load(open(f, "rb")))
            life = {}
            step = "predict"
            test_cells = set()
            for held in cfg["folds"]:
                for seed in cfg["seeds"]:
                    test_cells |= set(make_split(prep["chem"], held, seed, cfg["split"])["test"].tolist())
            with torch.no_grad():
                for ci in sorted(test_cells):
                    name = prep["cells"][ci]["name"]
                    curves, attn, _, _ = ds.read_samples_from_one_cell(name + ".pkl")
                    o = model(torch.tensor(curves[-1][None]), torch.tensor(attn[-1][None]),
                              DKP_embeddings=torch.tensor(np.asarray(emb[name], np.float32)[None]),
                              combined_masks=torch.tensor(mask(name + ".pkl")[None], dtype=torch.bfloat16))[0]
                    life[ci] = float(o.float().reshape(-1)[0]) * std + mean
            seen = set(split_recorder.MIX_large_train_files) | set(split_recorder.MIX_large_val_files)
        finally:
            os.chdir(cwd)
        step = "score"
        rows = []
        for held in cfg["folds"]:
            for seed in cfg["seeds"]:
                sp = make_split(prep["chem"], held, seed, cfg["split"])
                idx = window_idx(prep, sp["test"])
                idx = idx[prep["end_cycle"][idx] >= 100]
                if not len(idx):
                    continue
                pred = np.array([life[c] for c in prep["cell_idx"][idx]]) - prep["end_cycle"][idx]
                n_seen = sum(prep["cells"][c]["name"] + ".pkl" in seen for c in sp["test"])
                rows.append(dict(fold=held, regime="zero_shot", model="PBT_released", seed=seed, oracle_eol=False,
                                 test_cells_in_PBT_pretraining=f"{n_seen}/{len(sp['test'])}",
                                 **chem_metrics(prep["rul"][idx], pred, prep["cell_idx"][idx])))
        return rows, f"ran with checkpoint {ckpt.name}"
    except Exception as e:  # report, never crash the run
        return [], f"failed at step '{step}': {type(e).__name__}: {str(e)[:300]}"


def _anode(a):
    return "graphite" if a in ("graphite", "artificial graphite", "carbon") else a
