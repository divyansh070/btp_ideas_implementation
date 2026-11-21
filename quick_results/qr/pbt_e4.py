"""E4: released PBT on our LOCO protocol. E4a zero-shot, E4b fine-tuned on the held-out chemistry's few-shot cells.

PBT predicts total cycle life from the first N <= 100 cycles (300 points per cycle). For a test window ending at
cycle c we feed the first min(c, 100) cycles and score RUL = predicted life - c, so no window sees past its own end.
oracle_eol is False everywhere in E4.

Llama: PBT's Model.__init__ loads a Llama tokenizer (third_party/PBT/models/PBT.py:487-498) but never uses it, and
BatteryLifeConfig accepts text_config=None (default LlamaConfig(), no download, configuration_BatteryLifeLLM.py:112).
With PBT's cached prompt embeddings neither is needed, so the tokenizer is stubbed and no HF token is required.

E4b: PBT's finetune_model.py cannot be pointed at our split (dataset names map to PBT's own split lists; it calls
wandb.init and rewrites args from args.json), so `finetune` reproduces its recipes instead:
  FT  all parameters, lr 1e-5 (PBT's lower bound);
  AT  PBT's PBTtLayerWithAdapter (width 16) after the CyclePatch layer and every MoE layer, adapters + regression head
      trained (finetune_model.py 'AT' branch, add_adapters_to_PBT_withCP_flex, adapter_layers=-1), lr 2.5e-5
      (finetune_PBT.sh default);
both: Adam (wd = 0), MSE on label-scaled cycle life, batch 32, every prefix N = 1..100 of each training cell, best
epoch by target-val MAPE where epoch -1 (before any update; for FT the pretrained model itself) is a candidate.
PBT_selected picks zero-shot / FT / AT per fold x regime x seed by target-val MAPE. Differences from PBT's script:
fixed epoch count instead of patience 30 / least_epochs 50, bf16 autocast instead of accelerate's bf16 mixed
precision, no LR warm-up, dropout 0 (finetune_PBT.sh uses dropout=0 too).
"""
import contextlib
import copy
import json
import os
import pickle
import subprocess
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import numpy as np

from .features import make_split, window_idx
from .metrics import chem_metrics

PBT_REPO = "https://github.com/Ruifeng-Tan/PBT"
PBT_COMMIT = "74f03a73f04febe66877e08bec030d148e043fa0"  # audited in document.md
OPTIONAL = ["batteryml", "batteryml.data", "batteryml.data.battery_data", "wandb", "evaluate", "denseweight",
            "reformer_pytorch"]  # imported by PBT modules but unused for inference / FT


class _Stub(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        import importlib.machinery
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)  # transformers calls find_spec() on e.g. wandb

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


class _NoTokenizer:
    @staticmethod
    def from_pretrained(*_, **__):
        return types.SimpleNamespace(padding_side="right")


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_repo(cfg, work):
    """Local third_party/PBT if present (pinned commit), else a clone in `work` checked out at PBT_COMMIT."""
    local = Path(cfg["e4"].get("pbt_repo") or "")
    if local.is_dir() and (local / "models" / "PBT.py").exists():
        return local
    repo = Path(work) / "PBT"
    if not repo.exists():
        subprocess.run(["git", "clone", "-q", PBT_REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", PBT_COMMIT], check=True)
    return repo


def find_checkpoint(root):
    cands = sorted(Path(root, "pbt_ckpt").rglob("args.json"))
    pre = [p.parent for p in cands if json.loads(p.read_text()).get("dataset") == "MIX_large"]
    return (pre or [p.parent for p in cands] or [None])[0]


def pick_device(pref="auto"):
    import torch
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                torch.nn.Linear(8, 8).cuda()(torch.randn(4, 8, device="cuda")).sum().item()
            return "cuda"
        except RuntimeError as e:
            print("E4: bf16 autocast unusable on this GPU, using CPU:", e, flush=True)
    return "cpu"


class PBTAdapter:
    """The released PBT behind three calls: life(name) for N = 1..100, finetune(...), split_of(name)."""

    def __init__(self, repo, ckpt, data_root, emb, work, device):
        import torch
        for m in OPTIONAL:
            try:
                __import__(m)
            except ImportError:
                sys.modules[m] = _Stub(m)
        sys.path.insert(0, str(repo))
        self.repo, self.torch, self.device = Path(repo), torch, device
        with _cwd(repo):
            import joblib
            from accelerate import load_checkpoint_in_model
            from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
            from models import PBT
            from data_provider.data_loader import Dataset_PBT
            from data_provider.gate_masker import gate_masker
            from data_provider.data_split_recorder import split_recorder as sr
            PBT.AutoTokenizer = _NoTokenizer          # tokenizer is never used (see module docstring)
            args = json.loads((Path(ckpt) / "args.json").read_text())
            args |= dict(LLM_path="unused", dropout=0.0)
            pca = list(Path(ckpt).glob("*pca*.pkl"))
            if pca:
                args["pca_path"] = str(pca[0])
            elif not Path(args.get("pca_path", "")).exists():  # loaded but unused in forward (models/PBT.py:519)
                args["pca_path"] = str(Path(work) / "pca_unused.pkl")
                Path(work).mkdir(parents=True, exist_ok=True)
                pickle.dump(None, open(args["pca_path"], "wb"))
            from layers.Adapters import PBTtLayerWithAdapter
            self.base = PBT.Model(BatteryLifeConfig(BatteryElectrochemicalConfig(args), None))
            load_checkpoint_in_model(self.base, str(ckpt))
            self.base.to(device).eval()
            self.model, self.Adapter = self.base, PBTtLayerWithAdapter
            ls = joblib.load(Path(ckpt) / "label_scaler")
            self.mean, self.std = float(ls.mean_[-1]), float(np.sqrt(ls.var_[-1]))
            ds = Dataset_PBT.__new__(Dataset_PBT)
            ds.__dict__.update(args=Namespace(wo_DKPrompt=False), root_path=str(data_root), dataset="quick", seq_len=1,
                               early_cycle_threshold=100, charge_discharge_len=300, eval_cycle_min=1, eval_cycle_max=100,
                               need_keys=["current_in_A", "voltage_in_V", "charge_capacity_in_Ah",
                                          "discharge_capacity_in_Ah", "time_in_s"], ZN_coin_charge_first_file_names=[])
            self.ds = ds
            self.gate = {k: json.load(open(f"gate_data/{k}.json")) for k in ("cathodes", "anodes", "formats", "temperatures")}
            self.gm, self.args = gate_masker, args
            self.splits = {f: s for s, lst in (("train", sr.MIX_large_train_files), ("val", sr.MIX_large_val_files),
                                               ("test", sr.MIX_large_test_files)) for f in lst}
        self.emb, self._inputs = emb, {}
        self.info = dict(checkpoint=Path(ckpt).name, device=device, d_llm=args.get("d_llm"),
                         e_layers=args.get("e_layers"), d_layers=args.get("d_layers"))

    def split_of(self, name):
        return self.splits.get(name + ".pkl", "none")

    def _mask(self, fn):
        gm, a, g = self.gm, self.args, self.gate
        anode = g["anodes"][fn][0]
        anode = "graphite" if anode in ("graphite", "artificial graphite", "carbon") else anode
        parts = []
        for n, table, val in ((a["cathode_experts"], gm.MIX_large_cathodes2mask, "_".join(g["cathodes"][fn])),
                              (a["anode_experts"], gm.MIX_large_anode2mask, anode),
                              (a["format_experts"], gm.MIX_large_format2mask, g["formats"][fn][0]),
                              (a["temperature_experts"], gm.MIX_large_temperature2mask, g["temperatures"][fn])):
            m = np.zeros(n, np.float32)
            if val in table:
                m[table[val]] = 1
            parts.append(m)
        return np.concatenate(parts)  # order of data_provider/data_loader.py:900

    def inputs(self, name):
        """PBT's own samples for one cell: prefixes N = 1..min(100, EOL-1), each [100, 3, 300] with an N-cycle mask."""
        if name not in self._inputs:
            if name not in self.emb:
                raise KeyError(f"{name}: no PBT prompt embedding")
            with _cwd(self.repo):
                curves, attn, labels, eol = self.ds.read_samples_from_one_cell(name + ".pkl")
            if curves is None:
                raise ValueError(f"{name}: PBT could not build samples (no PBT life label or life <= 100)")
            self._inputs[name] = (np.asarray(curves, np.float32), np.asarray(attn, np.float32),
                                  np.asarray(self.emb[name], np.float32), self._mask(name + ".pkl"), float(eol))
        return self._inputs[name]

    def _forward(self, curves, attn, emb, mask):
        t, dev = self.torch, self.device
        with t.autocast(dev, dtype=t.bfloat16):
            o = self.model(t.as_tensor(curves, device=dev), t.as_tensor(attn, device=dev),
                           DKP_embeddings=t.as_tensor(emb, device=dev), combined_masks=t.as_tensor(mask, device=dev))[0]
        return o.float().reshape(-1)

    def life(self, name, bs=128):
        """Predicted cycle life for N = 1..K prefixes (index N-1)."""
        curves, attn, emb, mask, _ = self.inputs(name)
        self.model.eval()
        out = []
        with self.torch.no_grad():
            for i in range(0, len(curves), bs):
                n = len(curves[i:i + bs])
                out.append(self._forward(curves[i:i + bs], attn[i:i + bs], np.repeat(emb[None], n, 0),
                                         np.repeat(mask[None], n, 0)).cpu().numpy())
        return np.concatenate(out) * self.std + self.mean

    def reset(self):
        self.model = self.base  # the untouched pretrained model; fine-tunes never modify it

    def _with_adapters(self, size):
        """PBT's AT: adapters after the CyclePatch layer and every intra/inter MoE layer
        (finetune_model.py add_adapters_to_PBT_withCP_flex with adapter_layers=-1 -> all); train adapters + head."""
        m, conf = copy.deepcopy(self.base), Namespace(**self.args)
        m.flattenIntraCycleLayer = self.Adapter(conf, m.flattenIntraCycleLayer, adapter_size=size)
        for layers in (m.intra_MoE_layers, m.inter_MoE_layers):
            for i in range(len(layers)):
                layers[i] = self.Adapter(conf, layers[i], adapter_size=size)
        m.to(self.device)
        params = [p for n, p in m.named_parameters() if "adapter" in n or "regression_head" in n]
        return m, params

    def val_mape(self, val):
        return float(np.mean([np.mean(np.abs(self.life(n) - self.inputs(n)[4]) / self.inputs(n)[4]) for n in val]))

    def finetune(self, train, val, cfg, seed, method="FT"):
        """PBT's FT (all params) or AT (adapters + head) on `train` cells. Epoch -1 = before any update (for FT that
        is the pretrained model) is a selection candidate; best epoch by val MAPE on cycle life. Returns history."""
        t = self.torch
        t.manual_seed(seed)
        np.random.seed(seed)
        if method == "FT":
            self.model = copy.deepcopy(self.base)
            params = list(self.model.parameters())
        elif method == "AT":
            self.model, params = self._with_adapters(cfg["adapter_size"])
        else:
            raise ValueError(method)
        data = [self.inputs(n) for n in train]
        X = np.concatenate([d[0] for d in data])
        A = np.concatenate([d[1] for d in data])
        E = np.concatenate([np.repeat(d[2][None], len(d[0]), 0) for d in data])
        M = np.concatenate([np.repeat(d[3][None], len(d[0]), 0) for d in data])
        y = np.concatenate([np.full(len(d[0]), (d[4] - self.mean) / self.std, np.float32) for d in data])
        opt = t.optim.Adam(params, lr=cfg["lr"], weight_decay=0.0)
        best = self.val_mape(val)
        best_state, hist = copy.deepcopy(self.model.state_dict()), [dict(epoch=-1, train_loss_last=np.nan, val_mape_life=best)]
        rng = np.random.default_rng(seed)
        for ep in range(cfg["epochs"]):
            self.model.train()
            for b in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y) / cfg["batch"])))):
                opt.zero_grad()
                pred = self._forward(X[b], A[b], E[b], M[b])
                loss = t.nn.functional.mse_loss(pred, t.as_tensor(y[b], device=self.device))
                loss.backward()
                opt.step()
            v = self.val_mape(val)
            hist.append(dict(epoch=ep, train_loss_last=float(loss), val_mape_life=v))
            if v < best:
                best, best_state = v, copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(best_state)
        return hist


def _score(prep, test_cells, life_of, base):
    """Metrics on the test windows of `test_cells` with RUL = life[min(end, K) - 1] - end."""
    idx = window_idx(prep, test_cells)
    end = prep["end_cycle"][idx]
    lives = {c: life_of(c) for c in np.unique(prep["cell_idx"][idx])}
    pred = np.array([lives[c][min(e, len(lives[c])) - 1] for c, e in zip(prep["cell_idx"][idx], end)]) - end
    return dict(base, oracle_eol=False, **chem_metrics(prep["rul"][idx], pred, prep["cell_idx"][idx]))


def run_e4(prep, cfg, adapter, log=print):
    """Returns (rows, cell_rows, history_rows, notes). `adapter` is a PBTAdapter or a test double with the same calls:
    inputs(name), life(name), finetune(train, val, cfg, seed, method), reset(), split_of(name), info."""
    c = cfg["e4"]
    names = [x["name"] for x in prep["cells"]]
    usable, cell_rows, skipped, zero = {}, [], {}, {}

    def ok(ci):  # PBT can build inputs for this cell (embedding present, PBT life label > 100)
        if ci not in usable:
            try:
                adapter.inputs(names[ci])
                usable[ci] = True
            except (KeyError, ValueError) as e:
                usable[ci], skipped[names[ci]] = False, str(e)
                log(f"E4: skipping {names[ci]}: {e}")
        return usable[ci]

    def zero_life(ci):  # pretrained predictions, computed before any fine-tuning
        if ci not in zero:
            zero[ci] = adapter.life(names[ci])
        return zero[ci]

    rows, hist, t0 = [], [], time.time()
    # E4a zero-shot: every fold x seed, all test cells, and the subset that sat in PBT's own test split
    for seed in cfg["seeds"]:
        for held in cfg["folds"]:
            sp = make_split(prep["chem"], held, seed, cfg["split"])
            test = [ci for ci in sp["test"] if ok(ci)]
            for ci in sp["test"]:
                cell_rows.append(dict(fold=held, seed=seed, cell=names[ci], chem=prep["chem"][ci],
                                      pbt_split=adapter.split_of(names[ci]), used=bool(usable[ci])))
            for ci in list(sp["val"]) + list(sp["adapt"]):  # pretrained lives for the zero-shot val candidate
                if ok(ci):
                    zero_life(ci)
            pbt_test = [ci for ci in test if adapter.split_of(names[ci]) == "test"]
            for subset, cells in (("all_test_cells", test), ("pbt_test_split_only", pbt_test)):
                if cells:
                    rows.append(_score(prep, cells, zero_life, dict(fold=held, regime="zero_shot", model="PBT_zero_shot",
                                                                   subset=subset, seed=seed)))
    notes = dict(e4a_seconds=round(time.time() - t0, 1), skipped_cells=skipped)
    # E4b fine-tune on the same few-shot cells as E2, select on target val, score on the same blind test
    t1, seeds_done = time.time(), []
    for seed in cfg["seeds"] if c["finetune"]["enabled"] else []:
        if seeds_done:
            per_seed = (time.time() - t1) / len(seeds_done)
            if (time.time() - t0) + per_seed > 60 * c["budget_minutes"]:
                log(f"E4b: stopping after seeds {seeds_done}: next seed would exceed {c['budget_minutes']} min")
                break
        for held in cfg["folds"]:
            sp = make_split(prep["chem"], held, seed, cfg["split"])
            test = [ci for ci in sp["test"] if ok(ci)]
            val = [ci for ci in sp["val"] if ok(ci)]  # empty only with < 3 cells per chemistry (smoke test)
            for regime in c["finetune"]["regimes"]:
                k = int(regime.split("_")[1])
                train = [ci for ci in sp["adapt"][:k] if ok(ci)]
                if len(sp["adapt"]) < k or not train or not test:
                    continue
                v = val or train  # same fallback as regime_sets
                base = dict(fold=held, regime=regime, subset="all_test_cells", seed=seed, n_target_train_cells=len(train))
                eol = {ci: prep["cells"][ci]["eol"] for ci in v}
                zs_val = float(np.mean([np.mean(np.abs(zero[ci] - eol[ci]) / eol[ci]) for ci in v]))
                cands = {"PBT_zero_shot": _score(prep, test, zero_life, base | dict(model="PBT_zero_shot", selected_epoch=-1,
                                                                                    val_mape_life=zs_val))}
                for method, mcfg in c["finetune"]["methods"].items():
                    h = adapter.finetune([names[ci] for ci in train], [names[ci] for ci in v], c["finetune"] | mcfg,
                                         seed, method)
                    hist += [dict(fold=held, regime=regime, seed=seed, method=method, **r) for r in h]
                    sel = min(h, key=lambda r: r["val_mape_life"])
                    ft = {ci: adapter.life(names[ci]) for ci in test}
                    cands[f"PBT_{method}"] = _score(prep, test, ft.get, base | dict(
                        model=f"PBT_{method}", selected_epoch=sel["epoch"], val_mape_life=sel["val_mape_life"]))
                    rows.append(cands[f"PBT_{method}"])
                    log({k2: rows[-1][k2] for k2 in ("fold", "regime", "seed", "model", "selected_epoch", "rmse")})
                    adapter.reset()
                best = min(cands, key=lambda m: cands[m]["val_mape_life"])
                rows.append(cands[best] | dict(model="PBT_selected", chosen=best))
        seeds_done.append(seed)
    adapter.reset()
    notes |= dict(e4b_seconds=round(time.time() - t1, 1), e4b_seeds=seeds_done,
                  e4b_seeds_reduced=seeds_done != list(cfg["seeds"]) and c["finetune"]["enabled"])
    return rows, cell_rows, hist, notes


def build_adapter(data_root, prep, cfg, work):
    """Real adapter, or (None, reason) when something required is missing."""
    ckpt = find_checkpoint(data_root)
    if ckpt is None:
        return None, "skipped: no args.json in <data>/pbt_ckpt (download with RUN_E4=True)"
    for need in ("label_scaler",):
        if not (ckpt / need).exists():
            return None, f"skipped: checkpoint {ckpt.name} has no {need}"
    from .data import load_text_embeddings
    emb = load_text_embeddings(data_root)
    repo = ensure_repo(cfg, work)
    return PBTAdapter(repo, ckpt, data_root, emb, work, pick_device(cfg["e4"]["device"])), ""
