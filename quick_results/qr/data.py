"""Download (Colab only), read BatteryLife pickles, PBT life-label rule, synthetic cells for the smoke test.

Pickle format and label rule are ported from third_party/PBT (commit 74f03a7):
  - per-cell pickle: {'nominal_capacity_in_Ah', 'SOC_interval', 'cycle_data': [ {voltage_in_V, current_in_A,
    charge_capacity_in_Ah, discharge_capacity_in_Ah, time_in_s, cycle_number}, ... ]}  (data_provider/data_loader.py:1049-1061)
  - EOL rule: process_scripts/extract_life_labels.py:20-63
"""
import hashlib
import json
import pickle
import re
import time
import zipfile
from pathlib import Path

import numpy as np

ZENODO_DATA = "https://zenodo.org/api/records/17958489/files/{}.zip/content"    # BatteryLife v9, per-dataset zips
FIGSHARE_LABELS = "https://ndownloader.figshare.com/files/63780003"             # figshare 31148413, "Life labels.zip"
ZENODO_PBT = "https://zenodo.org/api/records/17972780/files/{}/content"         # PBT ckpt + prompt embeddings
ZENODO_RECORD = "https://zenodo.org/api/records/{}"                             # file sizes + md5
FIGSHARE_FILES = "https://api.figshare.com/v2/articles/{}/files"
PREFIX = {"SNL": "SNL_", "CALCE": "CALCE_", "HNEI": "HNEI_", "HUST": "HUST_"}
EMB_FILES = [f"{s}_DKP_embed_all_Llama.pkl" for s in ("training", "validation", "testing")]
STATS = ("mean", "std", "min", "max", "var", "median")
SIGNALS = ("V", "I_C", "Qd")


# ---------------------------------------------------------------- download (Colab)
def remote_meta():
    """{file name: (size, md5)} from the Zenodo and figshare APIs. Empty on failure (then Content-Length is the only check)."""
    import requests
    meta = {}
    for attempt in range(3):
        try:
            for rec in ("17958489", "17972780"):
                for f in requests.get(ZENODO_RECORD.format(rec), timeout=60).json()["files"]:
                    meta[f["key"]] = (int(f["size"]), f["checksum"].removeprefix("md5:"))
            for f in requests.get(FIGSHARE_FILES.format("31148413"), timeout=60).json():
                meta[f["name"]] = (int(f["size"]), f.get("computed_md5") or f.get("supplied_md5"))
            return meta
        except Exception as e:  # metadata is a check, not a requirement
            print(f"WARNING: file metadata lookup failed ({type(e).__name__}: {e}), attempt {attempt + 1}/3", flush=True)
            time.sleep(2 * (attempt + 1))
    return {}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _total_from(r, have):
    if r.status_code == 206:  # Content-Range: bytes <start>-<end>/<total>
        tot = r.headers.get("Content-Range", "").rpartition("/")[2]
        return int(tot) if tot.isdigit() else None
    n = r.headers.get("Content-Length")
    return int(n) if n and n.isdigit() else None


def fetch(url, dest, size=None, md5=None, retries=10, backoff=2.0, chunk=1 << 16):
    """Streaming download with HTTP Range resume of <dest>.tmp, retries with exponential backoff,
    size check against the record size / Content-Length and an md5 check before the rename.
    A finished dest whose size matches is skipped. Bytes of an unfinished chunk are lost on a cut (<= chunk)."""
    import requests
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    if dest.exists():
        if size is None or dest.stat().st_size == size:
            print(f"{dest.name}: present ({dest.stat().st_size} bytes{', size verified' if size else ''}), skipping", flush=True)
            return dest
        print(f"{dest.name}: size {dest.stat().st_size} != expected {size}, resuming", flush=True)
        dest.replace(tmp) if dest.stat().st_size < size else dest.unlink()
    expected = size
    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        try:
            if expected is None or have < expected:
                if expected is not None and have > expected:
                    tmp.unlink()
                    have = 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                    if r.status_code == 416 and expected is not None and have == expected:
                        pass  # already complete
                    else:
                        r.raise_for_status()
                        if have and r.status_code != 206:  # server ignored the Range header: restart
                            have = 0
                        expected = expected or _total_from(r, have)
                        print(f"{dest.name}: {'resuming at ' + str(have) if have else 'downloading'} "
                              f"of {expected} bytes (attempt {attempt}/{retries})", flush=True)
                        with open(tmp, "ab" if have else "wb") as f:
                            for b in r.iter_content(chunk):
                                f.write(b)
            got = tmp.stat().st_size
            if expected is not None and got != expected:
                raise IOError(f"short download: {got} of {expected} bytes")
            if expected is None:
                print(f"WARNING: {dest.name}: no size known, cannot verify length", flush=True)
            if md5 and _md5(tmp) != md5:
                tmp.unlink()  # corrupt, not merely short: start over
                raise IOError(f"md5 mismatch (expected {md5})")
            tmp.replace(dest)
            print(f"{dest.name}: OK ({got} bytes{', md5 verified' if md5 else ''})", flush=True)
            return dest
        except (requests.RequestException, IOError) as e:
            wait = min(backoff * 2 ** (attempt - 1), 60)
            print(f"{dest.name}: attempt {attempt}/{retries} failed: {type(e).__name__}: {e}; retry in {wait:.0f}s", flush=True)
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"{dest.name}: download failed after {retries} attempts ({url})")


def _extract_flat(zpath, out_dir, keep):
    """Extract members whose basename passes keep() into out_dir/<basename> (basename only: no path traversal)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not keep(name):
                continue
            dest = out_dir / name
            if not dest.exists():
                dest.write_bytes(z.read(info))
            n += 1
    if n == 0:
        raise RuntimeError(f"{zpath}: no matching members extracted into {out_dir}")
    return n


def download(root, datasets, pbt_ckpt=False):
    """Fetch only what we need. Layout matches PBT's root_path: <root>/<DS>/*.pkl, <root>/Life labels/*.json."""
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    meta = remote_meta()

    def get(url, dest, key):
        size, md5 = meta.get(key, (None, None))
        return fetch(url, dest, size, md5)

    for ds in datasets:
        z = get(ZENODO_DATA.format(ds), raw / f"{ds}.zip", f"{ds}.zip")
        print(ds, _extract_flat(z, root / ds, lambda n, p=PREFIX[ds]: n.startswith(p) and n.endswith(".pkl")), "cells")
    get(FIGSHARE_LABELS, raw / "life_labels.zip", "Life labels.zip")
    wanted = {f"{ds}_labels.json" for ds in datasets}
    _extract_flat(raw / "life_labels.zip", root / "Life labels", lambda n: n in wanted)
    get(ZENODO_PBT.format("aging_condition_prompt_embeddings.zip"), raw / "embeddings.zip", "aging_condition_prompt_embeddings.zip")
    _extract_flat(raw / "embeddings.zip", root / "embeddings", lambda n: n in EMB_FILES)
    if pbt_ckpt:
        z = get(ZENODO_PBT.format("PBT_checkpoints.zip"), raw / "PBT_checkpoints.zip", "PBT_checkpoints.zip")
        if not (root / "pbt_ckpt").exists():
            with zipfile.ZipFile(z) as f:
                f.extractall(root / "pbt_ckpt")  # zipfile strips absolute paths and '..'


# ---------------------------------------------------------------- metadata from cell names
def chem_of(name):
    ds = name.split("_")[0]
    if ds == "SNL":
        chem = name.split("_")[2]  # SNL_18650_<LFP|NMC|NCA>_...
        if chem not in ("LFP", "NMC", "NCA"):
            raise ValueError(f"cannot tag SNL chemistry from {name}")
        return chem
    return {"CALCE": "LCO", "HNEI": "NMC-LCO", "HUST": "LFP"}[ds]


def meta_of(name):
    ds = name.split("_")[0]
    if ds == "SNL":
        temp = float(re.search(r"_(-?\d+)C_", name).group(1))
    else:
        temp = {"CALCE": 25.0, "HNEI": 25.0, "HUST": 30.0}[ds]
    fmt = "prismatic" if ds == "CALCE" else "18650"   # third_party/PBT/gate_data/formats.json
    return dict(dataset=ds, chem=chem_of(name), fmt=fmt, temp=temp)


def nominal_of(name, data):
    if name.startswith("RWTH"):
        return 1.85
    if name.startswith("SNL_18650_NCA_25C_20-80"):
        return 3.2
    return float(data["nominal_capacity_in_Ah"])


# ---------------------------------------------------------------- PBT life-label rule
def eol_pbt(soh):
    """Port of extract_life_labels.py:32-63. soh: per-cycle max(Qd)/nominal/SOC_interval, cycle k at index k-1.

    Returns (eol, source). eol None means PBT drops the cell (last-cycle SOH >= 0.825). Never falls back to the last cycle.
    """
    soh = np.asarray(soh, float)
    last = soh[-1]
    if last >= 0.825:
        return None, "dropped: last-cycle SOH >= 0.825"
    if last > 0.8:  # (0.8, 0.825): regress cycle number on SOH over the last 20 cycles, evaluate at 0.80
        n = len(soh)
        x, y = soh[-20:], np.arange(n - 19, n + 1)
        a, b = np.polyfit(x, y, 1)
        return int(a * 0.80 + b), "extrapolated"
    return int(np.flatnonzero(soh <= 0.8)[0]) + 1, "threshold"


# ---------------------------------------------------------------- reading
def cycle_stats(cyc, nominal):
    sig = [np.asarray(cyc["voltage_in_V"], float),
           np.asarray(cyc["current_in_A"], float) / nominal,
           np.asarray(cyc["discharge_capacity_in_Ah"], float)]
    if any(s.size == 0 for s in sig):
        return np.full(18, np.nan), np.nan
    row = []
    for s in sig:
        row += [np.nanmean(s), np.nanstd(s), np.nanmin(s), np.nanmax(s), np.nanvar(s), np.nanmedian(s)]
    return np.array(row), np.nanmax(sig[2])


def read_cell(path):
    """One pickle -> per-cycle stats. Raw curves are discarded right away (HUST would not fit in Colab RAM)."""
    name = Path(path).stem
    with open(path, "rb") as f:
        data = pickle.load(f)
    nominal = nominal_of(name, data)
    lo, hi = data["SOC_interval"]
    rows, qmax = zip(*(cycle_stats(c, nominal) for c in data["cycle_data"]))
    stats, qmax = np.array(rows, np.float32), np.array(qmax, float)
    cell = dict(name=name, **meta_of(name), nominal=nominal, n_cycles=len(qmax), stats=stats, qmax=qmax.astype(np.float32))
    cell["eol"], cell["label_source"] = eol_pbt(qmax / nominal / (hi - lo))
    return cell


def load_cells(root, datasets, min_cycles=100, early_threshold=100):
    """Read every cell and apply PBT's filters. Returns (all cells with keep/drop reason, kept cells)."""
    root = Path(root)
    cells = []
    for ds in datasets:
        files = sorted((root / ds).glob(PREFIX[ds] + "*.pkl"))
        if not files:
            raise FileNotFoundError(f"no {ds} pickles under {root / ds}")
        for p in files:
            c = read_cell(p)
            if c["n_cycles"] < min_cycles:
                c["drop"] = f"fewer than {min_cycles} cycles"
            elif c["eol"] is None:
                c["drop"] = c["label_source"]
            elif c["eol"] <= early_threshold:
                c["drop"] = f"life <= {early_threshold} (PBT early_cycle_threshold)"
            else:
                c["drop"] = ""
            cells.append(c)
    kept = [c for c in cells if not c["drop"]]
    for c in kept:  # fail loudly: a kept cell must carry a real label
        if not isinstance(c["eol"], int) or c["eol"] <= 0 or c["label_source"] not in ("threshold", "extrapolated"):
            raise ValueError(f"{c['name']}: invalid cycle_life {c['eol']!r} ({c['label_source']})")
    return cells, kept


def load_pbt_labels(root, datasets):
    out = {}
    for ds in datasets:
        p = Path(root) / "Life labels" / f"{ds}_labels.json"
        if p.exists():
            out.update({k.removesuffix(".pkl"): v for k, v in json.loads(p.read_text()).items()})
    return out


def load_text_embeddings(root):
    """{cell_name: 4096-d Llama last-token vector}, PBT's cache (data_loader.py:631-645)."""
    out = {}
    for fn in EMB_FILES:
        p = Path(root) / "embeddings" / fn
        if p.exists():
            with open(p, "rb") as f:
                out.update(pickle.load(f))
    if not out:
        raise FileNotFoundError(f"no prompt embeddings under {Path(root) / 'embeddings'}")
    return {k: np.asarray(v, np.float32).reshape(-1) for k, v in out.items()}


# ---------------------------------------------------------------- synthetic data (smoke test only)
SYNTH = [  # name, life, end SOH (0.81 exercises the extrapolation branch)
    ("SNL_18650_LFP_25C_0-100_0.5-1C_a", 260, 0.78), ("SNL_18650_LFP_35C_0-100_0.5-2C_b", 230, 0.81),
    ("SNL_18650_NMC_25C_0-100_0.5-1C_a", 180, 0.78), ("SNL_18650_NMC_15C_0-100_0.5-2C_b", 160, 0.78),
    ("SNL_18650_NCA_25C_0-100_0.5-1C_a", 140, 0.78), ("SNL_18650_NCA_35C_0-100_0.5-2C_b", 125, 0.81),
]
VOFF = {"LFP": 0.0, "NMC": 0.4, "NCA": 0.5}


def make_synthetic(root, d_text=16, seed=0):
    """6 fake cells, 3 chemistries, written in the BatteryLife pickle layout plus labels and embeddings."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "SNL").mkdir(parents=True, exist_ok=True)
    (root / "embeddings").mkdir(exist_ok=True)
    (root / "Life labels").mkdir(exist_ok=True)
    labels, emb = {}, {}
    for name, life, end_soh in SYNTH:
        chem = chem_of(name)
        n = life + 15 if end_soh < 0.8 else int(life * 0.95)
        c = np.arange(1, n + 1)
        soh = 1 - 0.2 * (c / life) ** 1.3 + rng.normal(0, 1e-3, n)
        if end_soh > 0.8:
            soh = soh - soh[-1] + end_soh
        cycles = []
        for k in range(n):
            t = np.linspace(0, 1, 40)
            q = soh[k] * 1.1
            v = np.r_[3.0 + 0.6 * t, 3.4 - 0.9 * t * (2 - soh[k])] + VOFF[chem] + rng.normal(0, 2e-3, 80)
            cycles.append(dict(voltage_in_V=v, current_in_A=np.r_[np.full(40, 1.1), np.full(40, -1.1)] * (0.9 + 0.1 * soh[k]),
                               charge_capacity_in_Ah=np.r_[q * t, np.full(40, q)],
                               discharge_capacity_in_Ah=np.r_[np.zeros(40), q * t], time_in_s=np.arange(80.0),
                               cycle_number=k + 1))
        with open(root / "SNL" / f"{name}.pkl", "wb") as f:
            pickle.dump(dict(nominal_capacity_in_Ah=1.1, SOC_interval=[0, 1], cycle_data=cycles), f)
        labels[f"{name}.pkl"] = eol_pbt(soh)[0]
        emb[name] = rng.normal(size=d_text).astype(np.float32) + VOFF[chem]
    (root / "Life labels" / "SNL_labels.json").write_text(json.dumps(labels))
    with open(root / "embeddings" / EMB_FILES[0], "wb") as f:
        pickle.dump(emb, f)
