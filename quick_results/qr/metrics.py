"""Per-chemistry metrics in cycles. R² is computed inside one chemistry only, never pooled across chemistries."""
import numpy as np
import pandas as pd


def chem_metrics(true_rul, pred_rul, cell_idx, mape_floor=50):
    t, p = np.asarray(true_rul, float), np.asarray(pred_rul, float)
    err = p - t
    r2 = 1 - (err ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-9)
    cell_r2 = []
    for c in np.unique(cell_idx):
        m = cell_idx == c
        if m.sum() > 1 and t[m].var() > 0:
            cell_r2.append(1 - ((p[m] - t[m]) ** 2).sum() / ((t[m] - t[m].mean()) ** 2).sum())
    keep = t >= mape_floor
    return dict(rmse=float(np.sqrt((err ** 2).mean())), r2=float(r2),
                r2_cell_mean=float(np.mean(cell_r2)) if cell_r2 else np.nan,
                mape50=float((np.abs(err[keep]) / t[keep]).mean()) if keep.any() else np.nan,
                n_test_cells=int(len(np.unique(cell_idx))), n_test_windows=int(len(t)))


def summarise(df, keys, metrics=("rmse", "r2", "r2_cell_mean", "mape50")):
    """mean ± std over seeds as strings, one row per keys combination."""
    g = df.groupby(list(keys), sort=False)
    out = g[list(metrics)].agg(["mean", "std"])
    res = pd.DataFrame(index=out.index)
    for m in metrics:
        res[m] = [f"{a:.3g} ± {b:.2g}" if np.isfinite(a) else "n/a"
                  for a, b in zip(out[(m, "mean")], out[(m, "std")].fillna(0))]
    res["seeds"] = g.size()
    return res.reset_index()


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
