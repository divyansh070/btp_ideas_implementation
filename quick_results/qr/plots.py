import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def e1_figure(prep, path):
    """Cycle life per chemistry; RUL labels in cycles vs fraction-of-life labels per chemistry."""
    chems = sorted(set(prep["chem"]))
    eol = {c: [x["eol"] for x in prep["cells"] if x["chem"] == c] for c in chems}
    wchem = prep["chem"][prep["cell_idx"]]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, c in enumerate(chems):
        ax[0].scatter(np.full(len(eol[c]), i) + np.random.default_rng(i).uniform(-.15, .15, len(eol[c])), eol[c], s=14)
    ax[0].set(xticks=range(len(chems)), xticklabels=chems, ylabel="cycle life (EOL, cycles)", yscale="log",
              title="Cycle life per chemistry (one dot = one cell)")
    ax[1].boxplot([prep["rul"][wchem == c] for c in chems], showfliers=False)
    ax[1].set(xticks=range(1, len(chems) + 1), xticklabels=chems, ylabel="RUL label (cycles)", title="Label: RUL in cycles")
    ax[2].boxplot([prep["frac"][wchem == c] for c in chems], showfliers=False)
    ax[2].set(xticks=range(1, len(chems) + 1), xticklabels=chems, ylabel="RUL / EOL", ylim=(0, 1),
              title="Label: fraction of life")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def e3_gate_figure(gates, path, regime):
    """Mean gate weight on the held-out chemistry's test windows, per fold and routing variant."""
    d = gates[gates.regime == regime]
    if d.empty:
        return False
    folds = list(dict.fromkeys(d.fold))
    variants = sorted(set(d.variant))
    fig, axes = plt.subplots(1, len(folds), figsize=(3.2 * len(folds), 3.4), squeeze=False)
    for ax, fold in zip(axes[0], folds):
        sub = d[d.fold == fold]
        for j, v in enumerate(variants):
            s = sub[sub.variant == v]
            m0, m1 = s.gate_e0.mean(), s.gate_e1.mean()
            ax.bar([j - .2, j + .2], [m0, m1], width=.38, color=["tab:blue", "tab:orange"])
        ax.set(xticks=range(len(variants)), xticklabels=variants, ylim=(0, 1),
               title=f"held out {fold}\n{sub.experts.iloc[0]}" + ("\nno matching expert" if not sub.has_match.iloc[0] else ""))
    axes[0][0].set_ylabel("mean gate weight (blue e0, orange e1)")
    fig.suptitle(f"E3 gate weights on held-out test windows ({regime})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
