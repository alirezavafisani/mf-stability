"""Build the two figures from results.json and init_results.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/Projects/mf-stability")
R = json.load(open(ROOT + "/results.json"))
res = R["results"]

BLUE, RED, GREY = "#cfe3f5", "#b3402f", "#3b6ea5"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444", "axes.linewidth": 0.8})

# ---------- figure 1, the five methods ----------
order = sorted(res, key=lambda n: -res[n]["cosine_mean"])
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
panels = [("cosine_all", "Are the latent dimensions the same?",
           "matched |cosine| between two runs"),
          ("ari_all", "Do the cells cluster the same way?",
           "adjusted Rand index between two runs")]

for ax, (key, title, ylab) in zip(axes, panels):
    data = [res[n][key] for n in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.5,
                    medianprops=dict(color=RED, linewidth=2),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                                    markeredgecolor="#888"))
    for p in bp["boxes"]:
        p.set_facecolor(BLUE); p.set_edgecolor(GREY)
    for i, n in enumerate(data, start=1):
        # jittered points so the reader sees 190 pairs, not just a summary
        ax.scatter(np.random.default_rng(i).normal(i, 0.055, len(n)), n,
                   s=3, color="#2c4f75", alpha=0.25, zorder=3)
        ax.text(i, 1.045, f"{np.mean(n):.3f}", ha="center", fontsize=8.5, color="#333")
    ax.set_title(title, fontsize=12, pad=18)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_ylim(0.35, 1.08)
    ax.axhline(1.0, color="#bbb", lw=0.7, ls=":", zorder=0)
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", rotation=18)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

fig.suptitle("Same matrix, same rank, only the random seed changes\n"
             f"PBMC 3k, {R['matrix_shape'][0]:,} cells x {R['matrix_shape'][1]:,} genes, "
             f"rank {R['rank']}, {R['seeds']} seeds, "
             f"{res[order[0]]['pairs']} run pairs per method", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(ROOT + "/stability.png", dpi=170)
print("wrote stability.png")

# ---------- figure 2, where the instability comes from ----------
ip = ROOT + "/init_results.json"
if os.path.exists(ip):
    I = json.load(open(ip))
    pairs = [("PCA randomized SVD", "PCA exact SVD", "PCA"),
             ("NMF random init", "NMF NNDSVDa init", "NMF"),
             ("DictLearning random init", "DictLearning SVD init", "DictionaryLearning")]
    pairs = [p for p in pairs if p[0] in I and p[1] in I]
    if pairs:
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        x = np.arange(len(pairs)); w = 0.34
        seeded = [I[a]["ari_mean"] for a, b, _ in pairs]
        derived = [I[b]["ari_mean"] for a, b, _ in pairs]
        b1 = ax.bar(x - w/2, seeded, w, label="initialised from the random seed",
                    color="#e0a89c", edgecolor=RED)
        b2 = ax.bar(x + w/2, derived, w, label="initialised from the data",
                    color=BLUE, edgecolor=GREY)
        for bars in (b1, b2):
            ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
        ax.set_xticks(x, [lab for _, _, lab in pairs])
        ax.set_ylim(0, 1.24)
        ax.set_ylabel("mean ARI between two runs")
        ax.set_title("Same algorithm, same data, same rank. Only the start point changes\n"
                     f"{I[pairs[0][0]]['pairs']} run pairs per bar",
                     fontsize=12, pad=12)
        # NMF runs the other way, and the figure should say so rather than hide it
        nmf_i = next((i for i, (_, _, lab) in enumerate(pairs) if lab == "NMF"), None)
        if nmf_i is not None:
            ax.annotate("NMF reverses:\nits data-derived start\nconcentrates 46% of the\nenergy in one component",
                        xy=(nmf_i + w/2, derived[nmf_i]), xytext=(nmf_i + 0.05, 1.02),
                        fontsize=8, color="#333", ha="center",
                        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        fig.savefig(ROOT + "/initialisation.png", dpi=170)
        print("wrote initialisation.png")
