"""
Stability of matrix factorization on single-cell gene expression.

Question: run the same decomposition twice on the same matrix, changing only the
random seed. Do you get the same answer?

Data  : 10x Genomics PBMC 3k, 2,700 human immune cells x 1,000 most variable genes,
        CPM-normalised then log1p, so every entry is non-negative and all five
        methods run on the identical matrix.
Design: 5 methods x 20 seeds, rank 10.
Metric: two, because they answer different questions.
        1. matched cosine, are the 10 latent dimensions themselves the same. They come
           out in arbitrary order and sign, so pair them across runs with the Hungarian
           algorithm on |cosine| first, then average.
        2. clustering ARI, the downstream consequence. Cluster cells in each run's
           embedding, then ask whether two runs group the same cells together.

Memory note: one method is processed at a time and its embeddings are freed before the
next starts, and each method's result is written to disk as soon as it finishes, so a
kill loses at most one method.
"""
import gc, json, os, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import (PCA, KernelPCA, NMF, DictionaryLearning,
                                   LatentDirichletAllocation)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.expanduser("~/Desktop/Projects/mf-stability")
PART = ROOT + "/partial_results.json"
M = np.load(ROOT + "/data/pbmc3k_log1p_hvg1000.npy").astype(np.float32)
K, SEEDS, NCLUST = 10, 20, 8
print(f"matrix {M.shape} {M.dtype}, rank {K}, {SEEDS} seeds per method", flush=True)

METHODS = {
    "PCA":                lambda s: PCA(n_components=K, random_state=s),
    "KernelPCA":          lambda s: KernelPCA(n_components=K, kernel="rbf", random_state=s,
                                              copy_X=False),
    "NMF":                lambda s: NMF(n_components=K, init="random", random_state=s,
                                        max_iter=400),
    "DictionaryLearning": lambda s: DictionaryLearning(n_components=K, random_state=s,
                                                       max_iter=20,
                                                       transform_algorithm="lasso_lars"),
    # online learning keeps LDA's memory flat; batch mode is what ran the machine out
    "LatentDirichlet":    lambda s: LatentDirichletAllocation(n_components=K, random_state=s,
                                                              max_iter=10,
                                                              learning_method="online",
                                                              batch_size=256),
}

def matched_cosine(A, B):
    """Mean |cosine| between columns of two embeddings after optimal pairing."""
    A, B = normalize(A, axis=0), normalize(B, axis=0)
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())

done = {}
if os.path.exists(PART):
    done = json.load(open(PART))
    print("resuming, already done:", list(done), flush=True)

for name, mk in METHODS.items():
    if name in done:
        continue
    embs, labs = [], []
    t0 = time.time()
    for s in range(SEEDS):
        E = np.asarray(mk(s).fit_transform(M), dtype=np.float32)
        embs.append(E)
        labs.append(KMeans(n_clusters=NCLUST, n_init=10, random_state=0).fit_predict(E))
        gc.collect()
    fit_s = time.time() - t0

    cos, ari = [], []
    for i in range(SEEDS):
        for j in range(i + 1, SEEDS):
            cos.append(matched_cosine(embs[i], embs[j]))
            ari.append(adjusted_rand_score(labs[i], labs[j]))
    done[name] = {
        "fit_seconds": round(fit_s, 1), "pairs": len(cos),
        "cosine_mean": float(np.mean(cos)), "cosine_std": float(np.std(cos)),
        "cosine_min": float(np.min(cos)),
        "ari_mean": float(np.mean(ari)), "ari_std": float(np.std(ari)),
        "ari_min": float(np.min(ari)),
        "cosine_all": [float(x) for x in cos], "ari_all": [float(x) for x in ari],
    }
    del embs, labs
    gc.collect()
    json.dump(done, open(PART, "w"), indent=1)
    r = done[name]
    print(f"{name:<20} {fit_s:6.1f}s fit | cosine {r['cosine_mean']:.4f} +/- {r['cosine_std']:.4f} "
          f"(min {r['cosine_min']:.4f}) | ARI {r['ari_mean']:.4f} +/- {r['ari_std']:.4f} "
          f"(min {r['ari_min']:.4f})", flush=True)

json.dump({"matrix_shape": list(M.shape), "rank": K, "seeds": SEEDS,
           "n_clusters": NCLUST, "results": done},
          open(ROOT + "/results.json", "w"), indent=1)

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = sorted(done, key=lambda n: -done[n]["cosine_mean"])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, key, title, ylab in [
    (axes[0], "cosine_all", "Are the latent dimensions the same?",
     "matched |cosine| between two runs"),
    (axes[1], "ari_all", "Do the cells cluster the same way?",
     "adjusted Rand index between two runs")]:
    bp = ax.boxplot([done[n][key] for n in order], labels=order,
                    patch_artist=True, widths=0.55)
    for p in bp["boxes"]:
        p.set_facecolor("#cfe3f5"); p.set_edgecolor("#3b6ea5")
    for m in bp["medians"]:
        m.set_color("#b3402f"); m.set_linewidth(2)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Stability of five matrix decompositions on PBMC 3k\n"
             f"{M.shape[0]:,} cells x {M.shape[1]:,} genes, rank {K}, {SEEDS} seeds, "
             f"{done[order[0]]['pairs']} run pairs per method", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(ROOT + "/stability.png", dpi=170)
print("\nwrote", ROOT + "/results.json and stability.png")
