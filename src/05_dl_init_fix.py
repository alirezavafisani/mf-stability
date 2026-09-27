"""
Corrected DictionaryLearning initialisation test, plus a diagnostic for the NMF result.

The first attempt passed only dict_init. sklearn's _dict_learning takes the caller's
initialisation only when BOTH code_init and dict_init are given (_dict_learning.py, the
`if code_init is not None and dict_init is not None` branch), so it silently fell back to
the exact SVD and both conditions measured the same thing. This passes both.

It also asks why NMF with NNDSVDa initialisation produced more stable factors but much
less stable clustering, by reporting how the embedding's energy is spread across the ten
components and how balanced the resulting clusters are.
"""
import gc, json, os, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import DictionaryLearning, NMF
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.expanduser("~/Desktop/Projects/mf-stability")
M = np.load(ROOT + "/data/pbmc3k_log1p_hvg1000.npy").astype(np.float32)
K, SEEDS, NCLUST = 10, 10, 8
n, p = M.shape

def matched_cosine(A, B):
    A, B = normalize(A, axis=0), normalize(B, axis=0)
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())

def score(embs, labs):
    cos = [matched_cosine(embs[i], embs[j]) for i in range(len(embs)) for j in range(i + 1, len(embs))]
    ari = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    return {"pairs": len(cos), "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)),
            "ari_mean": float(np.mean(ari)), "ari_min": float(np.min(ari))}

# ---- 1. DictionaryLearning, genuinely random init this time ----
def make_dl(seed, random_init):
    if not random_init:
        return DictionaryLearning(n_components=K, random_state=seed, max_iter=20,
                                  transform_algorithm="lasso_lars")
    r = np.random.default_rng(seed)
    D = r.random((K, p)); D /= np.linalg.norm(D, axis=1, keepdims=True)
    C = r.random((n, K))
    return DictionaryLearning(n_components=K, random_state=seed, max_iter=20,
                              code_init=C, dict_init=D, transform_algorithm="lasso_lars")

# sanity check first: do two random inits actually differ, and does the fit see them?
a, b = make_dl(0, True), make_dl(1, True)
print("random inits differ:", not np.allclose(a.dict_init, b.dict_init), flush=True)

out = {}
for label, rnd in [("DictLearning random init", True), ("DictLearning SVD init", False)]:
    embs, labs, t0 = [], [], time.time()
    for s in range(SEEDS):
        E = np.asarray(make_dl(s, rnd).fit_transform(M), dtype=np.float32)
        embs.append(E)
        labs.append(KMeans(n_clusters=NCLUST, n_init=10, random_state=0).fit_predict(E))
        gc.collect()
    out[label] = {"fit_seconds": round(time.time() - t0, 1), **score(embs, labs)}
    del embs, labs; gc.collect()
    r = out[label]
    print(f"{label:<26} {r['fit_seconds']:6.1f}s | cosine {r['cosine_mean']:.4f} "
          f"(min {r['cosine_min']:.4f}) | ARI {r['ari_mean']:.4f} (min {r['ari_min']:.4f})", flush=True)

json.dump(out, open(ROOT + "/dl_init_results.json", "w"), indent=1)

# ---- 2. why does NNDSVDa give stable factors but unstable clusters? ----
print("\nNMF diagnostic, energy share per component and cluster balance", flush=True)
diag = {}
for label, init in [("random", "random"), ("nndsvda", "nndsvda")]:
    shares, sizes = [], []
    for s in range(3):
        E = NMF(n_components=K, init=init, random_state=s, max_iter=400).fit_transform(M)
        e = (E ** 2).sum(axis=0); e = np.sort(e / e.sum())[::-1]
        shares.append(e)
        lab = KMeans(n_clusters=NCLUST, n_init=10, random_state=0).fit_predict(E)
        sizes.append(np.sort(np.bincount(lab, minlength=NCLUST))[::-1])
    sh = np.mean(shares, axis=0); sz = np.mean(sizes, axis=0)
    diag[label] = {"energy_share_sorted": [round(float(x), 4) for x in sh],
                   "top1_share": round(float(sh[0]), 4),
                   "mean_cluster_sizes_sorted": [int(x) for x in sz],
                   "largest_cluster_frac": round(float(sz[0] / n), 4)}
    print(f"  {label:<9} top component holds {sh[0]:.1%} of the energy, "
          f"largest cluster holds {sz[0]/n:.1%} of the 2,700 cells", flush=True)

json.dump(diag, open(ROOT + "/nmf_diagnostic.json", "w"), indent=1)
print("\nwrote dl_init_results.json and nmf_diagnostic.json")
