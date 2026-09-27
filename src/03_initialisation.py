"""
Where does the instability actually come from?

Experiment 02 found three of the five methods returning bit-identical answers across
20 seeds. That is not a property of the algorithm family, it is a property of how
scikit-learn initialises each one. This script tests that directly by running the same
algorithm twice, changing only the initialisation, and measuring stability both ways.

  PCA                 randomized SVD solver   vs   exact full SVD
  NMF                 random init             vs   NNDSVDa init (derived from the data)
  DictionaryLearning  random dictionary init  vs   default SVD-based init

If instability tracks the initialisation rather than the method, the paired rows will
flip while the algorithm stays the same.
"""
import gc, json, os, time
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA, NMF, DictionaryLearning
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.expanduser("~/Desktop/Projects/mf-stability")
M = np.load(ROOT + "/data/pbmc3k_log1p_hvg1000.npy").astype(np.float32)
K, SEEDS, NCLUST = 10, 10, 8
rng = np.random.default_rng(0)

def rand_dict(seed):
    r = np.random.default_rng(seed)
    D = r.random((K, M.shape[1])).astype(np.float64)
    return D / np.linalg.norm(D, axis=1, keepdims=True)

CONDITIONS = {
    "PCA randomized SVD":      lambda s: PCA(n_components=K, svd_solver="randomized", random_state=s),
    "PCA exact SVD":           lambda s: PCA(n_components=K, svd_solver="full", random_state=s),
    "NMF random init":         lambda s: NMF(n_components=K, init="random", random_state=s, max_iter=400),
    "NMF NNDSVDa init":        lambda s: NMF(n_components=K, init="nndsvda", random_state=s, max_iter=400),
    "DictLearning random init": lambda s: DictionaryLearning(n_components=K, random_state=s, max_iter=20,
                                                             dict_init=rand_dict(s),
                                                             transform_algorithm="lasso_lars"),
    "DictLearning SVD init":   lambda s: DictionaryLearning(n_components=K, random_state=s, max_iter=20,
                                                            transform_algorithm="lasso_lars"),
}

def matched_cosine(A, B):
    A, B = normalize(A, axis=0), normalize(B, axis=0)
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())

out = {}
print(f"matrix {M.shape}, rank {K}, {SEEDS} seeds per condition", flush=True)
for name, mk in CONDITIONS.items():
    embs, labs, t0 = [], [], time.time()
    for s in range(SEEDS):
        E = np.asarray(mk(s).fit_transform(M), dtype=np.float32)
        embs.append(E)
        labs.append(KMeans(n_clusters=NCLUST, n_init=10, random_state=0).fit_predict(E))
        gc.collect()
    cos = [matched_cosine(embs[i], embs[j]) for i in range(SEEDS) for j in range(i + 1, SEEDS)]
    ari = [adjusted_rand_score(labs[i], labs[j]) for i in range(SEEDS) for j in range(i + 1, SEEDS)]
    out[name] = {"fit_seconds": round(time.time() - t0, 1), "pairs": len(cos),
                 "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)),
                 "ari_mean": float(np.mean(ari)), "ari_min": float(np.min(ari))}
    del embs, labs
    gc.collect()
    json.dump(out, open(ROOT + "/init_results.json", "w"), indent=1)
    r = out[name]
    print(f"{name:<26} {r['fit_seconds']:6.1f}s | cosine {r['cosine_mean']:.4f} "
          f"(min {r['cosine_min']:.4f}) | ARI {r['ari_mean']:.4f} (min {r['ari_min']:.4f})", flush=True)
print("\nwrote", ROOT + "/init_results.json")
