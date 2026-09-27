import io, os, subprocess, tarfile, time
import numpy as np, scipy.io, scipy.sparse as sp

OUT = os.path.expanduser("~/Desktop/Projects/mf-stability")
os.makedirs(OUT + "/data", exist_ok=True)
tarp = OUT + "/data/pbmc3k.tar.gz"

if not os.path.exists(tarp):
    url = "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz"
    print("downloading...", flush=True)
    # 10x rejects the default Python user agent with HTTP 403, so go through curl
    subprocess.run(["curl", "-fsSL", "-A", "Mozilla/5.0", url, "-o", tarp], check=True)
print("tar size MB:", round(os.path.getsize(tarp) / 1e6, 1))

with tarfile.open(tarp) as tf:
    names = tf.getnames()
    print("members:", [n for n in names if not n.endswith("/")])
    mtx = [n for n in names if n.endswith("matrix.mtx")][0]
    gen = [n for n in names if n.endswith("genes.tsv")][0]
    bar = [n for n in names if n.endswith("barcodes.tsv")][0]
    X = scipy.io.mmread(io.BytesIO(tf.extractfile(mtx).read())).tocsr()   # genes x cells
    genes = [l.decode().split("\t")[1].strip() for l in tf.extractfile(gen)]
    cells = [l.decode().strip() for l in tf.extractfile(bar)]

print("\nRAW matrix (genes x cells):", X.shape, "| nnz:", X.nnz, "| sparsity:",
      round(100 * (1 - X.nnz / (X.shape[0] * X.shape[1])), 2), "%")
print("genes:", len(genes), "cells:", len(cells))

# cells x genes
A = X.T.tocsr()
# basic QC: cells with >=200 genes, genes in >=3 cells
gpc = np.asarray((A > 0).sum(axis=1)).ravel()
A = A[gpc >= 200]
cpg = np.asarray((A > 0).sum(axis=0)).ravel()
A = A[:, cpg >= 3]
print("after QC (cells x genes):", A.shape)

# CPM-style normalise then log1p  -> non-negative, works for all five methods
tot = np.asarray(A.sum(axis=1)).ravel()
A = sp.diags(1e4 / np.maximum(tot, 1)) @ A
A = A.log1p()

# top variable genes
Ad = np.asarray(A.todense())
v = Ad.var(axis=0)
idx = np.argsort(v)[::-1][:1000]
M = Ad[:, idx]
print("final dense matrix:", M.shape, "| min:", round(float(M.min()), 3),
      "| max:", round(float(M.max()), 2), "| all non-negative:", bool((M >= 0).all()))
np.save(OUT + "/data/pbmc3k_log1p_hvg1000.npy", M.astype(np.float32))
np.save(OUT + "/data/hvg_gene_names.npy", np.array([genes[i] for i in idx]))

# ---- time ONE run of each method at rank 10 ----
from sklearn.decomposition import PCA, KernelPCA, NMF, DictionaryLearning, LatentDirichletAllocation
K = 10
tests = {
    "PCA":                 lambda s: PCA(n_components=K, random_state=s),
    "KernelPCA":           lambda s: KernelPCA(n_components=K, kernel="rbf", random_state=s),
    "NMF":                 lambda s: NMF(n_components=K, init="random", random_state=s, max_iter=300),
    "DictionaryLearning":  lambda s: DictionaryLearning(n_components=K, random_state=s, max_iter=20, transform_algorithm="lasso_lars"),
    "LatentDirichlet":     lambda s: LatentDirichletAllocation(n_components=K, random_state=s, max_iter=10),
}
print("\n--- timing one run each at rank 10 ---")
for name, mk in tests.items():
    t = time.time()
    try:
        mdl = mk(0); mdl.fit(M)
        print(f"{name:<20} {time.time()-t:7.1f} s   ok")
    except Exception as e:
        print(f"{name:<20} {time.time()-t:7.1f} s   FAILED {type(e).__name__}: {str(e)[:70]}")
print("\nsaved matrix to", OUT + "/data/")
