import os
import time
import tracemalloc
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# ==========================================================================
#  SETTINGS -- multi-seed evaluation settings and dataset paths
# ==========================================================================
def _resolve_path(env_var, default_relative, fallback_absolute=None):
    """Resolves dataset path from environment variable, relative path, or user fallback."""
    if env_var in os.environ:
        return os.environ[env_var]
    if os.path.exists(default_relative):
        return default_relative
    if fallback_absolute and os.path.exists(fallback_absolute):
        return fallback_absolute
    return default_relative

PATH_SCENE15     = _resolve_path("PATH_SCENE15", "data/Scene15_multiview.mat", "/Users/siyakohli/Desktop/DSA/Recursion/research/Scene15_multiview.mat")
PATH_HANDWRITTEN = _resolve_path("PATH_HANDWRITTEN", "data/HandWritten.mat", "/Users/siyakohli/Downloads/HandWritten.mat")
PATH_CUB         = _resolve_path("PATH_CUB", "data/CUB_200_2011/images", "CUB_200_2011/images")
CUB_CLASSES      = 10

M         = 1.1
VIEW_EXP  = 2.0
LAMBDA_GR = 0.2
PCA_DIMS  = 30
N_ITER    = 50

N_SEEDS   = 20
SEEDS     = list(range(42, 42 + N_SEEDS))     # 42 = the seed used for the original Table III

OUTPUT_CSV = "mvfcm_multiseed_results.csv"
# ==========================================================================


# --------------------------------------------------------------------------
#  SECTION 1 -- helpers
# --------------------------------------------------------------------------
def relabel(y):
    _, out = np.unique(y, return_inverse=True)
    return out.astype(int)


def accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(r, c)) / len(y_true)


def prepare(views, seed):
    out = []
    for v in views:
        v = StandardScaler().fit_transform(np.asarray(v, dtype=np.float64))
        if v.shape[1] > PCA_DIMS:
            v = PCA(n_components=PCA_DIMS, random_state=seed).fit_transform(v)
        out.append(v)
    return out


def start_U(n, c, seed):
    rng = np.random.default_rng(seed)
    U = rng.random((n, c))
    return U / U.sum(axis=1, keepdims=True)


def sq_dist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def update_U(d2, m):
    d2 = np.maximum(d2, 1e-12)
    inv = d2 ** (-1.0 / (m - 1))
    return inv / inv.sum(axis=1, keepdims=True)


def centroids(X, U, m):
    Um = U ** m
    return Um.T @ X / (Um.sum(axis=0)[:, None] + 1e-12)


# --------------------------------------------------------------------------
#  SECTION 2 -- the algorithms, parameterized by seed for reproducibility
# --------------------------------------------------------------------------
def fcm(X, c, seed):
    U = start_U(X.shape[0], c, seed)
    for _ in range(N_ITER):
        U = update_U(sq_dist(X, centroids(X, U, M)), M)
    return U


def mvfcm_baseline(views, c, seed):
    U = start_U(views[0].shape[0], c, seed)
    for _ in range(N_ITER):
        d2 = sum(sq_dist(v, centroids(v, U, M)) for v in views) / len(views)
        U = update_U(d2, M)
    return U.argmax(axis=1)


def weighted_mvfcm(views, c, seed):
    p = 1.0 / (VIEW_EXP - 1.0)
    U = start_U(views[0].shape[0], c, seed)
    w = np.ones(len(views)) / len(views)
    for _ in range(N_ITER):
        Um = U ** M
        d2_each = [sq_dist(v, centroids(v, U, M)) for v in views]
        J = np.array([(Um * d).sum() for d in d2_each]) + 1e-12
        w = (1.0 / J) ** p
        w = w / w.sum()
        U = update_U(sum(wi * d for wi, d in zip(w, d2_each)), M)
    return U.argmax(axis=1)


def kernel_mvfcm(views, c, seed):
    X = np.hstack(views)
    d2 = sq_dist(X, X)
    sigma2 = np.median(d2[d2 > 0]) + 1e-12
    K = np.exp(-d2 / (2 * sigma2))
    U = start_U(X.shape[0], c, seed)
    for _ in range(N_ITER):
        Um = U ** M
        s = Um.sum(axis=0) + 1e-12
        third = np.einsum("ik,ij,jk->k", Um, K, Um) / (s ** 2)
        U = update_U(1.0 - 2.0 * (K @ Um) / s + third[None, :], M)
    return U.argmax(axis=1)


def graph_mvfcm(views, c, seed, alpha=None, knn=7):
    alpha = LAMBDA_GR if alpha is None else alpha
    X = np.hstack(views)
    A = kneighbors_graph(X, n_neighbors=knn, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T).toarray()
    deg = A.sum(axis=1) + 1e-12
    S = A / deg[:, None]
    U = start_U(X.shape[0], c, seed)
    for _ in range(N_ITER):
        d2 = sum(sq_dist(v, centroids(v, U, M)) for v in views) / len(views)
        U = update_U(d2, M)
        U = (1.0 - alpha) * U + alpha * (S @ U)
        U = U / U.sum(axis=1, keepdims=True)
    return U.argmax(axis=1)


def robust_mvfcm(views, c, seed):
    """Eq. (5) implementation: L1 loss, weighted-median centroids."""
    n = views[0].shape[0]
    U = start_U(n, c, seed)
    for _ in range(N_ITER):
        Um = U ** M
        d1 = np.zeros((n, c))
        for v in views:
            C = np.empty((c, v.shape[1]))
            order = np.argsort(v, axis=0)
            v_sorted = np.take_along_axis(v, order, axis=0)
            for j in range(c):
                w_sorted = Um[:, j][order]
                cum = np.cumsum(w_sorted, axis=0)
                pick = (cum >= cum[-1] / 2.0).argmax(axis=0)
                C[j] = v_sorted[pick, np.arange(v.shape[1])]
            d1 += np.abs(v[:, None, :] - C[None, :, :]).sum(axis=2)
        U = update_U(d1 / len(views), M)
    return U.argmax(axis=1)


def online_mvfcm(views, c, seed, epochs=5, batch=200):
    rng = np.random.default_rng(seed)
    X = np.hstack(views)
    n = X.shape[0]
    C = X[rng.choice(n, c, replace=False)].copy()
    seen = np.ones(c)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            B = X[order[s:s + batch]]
            if len(B) == 0:
                continue
            Ub = update_U(sq_dist(B, C), M) ** M
            for j in range(c):
                wj = Ub[:, j].sum()
                if wj > 1e-9:
                    lr = wj / (seen[j] + wj)
                    C[j] = (1 - lr) * C[j] + lr * (Ub[:, j] @ B) / wj
                    seen[j] += wj
    return update_U(sq_dist(X, C), M).argmax(axis=1)


def kmeans_all(views, c, seed):
    return KMeans(n_clusters=c, n_init=10, random_state=seed).fit_predict(np.hstack(views))


# --------------------------------------------------------------------------
#  SECTION 3 -- loaders
# --------------------------------------------------------------------------
def load_scene15():
    import scipy.io as sio
    d = sio.loadmat(PATH_SCENE15)
    return [d["X1"], d["X2"], d["X3"]], d["Y"].flatten()


def load_handwritten():
    import scipy.io as sio
    d = sio.loadmat(PATH_HANDWRITTEN)
    return [d["X"][0, i] for i in range(6)], d["truth"].flatten()


def load_mnist_usps():
    from sklearn.datasets import fetch_openml
    rng = np.random.default_rng(42)   # dataset construction itself is fixed, not swept
    mn = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    Xm = mn.data.astype(np.float32) / 255.0
    ym = mn.target.astype(int)
    try:
        us = fetch_openml("USPS", version=2, as_frame=False, parser="auto")
        Xu = us.data.astype(np.float32)
        Xu = (Xu - Xu.min()) / (Xu.max() - Xu.min() + 1e-12)
        yu = us.target.astype(int) % 10
        print("   using the real USPS corpus")
        v1, v2, y = [], [], []
        for dig in range(10):
            mi, ui = np.where(ym == dig)[0], np.where(yu == dig)[0]
            take = min(100, len(mi), len(ui))
            v1.append(Xm[rng.choice(mi, take, replace=False)])
            v2.append(Xu[rng.choice(ui, take, replace=False)])
            y += [dig] * take
        return [np.vstack(v1), np.vstack(v2)], np.array(y)
    except Exception as e:
        print(f"   USPS unavailable ({type(e).__name__}); using MNIST pixels + HOG as the two views")
        from skimage.feature import hog
        idx = rng.choice(len(Xm), 2000, replace=False)
        X, y = Xm[idx], ym[idx]
        v2 = np.array([hog(x.reshape(28, 28), orientations=8,
                           pixels_per_cell=(7, 7), cells_per_block=(1, 1))
                       for x in X])
        return [X, v2], y


def load_reuters():
    import nltk
    from nltk.corpus import reuters as rc
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        rc.fileids()
    except LookupError:
        nltk.download("reuters", quiet=True)
        nltk.download("punkt", quiet=True)
    cats = sorted(rc.categories(), key=lambda t: -len(rc.fileids(t)))[:4]
    docs, y = [], []
    for i, cat in enumerate(cats):
        for fid in rc.fileids(cat)[:200]:
            docs.append(" ".join(rc.words(fid)))
            y.append(i)
    v1 = TfidfVectorizer(max_features=500, stop_words="english").fit_transform(docs).toarray()
    v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=500).fit_transform(docs).toarray()
    return [v1, v2], np.array(y)


def load_cub():
    cache = "cub_features_v2.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return [z["v1"], z["v2"]], z["y"]
    import torch
    from torchvision import models, transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ds = ImageFolder(PATH_CUB, transform=tf)
    keep = [i for i, (_, l) in enumerate(ds.samples) if l < CUB_CLASSES]
    loader = DataLoader(torch.utils.data.Subset(ds, keep), batch_size=32, shuffle=False, num_workers=0)
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval()
    deep, hist, labs = [], [], []
    print("   extracting ResNet18 features (one-off)...")
    with torch.no_grad():
        for i, (imgs, lbl) in enumerate(loader):
            deep.append(net(norm(imgs)).numpy())
            for img in imgs:
                px = img.reshape(3, -1).T.numpy()
                h, _ = np.histogramdd(px, bins=(6, 6, 6), range=((0, 1), (0, 1), (0, 1)))
                h = h.ravel()
                hist.append(h / (h.sum() + 1e-10))
            labs.append(lbl.numpy())
            if (i + 1) % 10 == 0:
                print(f"      batch {i+1}/{len(loader)}")
    v1, v2, y = np.vstack(deep), np.array(hist), np.hstack(labs)
    np.savez_compressed(cache, v1=v1, v2=v2, y=y)
    return [v1, v2], y


DATASETS = [
    ("Scene-15",      load_scene15),
    ("Handwritten",   load_handwritten),
    ("MNIST+USPS",    load_mnist_usps),
    ("Reuters-21578", load_reuters),
    ("CUB-200-2011",  load_cub),
]


# --------------------------------------------------------------------------
#  SECTION 4 -- run one dataset across all seeds
# --------------------------------------------------------------------------
def run_one(method_name, method_type, fn, y):
    """Time + measure memory for one (method, seed) run; return a result row."""
    tracemalloc.start()
    t0 = time.time()
    pred = relabel(fn())
    dt = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "method": method_name, "type": method_type,
        "ACC": accuracy(y, pred), "NMI": normalized_mutual_info_score(y, pred),
        "ARI": adjusted_rand_score(y, pred),
        "seconds": dt, "peak_mem_mb": peak / (1024 ** 2),
    }


def run_dataset_multiseed(name, loader):
    print("\n" + "=" * 74)
    print(name)
    print("=" * 74)

    views_raw, y = loader()
    y = relabel(y)
    c = len(np.unique(y))
    n_views = len(views_raw)
    print(f"   {len(y)} samples | {c} classes | {n_views} views | {len(SEEDS)} seeds")

    rows = []
    for seed in SEEDS:
        views = prepare(views_raw, seed)

        variant_jobs = [
            ("WMV-FCM",  lambda: weighted_mvfcm(views, c, seed)),
            ("K-MVFCM",  lambda: kernel_mvfcm(views, c, seed)),
            ("GR-MVFCM", lambda: graph_mvfcm(views, c, seed)),
            ("R-MVFCM",  lambda: robust_mvfcm(views, c, seed)),
            ("O-MVFCM",  lambda: online_mvfcm(views, c, seed)),
        ]
        baseline_jobs = [
            ("MV-FCM",  lambda: mvfcm_baseline(views, c, seed)),
            ("k-means", lambda: kmeans_all(views, c, seed)),
        ]

        for mname, fn in variant_jobs + baseline_jobs:
            r = run_one(mname, "variant" if (mname, fn) in variant_jobs else "baseline", fn, y)
            r["dataset"], r["seed"] = name, seed
            rows.append(r)

        # best-single-view: per-view FCM, keep the highest-ARI view THIS seed
        best = None
        for i in range(n_views):
            r = run_one(f"FCM-view{i+1}", "baseline", lambda i=i: fcm(views[i], c, seed).argmax(axis=1), y)
            if best is None or r["ARI"] > best["ARI"]:
                best = r
        best["method"] = "best-single-view"
        best["dataset"], best["seed"] = name, seed
        rows.append(best)

        print(f"   seed {seed:3d} done "
              f"(WMV ARI={[r for r in rows if r['seed']==seed and r['method']=='WMV-FCM'][0]['ARI']:.3f}, "
              f"k-means ARI={[r for r in rows if r['seed']==seed and r['method']=='k-means'][0]['ARI']:.3f})")

    return rows


# --------------------------------------------------------------------------
#  SECTION 5 -- run everything
# --------------------------------------------------------------------------
def main():
    print(f"settings: M={M}  VIEW_EXP={VIEW_EXP}  LAMBDA_GR={LAMBDA_GR}  "
          f"PCA_DIMS={PCA_DIMS}  N_ITER={N_ITER}  SEEDS={SEEDS[0]}..{SEEDS[-1]} (n={len(SEEDS)})")
    all_rows = []
    for name, loader in DATASETS:
        try:
            all_rows += run_dataset_multiseed(name, loader)
        except FileNotFoundError as e:
            print(f"\n   SKIPPED {name}: file not found -> {e.filename}")
        except Exception as e:
            print(f"\n   SKIPPED {name}: {type(e).__name__}: {e}")

    if not all_rows:
        print("\nNothing ran. Check the paths at the top of the file or set environment variables.")
        return

    df = pd.DataFrame(all_rows)
    # column names/order matching mvfcm_stats.py expectations
    df = df.rename(columns={"seconds": "runtime_sec"})
    df = df[["dataset", "method", "type", "seed", "ACC", "NMI", "ARI", "runtime_sec", "peak_mem_mb"]]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")
    print("Next: python mvfcm_stats.py  (mean+/-std, Wilcoxon+Holm-Bonferroni, revised Table III)")


if __name__ == "__main__":
    main()
