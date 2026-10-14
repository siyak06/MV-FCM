"""
Sensitivity / ablation sweeps for the MV-FCM comparative study, answering
Review 2 point 3 (PCA dimensionality) and point 4 (hyperparameter
sensitivity: m, lambda, k-NN size, kernel bandwidth), and the matching
asks in Review 3.

Reuses the corrected algorithm implementations from mvfcm_multiseed_v2.py
(imported, not copied) so the sweep is guaranteed to test the same code
that produced Table III -- monkeypatching that module's globals (M,
LAMBDA_GR, PCA_DIMS) per sweep point rather than re-implementing anything.

Two sweeps, run on two representative datasets (CUB-200-2011, where the
margins between methods are close, and Handwritten, where they are not):

  1. PCA_SWEEP: PCA_DIMS in {10, 20, 30, 50, 100}, every method re-run,
     5 seeds per point (this affects every method, since PCA is shared
     preprocessing -- Section IV-D -- so the sweep has to cover all 8).

  2. HYPERPARAM_SWEEP: one parameter varied at a time, others held at the
     paper's defaults (m=1.1, LAMBDA_GR=0.2, knn=7, kernel bandwidth
     multiplier=1.0, online batch=200/epochs=5), 5 seeds per point:
       - m in {1.1, 1.3, 1.5, 1.7, 2.0}            -- affects every method
       - lambda (graph strength) in {0.05,0.1,0.2,0.5,1.0} -- GR-MVFCM only
       - k-NN size in {5, 7, 10, 15, 20}             -- GR-MVFCM only
       - kernel bandwidth multiplier in {0.5, 1, 2}  -- K-MVFCM only
       - online batch size in {50, 100, 200, 400}    -- O-MVFCM only
       - online epochs in {2, 5, 10}                 -- O-MVFCM only

  NOT swept, and why: clipping threshold (Robust MV-FCM's actual
  implementation is an L1/weighted-median loss with no clip step --
  see the note on run_robust_mvfcm in mvfcm_multiseed_v2.py) and a
  fixed online learning rate eta (Online MV-FCM uses a running-average
  update, w_j/(seen_j+w_j), not a free eta -- batch size and epoch
  count are the closest real hyperparameters, and are swept instead).

Output: mvfcm_pca_sweep.csv, mvfcm_hyperparam_sweep.csv
Columns: sweep, dataset, param_name, param_value, method, seed, ACC, NMI, ARI
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import mvfcm_multiseed_v2 as base   # the corrected, Table-III-producing implementation

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SWEEP_SEEDS = [42, 43, 44, 45, 46]          # 5 seeds per sweep point (cheaper than the full 20)
SWEEP_DATASETS = [("CUB-200-2011", base.load_cub), ("Handwritten", base.load_handwritten)]

PCA_VALUES = [10, 20, 30, 50, 100]
M_VALUES = [1.1, 1.3, 1.5, 1.7, 2.0]
LAMBDA_VALUES = [0.05, 0.1, 0.2, 0.5, 1.0]
KNN_VALUES = [5, 7, 10, 15, 20]
KERNEL_BW_MULT_VALUES = [0.5, 1.0, 2.0]
ONLINE_BATCH_VALUES = [50, 100, 200, 400]
ONLINE_EPOCHS_VALUES = [2, 5, 10]

# paper defaults, restored after each sweep point
DEFAULT_M, DEFAULT_LAMBDA, DEFAULT_PCA = base.M, base.LAMBDA_GR, base.PCA_DIMS


def reset_defaults():
    base.M, base.LAMBDA_GR, base.PCA_DIMS = DEFAULT_M, DEFAULT_LAMBDA, DEFAULT_PCA


# --------------------------------------------------------------------------
# Local variant of kernel_mvfcm with a bandwidth multiplier
# (base.kernel_mvfcm hardcodes sigma2 = median pairwise sq. distance;
#  this is the smallest change that lets bandwidth be swept without
#  touching the module that produced Table III)
# --------------------------------------------------------------------------
def kernel_mvfcm_bw(views, c, seed, bw_mult):
    X = np.hstack(views)
    d2 = base.sq_dist(X, X)
    sigma2 = bw_mult * (np.median(d2[d2 > 0]) + 1e-12)
    K = np.exp(-d2 / (2 * sigma2))
    U = base.start_U(X.shape[0], c, seed)
    for _ in range(base.N_ITER):
        Um = U ** base.M
        s = Um.sum(axis=0) + 1e-12
        third = np.einsum("ik,ij,jk->k", Um, K, Um) / (s ** 2)
        U = base.update_U(1.0 - 2.0 * (K @ Um) / s + third[None, :], base.M)
    return U.argmax(axis=1)


# --------------------------------------------------------------------------
# Metric helper
# --------------------------------------------------------------------------
def score(y, pred):
    pred = base.relabel(pred)
    return {
        "ACC": base.accuracy(y, pred),
        "NMI": normalized_mutual_info_score(y, pred),
        "ARI": adjusted_rand_score(y, pred),
    }


# --------------------------------------------------------------------------
# Sweep 1: PCA dimensionality -- every method, both datasets
# --------------------------------------------------------------------------
def run_pca_sweep():
    rows = []
    for dataset_name, loader in SWEEP_DATASETS:
        print(f"\n=== PCA sweep: {dataset_name} ===")
        try:
            views_raw, y = loader()
        except FileNotFoundError as e:
            print(f"  SKIPPED {dataset_name}: file not found -> {e.filename}")
            continue
        except Exception as e:
            print(f"  SKIPPED {dataset_name}: {type(e).__name__}: {e}")
            continue

        y = base.relabel(y)
        c = len(np.unique(y))

        for pca_dim in PCA_VALUES:
            reset_defaults()
            base.PCA_DIMS = pca_dim
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                jobs = [
                    ("WMV-FCM",  lambda: base.weighted_mvfcm(views, c, seed)),
                    ("K-MVFCM",  lambda: base.kernel_mvfcm(views, c, seed)),
                    ("GR-MVFCM", lambda: base.graph_mvfcm(views, c, seed)),
                    ("R-MVFCM",  lambda: base.robust_mvfcm(views, c, seed)),
                    ("O-MVFCM",  lambda: base.online_mvfcm(views, c, seed)),
                    ("MV-FCM",   lambda: base.mvfcm_baseline(views, c, seed)),
                    ("k-means",  lambda: base.kmeans_all(views, c, seed)),
                ]
                for mname, fn in jobs:
                    pred = fn()
                    m = score(y, pred)
                    rows.append({"sweep": "PCA", "dataset": dataset_name,
                                 "param_name": "pca_dims", "param_value": pca_dim,
                                 "method": mname, "seed": seed, **m})
            print(f"  PCA_DIMS={pca_dim:>4d}  done ({len(SWEEP_SEEDS)} seeds x 7 methods)")
    reset_defaults()
    return rows


# --------------------------------------------------------------------------
# Sweep 2: hyperparameters, one at a time
# --------------------------------------------------------------------------
def run_hyperparam_sweep():
    rows = []
    for dataset_name, loader in SWEEP_DATASETS:
        print(f"\n=== Hyperparameter sweep: {dataset_name} ===")
        try:
            views_raw, y = loader()
        except FileNotFoundError as e:
            print(f"  SKIPPED {dataset_name}: file not found -> {e.filename}")
            continue
        except Exception as e:
            print(f"  SKIPPED {dataset_name}: {type(e).__name__}: {e}")
            continue

        y = base.relabel(y)
        c = len(np.unique(y))

        # -- m: affects every method --
        for m_val in M_VALUES:
            reset_defaults()
            base.M = m_val
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                jobs = [
                    ("WMV-FCM",  lambda: base.weighted_mvfcm(views, c, seed)),
                    ("K-MVFCM",  lambda: base.kernel_mvfcm(views, c, seed)),
                    ("GR-MVFCM", lambda: base.graph_mvfcm(views, c, seed)),
                    ("R-MVFCM",  lambda: base.robust_mvfcm(views, c, seed)),
                    ("O-MVFCM",  lambda: base.online_mvfcm(views, c, seed)),
                    ("MV-FCM",   lambda: base.mvfcm_baseline(views, c, seed)),
                ]
                for mname, fn in jobs:
                    met = score(y, fn())
                    rows.append({"sweep": "hyperparam", "dataset": dataset_name,
                                 "param_name": "m", "param_value": m_val,
                                 "method": mname, "seed": seed, **met})
            print(f"  m={m_val}  done")
        reset_defaults()

        # -- lambda (graph propagation strength): GR-MVFCM only --
        for lam in LAMBDA_VALUES:
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                met = score(y, base.graph_mvfcm(views, c, seed, alpha=lam))
                rows.append({"sweep": "hyperparam", "dataset": dataset_name,
                             "param_name": "lambda_gr", "param_value": lam,
                             "method": "GR-MVFCM", "seed": seed, **met})
        print(f"  lambda sweep done")

        # -- k-NN size: GR-MVFCM only --
        for knn in KNN_VALUES:
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                met = score(y, base.graph_mvfcm(views, c, seed, knn=knn))
                rows.append({"sweep": "hyperparam", "dataset": dataset_name,
                             "param_name": "knn", "param_value": knn,
                             "method": "GR-MVFCM", "seed": seed, **met})
        print(f"  k-NN sweep done")

        # -- kernel bandwidth multiplier: K-MVFCM only --
        for bw in KERNEL_BW_MULT_VALUES:
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                met = score(y, kernel_mvfcm_bw(views, c, seed, bw))
                rows.append({"sweep": "hyperparam", "dataset": dataset_name,
                             "param_name": "kernel_bw_mult", "param_value": bw,
                             "method": "K-MVFCM", "seed": seed, **met})
        print(f"  kernel bandwidth sweep done")

        # -- online batch size: O-MVFCM only --
        for batch in ONLINE_BATCH_VALUES:
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                met = score(y, base.online_mvfcm(views, c, seed, batch=batch))
                rows.append({"sweep": "hyperparam", "dataset": dataset_name,
                             "param_name": "online_batch", "param_value": batch,
                             "method": "O-MVFCM", "seed": seed, **met})
        print(f"  online batch-size sweep done")

        # -- online epochs: O-MVFCM only --
        for epochs in ONLINE_EPOCHS_VALUES:
            for seed in SWEEP_SEEDS:
                views = base.prepare(views_raw, seed)
                met = score(y, base.online_mvfcm(views, c, seed, epochs=epochs))
                rows.append({"sweep": "hyperparam", "dataset": dataset_name,
                             "param_name": "online_epochs", "param_value": epochs,
                             "method": "O-MVFCM", "seed": seed, **met})
        print(f"  online epochs sweep done")

    reset_defaults()
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.time()
    pca_rows = run_pca_sweep()
    if pca_rows:
        pd.DataFrame(pca_rows).to_csv("mvfcm_pca_sweep.csv", index=False)
        print(f"\nWrote mvfcm_pca_sweep.csv ({len(pca_rows)} rows)")
    else:
        print("\nNo PCA sweep rows collected. Check dataset availability.")

    hp_rows = run_hyperparam_sweep()
    if hp_rows:
        pd.DataFrame(hp_rows).to_csv("mvfcm_hyperparam_sweep.csv", index=False)
        print(f"Wrote mvfcm_hyperparam_sweep.csv ({len(hp_rows)} rows)")
    else:
        print("\nNo hyperparameter sweep rows collected. Check dataset availability.")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
