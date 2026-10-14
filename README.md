# Multi-View Fuzzy C-Means (MV-FCM): Comparative Study & Sensitivity Benchmark

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-MV--FCM-blue)](https://github.com/siyak06/MV-FCM)

This repository contains the official benchmark code, statistical testing pipeline, and experimental results for the paper:

> **"A Comparative Empirical Study and Sensitivity Analysis of Multi-View Fuzzy C-Means Clustering Algorithms"**  
> *Siya Kohli et al.*

---

## 📂 Repository Contents

```
MV-FCM/
├── mvfcm_multiseed_v2.py          # Main experiment: loaders, 8 methods, 20-seed runs, runtime & memory
├── mvfcm_sweeps.py                # Sensitivity sweeps: PCA dimensions and hyperparameters (feeds Table III)
├── mvfcm_stats.py                 # Statistical testing: Wilcoxon signed-rank + Holm-Bonferroni (feeds † marks)
├── check_dataset_facts.py         # Verification script: CUB counts, Reuters overlap, MNIST+USPS path
├── mvfcm_multiseed_results.csv    # 20-seed raw clustering results (feeds Table II & Fig. 1)
├── mvfcm_pca_sweep.csv            # PCA dimensionality sweep results (seeds 42..46)
├── mvfcm_hyperparam_sweep.csv     # Hyperparameter sweep results (m, lambda, knn, kernel bw, batch, epochs)
├── mvfcm_stats_output.csv         # Output of stats script: p-values, Holm adjustment, dagger flags
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md                      # Documentation and reproduction guide
```

---

## 📌 Core Scripts

### 1. Main Experiment: `mvfcm_multiseed_v2.py`
* **Purpose**: Implements all 8 clustering methods, dataset preprocessing, and 20 random seed runs (seeds 42 to 61).
* **Measures**: Accuracy (ACC), Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), runtime in seconds, and peak RAM consumption (MB).
* **Output**: Produces `mvfcm_multiseed_results.csv`, which directly feeds **Table II** and **Fig. 1** in the paper.
* **Usage**:
  ```bash
  python mvfcm_multiseed_v2.py
  ```

### 2. Sensitivity & Ablation Sweeps: `mvfcm_sweeps.py`
* **Purpose**: Reuses the core algorithm implementations from `mvfcm_multiseed_v2.py` to evaluate parameter sensitivity across representative datasets (CUB-200-2011 and Handwritten) over 5 random seeds (42 to 46):
  * **PCA Dimensionality**: $\text{PCA\_DIMS} \in \{10, 20, 30, 50, 100\}$ across all methods.
  * **Fuzziness Exponent $m$**: $m \in \{1.1, 1.3, 1.5, 1.7, 2.0\}$.
  * **Graph Regularizer $\lambda$**: $\lambda \in \{0.05, 0.1, 0.2, 0.5, 1.0\}$ (GR-MVFCM).
  * **$k$-NN Size**: $k \in \{5, 7, 10, 15, 20\}$ (GR-MVFCM).
  * **Kernel Bandwidth**: Multipliers $\{0.5, 1.0, 2.0\}$ (K-MVFCM).
  * **Online Mini-Batch Size & Epochs**: Batches $\{50, 100, 200, 400\}$, Epochs $\{2, 5, 10\}$ (O-MVFCM).
* **Output**: Produces `mvfcm_pca_sweep.csv` and `mvfcm_hyperparam_sweep.csv`, which feed **Table III**.
* **Usage**:
  ```bash
  python mvfcm_sweeps.py
  ```

### 3. Statistical Testing: `mvfcm_stats.py`
* **Purpose**: Performs statistical validation on `mvfcm_multiseed_results.csv`:
  * Evaluates each dataset $\times$ metric cell (15 cells total).
  * Identifies the best-performing method and the runner-up.
  * Runs a two-sided paired **Wilcoxon signed-rank test** across the 20 seeds.
  * Applies **Holm-Bonferroni correction** over all 15 comparisons.
  * Identifies cases where the lead is not statistically significant ($p_{\text{adj}} \ge 0.05$), marking them with a dagger (`†`).
  * Computes sample mean $\pm$ sample standard deviation ($\text{ddof}=1$).
* **Output**: Produces `mvfcm_stats_output.csv` and prints the formatted summary.
* **Usage**:
  ```bash
  python mvfcm_stats.py
  ```

### 4. Dataset Verification: `check_dataset_facts.py`
* **Purpose**: Verifies empirical properties reported in the paper:
  * Counts CUB-200-2011 sample images per class and verifies against cache.
  * Checks Reuters-21578 top-4 categories and cross-category document overlap.
  * Checks whether the real USPS corpus was downloaded or whether the HOG fallback was used.
* **Usage**:
  ```bash
  python check_dataset_facts.py
  ```

---

## 📊 Precomputed Data & Results Included

The repository includes the exact per-seed result files cited in the paper:
* **`mvfcm_multiseed_results.csv`**: Full 20-seed evaluation table across all 5 datasets and 8 algorithms.
* **`mvfcm_pca_sweep.csv`**: PCA sweep results across dimensions $\{10, 20, 30, 50, 100\}$.
* **`mvfcm_hyperparam_sweep.csv`**: Hyperparameter sweep results across all varied parameters.
* **`mvfcm_stats_output.csv`**: Statistical test table with $p$-values, adjusted $p$-values, and significance flags.

---

## 📥 Dataset Acquisition

To keep the repository lightweight, raw datasets and extracted feature caches (`cub_features_v2.npz`) are not hosted in Git. Datasets can be obtained as follows:

| Dataset | Views | Samples | Acquisition Details |
| :--- | :--- | :--- | :--- |
| **Scene-15** | 3 views | 4,485 | Place `Scene15_multiview.mat` in `data/` or set `PATH_SCENE15`. |
| **Handwritten** | 6 views | 2,000 | UCI Multiple Features: Place `HandWritten.mat` in `data/` or set `PATH_HANDWRITTEN`. |
| **MNIST + USPS** | 2 views | 1,000 | Automatically fetched via `scikit-learn`'s `fetch_openml`. |
| **Reuters-21578**| 2 views | 800 | Automatically downloaded via `nltk.download("reuters")`. |
| **CUB-200-2011** | 2 views | 586 | Download images from Caltech-UCSD Birds-200-2011 and place in `CUB_200_2011/images/` or set `PATH_CUB`. Feature extraction runs automatically via ResNet-18. |

---

## ⚙️ Environment Setup

```bash
git clone https://github.com/siyak06/MV-FCM.git
cd MV-FCM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📜 Citation

```bibtex
@article{kohli2026multiview,
  title={A Comparative Empirical Study and Sensitivity Analysis of Multi-View Fuzzy C-Means Clustering Algorithms},
  author={Kohli, Siya and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
