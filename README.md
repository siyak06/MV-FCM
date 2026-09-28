# Multi-View Fuzzy C-Means (MV-FCM): Comparative Study & Sensitivity Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Clustering: Fuzzy C-Means](https://img.shields.io/badge/Clustering-MV--FCM-orange.svg)]()

This repository contains the official implementation, evaluation suite, and sensitivity/ablation benchmarks for the research paper:

> **"A Comparative Empirical Study and Sensitivity Analysis of Multi-View Fuzzy C-Means Clustering Algorithms"**  
> *Siya Kohli et al.*

It includes multi-seed comparative evaluations across 5 real-world multimodal datasets, statistical hypothesis testing (Wilcoxon signed-rank tests with Holm-Bonferroni correction), and parameter sensitivity sweeps addressing PCA dimensionality, fuzziness exponent $m$, graph regularizer $\lambda$, $k$-NN graph neighborhood size, kernel bandwidth, and online streaming mini-batch parameters.

---

## 📌 Method Overview

This repository provides reproducible implementations of core Multi-View Fuzzy C-Means variants alongside standard clustering baselines:

| Algorithm | Method Name | Formulation & Distinctive Properties |
| :--- | :--- | :--- |
| **WMV-FCM** | Weighted MV-FCM | Dynamically computes per-view importance weights $w_v$ via an inverse power formulation $w_v \propto (1 / J_v)^{1 / (p - 1)}$. |
| **K-MVFCM** | Kernel MV-FCM | Maps non-linearly separable data into Hilbert space using an RBF kernel with an adaptive median pairwise bandwidth heuristic. |
| **GR-MVFCM** | Graph-Regularized MV-FCM | Enforces manifold smoothness using normalized $k$-NN Laplacian graph propagation: $U \leftarrow (1-\lambda)U + \lambda S U$. |
| **R-MVFCM** | Robust MV-FCM | Employs an $L_1$ distance loss with coordinate-wise weighted median centroid estimation to suppress heavy-tailed noise and outliers. |
| **O-MVFCM** | Online MV-FCM | Incremental streaming mini-batch clustering with running-average cluster center updates $w_j / (\text{seen}_j + w_j)$. |
| **MV-FCM** | Baseline MV-FCM | Standard multi-view fuzzy objective averaging squared Euclidean distance across all views. |
| **$k$-Means** | Baseline Concatenated | Hard partitioning baseline fit on concatenated standardized multiview vectors. |
| **Best Single-View** | Baseline Single-View | Single-view FCM executed on each individual modality; records performance of the optimal view per seed. |

---

## 📂 Repository Structure

```
.
├── mvfcm_multiseed_v2.py      # Core algorithms, dataset loaders, 20-seed benchmark harness
├── sensitivity.py             # Sensitivity sweeps: PCA dimensions, m, lambda, knn, kernel bw, batch size
├── mvfcm_stats.py             # Statistical tests (Mean ± Std, Wilcoxon, Holm-Bonferroni, Table III)
├── requirements.txt           # Pinned library dependencies
├── LICENSE                    # MIT License
├── README.md                  # Project documentation & reproduction instructions
└── data/                      # Dataset directory (see data/README.md)
    └── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/siyak06/Agrimesh.git
   cd Agrimesh
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📊 Supported Datasets

The benchmark evaluates clustering algorithms across 5 multimodal benchmarks:

| Dataset | Modalities / Views | Samples | Classes | Download / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Scene-15** | 3 views (GIST, PHOG, LBP) | 4,485 | 15 | Place `Scene15_multiview.mat` in `data/` |
| **Handwritten** | 6 views (Fourier, Profile, Karhunen, Pixels, Moments, Morphological) | 2,000 | 10 | Place `HandWritten.mat` in `data/` |
| **MNIST + USPS** | 2 views (MNIST pixels + USPS digits / HOG) | 1,000 | 10 | Auto-fetched via `fetch_openml` |
| **Reuters-21578**| 2 views (Word TF-IDF + Char n-gram TF-IDF) | 800 | 4 | Auto-fetched via `nltk.download("reuters")` |
| **CUB-200-2011** | 2 views (ResNet-18 deep embeddings + 3D RGB color histogram) | 586 (10 cls) | 10 | Place `CUB_200_2011/images` in `data/` |

> *Tip: You can customize dataset paths via environment variables `PATH_SCENE15`, `PATH_HANDWRITTEN`, and `PATH_CUB` without altering code.*

---

## 🚀 Running Experiments

### 1. Multi-Seed Benchmark (Table III Reproduction)
Runs all 8 algorithms across 20 random seeds (seeds 42 to 61) to track clustering Accuracy (ACC), Normalized Mutual Information (NMI), Adjusted Rand Index (ARI), runtime in seconds, and peak memory consumption (MB):

```bash
python mvfcm_multiseed_v2.py
```
*Output*: Generates `mvfcm_multiseed_results.csv`.

---

### 2. Statistical Significance & Revised Table III
Computes **Mean ± Standard Deviation**, runs **paired Wilcoxon signed-rank tests** against the baseline (`MV-FCM`), applies **Holm-Bonferroni step-down correction** to control the family-wise error rate, and formats the summary:

```bash
python mvfcm_stats.py --input mvfcm_multiseed_results.csv --output mvfcm_table3_summary.csv --latex
```
*Output*:
- Console display of revised **Table III** in Markdown.
- `mvfcm_table3_summary.csv` containing raw means, standard deviations, and adjusted $p$-values.
- LaTeX code for direct insertion into papers.

---

### 3. Reviewer Sensitivity & Ablation Sweeps
Runs the sensitivity sweeps addressing reviewer inquiries:
1. **PCA Dimensionality Sweep**: Tests $\text{PCA\_DIMS} \in \{10, 20, 30, 50, 100\}$ across all 7 algorithms (5 seeds each).
2. **Hyperparameter Sensitivity Sweeps**:
   - Fuzziness exponent $m \in \{1.1, 1.3, 1.5, 1.7, 2.0\}$ (all methods)
   - Graph regularizer strength $\lambda \in \{0.05, 0.1, 0.2, 0.5, 1.0\}$ (GR-MVFCM)
   - $k$-NN graph neighborhood size $k \in \{5, 7, 10, 15, 20\}$ (GR-MVFCM)
   - RBF kernel bandwidth multiplier $\in \{0.5, 1.0, 2.0\}$ (K-MVFCM)
   - Online streaming batch size $\in \{50, 100, 200, 400\}$ (O-MVFCM)
   - Online epochs $\in \{2, 5, 10\}$ (O-MVFCM)

```bash
python sensitivity.py
```
*Output*: `mvfcm_pca_sweep.csv` and `mvfcm_hyperparam_sweep.csv`.

---

## 📈 Evaluation Metrics

All methods are evaluated using standard external cluster validation metrics:
- **Clustering Accuracy (ACC)**: Computed using the Kuhn-Munkres (Hungarian) optimal bipartite matching between cluster assignments and ground truth:
  $$\text{ACC} = \max_{\pi} \frac{1}{N} \sum_{i=1}^N \mathbf{1}\{y_i = \pi(\hat{y}_i)\}$$
- **Normalized Mutual Information (NMI)**: Information-theoretic overlap metric scaled between $[0, 1]$.
- **Adjusted Rand Index (ARI)**: Pair-counting metric adjusted for chance agreement $[-1, 1]$.

---

## 📜 Citation

If you use this benchmark suite or algorithm implementations in your research, please cite:

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
