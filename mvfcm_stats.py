"""
Statistical Analysis and Table III Generation for MV-FCM Comparative Study.

Computes:
  1. Mean ± Standard Deviation across random seeds for ACC, NMI, ARI, runtime (s), and peak memory (MB).
  2. Pairwise Wilcoxon signed-rank tests comparing each multi-view variant
     against standard baselines (MV-FCM and best-single-view) paired by seed.
  3. Holm-Bonferroni family-wise error rate (FWER) correction on p-values.
  4. Formatted Table III output in both Markdown and CSV (with LaTeX export option).

Usage:
  python mvfcm_stats.py
  python mvfcm_stats.py --input mvfcm_multiseed_results.csv --output mvfcm_table3_summary.csv --latex
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm_bonferroni(p_values):
    """
    Applies the Holm-Bonferroni step-down procedure to control the family-wise error rate.
    Returns adjusted p-values matching the original input order.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    
    adj = np.empty(n, dtype=float)
    running_max = 0.0
    for i in range(n):
        val = (n - i) * sorted_p[i]
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)
        
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def format_stat(mean_val, std_val):
    """Formats value as 'mean ± std'."""
    return f"{mean_val:.3f} ± {std_val:.3f}"


def run_statistical_analysis(df, baseline_name="MV-FCM", alpha=0.05):
    """
    Computes mean ± std and conducts Wilcoxon signed-rank tests paired by seed against baseline_name.
    """
    datasets = df["dataset"].unique()
    methods = df["method"].unique()
    metrics = ["ACC", "NMI", "ARI"]
    
    summary_rows = []
    
    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        base_df = ds_df[ds_df["method"] == baseline_name]
        
        has_baseline = len(base_df) > 0
        
        # Collect p-values for Holm-Bonferroni correction within this dataset
        p_val_records = []
        
        for m in methods:
            m_df = ds_df[ds_df["method"] == m]
            if len(m_df) == 0:
                continue
                
            row = {
                "dataset": ds,
                "method": m,
                "type": m_df["type"].iloc[0] if "type" in m_df.columns else "unknown",
                "seeds_count": len(m_df),
            }
            
            # Means and standard deviations
            for met in ["ACC", "NMI", "ARI", "runtime_sec", "peak_mem_mb"]:
                if met in m_df.columns:
                    val_mean = m_df[met].mean()
                    val_std = m_df[met].std()
                    row[f"{met}_mean"] = val_mean
                    row[f"{met}_std"] = val_std
                    row[f"{met}_str"] = format_stat(val_mean, val_std)
            
            # Wilcoxon paired test vs baseline
            if has_baseline and m != baseline_name:
                # Merge on seed to pair observations
                paired = pd.merge(
                    m_df[["seed", "ACC", "NMI", "ARI"]],
                    base_df[["seed", "ACC", "NMI", "ARI"]],
                    on="seed",
                    suffixes=("_method", "_base"),
                )
                if len(paired) >= 5:
                    for met in metrics:
                        diff = paired[f"{met}_method"] - paired[f"{met}_base"]
                        if np.all(diff == 0):
                            p = 1.0
                        else:
                            try:
                                res = wilcoxon(paired[f"{met}_method"], paired[f"{met}_base"], zero_method="wilcox")
                                p = res.pvalue
                            except Exception:
                                p = 1.0
                        p_val_records.append({"method": m, "metric": met, "p_raw": p})
            
            summary_rows.append(row)
            
        # Apply Holm-Bonferroni correction per metric across methods
        if p_val_records:
            pval_df = pd.DataFrame(p_val_records)
            for met in metrics:
                sub = pval_df[pval_df["metric"] == met]
                if len(sub) > 0:
                    adj_p = holm_bonferroni(sub["p_raw"].values)
                    pval_df.loc[sub.index, "p_adj"] = adj_p
                    
            # Map adjusted p-values back into summary rows
            for r in summary_rows:
                if r["dataset"] == ds:
                    for met in metrics:
                        match = pval_df[(pval_df["method"] == r["method"]) & (pval_df["metric"] == met)]
                        if len(match) > 0:
                            p_adj_val = match["p_adj"].iloc[0]
                            r[f"{met}_p_adj"] = p_adj_val
                            # Significance marker
                            marker = ""
                            if p_adj_val < 0.001:
                                marker = "***"
                            elif p_adj_val < 0.01:
                                marker = "**"
                            elif p_adj_val < 0.05:
                                marker = "*"
                            r[f"{met}_sig"] = marker
                            if marker:
                                r[f"{met}_str"] += f" ({marker})"
                        else:
                            r[f"{met}_p_adj"] = np.nan
                            r[f"{met}_sig"] = ""

    return pd.DataFrame(summary_rows)


def print_table3_markdown(summary_df):
    """Prints Table III formatted as GitHub Markdown."""
    print("\n" + "=" * 90)
    print("REVISED TABLE III: Multi-Seed Clustering Performance (Mean ± Std)")
    print("Significance vs MV-FCM baseline: * p < 0.05, ** p < 0.01, *** p < 0.001 (Holm-Bonferroni)")
    print("=" * 90)
    
    cols = ["dataset", "method", "ACC_str", "NMI_str", "ARI_str", "runtime_sec_str", "peak_mem_mb_str"]
    rename_cols = {
        "dataset": "Dataset",
        "method": "Method",
        "ACC_str": "Accuracy (ACC)",
        "NMI_str": "NMI",
        "ARI_str": "ARI",
        "runtime_sec_str": "Runtime (s)",
        "peak_mem_mb_str": "Peak RAM (MB)",
    }
    
    display_df = summary_df[[c for c in cols if c in summary_df.columns]].rename(columns=rename_cols)
    print(display_df.to_markdown(index=False))


def generate_latex_table(summary_df):
    """Generates LaTeX table code for direct publication inclusion."""
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Empirical Comparison of Multi-View Fuzzy C-Means Variants (Mean $\pm$ Std).}")
    latex.append(r"\label{tab:multiview_fcm_results}")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Dataset} & \textbf{Method} & \textbf{ACC} & \textbf{NMI} & \textbf{ARI} & \textbf{Time (s)} & \textbf{Peak RAM (MB)} \\")
    latex.append(r"\hline")

    current_ds = None
    for _, row in summary_df.iterrows():
        ds_str = row["dataset"] if row["dataset"] != current_ds else ""
        current_ds = row["dataset"]
        acc = row.get("ACC_str", "-").replace("±", r"$\pm$")
        nmi = row.get("NMI_str", "-").replace("±", r"$\pm$")
        ari = row.get("ARI_str", "-").replace("±", r"$\pm$")
        sec = row.get("runtime_sec_str", "-").replace("±", r"$\pm$")
        mem = row.get("peak_mem_mb_str", "-").replace("±", r"$\pm$")
        latex.append(f"{ds_str} & {row['method']} & {acc} & {nmi} & {ari} & {sec} & {mem} \\\\")

    latex.append(r"\hline")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="MV-FCM Statistical Significance & Table III Generator")
    parser.add_argument("--input", default="mvfcm_multiseed_results.csv", help="Input CSV path from mvfcm_multiseed_v2.py")
    parser.add_argument("--output", default="mvfcm_table3_summary.csv", help="Output summary CSV path")
    parser.add_argument("--baseline", default="MV-FCM", help="Baseline method name for paired Wilcoxon testing")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table code")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        print("Run 'python mvfcm_multiseed_v2.py' first to generate multi-seed experimental results.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    summary_df = run_statistical_analysis(df, baseline_name=args.baseline)

    print_table3_markdown(summary_df)

    summary_df.to_csv(args.output, index=False)
    print(f"\nSaved statistical summary to: {args.output}")

    if args.latex:
        print("\n=== LaTeX Code ===")
        print(generate_latex_table(summary_df))


if __name__ == "__main__":
    main()
