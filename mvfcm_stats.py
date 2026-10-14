"""Statistics for the MV-FCM paper: mean +/- std, Wilcoxon signed-rank, Holm-Bonferroni.

Input : mvfcm_multiseed_results.csv (columns dataset, method, type, seed, ACC, NMI, ARI, ...)
Procedure (same as the paper):
  * for every dataset x metric (5 x 3 = 15 cells) take the method with the best mean
    and the runner-up by mean, among all eight methods (five variants, MV-FCM, k-means,
    best-single-view);
  * two-sided paired Wilcoxon signed-rank test over the 20 seeds (scipy defaults);
  * Holm-Bonferroni adjustment over the 15 p-values; significant if adjusted p < 0.05;
  * a dagger marks a best mean whose lead is NOT significant.
Note: best-single-view picks the view with the highest ARI per seed using the labels,
so it is an oracle baseline.
"""
import sys
import pandas as pd
from scipy.stats import wilcoxon

CSV = sys.argv[1] if len(sys.argv) > 1 else "mvfcm_multiseed_results.csv"
r = pd.read_csv(CSV).replace({"R-MVFCM": "L1-MVFCM"})
METRICS = ["ACC", "NMI", "ARI"]

rows = []
for d in r.dataset.unique():
    for m in METRICS:
        pv = r[r.dataset == d].pivot(index="seed", columns="method", values=m)
        mu = pv.mean().sort_values(ascending=False)
        a, b = mu.index[0], mu.index[1]
        rows.append(dict(dataset=d, metric=m, best=a, runner_up=b,
                         mean_best=mu.iloc[0], mean_runner=mu.iloc[1],
                         p=wilcoxon(pv[a], pv[b]).pvalue))
t = pd.DataFrame(rows)

order = t.p.sort_values().index           # Holm step-down
n, run, adj = len(t), 0.0, {}
for k, i in enumerate(order):
    run = max(run, min(1.0, (n - k) * t.p[i]))
    adj[i] = run
t["p_holm"] = pd.Series(adj)
t["significant"] = t.p_holm < 0.05
t["dagger"] = ~t.significant

pd.set_option("display.width", 200)
print(f"{n} comparisons, Holm-Bonferroni over all {n}\n")
print(t.round(4).to_string(index=False))

print("\nmean +/- std (ddof=1) per dataset, method, metric")
g = r.groupby(["dataset", "method"])[METRICS].agg(["mean", "std"]).round(3)
print(g.to_string())
t.to_csv("mvfcm_stats_output.csv", index=False)