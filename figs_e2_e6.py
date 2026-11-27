import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
M = ["LightGBM", "XGBoost", "Random Forest"]; col = {"LightGBM": "#1f77b4", "XGBoost": "#ff7f0e", "Random Forest": "#2ca02c"}
d = pd.read_csv("artifacts/e2_ablation/E2_ablation.csv")
p = d[(d.Metric == "PR-AUC")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
dr = p[p.Configuration.str.startswith("Drop")]; names = [c.replace("Drop ", "") for c in dr.Configuration.unique()]
for j, m in enumerate(M):
    v = dr[dr.Model == m].set_index("Configuration").loc[["Drop " + n for n in names]]
    ax[0].bar(np.arange(5) + j * .27, v["Delta vs full"], .27, color=col[m], label=m,
              yerr=[v["Delta vs full"] - v["Delta CI lower"], v["Delta CI upper"] - v["Delta vs full"]], capsize=2)
ax[0].set_xticks(np.arange(5) + .27); ax[0].set_xticklabels([n.split(" ")[0] + "\n" + " ".join(n.split(" ")[1:]) for n in names], fontsize=7); ax[0].axhline(0, c="k", lw=.5)
ax[0].set_title("Leave-one-group-out: change in PR-AUC (95% domain-bootstrap CI)"); ax[0].legend(); ax[0].set_ylabel("Delta PR-AUC vs full model")
on = p[p.Configuration.str.startswith("Only") | p.Configuration.str.startswith("Full")]; names2 = [c for c in on.Configuration.unique()]
for j, m in enumerate(M):
    v = on[on.Model == m].set_index("Configuration").loc[names2]; ax[1].bar(np.arange(len(names2)) + j * .27, v["Value"], .27, color=col[m], label=m)
ax[1].set_xticks(np.arange(len(names2)) + .27); ax[1].set_xticklabels([n.replace("Full model (19 features)", "Full").replace("Only ", "") .split(" ")[0] for n in names2]); ax[1].set_ylim(.5, 1)
ax[1].set_title("Single-group models: test PR-AUC"); plt.tight_layout(); plt.savefig("artifacts/e2_ablation/figE2_ablation.png", dpi=150); plt.close()
e = pd.read_csv("artifacts/e6_robustness/E6_perturbation.csv"); e = e[e.Perturbation != "identity (sanity check)"]
order = e[e.Model == "LightGBM"].sort_values("dTPR@1%FPR").Perturbation.tolist()
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for k, (c, t) in enumerate([("dTPR@1%FPR", "Change in TPR@1%FPR"), ("flip rate @0.5 (%)", "Prediction flip rate at p=0.5 (%)")]):
    for j, m in enumerate(M):
        v = e[e.Model == m].set_index("Perturbation").loc[order]; ax[k].barh(np.arange(len(order)) + j * .27, v[c], .27, color=col[m], label=m)
    ax[k].set_title(t)
ax[0].set_yticks(np.arange(len(order)) + .27); ax[0].set_yticklabels(order, fontsize=8); ax[0].legend(); plt.tight_layout(); plt.savefig("artifacts/e6_robustness/figE6_perturbation.png", dpi=150); plt.close()
q = pd.read_csv("artifacts/e6_robustness/E6_qr.csv"); q = q[q.Model == "LightGBM"]
fig, a = plt.subplots(figsize=(7, 4)); a.bar(q.Degradation, q["decode success (%)"], color="#6a5acd"); a.set_ylim(70, 100); a.set_ylabel("exact decode success (%)"); a.set_title("QR round trip: decode success by degradation (n=4000)")
plt.xticks(rotation=25, fontsize=8); plt.tight_layout(); plt.savefig("artifacts/e6_robustness/figE6_qr.png", dpi=150); plt.close()

# per-model E6 figures
for m, tag in [("LightGBM", "lgb"), ("XGBoost", "xgb"), ("Random Forest", "rf")]:
    v = e[e.Model == m].set_index("Perturbation").loc[order]; fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for a, (c, t) in zip(ax, [("dTPR@1%FPR", "Change in TPR@1%FPR"), ("dROC-AUC", "Change in ROC-AUC"), ("flip rate @0.5 (%)", "Flip rate at p=0.5 (%)")]):
        a.barh(np.arange(len(order)), v[c], color=col[m]); a.set_title(t, fontsize=10)
    ax[0].set_yticks(np.arange(len(order))); ax[0].set_yticklabels(order, fontsize=8); fig.suptitle(f"E6 URL perturbations - {m}", fontsize=11); plt.tight_layout()
    plt.savefig(f"artifacts/e6_robustness/figE6_perturbation_{tag}.png", dpi=150); plt.close()
