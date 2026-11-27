import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
R = json.load(open("artifacts/e4_hybrid/e4_raw.json")); rows = R["rows"]; rng = np.random.default_rng(42)
M = ["LightGBM", "XGBoost", "Random Forest"]
def tpr(y, p, f=.01): a, b, _ = roc_curve(y, p); return float(np.interp(f, a, b))
def cm(y, pred): tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum(); return tp, fp, fn, tn
out = []; chk = []
for sset in ["stale", "fresh"]:
    sub = [r for r in rows if r["set"] == sset and all("error" not in r["results"][m] for m in M)]
    y = np.array([r["label"] for r in sub]); n = len(y)
    L = [r["results"]["LightGBM"]["checks"] for r in sub]
    for lab in (1, 0):
        idx = [i for i in range(n) if y[i] == lab]
        chk.append({"Set": sset, "Label": "malicious" if lab else "benign", "n": len(idx),
                    "DNS resolved (%)": 100 * np.mean([L[i]["dns"].get("resolved", False) for i in idx]),
                    "SSL available (%)": 100 * np.mean([bool(L[i]["ssl"].get("available")) for i in idx]),
                    "WHOIS/RDAP age found (%)": 100 * np.mean([L[i]["domain_age"].get("days_since_registration") is not None for i in idx]),
                    "HTTP status obtained (%)": 100 * np.mean([L[i]["http"].get("status_code") is not None for i in idx])})
    for m in M:
        b = np.array([r["results"][m]["base_malicious_probability"] for r in sub]); f = np.array([r["results"][m]["final_malicious_probability"] for r in sub])
        fp_ = np.array([r["results"][m]["final_prediction"] for r in sub])
        pred_base = (b >= .5).astype(int); pred_h = (fp_ == "malicious").astype(int); pred_hs = (fp_ != "benign").astype(int)
        bs = []
        for _ in range(500):
            ii = rng.integers(0, n, n)
            if len(set(y[ii])) < 2: continue
            bs.append([average_precision_score(y[ii], f[ii]) - average_precision_score(y[ii], b[ii]), roc_auc_score(y[ii], f[ii]) - roc_auc_score(y[ii], b[ii]), tpr(y[ii], f[ii]) - tpr(y[ii], b[ii])])
        bs = np.array(bs)
        for name, p, pr in [("ML only", b, pred_base), ("Hybrid (ML + policy + external checks)", f, pred_h), ("Hybrid, suspicious counted as flagged", f, pred_hs)]:
            tp, fp, fn, tn = cm(y, pr); pre = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            out.append({"Set": sset, "Model": m, "System": name, "n": n, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p), "TPR@1%FPR": tpr(y, p),
                        "Precision": pre, "Recall": rec, "F1": 2 * pre * rec / max(pre + rec, 1e-9), "FPR": fp / max(fp + tn, 1), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                        "dPR-AUC CI": f"[{np.quantile(bs[:,0],.025):+.3f}, {np.quantile(bs[:,0],.975):+.3f}]" if "Hybrid (ML" in name else "",
                        "dTPR@1%FPR CI": f"[{np.quantile(bs[:,2],.025):+.3f}, {np.quantile(bs[:,2],.975):+.3f}]" if "Hybrid (ML" in name else ""})
O = pd.DataFrame(out); C = pd.DataFrame(chk); O.to_csv("artifacts/e4_hybrid/E4_hybrid.csv", index=False); C.to_csv("artifacts/e4_hybrid/E4_checks_by_label.csv", index=False)
print(C.round(1).to_string()); print(O[O.Model == "LightGBM"].round(3).drop(columns=["TP", "FP", "FN", "TN"]).to_string())
# figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, s in zip(ax, ["stale", "fresh"]):
    d = O[(O.Set == s) & (O.System.isin(["ML only", "Hybrid (ML + policy + external checks)"]))]
    w = .35
    for j, sysn in enumerate(["ML only", "Hybrid (ML + policy + external checks)"]):
        v = d[d.System == sysn].set_index("Model").loc[M]; a.bar(np.arange(3) + j * w, v["PR-AUC"], w, label=sysn.split(" (")[0])
    a.set_xticks(np.arange(3) + w / 2); a.set_xticklabels(M); a.set_ylim(.5, 1); a.set_title(f"{s} set: PR-AUC"); a.legend()
plt.tight_layout(); plt.savefig("artifacts/e4_hybrid/figE4_prauc.png", dpi=150); plt.close()
fig, a = plt.subplots(figsize=(7, 4)); c = C.copy(); c["g"] = c.Set + "/" + c.Label
cols = ["DNS resolved (%)", "SSL available (%)", "WHOIS/RDAP age found (%)", "HTTP status obtained (%)"]
for j, col in enumerate(cols): a.bar(np.arange(len(c)) + j * .2, c[col], .2, label=col)
a.set_xticks(np.arange(len(c)) + .3); a.set_xticklabels(c.g, rotation=15); a.legend(fontsize=7); a.set_title("External-check availability by label (confound)")
plt.tight_layout(); plt.savefig("artifacts/e4_hybrid/figE4_checks.png", dpi=150); plt.close()
