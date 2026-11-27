import sys, numpy as np, pandas as pd
from multiprocessing import Pool
from url_ml.thesis_features import extract, FEATURES
d = pd.read_csv("Dataset_Hybrid.csv", usecols=["url", "registered_domain"] + FEATURES + ["Label"])
def work(a): return [extract(u, r) for u, r in a]
if __name__ == "__main__":
    pairs = list(zip(d.url, d.registered_domain)); ch = [pairs[i:i + 5000] for i in range(0, len(pairs), 5000)]
    with Pool(6) as p: out = [r for part in p.map(work, ch) for r in part]
    F = pd.DataFrame(out)
    rows = [{"Feature": f, "Exact match (|diff|<1e-4)": float(np.mean(np.abs(F[f].values - d[f].values) < 1e-4)),
             "Mean |diff|": float(np.mean(np.abs(F[f].values - d[f].values)))} for f in FEATURES]
    r = pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv("artifacts/extractor_validation.csv", index=False)
    for f in ["HasBrandInSubdomainOrPath", "IsShortener", "HasEmbeddedURL", "IsPunycode", "HasAtSymbol"]:
        print(f, pd.crosstab(d[f], F[f]).values.tolist())
