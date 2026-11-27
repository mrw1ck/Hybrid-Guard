"""E6: robustness of the frozen calibrated models to semantics-preserving URL perturbations + QR round trip.
Stratified sample of the locked test partition; each perturbation is re-featurised with url_ml/thesis_features.py and re-scored.
Reported: ROC/PR-AUC, TPR@1%FPR at the frozen validation-derived operating point's FPR, mean |dp|, flip rate at tau=0.5 and at tau_high."""
import json, re, sys, random, time
from urllib.parse import urlsplit, urlunsplit
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from url_ml.thesis_features import extract, FEATURES
N = int(sys.argv[1]); NP = int(sys.argv[2])
rng = random.Random(42)
def sp(u):
    s = urlsplit(u); return s
def www(u):
    s = sp(u); h = s.netloc
    return u if h.startswith("www.") or "@" in h else urlunsplit(s._replace(netloc="www." + h))
def nowww(u):
    s = sp(u); return urlunsplit(s._replace(netloc=s.netloc[4:])) if s.netloc.startswith("www.") else u
def https(u): return re.sub(r"^http://", "https://", u)
def http(u): return re.sub(r"^https://", "http://", u)
def slash(u):
    s = sp(u); return u if s.path or s.query or s.fragment else u + "/"
def upper_host(u):
    s = sp(u); return urlunsplit(s._replace(netloc=s.netloc.upper()))
def tracking(u): return u + ("&" if sp(u).query else "?") + "utm_source=newsletter&utm_medium=email"
def fragment(u): return u + "#top" if "#" not in u else u
def pct_enc(u):
    s = sp(u); p = "".join(f"%{ord(c):02X}" if c.isalpha() and rng.random() < .15 else c for c in s.path)
    return urlunsplit(s._replace(path=p))
def port443(u):
    s = sp(u); return u if ":" in s.netloc.split("@")[-1] else urlunsplit(s._replace(netloc=s.netloc + (":443" if s.scheme == "https" else ":80")))
def benign_pad(u):
    s = sp(u); return urlunsplit(s._replace(path=(s.path.rstrip("/") + "/index.html") if not s.path.rsplit("/", 1)[-1].count(".") else s.path))
def sub_insert(u):
    s = sp(u); return urlunsplit(s._replace(netloc="m." + s.netloc)) if "@" not in s.netloc else u
def long_pad(u):                                # adversarial: dilutes ratio features with benign-looking path text
    s = sp(u); return urlunsplit(s._replace(path=s.path.rstrip("/") + "/" + "/".join(["products", "catalog", "2024", "news", "about-us"])))
PERT = {"identity (sanity check)": lambda u: u, "add www.": www, "strip www.": nowww, "http->https": https, "https->http": http,
        "add trailing slash": slash, "uppercase host": upper_host, "add tracking params": tracking, "add #fragment": fragment,
        "percent-encode 15% path letters": pct_enc, "explicit default port": port443, "append /index.html": benign_pad,
        "prefix subdomain m.": sub_insert, "benign path padding (adversarial)": long_pad}

df = pd.read_csv("Dataset_Hybrid.csv", usecols=["url", "Label", "partition"]); te = df[df.partition == "test"]
s = pd.concat([g.sample(min(len(g), N // 2), random_state=42) for _, g in te.groupby("Label")]).reset_index(drop=True)
y = s.Label.values; urls = s.url.tolist()
arts = {k: joblib.load(f"artifacts/{v}_thesis/url_phishing_model.joblib") for k, v in {"LightGBM": "lgb", "XGBoost": "xgb", "Random Forest": "rf"}.items()}
for a in arts.values():
    try: a["model"].set_params(n_jobs=NP)
    except Exception: pass
def score(us):
    X = pd.DataFrame([extract(u) for u in us])[FEATURES].astype("float32").values
    return {k: a["calibrator"].predict(a["model"].predict_proba(X)[:, 1]) for k, a in arts.items()}
thr = {k: json.load(open(f"artifacts/{v}_thesis/report/summary.json")).get("thresholds", {}) for k, v in {"LightGBM": "lgb", "XGBoost": "xgb", "Random Forest": "rf"}.items()}
print("thresholds", thr, flush=True)
def tpr_at(yv, p, fpr=.01):
    f, t, _ = roc_curve(yv, p); return float(np.interp(fpr, f, t))
base = score(urls); rows = []
for pn, fn in PERT.items():
    pu = [fn(u) for u in urls]; changed = float(np.mean([a != b for a, b in zip(urls, pu)]))
    sc = score(pu)
    for m in arts:
        p, b = sc[m], base[m]
        rows.append({"Perturbation": pn, "Model": m, "% URLs changed": 100 * changed, "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
                     "TPR@1%FPR": tpr_at(y, p), "dROC-AUC": roc_auc_score(y, p) - roc_auc_score(y, b), "dPR-AUC": average_precision_score(y, p) - average_precision_score(y, b),
                     "dTPR@1%FPR": tpr_at(y, p) - tpr_at(y, b), "mean |dp|": float(np.mean(np.abs(p - b))),
                     "flip rate @0.5 (%)": 100 * float(np.mean((p >= .5) != (b >= .5)))})
    print(pn, "done", flush=True)
import os; os.makedirs("artifacts/e6_robustness", exist_ok=True)
pd.DataFrame(rows).to_csv("artifacts/e6_robustness/E6_perturbation.csv", index=False)

# ---------------- QR round trip
import qrcode, cv2
from multiprocessing import Pool
DEG = {"clean": dict(), "scale 0.5x": dict(scale=.5), "gaussian blur k=5": dict(blur=5), "gaussian noise s=25": dict(noise=25),
       "rotate 15 deg": dict(rot=15), "jpeg q=30": dict(jpeg=30), "low contrast": dict(contrast=.35)}
def qr_one(a):
    u, = a; res = {}
    try:
        q = qrcode.QRCode(border=4, box_size=6, error_correction=qrcode.constants.ERROR_CORRECT_M); q.add_data(u); q.make(fit=True)
        im = np.array(q.make_image().convert("L"), dtype=np.uint8)
    except Exception: return {k: None for k in DEG}
    r = np.random.RandomState(abs(hash(u)) % (2**32)); det = cv2.QRCodeDetector()
    for name, d in DEG.items():
        x = im.copy()
        if "scale" in d: x = cv2.resize(x, None, fx=d["scale"], fy=d["scale"], interpolation=cv2.INTER_AREA)
        if "blur" in d: x = cv2.GaussianBlur(x, (d["blur"], d["blur"]), 0)
        if "noise" in d: x = np.clip(x + r.normal(0, d["noise"], x.shape), 0, 255).astype(np.uint8)
        if "rot" in d:
            h, w = x.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), d["rot"], 1); x = cv2.warpAffine(x, M, (w, h), borderValue=255)
        if "jpeg" in d: x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, d["jpeg"]])[1], 0)
        if "contrast" in d: x = (128 + (x.astype(float) - 128) * d["contrast"]).astype(np.uint8)
        try: out, _, _ = det.detectAndDecode(x)
        except Exception: out = ""
        res[name] = out
    return res
if __name__ == "__main__":
    nq = min(len(urls), 4000); idx = np.random.RandomState(1).choice(len(urls), nq, replace=False)
    qu = [urls[i] for i in idx]; T = time.time()
    with Pool(NP) as p: dec = p.map(qr_one, [(u,) for u in qu], chunksize=50)
    print("qr decode", (time.time() - T) / 60, "min", flush=True)
    yq = y[idx]; bq = {k: v[idx] for k, v in base.items()}; rows = []
    for name in DEG:
        ok = np.array([d[name] is not None and d[name] == u for d, u in zip(dec, qu)]); dec_any = np.array([bool(d[name]) for d in dec])
        # decoded-but-different strings are scored too (text corruption)
        du = [d[name] if d[name] else u for d, u in zip(dec, qu)]; sc = score(du)
        for m in arts:
            p = sc[m]; keep = dec_any
            rows.append({"Degradation": name, "Model": m, "n": nq, "decode success (%)": 100 * float(ok.mean()), "decoded-any (%)": 100 * float(dec_any.mean()),
                         "mean |dp| (decoded)": float(np.mean(np.abs(p[keep] - bq[m][keep]))) if keep.any() else np.nan,
                         "flip rate @0.5 (decoded, %)": 100 * float(np.mean((p[keep] >= .5) != (bq[m][keep] >= .5))) if keep.any() else np.nan,
                         "effective TPR@1%FPR (undecoded counted as missed)": float(np.nan)})
    pd.DataFrame(rows).to_csv("artifacts/e6_robustness/E6_qr.csv", index=False)
    print("DONE")
