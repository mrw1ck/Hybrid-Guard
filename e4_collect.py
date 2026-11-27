"""E4 hybrid: run the deployed Hybrid-Guard analyzer (UrlAnalyzerService: DNS + SSL + RDAP domain age + HTTP content + risk policy)
with the thesis models supplying the base probability. Every sampled URL gets the full analyzer path; ML-only / cascade variants are
derived offline in e4_analyze.py.
Samples:  A) 'stale' stratified sample of the locked test partition (historical URLs; many are dead -> DNS/HTTP signals are confounded by age)
          B) 'fresh' live set: OpenPhish community feed (label 1) + Tranco top-100k homepages (label 0), fetched at run time.
Safety: page bodies are streamed and truncated at 64 KB, never executed or saved."""
import io, json, sys, time, zipfile, types, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, joblib, httpx
from url_ml.feature_engineering import extract_url_features, normalize_url
from url_ml.thesis_features import extract, FEATURES
import webapp.app.services.external_checks as ec
from webapp.app.services.analyzer_service import UrlAnalyzerService

N_STALE_MAL, N_STALE_BEN, N_LIVE = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); WORKERS = int(sys.argv[4])
SEED = 42; MAXB = 65536
class Capped(httpx.Client):                       # stream + truncate instead of downloading whole bodies
    def get(self, url, **kw):
        with self.stream("GET", url) as r:
            buf = b""
            for ch in r.iter_bytes():
                buf += ch
                if len(buf) > MAXB: break
            return types.SimpleNamespace(status_code=r.status_code, headers=r.headers, history=r.history, text=buf.decode("utf-8", "replace"))
ec.httpx = types.SimpleNamespace(Client=Capped, Timeout=httpx.Timeout, RequestError=httpx.RequestError, HTTPError=httpx.HTTPError)

class MemCache:
    def __init__(s): s.d = {}; s.l = threading.Lock()
    def get(s, k, ttl):
        with s.l: return dict(s.d[k]) if k in s.d else None
    def set(s, k, v):
        with s.l: s.d[k] = dict(v)
    def save(s): pass

models = {"LightGBM": "artifacts/lgb_thesis", "XGBoost": "artifacts/xgb_thesis", "Random Forest": "artifacts/rf_thesis"}
arts = {k: joblib.load(f"{v}/url_phishing_model.joblib") for k, v in models.items()}
for a in arts.values():
    m = a["model"]
    if hasattr(m, "set_params"):
        try: m.set_params(n_jobs=4)
        except Exception: pass
def base_probs(urls):
    X = pd.DataFrame([extract(u) for u in urls])[FEATURES].astype("float32").values
    return {k: a["calibrator"].predict(a["model"].predict_proba(X)[:, 1]) for k, a in arts.items()}

# ---------- samples
te = pd.read_csv("Dataset_Hybrid.csv"); te = te[te.partition == "test"]
rs = np.random.RandomState(SEED)
mal = te[te.Label == 1].sample(N_STALE_MAL, random_state=SEED); ben = te[te.Label == 0].sample(N_STALE_BEN, random_state=SEED)
rows = [{"set": "stale", "label": int(r.Label), "raw_url": (r.url_raw if r.scheme_added else r.url), "dataset_url": r.url} for r in pd.concat([mal, ben]).itertuples()]
op = urllib.request.urlopen(urllib.request.Request("https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt", headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read().decode().split()
op = [u for u in op if u.startswith("http")]; op = list(np.random.RandomState(SEED).permutation(op)[:N_LIVE])
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(urllib.request.Request("https://tranco-list.eu/top-1m.csv.zip", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()))
tr = pd.read_csv(z.open(z.namelist()[0]), header=None, names=["rank", "domain"], nrows=100000)
tr = tr.sample(N_LIVE, random_state=SEED)
rows += [{"set": "fresh", "label": 1, "raw_url": u, "dataset_url": u, "source": "openphish"} for u in op]
rows += [{"set": "fresh", "label": 0, "raw_url": f"https://{r.domain}/", "dataset_url": f"https://{r.domain}/", "source": f"tranco_rank_{r.rank}"} for r in tr.itertuples()]
print("samples:", pd.DataFrame(rows).groupby(["set", "label"]).size().to_dict(), flush=True)

# ---------- base probability adapter (precomputed -> thread-safe lookup)
canon = {}
def canonical(raw):
    n = normalize_url(raw); p = __import__("urllib.parse", fromlist=["x"]).urlparse(n)
    return f"{p.scheme}://{p.netloc}/" if p.scheme and p.netloc and p.path == "" else n
urls = [canonical(r["raw_url"]) for r in rows]
BP = base_probs(urls)
class Adapter:
    def __init__(s, name): s.name = name; s.idx = {u: i for i, u in enumerate(urls)}
    def predict(s, url):
        p = float(BP[s.name][s.idx[url]]); return extract_url_features(url), int(p >= 0.5), p
    def rank_features(s, f, top_k=8): return []
cache = MemCache(); T0 = time.time(); done = [0]
def run(i):
    r = rows[i]; out = {}
    for name in models:
        try: res = UrlAnalyzerService(Adapter(name), cache).analyze(r["raw_url"])
        except Exception as e:
            out[name] = {"error": repr(e)[:200]}; continue
        res["checks"].get("http", {}).pop("content_preview", None); res["checks"].get("http", {}).pop("headers", None)
        out[name] = {k: res[k] for k in ["base_malicious_probability", "final_malicious_probability", "final_prediction", "fusion_reasons", "policy", "checks"]}
        if name != "LightGBM": out[name]["checks"] = None            # network checks are model-independent (cached)
    done[0] += 1
    if done[0] % 50 == 0: print(f"[{(time.time()-T0)/60:5.1f} min] {done[0]}/{len(rows)}", flush=True)
    return {**r, "results": out}
with ThreadPoolExecutor(WORKERS) as ex: out = list(ex.map(run, range(len(rows))))
json.dump({"collected_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": out}, open("artifacts/e4_hybrid/e4_raw.json", "w"), default=str)
print("DONE", (time.time() - T0) / 60, flush=True)
