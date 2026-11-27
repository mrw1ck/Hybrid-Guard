"""checker_model2.py — verify every number in handoff against results JSON."""
import json, sys
from pathlib import Path

BASE = Path(__file__).parent
results_path = BASE / "results" / "model2_lgbm.json"
handoff_path = BASE / "reports" / "MODEL2_lgbm_HANDOFF.md"

r = json.loads(results_path.read_text(encoding="utf-8"))
handoff = handoff_path.read_text(encoding="utf-8")

errors = []

def check(name, value, text, fmt=".4f"):
    s = format(value, fmt)
    if s not in text:
        errors.append(f"FAIL: {name}={s} not found in handoff")
    else:
        print(f"  OK: {name}={s}")

print("Checking results vs handoff …")
check("test.roc_auc",  r["test"]["roc_auc"],  handoff)
check("test.pr_auc",   r["test"]["pr_auc"],   handoff)
check("test.ece",      r["test"]["ece"],       handoff)
check("tau_high", r["thresholds"]["tau_high"], handoff)
check("tau_low",  r["thresholds"]["tau_low"],  handoff)
check("latency_us_mean", r["latency"]["latency_us_mean"], handoff, ".1f")

# Check FROZEN.json exists and predates test results
frozen_path = BASE / "artifacts" / "FROZEN.json"
if not frozen_path.exists():
    errors.append("FAIL: artifacts/FROZEN.json does not exist")
else:
    frozen = json.loads(frozen_path.read_text())
    print(f"  OK: FROZEN.json exists (written {frozen.get('written_at','?')})")

# Check parquets
import pandas as pd
for name, path in [
    ("oof_train", BASE/"predictions"/"model2_lgbm_oof_train.parquet"),
    ("test",      BASE/"predictions"/"model2_lgbm_test.parquet"),
]:
    if not path.exists():
        errors.append(f"FAIL: {path} missing")
        continue
    df = pd.read_parquet(path)
    expected = ["url_sha256","registered_domain","partition","fold","label","p_raw","p_cal","risk_level"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"FAIL: {name} parquet missing cols: {missing}")
    else:
        print(f"  OK: {name} parquet schema correct ({len(df):,} rows)")

if errors:
    print("\n=== CHECKER FAILED ===")
    for e in errors: print(" ", e)
    sys.exit(1)
else:
    print("\n=== ALL CHECKS PASSED ===")
