"""Model service for the Chapter-3 models (artifacts/{lgb,xgb,rf}_thesis).

Differences from the legacy ModelService:
  * features come from url_ml.thesis_features (the 19 features the models were trained on);
  * the probability is the ISOTONIC-CALIBRATED score, not the raw model output;
  * the operating thresholds tau_high / tau_low are the frozen values stored in the artifact;
  * IsShortener / HasEmbeddedURL URLs can never be classified as benign (hard rule of Section 3.5.3).
The legacy 22-feature dictionary is merged in so risk_policy (which reads is_https, tld, subdom_cnt ...) keeps working.
"""
import joblib
import numpy as np
import pandas as pd

from url_ml.feature_engineering import extract_url_features
from url_ml.thesis_features import FEATURES, extract


class ThesisModelService:
    def __init__(self, model_path):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.calibrator = artifact["calibrator"]
        self.feature_names = list(artifact["feature_names"])
        if self.feature_names != FEATURES:
            raise RuntimeError("Artifact feature order differs from url_ml.thesis_features.FEATURES")
        self.tau_high = float(artifact["thresholds"]["tau_high"])
        self.tau_low = float(artifact["thresholds"]["tau_low"])
        # Section 3.5.3: Safe if score <= tau_low, so the Suspicious band starts just above tau_low (the isotonic
        # calibrator is a step function and many URLs sit exactly on tau_low).
        self.thresholds = {"malicious": self.tau_high, "suspicious": float(np.nextafter(self.tau_low, 1.0))}
        self.model_type = artifact.get("training", {}).get("model_type", "thesis")
        self.version = artifact.get("training", {}).get("version", "")
        # native importances are used only to rank features in the debug panel
        self.feature_stats = artifact.get("feature_stats")

    def _thesis_features(self, url):
        row = extract(url)
        return row, pd.DataFrame([{n: row[n] for n in self.feature_names}]).astype("float32")

    def predict(self, url: str):
        thesis, frame = self._thesis_features(url)
        raw = float(self.model.predict_proba(frame)[0][1])
        calibrated = float(self.calibrator.predict([raw])[0])
        legacy = extract_url_features(url)                      # keys used by risk_policy (is_https, tld, ...)
        features = {**legacy, **{f"model_{k}": v for k, v in thesis.items()}, "raw_model_probability": raw}
        features["hard_url"] = int(thesis["IsShortener"] == 1 or thesis["HasEmbeddedURL"] == 1)
        return features, int(calibrated >= self.tau_high), calibrated

    def rank_features(self, features: dict, top_k: int = 8) -> list[dict]:
        imp = getattr(self.model, "feature_importances_", None)
        imp = np.asarray(imp if imp is not None else np.zeros(len(self.feature_names)), float)
        ranked = []
        for name, w in zip(self.feature_names, imp):
            v = float(features.get(f"model_{name}", 0.0)); ranked.append({"feature": name, "value": v, "importance": float(w), "debug_score": float(w) * abs(v)})
        ranked.sort(key=lambda r: r["importance"], reverse=True)
        return ranked[:top_k]
