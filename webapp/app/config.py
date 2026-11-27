from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


# Chapter-3 models: grouped 5-fold CV, 50 Optuna trials, isotonic calibration, frozen tau_high/tau_low.
THESIS_MODEL_PATHS = {
    "lgb": ARTIFACTS_DIR / "lgb_thesis" / "url_phishing_model.joblib",
    "xgb": ARTIFACTS_DIR / "xgb_thesis" / "url_phishing_model.joblib",
    "rf": ARTIFACTS_DIR / "rf_thesis" / "url_phishing_model.joblib",
}
# Legacy artifacts (22 features, uncalibrated, trained on a random URL-level split). Only used if no thesis artifact exists.
LEGACY_MODEL_PATHS = {
    "lgb": _first_existing(ARTIFACTS_DIR / "lgb_final" / "url_phishing_model.joblib", ARTIFACTS_DIR / "lgb_model" / "url_phishing_model.joblib"),
    "rf": _first_existing(ARTIFACTS_DIR / "rf_final" / "url_phishing_model.joblib", ARTIFACTS_DIR / "url_phishing_model.joblib"),
}
MODEL_PATHS = {m: p for m, p in THESIS_MODEL_PATHS.items() if p.exists()} or {m: p for m, p in LEGACY_MODEL_PATHS.items() if p.exists()}
MODEL_PATH = next(iter(MODEL_PATHS.values()), None)
CACHE_PATH = ARTIFACTS_DIR / "web_external_cache.json"

# Best model by cross-validated PR-AUC (Chapter-3 protocol); the three models are statistically tied on the test set.
DEFAULT_MODEL_MODE = "lgb"

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_CACHE_HOURS = 24
DEFAULT_MAX_CONTENT_BYTES = 4096

# Fallback thresholds for legacy (uncalibrated) artifacts only. Thesis models carry their own frozen tau_high / tau_low.
MALICIOUS_THRESHOLD = 0.5
SUSPICIOUS_THRESHOLD = 0.35
UNCERTAIN_THRESHOLD = 0.18

RISKY_TLDS = {
    "rf.gd",
    "tk",
    "ga",
    "gq",
    "cf",
    "ml",
    "vip",
    "xyz",
    "top",
    "click",
    "work",
    "support",
    "zip",
    "mov",
    "country",
    "stream",
}

TRUSTED_TLDS = {
    "gov",
    "gov.vn",
    "edu",
    "edu.vn",
}

SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "password",
    "signin",
    "account",
    "bank",
    "wallet",
    "secure",
    "update",
    "confirm",
    "billing",
    "token",
    "otp",
    "payment",
    "invoice",
}

KNOWN_LEGIT_LOGIN_DOMAINS = {
    "facebook.com",
    "fb.com",
    "instagram.com",
    "messenger.com",
    "whatsapp.com",
    "google.com",
    "youtube.com",
    "youtu.be",
    "microsoft.com",
    "live.com",
    "outlook.com",
    "apple.com",
    "icloud.com",
    "amazon.com",
    "aws.amazon.com",
    "github.com",
    "gitlab.com",
    "paypal.com",
    "stripe.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "netflix.com",
    "wikipedia.org",
    "cloudflare.com",
    "yahoo.com",
    "tiktok.com",
    "zalo.me",
    "vnexpress.net",
}
