from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


LGB_MODEL_PATH = _first_existing(
    ARTIFACTS_DIR / "lgb_final" / "url_phishing_model.joblib",
    ARTIFACTS_DIR / "lgb_model" / "url_phishing_model.joblib",
)
RF_MODEL_PATH = _first_existing(
    ARTIFACTS_DIR / "rf_final" / "url_phishing_model.joblib",
    ARTIFACTS_DIR / "url_phishing_model.joblib",
)
DT_MODEL_PATH = _first_existing(
    ARTIFACTS_DIR / "dt_final" / "url_phishing_model.joblib",
    ARTIFACTS_DIR / "dt_model" / "url_phishing_model.joblib",
)
MODEL_PATH = LGB_MODEL_PATH
CACHE_PATH = ARTIFACTS_DIR / "web_external_cache.json"

MODEL_PATHS = {
    "lgb": LGB_MODEL_PATH,
    "rf": RF_MODEL_PATH,
    "dt": DT_MODEL_PATH,
}

DEFAULT_MODEL_MODE = "lgb"

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_CACHE_HOURS = 24
DEFAULT_MAX_CONTENT_BYTES = 4096

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
