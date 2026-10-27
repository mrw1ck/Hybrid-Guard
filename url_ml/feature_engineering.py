"""Static lexical features for Model 1 (Table 3.5, 25 features).

URLs are treated strictly as inert strings: nothing here resolves, fetches or opens a URL.
tldextract runs fully offline on its bundled Public Suffix List snapshot.
"""
import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qsl, unquote, urlsplit

import tldextract

LISTS_DIR = Path(__file__).resolve().parent / "lists"

FEATURES: List[str] = [
    "URLLength", "HostnameLength", "PathLength", "QueryLength", "NoOfDots", "NoOfHyphens",
    "NoOfDigits", "NoOfSlashes", "NoOfQueryParams",
    "DigitRatio", "SpecialCharRatio", "NoOfObfuscatedChar", "HostEntropy", "PathEntropy",
    "HasAtSymbol",
    "IsIPHost", "SubdomainDepth", "IsSuspiciousTLD", "IsPunycode", "HasHTTPS", "HasNonStandardPort",
    "NoOfSuspiciousKeywords", "HasBrandInSubdomainOrPath",
    "IsShortener", "HasEmbeddedURL",
]

# Alias for backward compatibility across existing scripts
NUMERIC_FEATURES: List[str] = FEATURES

# Offline extractor: empty suffix_list_urls + no cache dir -> bundled snapshot only, never downloads.
TLD_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    cache_dir=None,
    include_psl_private_domains=True,
)


def _load(name: str) -> List[str]:
    list_path = LISTS_DIR / name
    if not list_path.exists():
        return []
    lines = list_path.read_text(encoding="utf-8").splitlines()
    return [l.strip().lower() for l in lines if l.strip() and not l.startswith("#")]


KEYWORDS: List[str] = _load("keywords.txt")
BRANDS: List[str] = _load("brands.txt")
SUSPICIOUS_TLDS = set(_load("suspicious_tlds.txt"))
SHORTENERS = set(_load("shorteners.txt"))

KEYWORD_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(KEYWORDS, key=len, reverse=True)),
    re.IGNORECASE,
)
# Short brands (ups, dhl, aol, irs, ...) only as whole alphanumeric tokens, to avoid "groups" -> ups.
BRAND_RE = re.compile(
    "|".join(
        re.escape(b) if len(b) >= 5 else r"(?<![a-z0-9])" + re.escape(b) + r"(?![a-z0-9])"
        for b in BRANDS
    ),
    re.IGNORECASE,
)
OBFUSCATED_RE = re.compile(r"%[0-9A-Fa-f]{2}")
EMBEDDED_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
SPECIAL_CHARS = set("@~_%=&?!*,;$+")


def normalize_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        return f"http://{url}"
    return url


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def registered_domain(host: str) -> str:
    """eTLD+1 with private PSL domains; IP hosts and hosts without a known suffix use the host."""
    if is_ip(host):
        return host
    ext = TLD_EXTRACT(host)
    rd = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain
    return rd or host


def extract_static_features(url: str, original_host: str = None, scheme_added: bool = False) -> Dict[str, Any]:
    url_str = (url or "").strip()
    has_scheme = "://" in url_str
    parsed_target = url_str if has_scheme else f"http://{url_str}"

    parts = urlsplit(parsed_target)
    host = (parts.hostname or "").lower().strip(".")
    path, query = parts.path or "", parts.query or ""
    n = len(url_str)
    digits = sum(ch.isdigit() for ch in url_str)
    ip = is_ip(host)

    if ip:
        sub, reg = "", host
    else:
        ext = TLD_EXTRACT(host)
        sub = ext.subdomain or ""
        reg = getattr(ext, "top_domain_under_public_suffix", None) or ext.registered_domain or host
    tld = "" if ip else host.rsplit(".", 1)[-1]

    try:
        port = parts.port
    except ValueError:
        port = None

    # Everything outside the registered domain: subdomain labels, path, query.
    outside = (sub + " " + path + " " + query).lower()
    query_values = [v for _, v in parse_qsl(query, keep_blank_values=True)]
    embedded_src = unquote(path) + " " + " ".join(unquote(v) for v in query_values)
    orig = original_host if original_host is not None else host

    host_ent = entropy(host)
    path_ent = entropy(f"{path}?{query}" if query else path)
    special_cnt = sum(ch in SPECIAL_CHARS for ch in url_str)
    sub_depth = len([p for p in sub.split(".") if p]) if sub else 0

    features = {
        "URLLength": n,
        "HostnameLength": len(host),
        "PathLength": len(path),
        "QueryLength": len(query),
        "NoOfDots": url_str.count("."),
        "NoOfHyphens": url_str.count("-"),
        "NoOfDigits": digits,
        "NoOfSlashes": path.count("/"),
        "NoOfQueryParams": len(parse_qsl(query, keep_blank_values=True)),
        "DigitRatio": digits / n if n else 0.0,
        "SpecialCharRatio": special_cnt / n if n else 0.0,
        "NoOfObfuscatedChar": len(OBFUSCATED_RE.findall(url_str)),
        "HostEntropy": host_ent,
        "PathEntropy": path_ent,
        "HasAtSymbol": int("@" in (parts.netloc or "") or "@" in path),
        "IsIPHost": int(ip),
        "SubdomainDepth": sub_depth,
        "IsSuspiciousTLD": int(tld in SUSPICIOUS_TLDS),
        "IsPunycode": int(any(l.startswith("xn--") for l in host.split("."))
                          or any(ord(c) > 127 for c in orig)),
        "HasHTTPS": int(parts.scheme.lower() == "https" and has_scheme and not scheme_added),
        "HasNonStandardPort": int(port is not None and port not in (80, 443)),
        "NoOfSuspiciousKeywords": len(KEYWORD_RE.findall((host + " " + path + " " + query).lower())),
        "HasBrandInSubdomainOrPath": int(bool(BRAND_RE.search(outside))),
        "IsShortener": int(reg in SHORTENERS or host in SHORTENERS),
        "HasEmbeddedURL": int(bool(EMBEDDED_RE.search(embedded_src))),

        # Backward-compatible convenience aliases for risk policies and heuristics
        "url": url_str,
        "dom": host,
        "tld": tld,
        "is_https": int(parts.scheme.lower() == "https" and has_scheme and not scheme_added),
        "subdom_cnt": sub_depth,
        "entropy": host_ent,
        "path_len": len(path),
        "query_len": len(query),
        "is_ip": int(ip),
        "spec_ratio": special_cnt / n if n else 0.0,
        "special_cnt": special_cnt,
        "digit_ratio": digits / n if n else 0.0,
        "url_len": n,
    }

    return features


def extract_url_features(url: str) -> Dict[str, Any]:
    """Drop-in alias for extract_static_features to preserve compatibility."""
    return extract_static_features(url)