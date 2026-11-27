import re
"""Re-implementation of the 19 static features stored in Dataset.csv (the original extractor is not in the repo).
Validated against Dataset.csv by validate_extractor.py. Input: a full URL string with scheme (the 'url' column)."""
import math, re
from collections import Counter
from urllib.parse import urlsplit

FEATURES = ["HostRatio", "PathRatio", "NoOfDots", "NoOfHyphens", "NoOfDigits", "NoOfSlashes", "NoOfQueryParams", "DigitRatio",
            "SpecialCharRatio", "NoOfObfuscatedChar", "HostEntropy", "PathEntropy", "HasAtSymbol", "SubdomainRatio", "IsPunycode",
            "HasNonStandardPort", "HasBrandInSubdomainOrPath", "IsShortener", "HasEmbeddedURL"]
SPECIAL = set("!$&*+,;=?@_~")
SHORTENERS = {"tinyurl.com", "ow.ly", "bit.ly", "cutt.ly", "rebrand.ly", "t.co", "x.co", "rb.gy", "qrs.ly", "goo.gl", "linktr.ee", "lnkd.in",
              "s.id", "is.gd", "bitly.com", "www.cutt.ly", "qrco.de", "tiny.cc", "db.tt", "www.tiny.cc", "tr.im", "bc.vc", "urlz.fr", "bitly.ws"}
BRANDS = ["paypal", "amazon", "dropbox", "apple", "aol", "google", "itunes", "alibaba", "ebay", "dhl", "yahoo", "facebook", "chase", "adobe",
          "microsoft", "outlook", "wellsfargo", "visa", "bankofamerica", "hotmail", "santander", "hsbc", "gmail", "netflix", "icloud", "office365",
          "onedrive", "linkedin", "docusign", "twitter", "mastercard", "whatsapp", "fedex", "citibank", "sharepoint", "barclays", "amex", "walmart",
          "instagram", "steamcommunity", "usbank", "capitalone", "americanexpress", "coinbase", "binance", "blockchain", "metamask", "telegram"]
_BRAND_RE = re.compile("|".join(BRANDS))
_PCT = re.compile(r"%[0-9a-fA-F]{2}")
_EMB = re.compile(r"https?(?::|%3a)(?://|%2f%2f)", re.I)


def _ent(s):
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


try:
    import tldextract
    _TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)   # bundled snapshot, no network; private suffixes (workers.dev) on
    _TLD_PRIV = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
except Exception:                                                           # pragma: no cover
    _TLD = _TLD_PRIV = None
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _regdom(host):
    """Registered domain (eTLD+1). IPs are returned whole. Falls back to a coarse heuristic without tldextract."""
    if not host or _IPV4.match(host) or ":" in host: return host
    if _TLD_PRIV is not None:
        e = _TLD_PRIV(host)
        if e.domain and e.suffix: return f"{e.domain}.{e.suffix}"
        return host
    p = host.split(".")
    if len(p) <= 2: return host
    if len(p[-1]) == 2 and len(p[-2]) <= 3 and len(p) >= 3: return ".".join(p[-3:])
    return ".".join(p[-2:])


def extract(url, registered_domain=None):
    try:
        sp = urlsplit(url); host = (sp.hostname or "").lower(); port = sp.port
    except ValueError:
        sp = urlsplit("http://invalid/"); host = ""; port = None
    n = max(len(url), 1); path = sp.path; q = sp.query
    rd = registered_domain or _regdom(host)
    sub = host[:-len(rd) - 1] if host.endswith("." + rd) else ""
    text = (sub + " " + path).lower()
    rest = url.split("://", 1)[-1]
    nd = sum(c.isdigit() for c in url)
    return {"HostRatio": len(host) / n, "PathRatio": len(path) / n, "NoOfDots": url.count("."), "NoOfHyphens": url.count("-"),
            "NoOfDigits": nd, "NoOfSlashes": path.count("/"), "NoOfQueryParams": len([x for x in q.split("&") if x]) if q else 0,
            "DigitRatio": nd / n, "SpecialCharRatio": sum(c in SPECIAL for c in url) / n, "NoOfObfuscatedChar": len(_PCT.findall(url)),
            "HostEntropy": _ent(host), "PathEntropy": _ent(path + q), "HasAtSymbol": int("@" in sp.netloc or "@" in path),
            "SubdomainRatio": len(sub) / len(host) if host else 0.0, "IsPunycode": int("xn--" in host),
            "HasNonStandardPort": int(port not in (None, 80, 443)), "HasBrandInSubdomainOrPath": int(bool(_BRAND_RE.search(text))),
            "IsShortener": int(host in SHORTENERS), "HasEmbeddedURL": int(bool(_EMB.search(rest)))}
