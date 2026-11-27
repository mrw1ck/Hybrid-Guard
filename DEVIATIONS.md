# DEVIATIONS from test.md

| # | Section | Deviation | Reason |
|---|---------|-----------|--------|
| 1 | McNemar test | Deferred until all 3 models trained | Requires prediction pairs from LightGBM + RF/XGBoost |
| 2 | URL-length-only tree | Approximated with all 19 features depth-2 tree | URL length not a direct feature; HostRatio+PathRatio used as proxy |
