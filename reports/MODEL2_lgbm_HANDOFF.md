# MODEL2_lgbm — HANDOFF REPORT
Generated: 2026-11-22T19:16:02.104338+00:00

## Model Identity
- **Model ID**: model2_lgbm
- **Feature set**: 19 URL lexical features (no QR features)
- **Algorithm**: LightGBM (gradient boosting)
- **Calibration**: Isotonic Regression on OOF predictions

## Dataset
- Total rows: 650,770
- Train: 520,616 | Test: 130,154
- Train positives: 208,246 | Train negatives: 312,370
- Shared domains (train/test): 0 ✓

## Cross-Validation (5-fold StratifiedGroupKFold)
| Fold | PR-AUC | ROC-AUC |
|------|--------|---------|
| 0 | 0.9660 | 0.9708 |
| 1 | 0.9673 | 0.9716 |
| 2 | 0.9663 | 0.9693 |
| 3 | 0.9671 | 0.9717 |
| 4 | 0.9609 | 0.9663 |
| **Mean** | **0.9655 ± 0.0024** | **0.9699 ± 0.0020** |

## Test Set Evaluation (locked after FROZEN.json)
| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.9644 | [0.9526, 0.9734] |
| PR-AUC  | 0.9602  | [0.9410, 0.9720] |
| ECE     | 0.0205 | — |

## Thresholds
| Threshold | Value | FPR | FNR |
|-----------|-------|-----|-----|
| tau_high (FPR≤1%) | 0.8267 | 0.0080 | 0.2205 |
| tau_low  (FNR≤2%) | 0.0631  | 0.4383  | 0.0167  |

## Latency
| Metric | Value |
|--------|-------|
| Mean   | 1488.7 µs/URL |
| P50    | 1457.3 µs/URL |
| P95    | 1919.9 µs/URL |
| P99    | 2109.4 µs/URL |

## Reference Models
| Model | PR-AUC |
|-------|--------|
| Majority class baseline | 0.4000 |
| Depth-2 decision tree   | 0.7405 |
| **LightGBM (Model 2)**  | **0.9602** |

## Best Hyperparameters
```json
{
  "n_estimators": 857,
  "num_leaves": 257,
  "learning_rate": 0.04408991842576282,
  "feature_fraction": 0.6798015797304817,
  "bagging_fraction": 0.8382451265278859,
  "min_child_samples": 25,
  "max_depth": 11,
  "reg_alpha": 2.1020120126802353,
  "reg_lambda": 0.03938811000272599
}
```

## Output Files
| File | Description |
|------|-------------|
| `models/model2_lgbm.txt` | LightGBM booster (text format) |
| `models/model2_lgbm_calibrator.joblib` | Isotonic calibrator |
| `artifacts/FROZEN.json` | Test-lock checkpoint |
| `predictions/model2_lgbm_oof_train.parquet` | OOF predictions (train) |
| `predictions/model2_lgbm_test.parquet` | Test predictions |
| `results/model2_lgbm.json` | All metrics (machine-readable) |
