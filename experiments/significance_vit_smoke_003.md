# Statistical Significance (Paired Bootstrap)

This report uses paired bootstrap resampling over per-sample errors.

- ViT artifact: `reports\evaluation\vit\vit_smoke_003\forecasts.npz`
- Baseline artifact: `reports\evaluation\baseline\persistence_vit_smoke_003\forecasts.npz`

## Result

Improvement is defined as `baseline_error - vit_error` (positive means ViT is better).

| Metric | N samples | N bootstrap | Mean improvement | 95% CI | p (one-sided) |
|---|---:|---:|---:|---:|---:|
| MAE | 4 | 5000 | -72.127792 | [-72.351425, -71.904167] | 1.000000 |
