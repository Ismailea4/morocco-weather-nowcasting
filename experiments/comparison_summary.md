# Model Comparison

- Event threshold: `0.5`
- ViT artifact: `reports/evaluation/vit/vit_smoke_003/forecasts.npz`
- Baseline artifact: `reports/evaluation/baseline/persistence_vit_smoke_003/forecasts.npz`

## Summary Metrics

| Model | RMSE ↓ | MAE ↓ | SSIM ↑ | CSI ↑ | POD ↑ | FAR ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 4.6424 | 1.9909 | 0.8880 | 0.8279 | 0.9048 | 0.0930 |
| ViT | 141.0781 | 74.1187 | 0.3804 | 0.6195 | 0.9696 | 0.3682 |

## Per-Channel Metrics

Per-channel arrays are ordered by channel axis in the stored tensor (default assumes `(B,T,C,H,W)`).

- Baseline MAE per channel: `[2.5124 2.2328 0.7441 2.4742]`
- Baseline RMSE per channel: `[3.5426 5.8788 1.5588 6.0552]`
- ViT MAE per channel: `[281.7885   5.2381   3.7597   5.6882]`
- ViT RMSE per channel: `[281.9689   6.467    4.0592   6.882 ]`

## Statistical Significance

Paired bootstrap over per-sample MAE differences:

- Report: `experiments/significance_vit_smoke_003.md`
- Result: mean improvement = `-72.127792` (baseline_error - vit_error), 95% CI `[-72.351425, -71.904167]`, one-sided p = `1.000000`

## Computational Efficiency

- Report: `experiments/efficiency_vit_smoke_003.md`
- ViT params: `891,008` | checkpoint: `3.409 MB` | CPU latency (synthetic): `15.59 ms/sample` (mean)
