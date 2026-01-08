# Efficiency Report

- Checkpoint: `models/vit/vit_smoke_003/best.pt`
- Device: `cpu`

## Model Size

| Item | Value |
|---|---:|
| Parameters | 891008 |
| Checkpoint size (MB) | 3.409 |

## Inference Latency (Synthetic Input)

Benchmark uses random normal input with the configured tensor shape.

| Item | Value |
|---|---:|
| batch_size | 1 |
| n_warmup | 5 |
| n_iters | 20 |
| ms_per_batch_mean | 15.5884 |
| ms_per_batch_p50 | 15.7841 |
| ms_per_batch_p90 | 17.1762 |
| ms_per_batch_p99 | 18.1860 |
| ms_per_sample_mean | 15.5884 |

## Config Snapshot

```json
{
  "in_channels": 4,
  "out_channels": 4,
  "image_size": 256,
  "patch_size": 16,
  "embed_dim": 128,
  "depth": 2,
  "num_heads": 4,
  "mlp_ratio": 4.0,
  "dropout": 0.1,
  "attn_dropout": 0.0,
  "temporal_fusion": "attention",
  "temporal_heads": 2,
  "t_out": 2
}
```
