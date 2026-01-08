# Model Card: ViT Nowcasting

## Model Overview
| Property | Value |
|----------|-------|
| **Name** | ViT Nowcasting |
| **Version** | 1.0 |
| **Implementation** | `src/models/vit_nowcasting.py` |
| **Task** | Multi-step, multi-channel weather nowcasting |
| **Inputs** | Tensor `(B, T_in, C, H, W)` |
| **Outputs** | Tensor `(B, T_out, C, H, W)` |

## Architecture Summary

```
Input: (B, T_in=4, C=4, H=256, W=256)
    ↓
PatchEmbed (Conv2d, kernel=stride=patch_size)
    ↓
Spatial Transformer Blocks × depth
    ↓
Temporal Fusion (Attention or Conv)
    ↓
Linear Head + Unpatchify
    ↓
Output: (B, T_out=2, C=4, H=256, W=256)
```

See [docs/VIT_ARCHITECTURE.md](../../docs/VIT_ARCHITECTURE.md) for detailed diagram.

## Architecture Variants

| Variant | Patch | Embed | Depth | Heads | Params | Use Case |
|---------|-------|-------|-------|-------|--------|----------|
| Small | 32 | 128 | 2 | 4 | ~300K | Fast prototyping |
| Base | 16 | 256 | 4 | 8 | ~1M | Balanced |
| Large | 8 | 384 | 6 | 12 | ~5M | High accuracy |
| Deep | 16 | 256 | 8 | 8 | ~2M | Complex patterns |

Configs available in `configs/ablation/`.

## Intended Use
- **Primary**: Short-horizon nowcasting (0-2 hours) for satellite/weather data
- **Input Data**: Preprocessed gridded data from EUMETSAT SEVIRI + auxiliary wind fields
- **Output**: Deterministic forecasts (no uncertainty quantification)

## Data Requirements
- **Format**: `data/combined/combined_*.npz` with key `data: (C, H, W)`
- **Channels**: 4 channels (satellite, wind_u, wind_v, wind_speed)
- **Temporal**: 15-minute cadence, T_in=4 input frames, T_out=2 output frames
- **Resolution**: 256×256 pixels (configurable)

## Training

### Entry Points
```bash
# Standard training
py -3.12 -m src.train.train_vit --config configs/vit_config_l2.yaml

# Ablation studies
py -3.12 -m src.train.run_ablation --configs configs/ablation/*.yaml
```

### Training Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| optimizer | AdamW | Adam, AdamW, Lion, SGD |
| lr | 0.001 | Learning rate |
| warmup_steps | 100 | Linear warmup steps |
| weight_decay | 0.0001 | L2 regularization |
| grad_clip | 1.0 | Gradient clipping norm |
| amp | true | Mixed precision (CUDA) |

### LR Schedule
- Linear warmup → Cosine decay to `min_lr`

## Evaluation Metrics
| Metric | Type | Description |
|--------|------|-------------|
| MAE | Regression | Mean Absolute Error |
| RMSE | Regression | Root Mean Squared Error |
| SSIM | Perceptual | Structural Similarity |
| CSI | Event | Critical Success Index |
| POD | Event | Probability of Detection |
| FAR | Event | False Alarm Rate |

Implementation: `src/eval/metrics.py`

## Artifacts
| Artifact | Location |
|----------|----------|
| Checkpoints | `models/vit/<run_id>/` |
| Metrics | `experiments/vit_nowcasting/<run_id>/metrics.jsonl` |
| Forecasts | `reports/evaluation/vit/<run_id>/forecasts.npz` |
| Config | `experiments/vit_nowcasting/<run_id>/config.yaml` |

## Interpretability
- **Spatial Attention**: Per-block attention weights showing which patches the model focuses on
- **Temporal Attention**: Weights showing how the model aggregates information across input timesteps
- **Visualization**: `src/eval/visualize.py` provides attention heatmap functions

## Limitations
1. **Over-smoothing**: MSE loss can cause regression-to-mean, blurring fine details
2. **Single-scale**: Fixed patch size may miss multi-scale features
3. **No Uncertainty**: Deterministic output, no probabilistic forecasts
4. **Data Requirements**: Needs contiguous time series with consistent cadence
5. **Domain Shift**: Performance may degrade with different satellite products or regions

## Known Failure Modes
- Attention maps can be misleading without proper aggregation
- Sensitivity to missing frames / cadence gaps
- Over-confident predictions during rapid weather changes
- Poor performance on rare/extreme events

## Compute Requirements

| Configuration | GPU Memory | CPU Memory | Inference Time |
|---------------|------------|------------|----------------|
| Small (P=32) | ~1 GB | ~2 GB | ~5 ms/sample |
| Base (P=16) | ~2 GB | ~4 GB | ~15 ms/sample |
| Large (P=8) | ~6 GB | ~8 GB | ~50 ms/sample |

*Measured on synthetic data, actual times may vary.*

## Reproducibility
- Random seed: 42 (default)
- Git commit hash logged in `meta.json`
- Full config saved with each run
- Dataset fingerprint tracked

## Citation
```bibtex
@software{vit_nowcasting,
  title = {Vision Transformer for Weather Nowcasting},
  year = {2026},
  url = {github.com/morocco-weather-nowcasting}
}
```
