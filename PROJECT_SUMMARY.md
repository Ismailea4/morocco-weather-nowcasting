# Morocco Weather Nowcasting - Project Summary

> **Vision Transformer (ViT) for Weather Nowcasting**  
> Completed: Phase 1 (Notebook Prototyping) & Phase 2 (Production Code)

---

## 📋 Overview

This project implements a **Vision Transformer (ViT)** model for weather nowcasting over Morocco, predicting future weather frames from historical satellite/reanalysis data.

### Key Features
- **ViT Architecture** with temporal attention fusion
- **4-channel input** (multi-variable weather data)
- **Multi-step forecasting** (T_in=4 → T_out=2)
- **6 evaluation metrics**: MAE, RMSE, SSIM, CSI, POD, FAR

---

## 📁 Project Structure

```
morocco-weather-nowcasting/
├── configs/
│   ├── vit_config_l2.yaml        # Main training config (30 epochs)
│   ├── vit_config_smoke.yaml     # Quick smoke test config
│   └── ablation/                 # Ablation study configs
├── data/
│   └── combined/                 # NPZ files (combined_*.npz)
├── src/
│   ├── models/
│   │   └── vit_nowcasting.py     # ViT model architecture
│   ├── train/
│   │   ├── train_vit.py          # CLI training script
│   │   └── run_ablation.py       # Ablation runner
│   ├── eval/
│   │   ├── metrics.py            # All 6 metrics (summary_table)
│   │   └── visualize.py          # Plotting utilities
│   └── datasets/
│       └── morocco_dataset.py    # Data loading
├── models/
│   └── vit/
│       └── vit_30ep/             # ✅ Best trained model
│           ├── best.pt           # Best checkpoint (epoch 18)
│           └── last.pt           # Final checkpoint (epoch 30)
├── experiments/
│   └── vit_nowcasting/
│       └── vit_30ep/             # Training logs & artifacts
│           ├── config.yaml
│           ├── metrics.jsonl
│           └── meta.json
├── notebook/
│   └── 03_vit_advanced_prototype.ipynb  # Exploration notebook
└── docs/
    ├── VIT_ARCHITECTURE.md       # Architecture documentation
    └── ...
```

---

## 🏆 Best Model

| Property | Value |
|----------|-------|
| **Location** | `models/vit/vit_30ep/best.pt` |
| **Best Epoch** | 18 |
| **Val Loss** | 65.36 |
| **Parameters** | ~5.3M |

### Final Metrics (Best Checkpoint)

| Metric | Value |
|--------|-------|
| **MAE** | 6.01 |
| **RMSE** | 8.31 |
| **SSIM** | 0.622 |
| **CSI** | 0.627 |
| **POD** | 0.975 |
| **FAR** | 0.363 |

---

## 🚀 Commands Used

### 1. Smoke Test (Verify Setup)
```bash
py -3.12 -m src.train.train_vit --config configs/vit_config_smoke.yaml
```

### 2. Ablation Dry Run
```bash
py -3.12 -m src.train.run_ablation --configs "configs/ablation/*.yaml" --dry-run
```

### 3. Full Training (30 Epochs)
```bash
py -3.12 -m src.train.train_vit --config configs/vit_config_l2.yaml --run-id vit_30ep
```

### 4. Check Training Metrics
```bash
type "experiments\vit_nowcasting\vit_30ep\metrics.jsonl"
```

---

## ⚙️ Training Configuration

From `configs/vit_config_l2.yaml`:

```yaml
model:
  embed_dim: 256
  depth: 6
  num_heads: 8
  patch_size: 16
  mlp_ratio: 4.0
  dropout: 0.1
  temporal_fusion: attention

training:
  epochs: 30
  batch_size: 2
  lr: 0.0005
  weight_decay: 0.0001
  warmup_steps: 50
  total_steps: 6000
  grad_clip: 1.0

data:
  t_in: 4
  t_out: 2
  image_size: 256
  train_split: 0.8
```

---

## 📊 Training Progress

The model converged well over 30 epochs:

| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 1 | 20,768 | 20,768 | Initial |
| 5 | 1,247 | 1,232 | Rapid improvement |
| 10 | 152.5 | 139.4 | Stabilizing |
| 18 | 72.1 | **65.36** | ⭐ Best |
| 30 | 55.8 | 68.4 | Slight overfit |

---

## 🔬 Model Architecture

**ViTNowcaster** - Vision Transformer adapted for spatiotemporal forecasting:

1. **Patch Embedding**: Conv2d projects 16×16 patches to 256-dim tokens
2. **Positional Encoding**: Learnable position embeddings (256 patches)
3. **Transformer Blocks** (×6): LayerNorm → MHA → LayerNorm → MLP
4. **Temporal Fusion**: Cross-frame attention aggregates T_in frames
5. **Prediction Head**: Linear projects to T_out × C × P² output patches

```
Input: (B, 4, 4, 256, 256)  →  Output: (B, 2, 4, 256, 256)
       [batch, T_in, C, H, W]         [batch, T_out, C, H, W]
```

---

## 📈 Evaluation Metrics

All metrics computed via `src/eval/metrics.py`:

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error (pixel-level) |
| **RMSE** | Root Mean Square Error |
| **SSIM** | Structural Similarity Index |
| **CSI** | Critical Success Index (event-based) |
| **POD** | Probability of Detection |
| **FAR** | False Alarm Rate |

---

## 📓 Notebook Exploration

The prototype notebook `notebook/03_vit_advanced_prototype.ipynb` includes:

1. ✅ Data loading & patch extraction visualization
2. ✅ ViT architecture design (inline)
3. ✅ Temporal fusion exploration (attention vs conv)
4. ✅ Hyperparameter search (embed_dim, depth, heads, patch_size, lr)
5. ✅ Training loop with LR scheduling
6. ✅ Baseline comparison (Persistence, dummy ConvLSTM)
7. ✅ Attention map visualizations (spatial & temporal)
8. ✅ 3-way prediction comparisons
9. ✅ Error analysis
10. ✅ Animated GIF generation

---

## 🛠️ Environment

- **Python**: 3.12
- **PyTorch**: with CUDA support
- **Key Dependencies**: numpy, pandas, matplotlib, seaborn, imageio, scipy

---

## 📝 Phases Completed

### Phase 1: Notebook Prototyping ✅
- Explored ViT architecture variants
- Ran hyperparameter search
- Visualized attention patterns
- Compared against baselines

### Phase 2: Production Code ✅
- Modular `src/` structure
- Config-driven training (`train_vit.py`)
- Ablation study runner
- All 6 metrics implemented
- Model checkpointing

### Phase 3: Reproducibility & Packaging
- 🔲 Coming next...

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `src/models/vit_nowcasting.py` | Model definition |
| `src/train/train_vit.py` | Training CLI |
| `src/eval/metrics.py` | Evaluation metrics |
| `configs/vit_config_l2.yaml` | Main config |
| `models/vit/vit_30ep/best.pt` | Best checkpoint |
| `experiments/vit_nowcasting/vit_30ep/metrics.jsonl` | Training logs |

---

## 🎯 Results Summary

The ViT model significantly outperforms the persistence baseline:

- **~97.5% detection rate** (POD) for weather events
- **63% CSI** indicating good event prediction
- **SSIM of 0.62** showing structural preservation
- Converged to stable validation loss by epoch 18

---

*Generated: January 8, 2026*
