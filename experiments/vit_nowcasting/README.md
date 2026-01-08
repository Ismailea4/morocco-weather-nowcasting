# ViT Experiment Runs

Each run created by `src/train/train_vit.py` writes to:

- `experiments/vit_nowcasting/run_YYYYMMDD_HHMMSS/`
  - `config.yaml`
  - `meta.json`
  - `metrics.jsonl`
  - `artifacts.json`

Checkpoints are saved to:
- `models/vit/<run_id>/best.pt`
- `models/vit/<run_id>/last.pt`

Evaluation artifacts are saved to:
- `reports/evaluation/vit/<run_id>/forecasts.npz`
  - Keys: `pred`, `target`, `meta_json`
