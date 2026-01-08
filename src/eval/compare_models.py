"""src.eval.compare_models

Infrastructure-only comparison runner.

It compares two saved forecast artifacts (NPZ) using shared metrics.
No baseline model code is required.

Example:
  py -3.12 src/eval/compare_models.py \
      --baseline reports/evaluation/baseline/run_x/forecasts.npz \
      --vit reports/evaluation/vit/run_y/forecasts.npz \
      --out experiments/comparison_summary/compare_run_x_vs_run_y.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os

import numpy as np

from src.eval.artifacts import load_forecast_npz
from src.eval.metrics import per_channel_mae, per_channel_rmse, summary_table


def _md_table_row(name: str, d: dict) -> str:
    return (
        f"| {name} | {d.get('rmse', float('nan')):.4f} | {d.get('mae', float('nan')):.4f} | "
        f"{d.get('ssim', float('nan')):.4f} | {d.get('csi', float('nan')):.4f} | "
        f"{d.get('pod', float('nan')):.4f} | {d.get('far', float('nan')):.4f} |\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, default="", help="Path to baseline forecasts.npz")
    ap.add_argument("--vit", type=str, required=True, help="Path to ViT forecasts.npz")
    ap.add_argument("--out", type=str, required=True, help="Output markdown file")
    ap.add_argument("--event-threshold", type=float, default=0.5)
    ap.add_argument("--channel-axis", type=int, default=-3)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    vit = load_forecast_npz(args.vit)
    vit_sum = summary_table(vit.pred, vit.target, threshold=args.event_threshold)
    vit_pcm = per_channel_mae(vit.pred, vit.target, channel_axis=args.channel_axis)
    vit_pcr = per_channel_rmse(vit.pred, vit.target, channel_axis=args.channel_axis)

    baseline_sum = None
    baseline_pcm = None
    baseline_pcr = None
    if args.baseline:
        base = load_forecast_npz(args.baseline)
        baseline_sum = summary_table(base.pred, base.target, threshold=args.event_threshold)
        baseline_pcm = per_channel_mae(base.pred, base.target, channel_axis=args.channel_axis)
        baseline_pcr = per_channel_rmse(base.pred, base.target, channel_axis=args.channel_axis)

    md = []
    md.append("# Model Comparison\n")
    md.append("\n")
    md.append(f"- Event threshold: `{args.event_threshold}`\n")
    md.append(f"- ViT artifact: `{_norm(args.vit)}`\n")
    if args.baseline:
        md.append(f"- Baseline artifact: `{_norm(args.baseline)}`\n")
    else:
        md.append("- Baseline artifact: *(not provided yet)*\n")

    md.append("\n## Summary Metrics\n\n")
    md.append("| Model | RMSE ↓ | MAE ↓ | SSIM ↑ | CSI ↑ | POD ↑ | FAR ↓ |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    if baseline_sum is not None:
        md.append(_md_table_row("Baseline", baseline_sum))
    md.append(_md_table_row("ViT", vit_sum))

    md.append("\n## Per-Channel Metrics\n\n")
    md.append("Per-channel arrays are ordered by channel axis in the stored tensor (default assumes `(B,T,C,H,W)`).\n\n")
    if baseline_pcm is not None:
        md.append(f"- Baseline MAE per channel: `{np.array2string(baseline_pcm, precision=4)}`\n")
        md.append(f"- Baseline RMSE per channel: `{np.array2string(baseline_pcr, precision=4)}`\n")
    md.append(f"- ViT MAE per channel: `{np.array2string(vit_pcm, precision=4)}`\n")
    md.append(f"- ViT RMSE per channel: `{np.array2string(vit_pcr, precision=4)}`\n")

    out_path.write_text("".join(md), encoding="utf-8")
    try:
        print(f"Wrote {os.path.relpath(out_path)}")
    except Exception:
        print("Wrote output markdown")


if __name__ == "__main__":
    main()
